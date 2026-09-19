use crate::job::Command;

fn check_length(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(format!("length mismatch: {} vs {}", a.len(), b.len()))
    }
}

fn check_matmul_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<(), String> {
    if a.len() != m * k {
        return Err(format!(
            "matmul: `a` has {} elements, expected m*k = {}",
            a.len(),
            m * k
        ));
    }
    if b.len() != k * n {
        return Err(format!(
            "matmul: `b` has {} elements, expected k*n = {}",
            b.len(),
            k * n
        ));
    }
    Ok(())
}

pub trait Backend {
    fn run(&mut self, command: &Command) -> Result<Vec<f32>, String>;

    /// Runs several commands and returns one result per input, in the same
    /// order. The default implementation just loops `run` — a backend only
    /// needs to override this if it can do better than one call per command
    /// (e.g. `CudaBackend`, which coalesces an all-`VectorAdd` batch into a
    /// single kernel launch).
    fn run_batch(&mut self, commands: &[&Command]) -> Vec<Result<Vec<f32>, String>> {
        commands.iter().map(|c| self.run(c)).collect()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn run(&mut self, command: &Command) -> Result<Vec<f32>, String> {
        match command {
            Command::VectorAdd { a, b } => {
                check_length(a, b)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            }
            Command::Reduce { a } => Ok(vec![a.iter().sum()]),
            Command::DotProduct { a, b } => {
                check_length(a, b)?;
                Ok(vec![a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()])
            }
            Command::MatMul { a, b, m, n, k } => {
                check_matmul_dims(a, b, *m, *n, *k)?;
                let mut c = vec![0.0f32; m * n];
                for row in 0..*m {
                    for col in 0..*n {
                        let mut sum = 0.0f32;
                        for kk in 0..*k {
                            sum += a[row * k + kk] * b[kk * n + col];
                        }
                        c[row * n + col] = sum;
                    }
                }
                Ok(c)
            }
        }
    }
}

use cust::prelude::*;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

// Field order matters here: Rust drops struct fields in declaration order
// (top to bottom), unlike C++'s reverse-declaration-order member destruction.
// `Context::drop` calls `cuDevicePrimaryCtxRelease`, `Module::drop` calls
// `cuModuleUnload`, and `Stream::drop` calls `cuStreamDestroy` — the latter
// two need the context to still be valid when they run. So `stream` and
// `module` are declared (and thus dropped) before `_ctx`, ensuring the
// context outlives everything that depends on it. Declaring `_ctx` first
// would release the context while `module`/`stream` still hold live CUDA
// driver handles into it, corrupting driver state (observed as intermittent
// `STATUS_ACCESS_VIOLATION` crashes during testing).
pub struct CudaBackend {
    stream: Stream,
    module: Module,
    _ctx: Context,
}

/// Must match `kernels/src/lib.rs`'s `REDUCE_BLOCK_SIZE` exactly — it sizes
/// the kernel's shared-memory array at compile time, and this value picks
/// the matching launch block size on the host. There's no way to share this
/// constant across the two crates (the host crate never links `kernels/` as
/// a Rust dependency, only consumes its compiled PTX via `build.rs`).
const REDUCE_BLOCK_SIZE: u32 = 256;

/// Must match `kernels/src/lib.rs`'s `MATMUL_TILE_SIZE` exactly — see
/// `REDUCE_BLOCK_SIZE`'s comment above for why this can't be shared across
/// crates directly.
const MATMUL_TILE_SIZE: u32 = 16;

impl CudaBackend {
    pub fn new(device: u32) -> Result<Self, String> {
        cust::init(CudaFlags::empty()).map_err(|e| format!("cust::init failed: {e}"))?;
        let dev = Device::get_device(device)
            .map_err(|e| format!("Device::get_device({device}) failed: {e}"))?;
        let ctx = Context::new(dev).map_err(|e| format!("Context::new failed: {e}"))?;
        ctx.set_flags(ContextFlags::SCHED_AUTO)
            .map_err(|e| format!("Context::set_flags failed: {e}"))?;
        let module =
            Module::from_ptx(PTX, &[]).map_err(|e| format!("Module::from_ptx failed: {e}"))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| format!("Stream::new failed: {e}"))?;
        Ok(Self {
            stream,
            module,
            _ctx: ctx,
        })
    }

    /// Uploads `a`/`b` (already known to be the same length), launches the
    /// `vector_add` kernel once over the whole buffer, and returns the
    /// result. Callers are responsible for length validation before calling
    /// this — it assumes `a.len() == b.len()`.
    fn launch_vector_add_flat(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let len = a.len();
        let a_gpu = a.as_dbuf().map_err(|e| format!("upload a failed: {e}"))?;
        let b_gpu = b.as_dbuf().map_err(|e| format!("upload b failed: {e}"))?;
        let mut out = vec![0.0f32; len];
        let out_gpu =
            DeviceBuffer::<f32>::zeroed(len).map_err(|e| format!("alloc out failed: {e}"))?;

        let function = self
            .module
            .get_function("vector_add")
            .map_err(|e| format!("get_function failed: {e}"))?;
        let (_, block_size) = function
            .suggested_launch_configuration(0, 0.into())
            .map_err(|e| format!("suggested_launch_configuration failed: {e}"))?;
        let grid_size = (len as u32).div_ceil(block_size);

        let stream = &self.stream;
        unsafe {
            launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    a_gpu.as_device_ptr(),
                    a_gpu.len(),
                    b_gpu.as_device_ptr(),
                    b_gpu.len(),
                    out_gpu.as_device_ptr(),
                )
            )
            .map_err(|e| format!("kernel launch failed: {e}"))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| format!("stream.synchronize failed: {e}"))?;
        out_gpu
            .copy_to(&mut out)
            .map_err(|e| format!("copy_to failed: {e}"))?;

        Ok(out)
    }

    /// Sums `a` via a block-level shared-memory tree reduction on the GPU
    /// (one partial sum per block), then finishes the (small) remaining sum
    /// on the host. Assumes `a` is non-empty — callers must short-circuit
    /// the empty case themselves (a zero-block launch is not meaningful).
    fn launch_reduce(&mut self, a: &[f32]) -> Result<f32, String> {
        let len = a.len();
        let a_gpu = a.as_dbuf().map_err(|e| format!("upload a failed: {e}"))?;

        let grid_size = (len as u32).div_ceil(REDUCE_BLOCK_SIZE);
        let partial_gpu = DeviceBuffer::<f32>::zeroed(grid_size as usize)
            .map_err(|e| format!("alloc partial_sums failed: {e}"))?;

        let function = self
            .module
            .get_function("reduce_sum")
            .map_err(|e| format!("get_function failed: {e}"))?;

        let stream = &self.stream;
        unsafe {
            launch!(
                function<<<grid_size, REDUCE_BLOCK_SIZE, 0, stream>>>(
                    a_gpu.as_device_ptr(),
                    a_gpu.len(),
                    partial_gpu.as_device_ptr(),
                )
            )
            .map_err(|e| format!("kernel launch failed: {e}"))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| format!("stream.synchronize failed: {e}"))?;

        let mut partials = vec![0.0f32; grid_size as usize];
        partial_gpu
            .copy_to(&mut partials)
            .map_err(|e| format!("copy_to failed: {e}"))?;

        Ok(partials.iter().sum())
    }

    /// Computes the dot product of `a` and `b` via the same shared-memory
    /// tree-reduction technique as `launch_reduce`, with an elementwise
    /// multiply folded into the per-thread value before reducing. Assumes
    /// `a.len() == b.len()` and both are non-empty — callers must validate
    /// length and short-circuit the empty case themselves.
    fn launch_dot_product(&mut self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        let len = a.len();
        let a_gpu = a.as_dbuf().map_err(|e| format!("upload a failed: {e}"))?;
        let b_gpu = b.as_dbuf().map_err(|e| format!("upload b failed: {e}"))?;

        let grid_size = (len as u32).div_ceil(REDUCE_BLOCK_SIZE);
        let partial_gpu = DeviceBuffer::<f32>::zeroed(grid_size as usize)
            .map_err(|e| format!("alloc partial_sums failed: {e}"))?;

        let function = self
            .module
            .get_function("dot_product")
            .map_err(|e| format!("get_function failed: {e}"))?;

        let stream = &self.stream;
        unsafe {
            launch!(
                function<<<grid_size, REDUCE_BLOCK_SIZE, 0, stream>>>(
                    a_gpu.as_device_ptr(),
                    a_gpu.len(),
                    b_gpu.as_device_ptr(),
                    b_gpu.len(),
                    partial_gpu.as_device_ptr(),
                )
            )
            .map_err(|e| format!("kernel launch failed: {e}"))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| format!("stream.synchronize failed: {e}"))?;

        let mut partials = vec![0.0f32; grid_size as usize];
        partial_gpu
            .copy_to(&mut partials)
            .map_err(|e| format!("copy_to failed: {e}"))?;

        Ok(partials.iter().sum())
    }

    /// Tiled `C = A * B` matmul. Assumes `a.len() == m*k` and `b.len() ==
    /// k*n` — callers must validate this themselves.
    fn launch_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let a_gpu = a.as_dbuf().map_err(|e| format!("upload a failed: {e}"))?;
        let b_gpu = b.as_dbuf().map_err(|e| format!("upload b failed: {e}"))?;
        let mut c = vec![0.0f32; m * n];
        let c_gpu = DeviceBuffer::<f32>::zeroed(m * n)
            .map_err(|e| format!("alloc c failed: {e}"))?;

        let function = self
            .module
            .get_function("matmul")
            .map_err(|e| format!("get_function failed: {e}"))?;

        let grid_size_x = (n as u32).div_ceil(MATMUL_TILE_SIZE);
        let grid_size_y = (m as u32).div_ceil(MATMUL_TILE_SIZE);

        let stream = &self.stream;
        unsafe {
            launch!(
                function<<<(grid_size_x, grid_size_y), (MATMUL_TILE_SIZE, MATMUL_TILE_SIZE), 0, stream>>>(
                    a_gpu.as_device_ptr(),
                    a_gpu.len(),
                    b_gpu.as_device_ptr(),
                    b_gpu.len(),
                    c_gpu.as_device_ptr(),
                    m,
                    n,
                    k,
                )
            )
            .map_err(|e| format!("kernel launch failed: {e}"))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| format!("stream.synchronize failed: {e}"))?;
        c_gpu
            .copy_to(&mut c)
            .map_err(|e| format!("copy_to failed: {e}"))?;

        Ok(c)
    }

    /// Coalesces every `VectorAdd` command in `commands` into a single
    /// kernel launch: concatenates all valid (length-matched, non-empty)
    /// pairs into one flat buffer, launches once, splits the result back by
    /// offset. Invalid (length-mismatched) commands get their own `Err`
    /// without touching the GPU; empty-but-valid pairs resolve to `Ok(vec![])`
    /// directly, never reaching the flat buffer. The match below is
    /// exhaustive over `Command`'s current variants; adding another variant
    /// will make it non-exhaustive and fail to *compile* here (not panic at
    /// runtime) until this function is updated to either handle it or be
    /// called only with `VectorAdd` commands, same as `run_batch` below
    /// already ensures.
    fn run_vector_add_batch(&mut self, commands: &[&Command]) -> Vec<Result<Vec<f32>, String>> {
        let pairs: Vec<(&[f32], &[f32])> = commands
            .iter()
            .map(|c| match c {
                Command::VectorAdd { a, b } => (a.as_slice(), b.as_slice()),
                Command::Reduce { .. } | Command::DotProduct { .. } | Command::MatMul { .. } => {
                    unreachable!("run_vector_add_batch is only ever called with VectorAdd commands")
                }
            })
            .collect();

        // Every index ends up `Some`: the loop below fills invalid indices
        // immediately, and the block after fills every remaining (valid)
        // index either from a successful launch or a shared launch error.
        let mut results: Vec<Option<Result<Vec<f32>, String>>> = vec![None; pairs.len()];
        let mut valid_indices = Vec::new();

        for (i, (a, b)) in pairs.iter().enumerate() {
            match check_length(a, b) {
                Err(e) => results[i] = Some(Err(e)),
                Ok(()) if a.is_empty() => results[i] = Some(Ok(Vec::new())),
                Ok(()) => valid_indices.push(i),
            }
        }

        if !valid_indices.is_empty() {
            let total_len: usize = valid_indices.iter().map(|&i| pairs[i].0.len()).sum();
            let mut flat_a = Vec::with_capacity(total_len);
            let mut flat_b = Vec::with_capacity(total_len);
            let mut offsets = Vec::with_capacity(valid_indices.len());
            for &i in &valid_indices {
                let (a, b) = pairs[i];
                offsets.push((flat_a.len(), a.len()));
                flat_a.extend_from_slice(a);
                flat_b.extend_from_slice(b);
            }

            match self.launch_vector_add_flat(&flat_a, &flat_b) {
                Ok(flat_out) => {
                    for (&i, &(offset, len)) in valid_indices.iter().zip(offsets.iter()) {
                        results[i] = Some(Ok(flat_out[offset..offset + len].to_vec()));
                    }
                }
                Err(e) => {
                    for &i in &valid_indices {
                        results[i] = Some(Err(e.clone()));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.expect("every job index must have been assigned a result"))
            .collect()
    }
}

impl Backend for CudaBackend {
    fn run(&mut self, command: &Command) -> Result<Vec<f32>, String> {
        match command {
            Command::VectorAdd { .. } => self
                .run_vector_add_batch(&[command])
                .into_iter()
                .next()
                .expect("run_vector_add_batch must return exactly one result per input"),
            Command::Reduce { a } => {
                if a.is_empty() {
                    return Ok(vec![0.0]);
                }
                self.launch_reduce(a).map(|v| vec![v])
            }
            Command::DotProduct { a, b } => {
                check_length(a, b)?;
                if a.is_empty() {
                    return Ok(vec![0.0]);
                }
                self.launch_dot_product(a, b).map(|v| vec![v])
            }
            Command::MatMul { a, b, m, n, k } => {
                check_matmul_dims(a, b, *m, *n, *k)?;
                self.launch_matmul(a, b, *m, *n, *k)
            }
        }
    }

    fn run_batch(&mut self, commands: &[&Command]) -> Vec<Result<Vec<f32>, String>> {
        let all_vector_add = !commands.is_empty()
            && commands
                .iter()
                .all(|c| matches!(c, Command::VectorAdd { .. }));

        if all_vector_add {
            self.run_vector_add_batch(commands)
        } else {
            commands.iter().map(|c| self.run(c)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_equal_length_vectors() {
        let mut backend = CpuBackend;
        let result = backend
            .run(&Command::VectorAdd {
                a: vec![1.0, 2.0, 3.0],
                b: vec![10.0, 20.0, 30.0],
            })
            .unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let mut backend = CpuBackend;
        let err = backend
            .run(&Command::VectorAdd {
                a: vec![1.0, 2.0],
                b: vec![1.0],
            })
            .unwrap_err();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn cuda_backend_new_accepts_device_zero() {
        CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");
    }

    #[test]
    fn cuda_backend_matches_cpu_backend() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new (requires an NVIDIA GPU + CUDA driver)");

        let command = Command::VectorAdd {
            a: vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            b: vec![10.0f32, 20.0, 30.0, 40.0, 50.0],
        };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cuda_backend_matches_cpu_backend_across_multiple_blocks() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new (requires an NVIDIA GPU + CUDA driver)");

        let len = 100_000;
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();
        let command = Command::VectorAdd { a, b };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cuda_backend_batch_matches_cpu_backend_per_job_with_different_lengths() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let commands = [
            Command::VectorAdd {
                a: vec![1.0, 2.0, 3.0],
                b: vec![10.0, 20.0, 30.0],
            },
            Command::VectorAdd {
                a: vec![1.0],
                b: vec![100.0],
            },
            Command::VectorAdd {
                a: vec![5.0, 6.0, 7.0, 8.0, 9.0],
                b: vec![0.5, 0.5, 0.5, 0.5, 0.5],
            },
        ];
        let command_refs: Vec<&Command> = commands.iter().collect();

        let results = cuda.run_batch(&command_refs);
        assert_eq!(results.len(), commands.len());

        for (command, result) in commands.iter().zip(results) {
            let expected = cpu.run(command).unwrap();
            assert_eq!(result.unwrap(), expected);
        }
    }

    #[test]
    fn cuda_backend_batch_isolates_a_length_mismatch_from_the_other_jobs() {
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let commands = [
            Command::VectorAdd {
                a: vec![1.0, 2.0],
                b: vec![10.0, 20.0],
            }, // valid
            Command::VectorAdd {
                a: vec![1.0, 2.0, 3.0],
                b: vec![1.0],
            }, // invalid: length mismatch
            Command::VectorAdd {
                a: vec![5.0],
                b: vec![50.0],
            }, // valid
        ];
        let command_refs: Vec<&Command> = commands.iter().collect();

        let results = cuda.run_batch(&command_refs);

        assert_eq!(results[0].as_ref().unwrap(), &vec![11.0, 22.0]);
        assert!(results[1].as_ref().unwrap_err().contains("length mismatch"));
        assert_eq!(results[2].as_ref().unwrap(), &vec![55.0]);
    }

    #[test]
    fn cuda_backend_batch_handles_an_empty_pair() {
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let commands = [
            Command::VectorAdd {
                a: vec![],
                b: vec![],
            },
            Command::VectorAdd {
                a: vec![1.0, 2.0],
                b: vec![10.0, 20.0],
            },
        ];
        let command_refs: Vec<&Command> = commands.iter().collect();

        let results = cuda.run_batch(&command_refs);

        assert_eq!(results[0].as_ref().unwrap(), &Vec::<f32>::new());
        assert_eq!(results[1].as_ref().unwrap(), &vec![11.0, 22.0]);
    }

    #[test]
    fn cuda_backend_batch_of_only_empty_pairs_does_not_touch_the_gpu_launch() {
        // Unlike `cuda_backend_batch_handles_an_empty_pair` (where a
        // non-empty job in the same batch keeps the concatenated flat
        // buffer non-empty regardless), this batch has no non-empty job at
        // all: `valid_indices` ends up empty, so the flat-buffer/launch path
        // must never run. Before the empty-vector fix (an earlier task),
        // a batch shaped like this reached the GPU launch with a
        // zero-length buffer and failed with `InvalidValue`.
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let commands = [Command::VectorAdd {
            a: vec![],
            b: vec![],
        }];
        let command_refs: Vec<&Command> = commands.iter().collect();

        let results = cuda.run_batch(&command_refs);

        assert_eq!(results[0].as_ref().unwrap(), &Vec::<f32>::new());
    }

    #[test]
    fn reduces_small_vector_on_cpu() {
        let mut backend = CpuBackend;
        let result = backend
            .run(&Command::Reduce {
                a: vec![1.0, 2.0, 3.0, 4.0],
            })
            .unwrap();
        assert_eq!(result, vec![10.0]);
    }

    #[test]
    fn cuda_backend_reduce_matches_cpu_backend_small() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let command = Command::Reduce {
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cuda_backend_reduce_matches_cpu_backend_across_multiple_blocks() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        // Bounded, repeating values (not a huge monotonically-growing range)
        // keep the true sum modest, so floating-point rounding differences
        // between the CPU's sequential summation order and the GPU's
        // tree-reduction summation order stay tiny relative to the result —
        // exact equality is not guaranteed here, only approximate.
        let len = 100_000;
        let a: Vec<f32> = (0..len).map(|i| (i % 7) as f32).collect();
        let command = Command::Reduce { a };

        let expected = cpu.run(&command).unwrap()[0];
        let actual = cuda.run(&command).unwrap()[0];

        let tolerance = expected.abs() * 1e-4 + 1e-3;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected ~{expected}, got {actual} (tolerance {tolerance})"
        );
    }

    #[test]
    fn cuda_backend_reduce_of_empty_vector_is_zero() {
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");
        let result = cuda.run(&Command::Reduce { a: vec![] }).unwrap();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn dot_product_small_vectors_on_cpu() {
        let mut backend = CpuBackend;
        let result = backend
            .run(&Command::DotProduct {
                a: vec![1.0, 2.0, 3.0],
                b: vec![4.0, 5.0, 6.0],
            })
            .unwrap();
        assert_eq!(result, vec![32.0]); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_product_rejects_mismatched_lengths_on_cpu() {
        let mut backend = CpuBackend;
        let err = backend
            .run(&Command::DotProduct {
                a: vec![1.0, 2.0],
                b: vec![1.0],
            })
            .unwrap_err();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn cuda_backend_dot_product_matches_cpu_backend_across_multiple_blocks() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let len = 100_000;
        let a: Vec<f32> = (0..len).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i % 3) as f32).collect();
        let command = Command::DotProduct { a, b };

        let expected = cpu.run(&command).unwrap()[0];
        let actual = cuda.run(&command).unwrap()[0];

        let tolerance = expected.abs() * 1e-4 + 1e-3;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected ~{expected}, got {actual} (tolerance {tolerance})"
        );
    }

    #[test]
    fn matmul_2x2_on_cpu() {
        let mut backend = CpuBackend;
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let result = backend
            .run(&Command::MatMul {
                a: vec![1.0, 2.0, 3.0, 4.0],
                b: vec![5.0, 6.0, 7.0, 8.0],
                m: 2,
                n: 2,
                k: 2,
            })
            .unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_rejects_wrong_sized_a_on_cpu() {
        let mut backend = CpuBackend;
        let err = backend
            .run(&Command::MatMul {
                a: vec![1.0, 2.0, 3.0], // should be m*k = 4
                b: vec![5.0, 6.0, 7.0, 8.0],
                m: 2,
                n: 2,
                k: 2,
            })
            .unwrap_err();
        assert!(err.contains("matmul"));
    }

    #[test]
    fn cuda_backend_matmul_2x2_matches_cpu_backend() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let command = Command::MatMul {
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![5.0, 6.0, 7.0, 8.0],
            m: 2,
            n: 2,
            k: 2,
        };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cuda_backend_matmul_non_square_matches_cpu_backend() {
        // m, n, k are all different from each other and larger than
        // MATMUL_TILE_SIZE (16), so a tile boundary is actually crossed in
        // every dimension. This is the test that specifically catches a
        // regression back to the upstream gemm_tiled example's row/col
        // swap bug (see the kernel's own comment in kernels/src/lib.rs):
        // that bug only produces wrong results when m != n, which this
        // test guarantees.
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let (m, n, k) = (64, 32, 48);
        let a: Vec<f32> = (0..(m * k)).map(|i| (i % 13) as f32 * 0.5).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i % 7) as f32 * 0.25).collect();
        let command = Command::MatMul { a, b, m, n, k };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (i, (a_val, e_val)) in actual.iter().zip(expected.iter()).enumerate() {
            let tolerance = e_val.abs() * 1e-3 + 1e-2;
            assert!(
                (a_val - e_val).abs() <= tolerance,
                "mismatch at index {i}: expected {e_val}, got {a_val}"
            );
        }
    }

    #[test]
    fn cuda_backend_matmul_partial_tile_matches_cpu_backend() {
        // Dimensions deliberately NOT multiples of MATMUL_TILE_SIZE (16), so
        // every tile in every dimension has a partial/ragged edge that
        // exercises the kernel's zero-padding bounds checks (row >= m,
        // col >= n, kk+tx >= k, kk+ty >= k) — untested by the other matmul
        // tests, which all use exact multiples of 16.
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

        let (m, n, k) = (20, 17, 33);
        let a: Vec<f32> = (0..(m * k)).map(|i| (i % 11) as f32 * 0.5).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i % 9) as f32 * 0.25).collect();
        let command = Command::MatMul { a, b, m, n, k };

        let expected = cpu.run(&command).unwrap();
        let actual = cuda.run(&command).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (i, (a_val, e_val)) in actual.iter().zip(expected.iter()).enumerate() {
            let tolerance = e_val.abs() * 1e-3 + 1e-2;
            assert!(
                (a_val - e_val).abs() <= tolerance,
                "mismatch at index {i}: expected {e_val}, got {a_val}"
            );
        }
    }
}
