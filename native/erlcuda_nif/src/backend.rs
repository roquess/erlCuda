pub trait Backend {
    fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String>;
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
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

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        let ctx = cust::quick_init().map_err(|e| format!("cust::quick_init failed: {e:?}"))?;
        let module =
            Module::from_ptx(PTX, &[]).map_err(|e| format!("Module::from_ptx failed: {e:?}"))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| format!("Stream::new failed: {e:?}"))?;
        Ok(Self {
            stream,
            module,
            _ctx: ctx,
        })
    }
}

impl Backend for CudaBackend {
    fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
        }

        let len = a.len();
        let a_gpu = a.as_dbuf().map_err(|e| format!("upload a failed: {e:?}"))?;
        let b_gpu = b.as_dbuf().map_err(|e| format!("upload b failed: {e:?}"))?;
        let mut out = vec![0.0f32; len];
        let out_gpu = out
            .as_slice()
            .as_dbuf()
            .map_err(|e| format!("alloc out failed: {e:?}"))?;

        let function = self
            .module
            .get_function("vector_add")
            .map_err(|e| format!("get_function failed: {e:?}"))?;
        let (_, block_size) = function
            .suggested_launch_configuration(0, 0.into())
            .map_err(|e| format!("suggested_launch_configuration failed: {e:?}"))?;
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
            .map_err(|e| format!("kernel launch failed: {e:?}"))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| format!("stream.synchronize failed: {e:?}"))?;
        out_gpu
            .copy_to(&mut out)
            .map_err(|e| format!("copy_to failed: {e:?}"))?;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_equal_length_vectors() {
        let mut backend = CpuBackend;
        let result = backend
            .vector_add(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
            .unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let mut backend = CpuBackend;
        let err = backend.vector_add(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn cuda_backend_matches_cpu_backend() {
        let mut cpu = CpuBackend;
        let mut cuda =
            CudaBackend::new().expect("CudaBackend::new (requires an NVIDIA GPU + CUDA driver)");

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];

        let expected = cpu.vector_add(&a, &b).unwrap();
        let actual = cuda.vector_add(&a, &b).unwrap();

        assert_eq!(actual, expected);
    }
}
