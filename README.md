# erlCuda

GPU compute for the BEAM. `erlCuda` lets Erlang and Elixir code launch CUDA
kernels written in Rust, using [Rustler](https://github.com/rusterlium/rustler)
NIFs as the bridge and the [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
toolchain to compile Rust to PTX.

Status: **early alpha**. Four kernels run end-to-end (Elixir -> Rustler NIF ->
dedicated GPU worker thread -> real CUDA kernel -> async result):
`vector_add`, `reduce`, `dot_product`, and `matmul`. This is backed by
explicit multi-GPU device selection, opportunistic per-device job batching,
and a reproducible pure-Rust-vs-full-stack latency benchmark (see Benchmarks
below) — the initial roadmap items are all addressed, though several are
explicitly noted below as best-effort or untested beyond the maintainer's own
machine rather than fully hardened.

## Why

NVIDIA's Rust support for CUDA (via the Rust-CUDA project: `rustc_codegen_nvvm`
+ `cust`/`cudarc`) makes it possible to write GPU kernels in real Rust instead
of hand-written CUDA C. Rustler already makes it possible to write BEAM NIFs
in Rust. `erlCuda` connects the two, so an Elixir/Erlang application can:

- write kernels in Rust, compiled to PTX for the `nvptx64-nvidia-cuda` target,
- launch them on the GPU from a NIF,
- get results back without blocking BEAM schedulers.

The interesting engineering problem isn't the kernel code itself, it's making
GPU execution behave inside the BEAM's cooperative scheduling model.

## Architecture

```
+-------------------------------------------------------------+
|                      Elixir / Erlang app                    |
|                                                               |
|   ErlCuda.launch(kernel, args)  ->  {:ok, job_id}            |
|   receive do {:erlcuda, ^job_id, {:ok, result}} -> ... end    |
+-------------------------------|------------------------------+
                                | NIF call (non-blocking)
                                v
+-------------------------------------------------------------+
|                    erlcuda_nif (Rust, Rustler)               |
|                                                               |
|  - validates args, encodes job                               |
|  - pushes job onto a channel to the GPU worker                |
|  - returns {:ok, job_id} to BEAM immediately                  |
+-------------------------------|------------------------------+
                                | mpsc channel
                                v
+-------------------------------------------------------------+
|                  GPU worker (dedicated OS thread)             |
|                                                               |
|  - owns the CUDA context for its device (one thread per device) |
|  - loads compiled kernel modules (PTX)                        |
|  - launches kernel, (optionally streams / async copies)       |
|  - on completion: OwnedEnv::send_and_clear(pid, {:erlcuda,     |
|    job_id, result})                                            |
+-------------------------------|------------------------------+
                                | cust host API
                                v
+-------------------------------------------------------------+
|                     GPU kernels (Rust, no_std)                |
|                                                               |
|  compiled with rustc_codegen_nvvm to nvptx64-nvidia-cuda PTX,  |
|  loaded and launched by the host-side worker above             |
+---------------------------------------------------------------+
```

### Why a dedicated GPU worker thread, not a NIF-per-call

A CUDA context is bound to the OS thread that created it. The BEAM runs NIFs
across a pool of scheduler threads (plus dirty schedulers), so naively calling
CUDA driver APIs directly from a regular NIF means the context migrates
between threads across calls, which the CUDA driver does not support cleanly.

`erlCuda` instead runs a single long-lived OS thread per GPU that owns the
CUDA context for its whole lifetime. NIFs never touch the GPU directly; they
only enqueue work on a channel and return immediately. This also means:

- BEAM schedulers are never blocked on `cudaDeviceSynchronize` or kernel
  execution time,
- kernel launches are naturally serialized per GPU (matching how a CUDA
  context is normally driven from one thread), while multiple GPUs map to
  multiple worker threads,
- results are delivered asynchronously via `OwnedEnv::send_and_clear`
  (Rustler's safe wrapper around `enif_send`), following the same
  pattern used by other BEAM libraries that wrap long-running native work
  (e.g. NIF resource + message-based completion instead of a dirty NIF that
  blocks a scheduler for the whole kernel duration).

Dirty schedulers are deliberately avoided as the primary mechanism: they help
with blocking calls, but a kernel launch is not actually CPU-bound blocking
work, and pinning the CUDA context's owning thread ourselves gives full
control over queueing, batching, and lifetime.

## Components

```
erlCuda/
├── lib/                    # Elixir public API (ErlCuda module)
├── src/
│   └── erlcuda.erl          # idiomatic Erlang wrapper (launch/2,3, launch_sync/2,3)
├── native/
│   └── erlcuda_nif/        # Rust crate, Rustler NIF + GPU worker thread
│       ├── src/
│       │   ├── lib.rs      # NIF entry points (launch_vector_add, launch_reduce,
│       │   │               # launch_dot_product, launch_matmul)
│       │   ├── worker.rs   # GPU worker thread, channel, async result delivery
│       │   ├── backend.rs  # Backend trait, CpuBackend, CudaBackend (owns
│       │   │               # Context/Module/Stream, launches the kernel)
│       │   └── job.rs      # Job struct crossing the channel
│       └── Cargo.toml
├── kernels/                 # Rust GPU kernels compiled to PTX
│   ├── src/
│   │   └── lib.rs           # #[no_std] kernel functions
│   └── Cargo.toml            # built with rustc_codegen_nvvm (Rust-CUDA)
├── test/
├── bench/
│   └── erlcuda_bench.exs    # full-stack latency benchmark (see Benchmarks below)
├── mix.exs
└── LICENSE
```

## Requirements

- NVIDIA GPU with a supported driver and CUDA toolkit installed.
- Rust toolchain: stable for `native/erlcuda_nif/`, plus the nightly toolchain
  pinned in `kernels/rust-toolchain.toml` for building the `kernels/` crate
  (installed automatically by `rustup` the first time a command runs inside
  `kernels/`).
- Erlang/OTP and Elixir.
- [Rustler](https://github.com/rusterlium/rustler), pulled in as a normal Mix
  dependency.

### Building `rustc_codegen_nvvm` (one-time, per machine)

`native/erlcuda_nif/build.rs` compiles `kernels/` to PTX via `cuda_builder`,
which needs the `rustc_codegen_nvvm` backend available as a dynamic library on
`PATH`. This repo deliberately has no root Cargo workspace (`native/erlcuda_nif/`
must stay on stable, `kernels/` on nightly), which rules out `cuda_builder`'s
own "build it for me" paths — so until this is scripted, it has to be built
once by hand from the pinned Rust-CUDA checkout:

```bash
# Locate Cargo's own checkout of the Rust-CUDA repo, fetched automatically
# as a git dependency of this crate, at commit
# 6a836d9236fc38e0fa7a71f7bdeda7a8f82bc8d5 — normally under
# ~/.cargo/git/checkouts/rust-cuda-*/6a836d9*/ (run `cargo build` once first
# if it doesn't exist yet, to make Cargo fetch it):
cd <path-to-that-checkout>
cargo build -p rustc_codegen_nvvm --release
```

Once built **inside that same checkout** (not an independently `git clone`d
copy — the auto-discovery below only looks in Cargo's own git-checkout cache,
and only accepts a checkout of this exact pinned commit), `native/erlcuda_nif/build.rs`
automatically finds it and extends its own `PATH` for the build — you don't
need to manually export `PATH` in every new shell session. The one-time
`cargo build -p rustc_codegen_nvvm --release` step above is still required if
you've never built it on this machine before.

Without that one-time build, `cargo build` in `native/erlcuda_nif/` fails to
locate the codegen backend.

## Usage

```elixir
defmodule Example do
  def run do
    {:ok, job_id} = ErlCuda.launch(:vector_add, [a, b])

    receive do
      {:erlcuda, ^job_id, {:ok, result}} -> result
      {:erlcuda, ^job_id, {:error, reason}} -> raise "GPU kernel failed: #{inspect(reason)}"
    end
  end
end
```

`ErlCuda.launch!/3` wraps the receive above with a timeout for the common
case; see `lib/erl_cuda.ex`.

Pass `device: N` as an option to target a specific GPU (defaults to
`device: 0`): `ErlCuda.launch(:vector_add, [a, b], device: 1)`.

Three more kernels are available besides `vector_add`, using the same
`launch`/`launch!` API:

```elixir
ErlCuda.launch!(:reduce, [a])
ErlCuda.launch!(:dot_product, [a, b])
ErlCuda.launch!(:matmul, [a_flat, b_flat, m, n, k])
```

`reduce` sums a single vector down to one value; `dot_product` takes two
equal-length vectors and returns their dot product as a single value;
`matmul` multiplies an `m x k` matrix by a `k x n` matrix (each passed
flattened, row-major) and returns the flattened `m x n` result.

Note on batching: the per-device worker's opportunistic batching (see
Roadmap below) only coalesces a batch into a single kernel launch when every
job in it is `vector_add` targeting the same device. If a drained batch
contains any `reduce`, `dot_product`, or `matmul` job — or a mix of kernel
types — the worker falls back to running each job in that batch one at a
time, in order, still returning the correct per-job result for each. This
was already true architecturally before `reduce`/`dot_product`/`matmul`
existed; it's called out here now that there's more than one non-`vector_add`
kernel to wonder about.

## Benchmarks

Two small programs measure per-job `vector_add` latency (1000-element `f32`
vectors, one job in flight at a time) at opposite ends of the stack:

- `native/erlcuda_nif/src/bin/bench_pure_cuda.rs` calls `CudaBackend::vector_add`
  directly, from plain Rust, with no BEAM, no NIF, and no channel involved.
- `bench/erlcuda_bench.exs` goes through the full stack: `ErlCuda.launch!/2` ->
  Rustler NIF -> mpsc channel -> GPU worker thread -> `OwnedEnv::send_and_clear`
  -> `receive` in Elixir.

At 1000 elements the actual GPU compute time is negligible either way, so the
gap between the two (if any) should mostly reflect NIF/channel/BEAM-message
overhead rather than kernel execution time.

Reproduce with:

```bash
cd native/erlcuda_nif && cargo run --release --features bench --bin bench_pure_cuda
mix run bench/erlcuda_bench.exs   # from the repo root
```

### A methodology caveat found while measuring this

Both programs are correct on their own, but running them as two separate,
freshly-launched OS processes back-to-back is not a clean way to compare
steady-state latency: an earlier review pass reported that whichever program
happened to run *second* in a pair tended to read noticeably higher,
regardless of which program it was — attributed to CUDA context
creation/teardown handoff contention between the outgoing and incoming
process, not to either program's own behavior.

Six alternating-order trials taken in one session on this machine did **not**
reproduce that flip. In every trial, `erlcuda_mean_us` came out higher than
`pure_cuda_mean_us`, whether erlCuda ran first or second:

| trial | 1st run | 1st value (us) | 2nd run | 2nd value (us) |
|-------|---------|-----------------|---------|-----------------|
| 1 | pure_cuda | 328.7 | erlcuda | 369.2 |
| 2 | erlcuda   | 383.6 | pure_cuda | 337.8 |
| 3 | pure_cuda | 334.4 | erlcuda | 371.3 |
| 4 | erlcuda   | 372.3 | pure_cuda | 337.7 |
| 5 | pure_cuda | 336.6 | erlcuda | 373.3 |
| 6 | erlcuda   | 376.4 | pure_cuda | 347.9 |

`pure_cuda_mean_us` ranged 328.7-347.9 (median ~337.2us); `erlcuda_mean_us`
ranged 369.2-383.6 (median ~372.8us), with no overlap between the two ranges
across those 6 runs — at the time, that looked like a real, order-independent
full-stack overhead of roughly 35-40us (~10-11%), distinct from the
position-dependent noise described above.

**A second independent session on the same machine and the same checkout
contradicted this in direction.** Re-running both programs 8 more times
(4 pure_cuda-first, 4 erlcuda-first) gave:

```
pure_cuda: 451.3, 431.6, 438.6, 445.1, 458.3, 446.4, 449.6, 443.0   (range 431.6-458.3)
erlcuda:   440.9, 424.5, 421.4, 419.3, 429.6, 427.6, 423.4, 429.6   (range 419.3-440.9)
```

This time `pure_cuda` read consistently *higher* than `erlcuda` by roughly
15-20us, the opposite of the first session's finding — and both sessions'
absolute magnitudes shifted too (420-460us here vs. 330-380us before), most
likely from ambient GPU/driver/thermal state at the time of each session.

**Taken together, the honest conclusion is that this benchmark cannot
reliably determine even the *direction* of erlCuda's overhead at this vector
size, let alone its magnitude.** Session 1's two ranges didn't overlap at
all; session 2's overlapped somewhat (431.6-440.9us shared by both), but its
means still moved in the opposite direction from session 1's. Either way,
the two sessions disagree with each other about which side was faster.
Whatever the true
NIF/channel overhead is at 1000 elements, it is evidently small enough to be
dominated by session-to-session noise (GPU clock/power state, driver
scheduling, background load) when measured this way — one-shot process
launches with 100 in-process repetitions each. A methodology that could
actually resolve this would need to interleave both measurements within a
single long-lived session (or otherwise control for cross-session GPU state),
which is out of scope for the simple manual comparison built here.

**This is a single-machine sample, not a tracked benchmark and not a general
performance claim, and — as demonstrated above — not even a stable one across
sessions on the same machine.** There's no CI job pinning these numbers, no
fixed hardware/driver baseline, and no statistical confidence interval behind
them. Run the commands yourself; do not treat either session's numbers, or
this project's, as a reliable verdict on which side is faster.

## Roadmap

- [x] `erlcuda_nif`: Rustler skeleton, dedicated GPU worker thread owning a
      CUDA context via `cust`.
- [x] Job/result encoding between Erlang terms and device buffers (flat
      `f32` arrays).
- [x] First kernel example (`kernels/`), `vector_add`, built with Rust-CUDA.
- [x] Async completion via `OwnedEnv::send_and_clear`, with `ErlCuda.launch/3`
      returning a job id and `ErlCuda.launch!/3` as a blocking convenience
      wrapper.
- [x] Multi-GPU: one worker thread and CUDA context per device
      (`worker::sender(device)`), explicit `device:` option in
      `ErlCuda.launch/3` and `ErlCuda.launch!/3` (default `0`). Routing to
      independent per-device worker threads is verified with stub backends;
      true concurrent execution across two or more physical GPUs is
      untested on the maintainer's single-GPU machine.
- [x] Batching: each device's worker thread opportunistically coalesces
      whatever jobs are already queued (up to `MAX_BATCH_SIZE = 256`) into
      a single kernel launch via `Backend::vector_add_batch`, with zero
      added latency for a lone in-flight job. Best-effort — not a
      guaranteed batch size or time window. (Real CUDA streams for
      intra-job pipelining were considered and not implemented; this item
      covers coalescing only.)
- [x] Benchmarks against a plain Rust baseline to measure NIF/IPC overhead:
      `bench_pure_cuda` (direct `CudaBackend::vector_add` call) vs.
      `bench/erlcuda_bench.exs` (full `ErlCuda.launch!/2` stack), both
      reproducible with a single command (see Benchmarks above). Two
      independent measurement sessions on the maintainer's machine
      disagreed about which side was faster — the honest finding is that
      this simple methodology cannot reliably pin down erlCuda's overhead
      at this vector size, not a specific overhead number.

## Credits

This project is a thin bridge between two existing pieces of work; the hard
technical parts belong to them:

- [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA) — the toolchain that
  makes it possible to write CUDA kernels in Rust and compile them to PTX
  (`rustc_codegen_nvvm`, `cust`, `cudarc` and related crates). `erlCuda`
  targets this ecosystem for all device-side kernel code.
- [Rustler](https://github.com/rusterlium/rustler) — safe Rust NIFs for
  Erlang/Elixir. `erlCuda`'s host-side bridge is built on it.

## License

Apache License 2.0. See [LICENSE](LICENSE).
