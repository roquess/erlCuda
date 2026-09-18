# erlCuda

GPU compute for the BEAM. `erlCuda` lets Erlang and Elixir code launch CUDA
kernels written in Rust, using [Rustler](https://github.com/rusterlium/rustler)
NIFs as the bridge and the [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
toolchain to compile Rust to PTX.

Status: **early alpha**. The `vector_add` kernel runs end-to-end (Elixir ->
Rustler NIF -> dedicated GPU worker thread -> real CUDA kernel -> async
result), with explicit multi-GPU device selection, opportunistic
per-device job batching, and a reproducible pure-Rust-vs-full-stack latency
benchmark (see Benchmarks below) — the initial roadmap items are all
addressed, though several are explicitly noted below as best-effort or
untested beyond the maintainer's own machine rather than fully hardened.

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
├── src/                    # Erlang sources, if any pure-Erlang glue is needed
├── native/
│   └── erlcuda_nif/        # Rust crate, Rustler NIF + GPU worker thread
│       ├── src/
│       │   ├── lib.rs      # NIF entry point (launch_vector_add)
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
# Locate (or clone) the Rust-CUDA checkout pinned by this repo's git
# dependencies, at commit 6a836d9236fc38e0fa7a71f7bdeda7a8f82bc8d5:
cd <path-to-rust-cuda-checkout>
cargo build -p rustc_codegen_nvvm --release
```

Then add both of these to `PATH` before building `native/erlcuda_nif/`:
- that checkout's `target/release/` directory (contains `rustc_codegen_nvvm`'s
  dynamic library),
- the CUDA toolkit's `nvvm/bin` directory (e.g.
  `%CUDA_PATH%\nvvm\bin` on Windows, `$CUDA_PATH/nvvm/bin` on Linux).

Without this, `cargo build` in `native/erlcuda_nif/` fails to locate the
codegen backend. This is tracked as a known gap to automate (e.g. a setup
script, or vendoring a prebuilt artifact) rather than something to repeat by
hand indefinitely.

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

Six alternating-order trials taken for this task on this machine did **not**
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

`pure_cuda_mean_us` ranged 328.7-347.9 (median ~337.2us) regardless of
position; `erlcuda_mean_us` ranged 369.2-383.6 (median ~372.8us) regardless
of position, with no overlap between the two ranges across all 6 runs. That
points to a real, order-independent full-stack overhead of roughly 35-40us
(~10-11% of the pure-Rust baseline) for this kernel/vector size on this
machine, rather than to the position-dependent noise described above.

Whether that specific ~35-40us gap generalizes, or whether a run on a
different day would surface the position-dependent flip instead (both
effects could plausibly coexist and one just didn't show up in six runs), is
not something this quick, manual measurement can settle either way.

**This is a single-machine, one-shot, manually-run sample, not a tracked
benchmark and not a general performance claim.** There's no CI job pinning
these numbers, no fixed hardware/driver baseline, and no statistical
confidence interval behind them — expect your own numbers, and possibly your
own qualitative pattern, to differ.

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
      reproducible with a single command (see Benchmarks above). Because a
      single back-to-back pair of freshly-launched processes can in
      principle be skewed by cross-process CUDA context handoff, the
      section above reports six alternating-order trials rather than one
      cherry-picked pair, together with the honest result: on the
      maintainer's machine this run, the full stack was consistently
      slower by roughly 35-40us regardless of run order, not just a
      one-off snapshot.

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
