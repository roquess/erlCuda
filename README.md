# erlCuda

GPU compute for the BEAM. `erlCuda` lets Erlang and Elixir code launch CUDA
kernels written in Rust, using [Rustler](https://github.com/rusterlium/rustler)
NIFs as the bridge and the [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
toolchain to compile Rust to PTX.

Status: **early alpha**. The `vector_add` kernel runs end-to-end (Elixir ->
Rustler NIF -> dedicated GPU worker thread -> real CUDA kernel -> async
result), with explicit multi-GPU device selection. Streams/batching and
benchmarks (see Roadmap below) are not implemented yet.

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
- [ ] Streams and batching for throughput once the single-kernel path works.
- [ ] Benchmarks against a plain Rust/cudarc baseline to measure NIF/IPC
      overhead.

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
