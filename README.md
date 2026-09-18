# erlCuda

GPU compute for the BEAM. `erlCuda` lets Erlang and Elixir code launch CUDA
kernels written in Rust, using [Rustler](https://github.com/rusterlium/rustler)
NIFs as the bridge and the [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
toolchain to compile Rust to PTX.

Status: **early design / pre-alpha**. Architecture below is the target shape;
most of it is not implemented yet. Contributions and design feedback welcome.

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
|   ErlCuda.launch(kernel, args)  ->  {:ok, ref}               |
|   receive do {:erlcuda, ^ref, {:ok, result}} -> ... end       |
+-------------------------------|------------------------------+
                                | NIF call (non-blocking)
                                v
+-------------------------------------------------------------+
|                    erlcuda_nif (Rust, Rustler)               |
|                                                               |
|  - validates args, encodes job                               |
|  - pushes job onto a channel to the GPU worker                |
|  - returns {:ok, ref} to BEAM immediately                     |
+-------------------------------|------------------------------+
                                | mpsc channel
                                v
+-------------------------------------------------------------+
|                  GPU worker (dedicated OS thread)             |
|                                                               |
|  - owns the single CUDA context for this process              |
|  - loads compiled kernel modules (PTX)                        |
|  - launches kernel, (optionally streams / async copies)       |
|  - on completion: enif_send(env, pid, {:erlcuda, ref, result}) |
+-------------------------------|------------------------------+
                                | cust / cudarc host API
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
- results are delivered asynchronously via `enif_send`, following the same
  pattern used by other BEAM libraries that wrap long-running native work
  (e.g. NIF resource + message-based completion instead of a dirty NIF that
  blocks a scheduler for the whole kernel duration).

Dirty schedulers are deliberately avoided as the primary mechanism: they help
with blocking calls, but a kernel launch is not actually CPU-bound blocking
work, and pinning the CUDA context's owning thread ourselves gives full
control over queueing, batching, and lifetime.

## Components (planned layout)

```
erlCuda/
├── lib/                    # Elixir public API (ErlCuda module)
├── src/                    # Erlang sources, if any pure-Erlang glue is needed
├── native/
│   └── erlcuda_nif/        # Rust crate, Rustler NIF + GPU worker thread
│       ├── src/
│       │   ├── lib.rs      # NIF entry points (load/launch/etc.)
│       │   ├── worker.rs   # GPU worker thread, channel, CUDA context owner
│       │   └── job.rs      # job/result encoding between BEAM terms and GPU calls
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
- Rust toolchain (stable, plus the nightly components required by
  `rustc_codegen_nvvm` for building the `kernels/` crate).
- Erlang/OTP and Elixir.
- [Rustler](https://github.com/rusterlium/rustler) precompiled or built from
  source for the target platform.

## Planned usage

```elixir
defmodule Example do
  def run do
    {:ok, ref} = ErlCuda.launch(:vector_add, [a, b])

    receive do
      {:erlcuda, ^ref, {:ok, result}} -> result
      {:erlcuda, ^ref, {:error, reason}} -> raise "GPU kernel failed: #{inspect(reason)}"
    end
  end
end
```

A synchronous helper (`ErlCuda.launch!/2`, wrapping the receive above with a
timeout) will be provided for the common case.

## Roadmap

- [ ] `erlcuda_nif`: load Rustler skeleton, spawn GPU worker thread, own a
      CUDA context via `cust`/`cudarc`.
- [ ] Job/result encoding between Erlang terms and device buffers (start with
      flat numeric arrays).
- [ ] First kernel example (`kernels/`) built with Rust-CUDA, e.g. vector add.
- [ ] Async completion via `enif_send`, with `ErlCuda.launch/2` returning a
      reference and `ErlCuda.launch!/2` as a blocking convenience wrapper.
- [ ] Multi-GPU: one worker thread per device, device selection in the API.
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
