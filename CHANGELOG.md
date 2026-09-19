# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-19

Initial release. Status: early alpha.

### Added

- Four GPU kernels, launched from Erlang and run end-to-end on real CUDA
  hardware: `vector_add`, `reduce`, `dot_product`, `matmul`.
- `erlcuda:launch/2,3` (async, returns a job id, result delivered as a
  message) and `erlcuda:launch_sync/2,3` (blocking convenience wrapper with
  a timeout).
- A dedicated GPU worker thread per device, each owning its own CUDA context
  for its entire lifetime, so NIF calls never touch the GPU directly and
  BEAM schedulers are never blocked on kernel execution.
- Multi-GPU support via an explicit `{device, N}` option on `launch`/
  `launch_sync` (defaults to device `0`). Routing to independent per-device
  worker threads is verified with stub backends; true concurrent execution
  across two or more physical GPUs is untested on the maintainer's
  single-GPU machine.
- Opportunistic batching: each device's worker coalesces queued jobs (up to
  `MAX_BATCH_SIZE = 256`) into a single kernel launch, but only when every
  job in the batch is the same kernel type targeting that device. A batch
  containing a mix of kernel types, or any non-`vector_add` job, falls back
  to running each job one at a time, in order. Best-effort, not a
  guaranteed batch size or time window.
- A reproducible latency benchmark comparing plain-Rust `CudaBackend`
  calls against the full `erlcuda:launch_sync/2` stack (see the README's
  Benchmarks section) — reported honestly, including two measurement
  sessions that disagreed on the direction of the overhead.
- Pure Erlang and Rust implementation: no Elixir dependency anywhere in the
  stack.
