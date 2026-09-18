use std::time::{Duration, Instant};

use erlcuda_nif::backend::{Backend, CudaBackend};

const VECTOR_LEN: usize = 1_000;
const REPETITIONS: usize = 100;

fn main() {
    let mut backend =
        CudaBackend::new(0).expect("CudaBackend::new(0) (requires an NVIDIA GPU + CUDA driver)");

    let a: Vec<f32> = (0..VECTOR_LEN).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..VECTOR_LEN).map(|i| (i as f32) * 2.0).collect();

    // Warm-up: absorbs the one-time cost already paid by `CudaBackend::new`
    // (context creation) plus first-launch PTX module JIT-loading, neither
    // of which reflects steady-state per-job latency.
    backend.vector_add(&a, &b).expect("warm-up vector_add failed");

    let mut total = Duration::ZERO;
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        backend.vector_add(&a, &b).expect("vector_add failed");
        total += start.elapsed();
    }

    let mean_us = total.as_micros() as f64 / REPETITIONS as f64;
    println!("pure_cuda_mean_us: {mean_us:.1}");
}
