use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vector_add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}

pub const REDUCE_BLOCK_SIZE: usize = 256;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn reduce_sum(a: &[f32], partial_sums: *mut f32) {
    #[address_space(shared)]
    static mut SDATA: [MaybeUninit<f32>; REDUCE_BLOCK_SIZE] =
        [MaybeUninit::uninit(); REDUCE_BLOCK_SIZE];

    let tid = thread::thread_idx_x() as usize;
    let idx = thread::index_1d() as usize;

    let value = if idx < a.len() { a[idx] } else { 0.0 };
    unsafe {
        SDATA[tid].write(value);
    }
    thread::sync_threads();

    // Stop the barrier-synchronized tree reduction once the active-thread
    // count would drop below a full warp (32): reaching `sync_threads()`
    // with an active mask that's a strict *subset* of a warp hangs on this
    // toolchain (confirmed by direct experiment — a `stride >= 32` loop
    // runs correctly every time; changing the bound to `stride > 0` so a
    // stride=16 step is reached hangs the kernel deterministically, even
    // though the exact same code is textbook-correct CUDA C and works with
    // nvcc). All strides here (128, 64, 32) are >= 32, so `tid < stride`
    // always selects one or more *whole* warps, never splits one.
    let mut stride = REDUCE_BLOCK_SIZE / 2;
    while stride >= 32 {
        if tid < stride {
            let sum = unsafe { SDATA[tid].assume_init() + SDATA[tid + stride].assume_init() };
            unsafe {
                SDATA[tid].write(sum);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    // The remaining 32 partial sums (exactly one warp's worth) are finished
    // serially by thread 0 alone. No further `sync_threads()` is needed —
    // nothing else touches shared memory from here — which sidesteps the
    // sub-warp-divergent-barrier issue entirely instead of working around
    // it with warp-shuffle intrinsics this crate doesn't expose.
    if tid == 0 {
        let mut total = 0.0f32;
        for i in 0..32 {
            total += unsafe { SDATA[i].assume_init() };
        }
        unsafe {
            *partial_sums.add(thread::block_idx_x() as usize) = total;
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn dot_product(a: &[f32], b: &[f32], partial_sums: *mut f32) {
    #[address_space(shared)]
    static mut SDATA: [MaybeUninit<f32>; REDUCE_BLOCK_SIZE] =
        [MaybeUninit::uninit(); REDUCE_BLOCK_SIZE];

    let tid = thread::thread_idx_x() as usize;
    let idx = thread::index_1d() as usize;

    let value = if idx < a.len() { a[idx] * b[idx] } else { 0.0 };
    unsafe {
        SDATA[tid].write(value);
    }
    thread::sync_threads();

    // See Task 1's `reduce_sum` for why this loop stops at `stride >= 32`
    // and finishes the last warp serially instead of continuing to
    // `stride > 0`: sub-warp-divergent `sync_threads()` hangs on this
    // toolchain.
    let mut stride = REDUCE_BLOCK_SIZE / 2;
    while stride >= 32 {
        if tid < stride {
            let sum = unsafe { SDATA[tid].assume_init() + SDATA[tid + stride].assume_init() };
            unsafe {
                SDATA[tid].write(sum);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut total = 0.0f32;
        for i in 0..32 {
            total += unsafe { SDATA[i].assume_init() };
        }
        unsafe {
            *partial_sums.add(thread::block_idx_x() as usize) = total;
        }
    }
}

pub const MATMUL_TILE_SIZE: usize = 16;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn matmul(a: &[f32], b: &[f32], c: *mut f32, m: usize, n: usize, k: usize) {
    const TILE_SIZE_2D: usize = MATMUL_TILE_SIZE * MATMUL_TILE_SIZE;

    #[address_space(shared)]
    static mut TILE_A: [MaybeUninit<f32>; TILE_SIZE_2D] = [MaybeUninit::uninit(); TILE_SIZE_2D];
    #[address_space(shared)]
    static mut TILE_B: [MaybeUninit<f32>; TILE_SIZE_2D] = [MaybeUninit::uninit(); TILE_SIZE_2D];

    let tx = thread::thread_idx_x() as usize;
    let ty = thread::thread_idx_y() as usize;

    // row from block_idx_y(), col from block_idx_x() — matching the host
    // launch's grid_size_y (sized by m) / grid_size_x (sized by n). The
    // upstream gemm_tiled example this is adapted from has these swapped,
    // which only works by coincidence for square matrices; this ordering is
    // dimensionally correct for any m/n/k, and the non-square test below
    // exists specifically to catch a regression back to the swapped form.
    let row = thread::block_idx_y() as usize * MATMUL_TILE_SIZE + ty;
    let col = thread::block_idx_x() as usize * MATMUL_TILE_SIZE + tx;

    let mut sum = 0.0f32;
    for kk in (0..k).step_by(MATMUL_TILE_SIZE) {
        if row < m && (kk + tx) < k {
            unsafe {
                TILE_A[ty * MATMUL_TILE_SIZE + tx].write(a[row * k + (kk + tx)]);
            }
        } else {
            unsafe {
                TILE_A[ty * MATMUL_TILE_SIZE + tx].write(0.0f32);
            }
        }
        if col < n && (kk + ty) < k {
            unsafe {
                TILE_B[ty * MATMUL_TILE_SIZE + tx].write(b[(kk + ty) * n + col]);
            }
        } else {
            unsafe {
                TILE_B[ty * MATMUL_TILE_SIZE + tx].write(0.0f32);
            }
        }
        thread::sync_threads();

        for i in 0..MATMUL_TILE_SIZE {
            sum += unsafe {
                TILE_A[ty * MATMUL_TILE_SIZE + i].assume_init()
                    * TILE_B[i * MATMUL_TILE_SIZE + tx].assume_init()
            };
        }
        thread::sync_threads();
    }

    if row < m && col < n {
        unsafe {
            *c.add(row * n + col) = sum;
        }
    }
}
