mod atoms {
    rustler::atoms! {
        ok,
        error,
        invalid_device,
    }
}

mod backend;
mod job;
mod worker;

use std::sync::atomic::{AtomicU64, Ordering};

use cust::device::Device;
use cust::CudaFlags;
use job::Job;
use rustler::{Encoder, Env, Term};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[rustler::nif]
fn launch_vector_add<'a>(env: Env<'a>, a: Vec<f32>, b: Vec<f32>, device: u32) -> Term<'a> {
    if let Err(reason) = validate_device(device) {
        return (atoms::error(), reason).encode(env);
    }

    let id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let pid = env.pid();

    worker::sender(device)
        .send(Job { id, pid, a, b })
        .expect("erlcuda GPU worker thread is not running");

    (atoms::ok(), id).encode(env)
}

/// Validates that `device` refers to a real GPU on this machine before any
/// job for it is enqueued. Rejecting an out-of-range `device` here keeps it
/// from ever reaching `worker::sender`/`get_or_spawn`, which mitigates (but,
/// per the Task 1 review note, does not fully eliminate) the `WORKERS` mutex
/// poisoning risk: `get_or_spawn`'s `entry(...).or_insert_with(...)` runs
/// `spawn_worker` while the lock is held, and `spawn_worker`'s own
/// `thread::Builder::spawn(...).expect(...)` call is synchronous, so it can
/// still poison the lock if launching the OS thread itself fails (e.g. under
/// resource exhaustion). A `CudaBackend::new` failure, by contrast, panics
/// later on the already-detached worker thread, after `get_or_spawn` has
/// returned and the lock has been released.
fn validate_device(device: u32) -> Result<(), rustler::Atom> {
    cust::init(CudaFlags::empty()).expect("cust::init failed");
    let count = Device::num_devices().expect("Device::num_devices failed");
    if device >= count {
        Err(atoms::invalid_device())
    } else {
        Ok(())
    }
}

rustler::init!("Elixir.ErlCuda.Native");
