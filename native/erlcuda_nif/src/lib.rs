pub(crate) mod atoms {
    rustler::atoms! {
        ok,
        error,
        invalid_device,
        erlcuda,
    }
}

pub mod backend;
mod job;
mod worker;

use std::sync::atomic::{AtomicU64, Ordering};

use cust::device::Device;
use cust::CudaFlags;
use job::{Command, Job};
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
        .send(Job {
            id,
            pid,
            command: Command::VectorAdd { a, b },
        })
        .expect("erlcuda GPU worker thread is not running");

    (atoms::ok(), id).encode(env)
}

/// Validates that `device` refers to a real GPU on this machine before any
/// job for it is enqueued, returning a clean `{:error, :invalid_device}`
/// for an out-of-range value instead of ever reaching `worker::sender`.
///
/// This is a separate concern from the `WORKERS` mutex poisoning risk that
/// used to exist in `get_or_spawn`: that risk (a panic during
/// `thread::Builder::spawn(...).expect(...)`, e.g. under OS resource
/// exhaustion, poisoning the lock for every other device) is now fixed via
/// double-checked locking in `get_or_spawn`/`get_or_spawn_with_sink`, which
/// spawns the worker thread outside of any lock on `WORKERS`.
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
