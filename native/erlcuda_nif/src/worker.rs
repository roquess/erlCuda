use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;
use std::thread;

use rustler::types::LocalPid;
use rustler::{Encoder, OwnedEnv};

use crate::backend::{Backend, CpuBackend};
use crate::job::Job;

mod atoms {
    rustler::atoms! {
        erlcuda,
        ok,
        error,
    }
}

static JOB_SENDER: OnceLock<Sender<Job>> = OnceLock::new();

pub fn sender() -> &'static Sender<Job> {
    JOB_SENDER.get_or_init(|| spawn_worker(|| CpuBackend))
}

fn spawn_worker<B, F>(make_backend: F) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();

    thread::Builder::new()
        .name("erlcuda-gpu-worker".into())
        .spawn(move || {
            let mut backend = make_backend();
            for job in rx {
                let result = backend.vector_add(&job.a, &job.b);
                send_outcome(&job.pid, job.id, result);
            }
        })
        .expect("failed to spawn erlcuda GPU worker thread");

    tx
}

fn send_outcome(pid: &LocalPid, id: u64, result: Result<Vec<f32>, String>) {
    let mut owned_env = OwnedEnv::new();
    owned_env.send_and_clear(pid, |env| match result {
        Ok(values) => (atoms::erlcuda(), id, (atoms::ok(), values)).encode(env),
        Err(reason) => (atoms::erlcuda(), id, (atoms::error(), reason)).encode(env),
    });
}
