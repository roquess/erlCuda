mod atoms {
    rustler::atoms! {
        ok,
    }
}

mod backend;
mod job;
mod worker;

use std::sync::atomic::{AtomicU64, Ordering};

use job::Job;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[rustler::nif]
fn launch_vector_add(env: rustler::Env, a: Vec<f32>, b: Vec<f32>) -> (rustler::Atom, u64) {
    let id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let pid = env.pid();

    worker::sender()
        .send(Job { id, pid, a, b })
        .expect("erlcuda GPU worker thread is not running");

    (atoms::ok(), id)
}

rustler::init!("Elixir.ErlCuda.Native");
