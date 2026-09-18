use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;
use std::thread;

use rustler::types::LocalPid;
use rustler::{Encoder, OwnedEnv};

use crate::backend::{Backend, CudaBackend};
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
    JOB_SENDER.get_or_init(|| {
        spawn_worker(|| CudaBackend::new().expect("failed to initialize CUDA backend"))
    })
}

fn spawn_worker<B, F>(make_backend: F) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    spawn_worker_with_sink(make_backend, send_outcome).0
}

/// Spawns the worker thread, draining `Job`s from a fresh channel and calling
/// `backend.vector_add` once per job, in order, feeding each outcome to
/// `on_outcome`. The thread exits once the channel is disconnected (i.e. every
/// `Sender<Job>` clone has been dropped).
///
/// This is split out from `spawn_worker` so tests can inject a stub `Backend`
/// and a recording `on_outcome` sink, observing the worker's thread lifecycle
/// and channel-draining behavior without needing a live BEAM environment
/// (which `send_outcome`'s `OwnedEnv`/`LocalPid` FFI calls require). Production
/// code always goes through `spawn_worker`, which wires up the real
/// `send_outcome`; only tests call this directly.
fn spawn_worker_with_sink<B, F, S>(
    make_backend: F,
    on_outcome: S,
) -> (Sender<Job>, thread::JoinHandle<()>)
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
    S: Fn(&LocalPid, u64, Result<Vec<f32>, String>) + Send + 'static,
{
    let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();

    let handle = thread::Builder::new()
        .name("erlcuda-gpu-worker".into())
        .spawn(move || {
            let mut backend = make_backend();
            for job in rx {
                let result = backend.vector_add(&job.a, &job.b);
                on_outcome(&job.pid, job.id, result);
            }
        })
        .expect("failed to spawn erlcuda GPU worker thread");

    (tx, handle)
}

fn send_outcome(pid: &LocalPid, id: u64, result: Result<Vec<f32>, String>) {
    let mut owned_env = OwnedEnv::new();
    // `send_and_clear` only errors if the recipient process has died — nothing
    // useful to do in that case (no BEAM-side error handler could receive it),
    // so it's discarded rather than treated as a worker failure.
    let _ = owned_env.send_and_clear(pid, |env| match result {
        Ok(values) => (atoms::erlcuda(), id, (atoms::ok(), values)).encode(env),
        Err(reason) => (atoms::erlcuda(), id, (atoms::error(), reason)).encode(env),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// A stub `Backend` that records every call it receives instead of doing
    /// any real computation, so tests can assert on call count, argument
    /// values, and call order.
    struct RecordingBackend {
        calls: Arc<Mutex<Vec<(Vec<f32>, Vec<f32>)>>>,
    }

    impl Backend for RecordingBackend {
        fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            self.calls.lock().unwrap().push((a.to_vec(), b.to_vec()));
            Ok(vec![])
        }
    }

    /// Builds a `LocalPid` without a live BEAM environment.
    ///
    /// SAFETY: `LocalPid::from_c_arg` just wraps its argument in a `Copy`
    /// struct; it performs no FFI call. The underlying `ErlNifPid` is a
    /// single machine word with no validity invariants (any bit pattern,
    /// including all-zero, is a legal value to hold), so zero-initializing it
    /// is sound. Nothing in these tests ever calls a method that would
    /// dereference it through the BEAM's NIF FFI table (no `encode`,
    /// `is_alive`, `Eq`, or `Ord`) — the value only exists to satisfy `Job`'s
    /// `pid` field so a `Job` can be constructed and sent through a plain
    /// `mpsc` channel outside of a NIF call.
    fn fake_pid() -> LocalPid {
        unsafe { LocalPid::from_c_arg(std::mem::zeroed()) }
    }

    /// Joins a worker thread's handle off the test thread, failing fast
    /// instead of hanging forever if `spawn_worker_with_sink`'s loop ever
    /// fails to terminate after the channel is dropped.
    fn join_within(handle: thread::JoinHandle<()>, timeout: Duration) {
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            handle.join().expect("worker thread panicked");
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(timeout)
            .expect("worker thread did not terminate within the timeout");
    }

    #[test]
    fn calls_backend_once_per_job_in_order_with_correct_arguments() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let calls_for_backend = Arc::clone(&calls);

        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let outcomes_for_sink = Arc::clone(&outcomes);

        let (tx, handle) = spawn_worker_with_sink(
            move || RecordingBackend {
                calls: calls_for_backend,
            },
            move |_pid, id, result| {
                outcomes_for_sink.lock().unwrap().push((id, result));
            },
        );

        for i in 1..=5u64 {
            tx.send(Job {
                id: i,
                pid: fake_pid(),
                a: vec![i as f32],
                b: vec![(i * 10) as f32],
            })
            .expect("worker thread should still be receiving");
        }

        // Dropping every sender disconnects the channel, which ends the
        // worker's `for job in rx` loop.
        drop(tx);
        join_within(handle, Duration::from_secs(5));

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 5, "backend should be called once per job");
        for (i, (a, b)) in calls.iter().enumerate() {
            let n = (i + 1) as f32;
            assert_eq!(a, &vec![n], "job {} called with wrong `a`", i + 1);
            assert_eq!(b, &vec![n * 10.0], "job {} called with wrong `b`", i + 1);
        }

        let outcomes = outcomes.lock().unwrap();
        let ids: Vec<u64> = outcomes.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4, 5],
            "outcomes should be produced in the order jobs were sent"
        );
    }

    #[test]
    fn terminates_cleanly_and_stops_calling_backend_once_channel_is_dropped() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let calls_for_backend = Arc::clone(&calls);

        let (tx, handle) = spawn_worker_with_sink(
            move || RecordingBackend {
                calls: calls_for_backend,
            },
            |_pid, _id, _result| {},
        );

        tx.send(Job {
            id: 1,
            pid: fake_pid(),
            a: vec![1.0],
            b: vec![2.0],
        })
        .unwrap();

        drop(tx);
        // If the loop failed to exit on channel disconnect, this would hang
        // and `join_within` would fail the test after its timeout instead of
        // blocking forever.
        join_within(handle, Duration::from_secs(5));

        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "backend must not be called again after the channel disconnects"
        );
    }
}
