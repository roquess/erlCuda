use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
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

/// Caps how many jobs a single batch/kernel launch can cover, bounding
/// worst-case memory and launch-configuration size under a flood of jobs.
/// Arbitrary but reasonable for now; not derived from a specific
/// measurement (that's what benchmarking, a separate later step, is for).
const MAX_BATCH_SIZE: usize = 256;

pub fn sender(device: u32) -> Sender<Job> {
    static WORKERS: OnceLock<Mutex<HashMap<u32, Sender<Job>>>> = OnceLock::new();
    let table = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
    get_or_spawn(table, device, move || {
        CudaBackend::new(device).expect("failed to initialize CUDA backend")
    })
}

/// Looks up the cached worker `Sender` for `device`, spawning a new worker
/// thread (and its backend) via `make_backend` only the first time `device`
/// is requested. Subsequent calls for the same `device` reuse the cached,
/// cheaply-cloneable `Sender` instead of spawning a second backend/thread for
/// it.
fn get_or_spawn<B, F>(
    table: &Mutex<HashMap<u32, Sender<Job>>>,
    device: u32,
    make_backend: F,
) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    let mut workers = table.lock().expect("erlcuda WORKERS mutex poisoned");
    workers
        .entry(device)
        .or_insert_with(|| spawn_worker(make_backend))
        .clone()
}

/// Test-only variant of `get_or_spawn` with an injectable outcome sink.
///
/// This mirrors the `spawn_worker`/`spawn_worker_with_sink` split below:
/// `get_or_spawn` always wires up the real `send_outcome`, which calls into
/// `OwnedEnv`/BEAM NIF FFI that is only valid when this code is actually
/// running as a loaded NIF inside the BEAM. Invoking it from a plain
/// `cargo test` process (as the routing test below does, via `fake_pid()`)
/// hits an uninitialized FFI function table and hard-aborts the process. So
/// tests exercise the per-device caching/routing logic through this
/// sink-injectable variant instead, with a no-op sink.
#[cfg(test)]
fn get_or_spawn_with_sink<B, F, S>(
    table: &Mutex<HashMap<u32, Sender<Job>>>,
    device: u32,
    make_backend: F,
    on_outcome: S,
) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
    S: Fn(&LocalPid, u64, Result<Vec<f32>, String>) + Send + 'static,
{
    let mut workers = table.lock().expect("erlcuda WORKERS mutex poisoned");
    workers
        .entry(device)
        .or_insert_with(|| spawn_worker_with_sink(make_backend, on_outcome).0)
        .clone()
}

fn spawn_worker<B, F>(make_backend: F) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    spawn_worker_with_sink(make_backend, send_outcome).0
}

/// Spawns the worker thread, draining `Job`s from a fresh channel and
/// opportunistically batching whatever's already queued (via `drain_batch`)
/// into a single `backend.vector_add_batch` call per batch, in order, feeding
/// each outcome to `on_outcome`. The thread exits once the channel is
/// disconnected (i.e. every `Sender<Job>` clone has been dropped).
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
            while let Ok(first) = rx.recv() {
                let batch = drain_batch(&rx, first, MAX_BATCH_SIZE);
                let inputs: Vec<(&[f32], &[f32])> = batch
                    .iter()
                    .map(|job| (job.a.as_slice(), job.b.as_slice()))
                    .collect();
                let results = backend.vector_add_batch(&inputs);
                for (job, result) in batch.into_iter().zip(results) {
                    on_outcome(&job.pid, job.id, result);
                }
            }
        })
        .expect("failed to spawn erlcuda GPU worker thread");

    (tx, handle)
}

/// Given a job that's already been received (via a blocking `recv`),
/// opportunistically drains any additional jobs already sitting in the
/// channel via non-blocking `try_recv`, up to `cap` jobs total. Stops as
/// soon as the channel has nothing more queued right now, or `cap` is hit.
///
/// This is intentionally a pure function over `&Receiver` (not a method on
/// the worker thread's loop) so it can be tested deterministically: a test
/// can pre-load a channel with jobs and call this directly, with no thread
/// spawning or timing involved.
fn drain_batch(rx: &Receiver<Job>, first: Job, cap: usize) -> Vec<Job> {
    let mut batch = vec![first];
    while batch.len() < cap {
        match rx.try_recv() {
            Ok(job) => batch.push(job),
            Err(_) => break,
        }
    }
    batch
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

    #[test]
    fn routes_different_devices_to_independent_backends_and_caches_the_sender() {
        let workers: OnceLock<Mutex<HashMap<u32, Sender<Job>>>> = OnceLock::new();
        let table = workers.get_or_init(|| Mutex::new(HashMap::new()));

        let calls_0: Arc<Mutex<Vec<(Vec<f32>, Vec<f32>)>>> = Arc::new(Mutex::new(Vec::new()));
        let calls_1: Arc<Mutex<Vec<(Vec<f32>, Vec<f32>)>>> = Arc::new(Mutex::new(Vec::new()));

        // Use a no-op outcome sink rather than the real `send_outcome`: the
        // real one calls into BEAM NIF FFI via `fake_pid()`, which is only
        // sound to invoke from inside an actual loaded NIF (see
        // `get_or_spawn_with_sink`'s doc comment) — this test runs as plain
        // `cargo test` code, so it exercises the device-routing/caching logic
        // through `get_or_spawn_with_sink` directly instead of going through
        // production's `get_or_spawn`.
        let noop_sink = |_pid: &LocalPid, _id: u64, _result: Result<Vec<f32>, String>| {};

        let c0 = calls_0.clone();
        let sender_0 =
            get_or_spawn_with_sink(table, 0, move || RecordingBackend { calls: c0 }, noop_sink);
        let c1 = calls_1.clone();
        let sender_1 =
            get_or_spawn_with_sink(table, 1, move || RecordingBackend { calls: c1 }, noop_sink);

        // Requesting device 0 again must reuse the cached sender, NOT spawn a
        // second worker/backend for the same device — the factory panicking
        // if invoked proves this.
        let sender_0_again = get_or_spawn_with_sink(
            table,
            0,
            || -> RecordingBackend { panic!("must not respawn device 0") },
            noop_sink,
        );

        sender_0
            .send(Job {
                id: 1,
                pid: fake_pid(),
                a: vec![1.0],
                b: vec![10.0],
            })
            .unwrap();
        sender_0_again
            .send(Job {
                id: 2,
                pid: fake_pid(),
                a: vec![2.0],
                b: vec![20.0],
            })
            .unwrap();
        sender_1
            .send(Job {
                id: 3,
                pid: fake_pid(),
                a: vec![3.0],
                b: vec![30.0],
            })
            .unwrap();

        drop(sender_0);
        drop(sender_0_again);
        drop(sender_1);

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let done_0 = calls_0.lock().unwrap().len() == 2;
            let done_1 = calls_1.lock().unwrap().len() == 1;
            if done_0 && done_1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for both device queues to drain (device 0: {}, device 1: {})",
                calls_0.lock().unwrap().len(),
                calls_1.lock().unwrap().len()
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(
            *calls_0.lock().unwrap(),
            vec![(vec![1.0], vec![10.0]), (vec![2.0], vec![20.0])]
        );
        assert_eq!(*calls_1.lock().unwrap(), vec![(vec![3.0], vec![30.0])]);
    }

    #[test]
    fn drain_batch_drains_everything_currently_queued_when_under_the_cap() {
        let (tx, rx) = mpsc::channel();
        for i in 2..=4u64 {
            tx.send(Job {
                id: i,
                pid: fake_pid(),
                a: vec![i as f32],
                b: vec![0.0],
            })
            .unwrap();
        }
        let first = Job {
            id: 1,
            pid: fake_pid(),
            a: vec![1.0],
            b: vec![0.0],
        };

        let batch = drain_batch(&rx, first, 256);

        assert_eq!(
            batch.iter().map(|j| j.id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn drain_batch_stops_at_the_cap_leaving_the_remainder_in_the_channel() {
        let (tx, rx) = mpsc::channel();
        for i in 1..=5u64 {
            tx.send(Job {
                id: i,
                pid: fake_pid(),
                a: vec![i as f32],
                b: vec![0.0],
            })
            .unwrap();
        }
        let first = rx.recv().unwrap();

        let batch = drain_batch(&rx, first, 3);

        assert_eq!(batch.iter().map(|j| j.id).collect::<Vec<_>>(), vec![1, 2, 3]);
        let remaining: Vec<u64> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|j| j.id)
            .collect();
        assert_eq!(remaining, vec![4, 5]);
    }
}
