use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread;

use rustler::types::LocalPid;
use rustler::{Encoder, OwnedEnv};

use crate::backend::{Backend, CudaBackend};
use crate::job::Job;

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
///
/// Uses double-checked locking so the spawn itself happens *outside* any
/// lock: if `make_backend`/thread creation panics (e.g. OS thread creation
/// failing under resource exhaustion), that panic can never occur while a
/// `MutexGuard` on `table` is alive, so it can never poison `table` for
/// other, unrelated devices. The accepted trade-off is a narrow, harmless
/// race: two concurrent first-time calls for the *same* device can each
/// spawn a worker thread, with only one kept in the map — the loser's
/// `Sender` is dropped immediately without ever being cloned or handed to a
/// caller, so its channel disconnects right away and that thread's
/// `while let Ok(first) = rx.recv()` loop exits almost immediately after
/// briefly constructing (and then dropping) its own backend. It never
/// lingers; the cost is a moment of wasted setup, not a leaked thread.
fn get_or_spawn<B, F>(
    table: &Mutex<HashMap<u32, Sender<Job>>>,
    device: u32,
    make_backend: F,
) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    if let Some(sender) = table
        .lock()
        .expect("erlcuda WORKERS mutex poisoned")
        .get(&device)
    {
        return sender.clone();
    }

    let sender = spawn_worker(make_backend);

    table
        .lock()
        .expect("erlcuda WORKERS mutex poisoned")
        .entry(device)
        .or_insert(sender)
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
    if let Some(sender) = table
        .lock()
        .expect("erlcuda WORKERS mutex poisoned")
        .get(&device)
    {
        return sender.clone();
    }

    let sender = spawn_worker_with_sink(make_backend, on_outcome).0;

    table
        .lock()
        .expect("erlcuda WORKERS mutex poisoned")
        .entry(device)
        .or_insert(sender)
        .clone()
}

fn spawn_worker<B, F>(make_backend: F) -> Sender<Job>
where
    B: Backend,
    F: FnOnce() -> B + Send + 'static,
{
    spawn_worker_with_sink(make_backend, send_outcome).0
}

#[cfg(test)]
thread_local! {
    /// Test-only hook consulted by `spawn_worker_with_sink`: when set (via
    /// `Cell::set`), overrides the stack size used for the *next* worker
    /// thread spawned on this same (test) thread.
    ///
    /// This exists to let a test deterministically force
    /// `thread::Builder::spawn` itself to fail, simulating "OS thread
    /// creation fails under resource exhaustion" — the real, documented
    /// risk the double-checked locking in `get_or_spawn`/
    /// `get_or_spawn_with_sink` guards against. An absurdly large stack size
    /// (e.g. `usize::MAX / 2`) is rejected by the OS immediately as an
    /// invalid parameter, with no actual memory reserved, so this is fast
    /// and side-effect-free.
    ///
    /// A panic from `make_backend` itself can't be used for this instead:
    /// `make_backend` only ever runs *inside* the newly spawned thread's own
    /// closure (see below), never synchronously on the caller, so it can
    /// never occur while the caller holds `WORKERS`'s `MutexGuard` — no
    /// matter how `get_or_spawn`/`get_or_spawn_with_sink` lock the table.
    /// (Constructing the backend synchronously on the caller instead, so a
    /// factory panic *would* land under the lock, was considered and
    /// rejected: `CudaBackend::new` creates a CUDA context that is current
    /// only on the thread that created it, so building it on the caller
    /// while every GPU call happens later on the worker thread would break
    /// real GPU execution.)
    ///
    /// `None` (the default) leaves the builder's normal default stack size
    /// untouched, so production code and every other test are unaffected.
    static TEST_STACK_SIZE_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Sets `TEST_STACK_SIZE_OVERRIDE` for the lifetime of this guard, resetting
/// it to `None` on drop — including if a panic unwinds through the scope
/// holding it — so a test using this can't leave the override set for
/// whichever test the harness runs next on the same OS thread.
#[cfg(test)]
struct StackSizeOverrideGuard;

#[cfg(test)]
impl StackSizeOverrideGuard {
    fn set(size: usize) -> Self {
        TEST_STACK_SIZE_OVERRIDE.with(|cell| cell.set(Some(size)));
        Self
    }
}

#[cfg(test)]
impl Drop for StackSizeOverrideGuard {
    fn drop(&mut self) {
        TEST_STACK_SIZE_OVERRIDE.with(|cell| cell.set(None));
    }
}

/// Spawns the worker thread, draining `Job`s from a fresh channel and
/// opportunistically batching whatever's already queued (via `drain_batch`)
/// into a single `backend.run_batch` call per batch, in order, feeding each
/// outcome to `on_outcome`. The thread exits once the channel is
/// disconnected (i.e. every `Sender<Job>` clone has been dropped).
///
/// This is split out from `spawn_worker` so tests can inject a stub `Backend`
/// and a recording `on_outcome` sink, observing the worker's thread lifecycle
/// and channel-draining behavior without needing a live BEAM environment
/// (which `send_outcome`'s `OwnedEnv`/`LocalPid` FFI calls require). Production
/// code always goes through `spawn_worker`, which wires up the real
/// `send_outcome`; only tests call this directly.
///
/// The `thread::Builder::spawn` call below consults `TEST_STACK_SIZE_OVERRIDE`
/// under `#[cfg(test)]` only; see that item's doc comment for why.
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

    #[allow(unused_mut)]
    let mut builder = thread::Builder::new().name("erlcuda-gpu-worker".into());
    #[cfg(test)]
    {
        if let Some(size) = TEST_STACK_SIZE_OVERRIDE.with(|cell| cell.get()) {
            builder = builder.stack_size(size);
        }
    }

    let handle = builder
        .spawn(move || {
            let mut backend = make_backend();
            while let Ok(first) = rx.recv() {
                let batch = drain_batch(&rx, first, MAX_BATCH_SIZE);
                let commands: Vec<&crate::job::Command> =
                    batch.iter().map(|job| &job.command).collect();
                let results = backend.run_batch(&commands);
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
        Ok(values) => (crate::atoms::erlcuda(), id, (crate::atoms::ok(), values)).encode(env),
        Err(reason) => (crate::atoms::erlcuda(), id, (crate::atoms::error(), reason)).encode(env),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Command;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    type RecordedCalls = Arc<Mutex<Vec<(Vec<f32>, Vec<f32>)>>>;

    /// A stub `Backend` that records every `VectorAdd` call it receives
    /// instead of doing any real computation, so tests can assert on call
    /// count, argument values, and call order. Every test in this module
    /// only ever constructs `Command::VectorAdd` jobs, so the `Reduce` arm
    /// is unreachable in practice; it exists only to satisfy exhaustiveness.
    struct RecordingBackend {
        calls: RecordedCalls,
    }

    impl Backend for RecordingBackend {
        fn run(&mut self, command: &Command) -> Result<Vec<f32>, String> {
            match command {
                Command::VectorAdd { a, b } => {
                    self.calls.lock().unwrap().push((a.clone(), b.clone()));
                    Ok(vec![])
                }
                Command::Reduce { .. } => {
                    unreachable!("RecordingBackend tests only ever construct VectorAdd jobs")
                }
            }
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

    fn vector_add_job(id: u64, a: Vec<f32>, b: Vec<f32>) -> Job {
        Job {
            id,
            pid: fake_pid(),
            command: Command::VectorAdd { a, b },
        }
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
            tx.send(vector_add_job(i, vec![i as f32], vec![(i * 10) as f32]))
                .expect("worker thread should still be receiving");
        }

        // Dropping every sender disconnects the channel, which ends the
        // worker's `while let Ok(first) = rx.recv()` loop.
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

        tx.send(vector_add_job(1, vec![1.0], vec![2.0])).unwrap();

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

        let calls_0: RecordedCalls = Arc::new(Mutex::new(Vec::new()));
        let calls_1: RecordedCalls = Arc::new(Mutex::new(Vec::new()));

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

        sender_0.send(vector_add_job(1, vec![1.0], vec![10.0])).unwrap();
        sender_0_again
            .send(vector_add_job(2, vec![2.0], vec![20.0]))
            .unwrap();
        sender_1.send(vector_add_job(3, vec![3.0], vec![30.0])).unwrap();

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
    fn a_panicking_spawn_for_one_device_does_not_poison_the_table_for_another() {
        let workers: OnceLock<Mutex<HashMap<u32, Sender<Job>>>> = OnceLock::new();
        let table = workers.get_or_init(|| Mutex::new(HashMap::new()));
        let noop_sink = |_pid: &LocalPid, _id: u64, _result: Result<Vec<f32>, String>| {};

        // Cache device 0 successfully first.
        let calls_0 = Arc::new(Mutex::new(Vec::new()));
        let sender_0 = get_or_spawn_with_sink(
            table,
            0,
            move || RecordingBackend { calls: calls_0 },
            noop_sink,
        );

        // Attempt device 1 with `thread::Builder::spawn` itself forced to
        // fail (see `TEST_STACK_SIZE_OVERRIDE`'s doc comment for why this,
        // rather than a panicking `make_backend`, is what genuinely
        // reproduces "a spawn-time panic under the lock"). The factory here
        // must never actually run — if it does, thread creation didn't fail
        // as expected, and this panic (on the worker thread, not the test
        // thread) is a loud signal to re-check that assumption rather than
        // silently doing nothing.
        //
        // `_override_guard` resets `TEST_STACK_SIZE_OVERRIDE` on drop (even
        // if a panic unwinds past this point), so a future edit that adds
        // panicking code here can't accidentally leave the override set for
        // whichever test the harness runs next on this same OS thread.
        let _override_guard = StackSizeOverrideGuard::set(usize::MAX / 2);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            get_or_spawn_with_sink(
                table,
                1,
                || -> RecordingBackend {
                    panic!("must not run: thread creation should have failed first")
                },
                noop_sink,
            )
        }));
        drop(_override_guard);
        assert!(
            panicked.is_err(),
            "expected the device-1 spawn to panic (thread creation should have failed)"
        );

        // Device 0's already-cached sender must still work — the table must
        // not be poisoned by device 1's panic.
        sender_0
            .send(vector_add_job(1, vec![1.0], vec![10.0]))
            .expect("device 0's worker should still be alive and receiving");

        // A fresh lookup for device 0 must also still succeed (proves the
        // mutex itself isn't poisoned, not just that the old sender clone
        // happens to still work).
        let sender_0_again = get_or_spawn_with_sink(
            table,
            0,
            || -> RecordingBackend { panic!("must not respawn device 0, it's already cached") },
            noop_sink,
        );
        sender_0_again
            .send(vector_add_job(2, vec![2.0], vec![20.0]))
            .expect("device 0's worker should still be alive and receiving");
    }

    #[test]
    fn drain_batch_drains_everything_currently_queued_when_under_the_cap() {
        let (tx, rx) = mpsc::channel();
        for i in 2..=4u64 {
            tx.send(vector_add_job(i, vec![i as f32], vec![0.0])).unwrap();
        }
        let first = vector_add_job(1, vec![1.0], vec![0.0]);

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
            tx.send(vector_add_job(i, vec![i as f32], vec![0.0])).unwrap();
        }
        let first = rx.recv().unwrap();

        let batch = drain_batch(&rx, first, 3);

        assert_eq!(
            batch.iter().map(|j| j.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let remaining: Vec<u64> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|j| j.id)
            .collect();
        assert_eq!(remaining, vec![4, 5]);
    }
}
