//! Single flight and prefetch: concurrent callers share one rebuild, rebuilds
//! can run ahead of need on the loader threads, and neither can deadlock.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

use pulsar_scenedb::{residency_stats, set_residency_budget, Residency, DEFAULT_RESIDENCY_BUDGET};

/// The cache is process-wide; tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

struct Blob(Vec<u8>);

fn size(blob: &Blob) -> usize {
    blob.0.len()
}

#[test]
fn concurrent_callers_share_one_rebuild() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = builds.clone();
    let value: Residency<Blob> = Residency::lazy(size, move || {
        counter.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        Ok(Blob(vec![3; 64]))
    });
    let before = residency_stats().joined;
    let barrier = Arc::new(Barrier::new(16));
    let threads: Vec<_> = (0..16)
        .map(|_| {
            let (value, barrier) = (value.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                value.get().unwrap()
            })
        })
        .collect();
    let results: Vec<Arc<Blob>> = threads.into_iter().map(|t| t.join().unwrap()).collect();

    assert_eq!(builds.load(Ordering::SeqCst), 1, "sixteen callers, one rebuild");
    assert!(
        results.windows(2).all(|pair| Arc::ptr_eq(&pair[0], &pair[1])),
        "everyone got the same value"
    );
    assert!(residency_stats().joined > before, "the others joined the flight");
}

#[test]
fn a_failed_rebuild_reaches_every_waiter_and_the_next_call_retries() {
    let _turn = SERIAL.lock().unwrap();
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = builds.clone();
    let value: Residency<Blob> = Residency::lazy(size, move || {
        std::thread::sleep(Duration::from_millis(100));
        match counter.fetch_add(1, Ordering::SeqCst) {
            0 => Err("first attempt fails".to_string()),
            _ => Ok(Blob(vec![1; 8])),
        }
    });
    let barrier = Arc::new(Barrier::new(6));
    let threads: Vec<_> = (0..6)
        .map(|_| {
            let (value, barrier) = (value.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                value.get().err()
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap().as_deref(), Some("first attempt fails"));
    }
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(value.get().unwrap().0.len(), 8, "a later call tries again");
    assert_eq!(builds.load(Ordering::SeqCst), 2);
}

#[test]
fn a_panicking_rebuild_does_not_leave_waiters_hanging() {
    let _turn = SERIAL.lock().unwrap();
    let value: Residency<Blob> = Residency::lazy(size, || {
        std::thread::sleep(Duration::from_millis(150));
        panic!("decoder blew up");
    });
    let builder = {
        let value = value.clone();
        std::thread::spawn(move || {
            let _ = value.get();
        })
    };
    std::thread::sleep(Duration::from_millis(40));
    let waiter = {
        let value = value.clone();
        std::thread::spawn(move || value.get().err())
    };
    assert!(builder.join().is_err(), "the builder thread panicked");
    assert_eq!(waiter.join().unwrap().as_deref(), Some("rebuild panicked"));
    assert!(!value.is_resident());
}

#[test]
fn prefetches_run_in_parallel_on_the_loader_threads() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
    let builds = Arc::new(AtomicUsize::new(0));
    let values: Vec<Residency<Blob>> = (0..8)
        .map(|_| {
            let builds = builds.clone();
            Residency::lazy(size, move || {
                builds.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(150));
                Ok(Blob(vec![9; 16]))
            })
        })
        .collect();
    let before = residency_stats().prefetches;

    let started = Instant::now();
    let pending: Vec<_> = values.iter().map(|v| v.prefetch()).collect();
    let requested = started.elapsed();
    for prefetch in pending {
        prefetch.wait().unwrap();
    }
    let total = started.elapsed();

    assert!(requested < Duration::from_millis(100), "prefetch returns at once: {requested:?}");
    // Eight 150 ms rebuilds one after another would take 1.2 s.
    assert!(total < Duration::from_millis(900), "ran in parallel: {total:?}");
    assert_eq!(builds.load(Ordering::SeqCst), 8);
    assert_eq!(residency_stats().prefetches, before + 8);
    assert!(values.iter().all(|v| v.is_resident()));
}

#[test]
fn a_get_during_a_prefetch_joins_it() {
    let _turn = SERIAL.lock().unwrap();
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = builds.clone();
    let value: Residency<Blob> = Residency::lazy(size, move || {
        counter.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));
        Ok(Blob(vec![2; 32]))
    });
    let prefetch = value.prefetch();
    let got = value.get().unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 1, "one rebuild serves both");
    assert!(Arc::ptr_eq(&got, &prefetch.wait().unwrap()));
}

#[test]
fn prefetching_a_resident_value_does_nothing() {
    let _turn = SERIAL.lock().unwrap();
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = builds.clone();
    let value: Residency<Blob> = Residency::lazy(size, move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(Blob(vec![1; 64]))
    });
    value.get().unwrap();
    let before = residency_stats().prefetches;
    let prefetch = value.prefetch();
    assert!(prefetch.is_ready());
    prefetch.wait().unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(residency_stats().prefetches, before);
}

#[test]
fn waiting_on_a_queued_prefetch_runs_it_instead_of_deadlocking_behind_busy_workers() {
    let _turn = SERIAL.lock().unwrap();
    let workers = pulsar_scenedb::asset_load_workers();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let started = Arc::new(AtomicUsize::new(0));

    // Occupy every loader thread with a rebuild that blocks until released.
    let blockers: Vec<Residency<Blob>> = (0..workers)
        .map(|_| {
            let (gate, started) = (gate.clone(), started.clone());
            Residency::lazy(size, move || {
                started.fetch_add(1, Ordering::SeqCst);
                let (open, signal) = &*gate;
                let mut open = open.lock().unwrap();
                while !*open {
                    open = signal.wait(open).unwrap();
                }
                Ok(Blob(vec![0; 1]))
            })
        })
        .collect();
    let pending: Vec<_> = blockers.iter().map(|b| b.prefetch()).collect();
    while started.load(Ordering::SeqCst) < workers {
        std::thread::sleep(Duration::from_millis(5));
    }

    // Every loader thread is busy, so this prefetch sits in the queue. Waiting
    // for it must run it here.
    let target: Residency<Blob> = Residency::lazy(size, || Ok(Blob(vec![5; 4])));
    let queued = target.prefetch();
    let began = Instant::now();
    assert_eq!(queued.wait().unwrap().0.len(), 4);
    assert!(began.elapsed() < Duration::from_secs(2), "did not wait for a free worker");

    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    for prefetch in pending {
        prefetch.wait().unwrap();
    }
}
