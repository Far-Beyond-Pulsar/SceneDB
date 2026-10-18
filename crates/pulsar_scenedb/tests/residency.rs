//! Budgeted residency: rebuildable data is kept within a byte budget, dropped
//! least-recently-used first, and rebuilt from its source on demand.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pulsar_scenedb::{residency_stats, set_residency_budget, Residency, DEFAULT_RESIDENCY_BUDGET};

/// The cache is process-wide; tests take turns and each sets its own budget.
static SERIAL: Mutex<()> = Mutex::new(());

struct Blob(Vec<u8>);

/// A value of `bytes` bytes that counts its rebuilds.
fn blob(bytes: usize, builds: &Arc<AtomicUsize>) -> Residency<Blob> {
    let builds = builds.clone();
    Residency::lazy(
        |blob: &Blob| blob.0.len(),
        move || {
            builds.fetch_add(1, Ordering::SeqCst);
            Ok(Blob(vec![1; bytes]))
        },
    )
}

#[test]
fn values_build_once_and_stay_while_within_budget() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(1000);
    let builds = Arc::new(AtomicUsize::new(0));
    let a = blob(300, &builds);
    assert!(!a.is_resident(), "lazy until first use");
    assert_eq!(a.get().unwrap().0.len(), 300);
    assert_eq!(a.get().unwrap().0.len(), 300);
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert!(a.is_resident());
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}

#[test]
fn the_least_recently_used_value_is_dropped_first_and_rebuilt_on_demand() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(1000);
    let builds = Arc::new(AtomicUsize::new(0));
    let (a, b, c) = (blob(400, &builds), blob(400, &builds), blob(400, &builds));
    a.get().unwrap();
    b.get().unwrap();
    a.get().unwrap(); // a is now more recent than b
    c.get().unwrap(); // 1200 > 1000: b goes

    assert!(a.is_resident() && c.is_resident());
    assert!(!b.is_resident(), "b was least recently used");
    assert_eq!(builds.load(Ordering::SeqCst), 3);

    b.get().unwrap(); // rebuilt; now a (older than c) goes
    assert_eq!(builds.load(Ordering::SeqCst), 4);
    assert!(b.is_resident() && c.is_resident() && !a.is_resident());
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}

#[test]
fn a_value_in_use_survives_its_own_eviction() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(500);
    let builds = Arc::new(AtomicUsize::new(0));
    let (a, b) = (blob(400, &builds), blob(400, &builds));
    let held = a.get().unwrap();
    b.get().unwrap(); // evicts a from the cache
    assert!(!a.is_resident());
    assert_eq!(held.0.len(), 400, "the caller's Arc is untouched");
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}

#[test]
fn one_value_over_the_whole_budget_is_still_kept() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(100);
    let builds = Arc::new(AtomicUsize::new(0));
    let big = blob(5000, &builds);
    big.get().unwrap();
    big.get().unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 1, "no thrash on a single huge value");
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}

#[test]
fn dropping_the_owner_releases_its_bytes() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
    let before = residency_stats().resident_bytes;
    let builds = Arc::new(AtomicUsize::new(0));
    let a = blob(4096, &builds);
    a.get().unwrap();
    assert_eq!(residency_stats().resident_bytes, before + 4096);
    drop(a);
    assert_eq!(residency_stats().resident_bytes, before);
}

#[test]
fn a_failed_rebuild_reports_the_error_and_leaves_nothing_resident() {
    let _turn = SERIAL.lock().unwrap();
    let broken: Residency<Blob> =
        Residency::lazy(|b: &Blob| b.0.len(), || Err("source is gone".to_string()));
    assert_eq!(broken.get().err().as_deref(), Some("source is gone"));
    assert!(!broken.is_resident());
}

#[test]
fn with_value_is_resident_at_once_and_counted() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
    let before = residency_stats();
    let made = Residency::with_value(
        Blob(vec![0; 2048]),
        |b: &Blob| b.0.len(),
        || Ok(Blob(vec![0; 2048])),
    );
    assert!(made.is_resident());
    assert_eq!(residency_stats().resident_bytes, before.resident_bytes + 2048);
}

#[test]
fn many_threads_sharing_a_small_budget_stay_consistent() {
    let _turn = SERIAL.lock().unwrap();
    set_residency_budget(8 * 1024);
    let builds = Arc::new(AtomicUsize::new(0));
    let values: Arc<Vec<Residency<Blob>>> =
        Arc::new((0..32).map(|_| blob(1024, &builds)).collect());

    let threads: Vec<_> = (0..8)
        .map(|t| {
            let values = values.clone();
            std::thread::spawn(move || {
                for round in 0..400usize {
                    let value = &values[(round * 7 + t * 3) % values.len()];
                    assert_eq!(value.get().unwrap().0.len(), 1024);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }

    let stats = residency_stats();
    assert!(
        stats.resident_bytes <= 8 * 1024,
        "within budget once everyone is idle: {stats:?}"
    );
    let resident = values.iter().filter(|v| v.is_resident()).count();
    assert_eq!(
        resident * 1024,
        stats.resident_bytes,
        "the accounted bytes match what is actually resident"
    );
    assert!(builds.load(Ordering::SeqCst) > 32, "eviction forced rebuilds");
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}
