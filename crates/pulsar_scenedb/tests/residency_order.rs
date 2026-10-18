//! Recency order: the cache evicts exactly what a reference LRU would, and
//! choosing a victim does not scan the resident set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pulsar_scenedb::{residency_stats, set_residency_budget, Residency, DEFAULT_RESIDENCY_BUDGET};

/// The cache is process-wide; tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

struct Blob(Vec<u8>);

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

/// Deterministic pseudo-random numbers, so a failure reproduces.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn eviction_order_matches_a_reference_lru_over_random_accesses() {
    let _turn = SERIAL.lock().unwrap();
    let base = residency_stats().resident_bytes;
    // Room for exactly ten of the 100-byte values.
    set_residency_budget(base + 1000);
    let builds = Arc::new(AtomicUsize::new(0));
    let values: Vec<Residency<Blob>> = (0..64).map(|_| blob(100, &builds)).collect();

    let mut model: Vec<usize> = Vec::new(); // least recent first
    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    for step in 0..5000 {
        // Skewed so some values stay hot and the order is not trivial.
        let pick = (rng.next() % 64).min(rng.next() % 64) as usize;
        values[pick].get().unwrap();
        model.retain(|&v| v != pick);
        model.push(pick);
        while model.len() > 10 {
            model.remove(0);
        }
        let resident: Vec<usize> = (0..64).filter(|&v| values[v].is_resident()).collect();
        let mut expected = model.clone();
        expected.sort_unstable();
        assert_eq!(resident, expected, "resident set diverged from the model at step {step}");
    }
    assert_eq!(residency_stats().resident_bytes, base + 1000);
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}

#[test]
fn choosing_a_victim_does_not_scan_every_resident_value() {
    let _turn = SERIAL.lock().unwrap();
    let base = residency_stats().resident_bytes;
    set_residency_budget(base + 20_000);
    let builds = Arc::new(AtomicUsize::new(0));
    let values: Vec<Residency<Blob>> = (0..100_000).map(|_| blob(1, &builds)).collect();

    // Every access past the first 20,000 evicts. A scan of the resident set
    // per eviction would be about 1.6e9 comparisons here.
    let started = std::time::Instant::now();
    for value in &values {
        value.get().unwrap();
    }
    let elapsed = started.elapsed();
    assert!(elapsed < std::time::Duration::from_secs(10), "took {elapsed:?}");

    let resident = values.iter().filter(|v| v.is_resident()).count();
    assert_eq!(resident, 20_000);
    assert!(values[99_999].is_resident() && !values[0].is_resident());
    set_residency_budget(DEFAULT_RESIDENCY_BUDGET);
}
