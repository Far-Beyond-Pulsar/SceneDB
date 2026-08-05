//! Fuzzes `Delta::from_bytes` — the raw wire-format parser for a frame's
//! state changes, the first thing untrusted network bytes go through.
//! Must never panic, never read out of bounds, and never over-allocate
//! relative to `data`'s own length (see the `*_huge_claimed_count*`
//! regression tests in `replication.rs` for the bug this exact shape of
//! fuzzing should have caught before a human had to go looking for it).
#![no_main]

use libfuzzer_sys::fuzz_target;
use pulsar_scenedb::Delta;

fuzz_target!(|data: &[u8]| {
    let _ = Delta::from_bytes(data);
});
