//! Fuzzes every built-in `Replicable::replicate_decode` impl — the actual
//! boundary where untrusted per-field bytes become owned Rust values
//! (`String`, `Vec<T>`, `Option<T>`, fixed-size float arrays, Pod scalars).
//! This is the core soundness fix's own decode path: before it existed,
//! reconstructing a non-`Pod` field from network bytes was undefined
//! behavior by construction, not just a bug that fuzzing might find.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pulsar_scenedb::Replicable;

fuzz_target!(|data: &[u8]| {
    let _ = String::replicate_decode(data);
    let _ = Vec::<u32>::replicate_decode(data);
    let _ = Vec::<u8>::replicate_decode(data);
    let _ = Vec::<String>::replicate_decode(data);
    let _ = Option::<u32>::replicate_decode(data);
    let _ = Option::<Vec<u32>>::replicate_decode(data);
    let _ = <[f32; 2]>::replicate_decode(data);
    let _ = <[f32; 3]>::replicate_decode(data);
    let _ = <[f32; 4]>::replicate_decode(data);
    let _ = f32::replicate_decode(data);
    let _ = u64::replicate_decode(data);
});
