//! Fuzzes the full "network bytes to mutated World" pipeline: parse a
//! `Delta` from raw bytes, then `apply` it to a fresh `World` against a
//! locally-registered schema — exactly what a real client does with
//! server traffic. This goes well beyond the pure-parsing targets: a
//! `Delta` that decodes successfully can still carry adversarial content
//! (bogus entity bits, entity/component-id combinations that don't match
//! any live archetype, field bytes with the wrong `Replicable` framing for
//! their declared encoding) that only surfaces once `apply` actually walks
//! the archetype/column machinery.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pulsar_scenedb::*;
use pulsar_scenedb_derive::Replicate;

#[derive(Replicate, Default)]
struct FuzzComponent {
    #[replicate(encoding = Pod, condition = Always)]
    position: [f32; 3],
    #[replicate(encoding = Serialized, condition = Always)]
    name: String,
    #[replicate(encoding = Pod, condition = Always)]
    tags: [f32; 2],
}

fuzz_target!(|data: &[u8]| {
    let Some(delta) = Delta::from_bytes(data) else {
        return;
    };

    let mut registry = ReplicationRegistry::new();
    FuzzComponent::register_replication(&mut registry);

    let mut world = World::new();
    // Must never panic no matter what a successfully-decoded-but-adversarial
    // Delta contains — a real `Err` return is fine, a crash is not.
    let _ = delta.apply(&mut world, &registry);
});
