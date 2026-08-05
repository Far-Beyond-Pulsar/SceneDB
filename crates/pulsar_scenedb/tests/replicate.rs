use pulsar_scenedb::{
    component_id, ReplicationEncoding, ReplicationCondition,
    EventChannel, ReplicationRegistry, Replicable, Delta, ComponentDelta, World,
};
use pulsar_scenedb_derive::Replicate;

#[derive(Replicate, Default)]
struct TestComp {
    #[replicate(encoding = Pod, condition = Always)]
    pos: [f32; 3],

    #[replicate(encoding = DeltaCompressed, condition = SimulatedOnly)]
    health: f32,

    #[replicate(encoding = Event, condition = Multicast)]
    on_damage: u32,
}

#[test]
fn replicate_derive_registers_schema() {
    let mut registry = ReplicationRegistry::new();
    TestComp::register_replication(&mut registry);

    let cid = component_id::<TestComp>();
    let schema = registry.schema(cid).unwrap();
    assert_eq!(schema.fields.len(), 3);
    assert!(matches!(schema.fields[0].encoding, ReplicationEncoding::Pod));
    assert!(matches!(schema.fields[1].encoding, ReplicationEncoding::DeltaCompressed));
    assert!(matches!(schema.fields[2].encoding, ReplicationEncoding::Event));
    assert_eq!(schema.fields[2].event_channel, Some(EventChannel::ReliableOrdered));
}

#[derive(Replicate, Default)]
struct EventChannelOverride {
    #[replicate(encoding = Event, condition = Multicast, event_channel = Unreliable)]
    on_ping: u32,
}

#[test]
fn replicate_derive_event_channel_override() {
    let mut registry = ReplicationRegistry::new();
    EventChannelOverride::register_replication(&mut registry);

    let cid = component_id::<EventChannelOverride>();
    let schema = registry.schema(cid).unwrap();
    assert_eq!(schema.fields.len(), 1);
    assert!(matches!(schema.fields[0].encoding, ReplicationEncoding::Event));
    assert_eq!(schema.fields[0].condition, ReplicationCondition::Multicast);
    assert_eq!(schema.fields[0].event_channel, Some(EventChannel::Unreliable));
}

#[derive(Replicate)]
struct GenericComp<T: pulsar_scenedb::Component> {
    #[replicate(encoding = Pod, condition = Always)]
    tag: u32,
    _marker: std::marker::PhantomData<T>,
}

// Written by hand rather than `#[derive(Default)]`: the derive adds a
// `T: Default` bound to the generated impl for every generic parameter
// regardless of whether it's actually needed (a well-known derive-macro
// over-constraint) — `PhantomData<T>` is `Default` unconditionally, so `T`
// need only be `Component` here, matching the struct's own bound.
impl<T: pulsar_scenedb::Component> Default for GenericComp<T> {
    fn default() -> Self {
        Self { tag: 0, _marker: std::marker::PhantomData }
    }
}

#[test]
fn replicate_derive_supports_type_generics() {
    let mut registry = ReplicationRegistry::new();
    GenericComp::<u64>::register_replication(&mut registry);

    let cid = component_id::<GenericComp<u64>>();
    let schema = registry.schema(cid).unwrap();
    assert_eq!(schema.fields.len(), 1);
    assert!(matches!(schema.fields[0].encoding, ReplicationEncoding::Pod));

    // A different instantiation of the same generic struct is a distinct
    // component type with its own schema slot.
    let mut registry2 = ReplicationRegistry::new();
    GenericComp::<i8>::register_replication(&mut registry2);
    let cid2 = component_id::<GenericComp<i8>>();
    assert_ne!(cid, cid2);
    assert!(registry2.schema(cid2).is_some());
}

// ── The actual soundness fix: owned/heap data replicated generically ────
//
// Before the `Replicable` trait, `Delta::apply` reconstructed components by
// reinterpreting raw bytes as the concrete type — sound only for `Pod`
// types. A `String`/`Vec<u32>` field would have read a heap pointer back
// from garbage/zeroed bytes: real undefined behavior. This test is the
// concrete case that must now work safely and correctly.

#[derive(Replicate, Default)]
struct Profile {
    #[replicate(encoding = Serialized, condition = Always)]
    name: String,
    #[replicate(encoding = Serialized, condition = Always)]
    tags: Vec<u32>,
}

#[test]
fn generic_owned_data_round_trips_through_delta_apply() {
    let mut registry = ReplicationRegistry::new();
    Profile::register_replication(&mut registry);
    let cid = component_id::<Profile>();

    // The entity handle is a value shared verbatim between peers (see
    // `Delta::apply`'s doc) — spawn it on a stand-in "server" world just to
    // get a real handle, independent of the "client" world that will
    // reconstruct it from nothing but wire bytes below.
    let mut server = World::new();
    let e = server.spawn();

    let mut name_bytes = Vec::new();
    "Alice".to_string().replicate_encode(&mut name_bytes);
    let mut tags_bytes = Vec::new();
    vec![1u32, 2, 3].replicate_encode(&mut tags_bytes);

    let blob = pulsar_scenedb::encode_archetype_key(&[cid]);
    let delta = Delta {
        frame: 0,
        base_frame: 0,
        spawned: vec![(e, blob)],
        despawned: vec![],
        component_deltas: vec![ComponentDelta {
            entity: e,
            component_type: cid,
            field_data: vec![name_bytes, tags_bytes],
        }],
        events: vec![],
    };

    // Fresh client world — this entity/component do not exist here at all
    // until `apply` reconstructs them purely from the encoded bytes above.
    let mut client = World::new();
    assert_eq!(delta.apply(&mut client, &registry), Ok(()));
    assert!(client.is_alive(e));

    let profile = client.get::<Profile>(e).expect("Profile component reconstructed");
    assert_eq!(profile.name, "Alice");
    assert_eq!(profile.tags, vec![1, 2, 3]);
}

#[test]
fn string_replicate_decode_rejects_invalid_utf8_without_panicking() {
    let bad = vec![0xFFu8, 0xFE, 0xFD];
    let result = String::replicate_decode(&bad);
    assert!(result.is_err(), "invalid UTF-8 must be rejected, not panic or produce garbage");
}
