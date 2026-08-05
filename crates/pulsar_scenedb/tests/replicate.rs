use pulsar_scenedb::{
    component_id, ReplicationEncoding, ReplicationCondition,
    EventChannel, ReplicationRegistry,
};
use pulsar_scenedb_derive::Replicate;

#[derive(Replicate)]
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

#[derive(Replicate)]
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
