use pulsar_scenedb::{ReplicationEncoding, ReplicationCondition, EventChannel, ReplicationRegistry};
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

    let cid = pulsar_scenedb::component_id::<TestComp>();
    let schema = registry.schema(cid).unwrap();
    assert_eq!(schema.fields.len(), 3);
    assert!(matches!(schema.fields[0].encoding, ReplicationEncoding::Pod));
    assert!(matches!(schema.fields[1].encoding, ReplicationEncoding::DeltaCompressed));
    assert!(matches!(schema.fields[2].encoding, ReplicationEncoding::Event));
    assert_eq!(schema.fields[2].event_channel, Some(EventChannel::ReliableOrdered));
}
