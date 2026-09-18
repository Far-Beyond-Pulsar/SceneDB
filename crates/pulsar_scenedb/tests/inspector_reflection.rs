//! A `#[derive(SceneStore)]` component must register itself with
//! `pulsar_reflection` and decode to structured JSON -- not raw hex --
//! exactly like a hand-written `#[derive(Reflectable)]` type. This is the
//! property the inspector's details pane depends on, and it is a separate
//! integration crate because the derive emits `::pulsar_scenedb::...` paths
//! that only resolve for a consumer of the crate.
#![cfg(feature = "telemetry")]

use pulsar_scenedb::{InspectorCpuRequest, SceneStore, World};

#[derive(Clone, Copy, Debug, PartialEq, SceneStore)]
struct ReflectionProbe {
    #[gpu]
    position: [f32; 3],
    #[gpu]
    id: u32,
}

#[test]
fn scene_store_component_decodes_via_reflection() {
    let mut world = World::new();
    world.spawn_bundle((ReflectionProbe {
        position: [1.0, 2.0, 3.0],
        id: 42,
    },));

    let meta = world.telemetry_snapshot_metadata();
    let arch = meta
        .archetypes
        .iter()
        .find(|a| !a.columns.is_empty())
        .expect("spawned archetype");
    let probe = arch
        .columns
        .iter()
        .find(|c| c.type_name.ends_with("ReflectionProbe"))
        .expect("ReflectionProbe column");

    let response = world
        .inspector_cpu_range(&InspectorCpuRequest {
            archetype_id: arch.id,
            component_id: probe.component_id,
            row_start: 0,
            row_count: 1,
        })
        .expect("range served");

    let rows = response
        .rows_reflected
        .expect("a SceneStore component must be reflected, not hex-dumped");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["id"], serde_json::json!(42));
    assert_eq!(rows[0]["position"], serde_json::json!([1.0, 2.0, 3.0]));
    assert!(
        response.rows_hex.is_empty(),
        "reflected columns must not also emit hex"
    );
}
