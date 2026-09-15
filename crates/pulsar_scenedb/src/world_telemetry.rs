//! World-level telemetry: a serializable snapshot of an archetype-based
//! [`crate::World`]'s live CPU-side state, for external inspector/debug
//! tooling.
//!
//! This complements, rather than replaces, [`crate::telemetry`]'s
//! `TelemetrySnapshot`: that type was built for the spatial-cell/GPU-store
//! layer (`CellStorage` + `SceneGpuStore`, fed by hand in
//! `examples/telemetry_demo.rs`), a separate storage layer from `World`'s
//! archetype-based ECS. `World` has no public API to enumerate its
//! archetypes/columns generically -- `ErasedColumn` is `pub(crate)`, by
//! design -- so this snapshot has to be built from inside the crate; that is
//! the entire reason this module exists as a small, deliberate addition here
//! rather than something downstream tooling can bolt on externally.
//!
//! Every column is dumped as raw hex bytes per row via `ErasedColumn`'s
//! type-erased accessors. That is meaningful for the `#[gpu(layout =
//! packed)]` POD components this crate's GPU mirror requires (the common
//! case for any component reaching the GPU), but is **not** guaranteed
//! meaningful for arbitrary non-POD component types -- a `String` field, for
//! instance, dumps its `Vec` internals (pointer/len/cap), not its text. This
//! mirrors the same caveat `telemetry::ColumnData::rows_hex` already
//! documents for the cell-based snapshot.

use serde::Serialize;
use serde_json::Value;

use crate::component::{self, ComponentId, ErasedColumn};
use crate::world::World;
use pulsar_reflection::RUNTIME_TYPE_REGISTRY;

/// Full snapshot of a [`World`]'s live entities and components, in
/// archetype-major order.
#[derive(Serialize)]
pub struct WorldSnapshot {
    /// Total live entity count across all archetypes.
    pub entity_count: usize,
    pub archetypes: Vec<ArchetypeSnapshot>,
    /// SceneDB-owned GPU storage snapshot, when this world has a mirror.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<crate::telemetry::GpuSnapshot>,
    /// Optional SceneDB-owned projections, including the GPU mirror and upload state.
    #[serde(flatten)]
    pub extensions: std::collections::BTreeMap<String, Value>,
}

impl WorldSnapshot {
    /// Attach a serializable projection owned by SceneDB, such as GPU mirror or upload diagnostics.
    pub fn insert_snapshot_extension(&mut self, key: impl Into<String>, value: Value) {
        self.extensions.insert(key.into(), value);
    }
}

#[derive(Serialize)]
pub struct ArchetypeSnapshot {
    pub id: u32,
    pub entity_count: usize,
    /// Packed [`crate::Entity::bits`] for every live row, same order as
    /// each column's `rows_hex`.
    pub entities: Vec<u64>,
    pub columns: Vec<ArchetypeColumnSnapshot>,
}

#[derive(Serialize)]
pub struct ArchetypeColumnSnapshot {
    pub component_id: u32,
    /// `std::any::type_name::<T>()` recorded when this component type was
    /// first registered. Display only -- see [`component::type_name`].
    pub type_name: &'static str,
    pub element_size: usize,
    /// Structured reflection-decoded value per row when the type is registered.
    pub rows_reflected: Option<Vec<serde_json::Value>>,
    /// Hex-encoded raw bytes, row-major: `[row0_bytes, row1_bytes, ...]`,
    /// same order as the owning [`ArchetypeSnapshot::entities`].
    pub rows_hex: Vec<String>,
}

impl World {
    /// Collect a [`WorldSnapshot`] of this world's current archetypes.
    ///
    /// `O(total component bytes across all live entities)` -- copies every
    /// live column's raw bytes as hex. Intended for periodic polling by
    /// external inspector/debug tooling (e.g. once every few frames), not
    /// the hot render/simulation path.
    pub fn telemetry_snapshot(&self) -> WorldSnapshot {
        let archetypes: Vec<ArchetypeSnapshot> = self
            .archetypes
            .iter()
            .map(|archetype| {
                let columns: Vec<ArchetypeColumnSnapshot> = archetype
                    .columns
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, col)| {
                        let col = col.as_ref()?;
                        let component_id = idx as u32;
                        let element_size = col.element_size();
                        let len = col.len();

                        let is_reflectable = RUNTIME_TYPE_REGISTRY
                            .get_by_id(ErasedColumn::type_id(&**col))
                            .is_some();
                        let mut rows_hex = Vec::new();
                        let mut rows_reflected_vec = Vec::new();

                        if is_reflectable {
                            rows_reflected_vec.reserve(len);
                            for row in 0..len {
                                let value = match RUNTIME_TYPE_REGISTRY
                                    .serialize_json_for_any(col.get_any(row))
                                {
                                    Ok(value) => value,
                                    Err(error) => serde_json::json!({
                                        "__reflection_error__": error.to_string(),
                                    }),
                                };
                                rows_reflected_vec.push(value);
                            }
                        } else {
                            rows_hex.reserve(len);
                            for row in 0..len {
                                // SAFETY: `row < len == col.len()`.
                                let ptr = unsafe { col.get_raw(row) } as *const u8;
                                // SAFETY: `ptr` is valid for `element_size` bytes.
                                let bytes =
                                    unsafe { std::slice::from_raw_parts(ptr, element_size) };
                                rows_hex.push(hex_encode(bytes));
                            }
                        }

                        Some(ArchetypeColumnSnapshot {
                            component_id,
                            type_name: component::type_name(ComponentId(component_id)),
                            element_size,
                            rows_reflected: is_reflectable.then_some(rows_reflected_vec),
                            rows_hex,
                        })
                    })
                    .collect();

                ArchetypeSnapshot {
                    id: archetype.id.0,
                    entity_count: archetype.entities.len(),
                    entities: archetype.entities.iter().map(|e| e.bits()).collect(),
                    columns,
                }
            })
            .collect();

        WorldSnapshot {
            entity_count: archetypes.iter().map(|a| a.entity_count).sum(),
            gpu: self.gpu_mirror().map(|mirror| crate::telemetry::collect_gpu_snapshot(mirror.store())),
            extensions: std::collections::BTreeMap::new(),
            archetypes,
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(hex_char(b >> 4));
        s.push(hex_char(b & 0x0f));
    }
    s
}

fn hex_char(v: u8) -> char {
    if v < 10 {
        (b'0' + v) as char
    } else {
        (b'a' + v - 10) as char
    }
}
