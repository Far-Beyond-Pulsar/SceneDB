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

use crate::component::{self, ComponentId};
use crate::world::World;

/// Full snapshot of a [`World`]'s live entities and components, in
/// archetype-major order.
#[derive(Serialize)]
pub struct WorldSnapshot {
    /// Total live entity count across all archetypes.
    pub entity_count: usize,
    pub archetypes: Vec<ArchetypeSnapshot>,
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

                        let mut rows_hex = Vec::with_capacity(len);
                        for row in 0..len {
                            // SAFETY: `row < len == col.len()`.
                            let ptr = unsafe { col.get_raw(row) } as *const u8;
                            // SAFETY: `ptr` points at a live element of this
                            // column, valid for `element_size` bytes.
                            let bytes =
                                unsafe { std::slice::from_raw_parts(ptr, element_size) };
                            rows_hex.push(hex_encode(bytes));
                        }

                        Some(ArchetypeColumnSnapshot {
                            component_id,
                            type_name: component::type_name(ComponentId(component_id)),
                            element_size,
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
