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
//! Every column is always dumped as raw hex bytes per row via
//! `ErasedColumn`'s type-erased accessors -- a universal fallback that is
//! meaningful for the `#[gpu(layout = packed)]` POD components this crate's
//! GPU mirror requires, but **not** guaranteed meaningful for arbitrary
//! non-POD component types (a `String` field, for instance, dumps its `Vec`
//! internals, not its text).
//!
//! When a column's type is additionally registered with
//! `pulsar_reflection` (`#[derive(Reflectable)]`/`#[pulsar_type]` --
//! `pulsar_scenedb::token::TypeToken` already threads every `#[gpu]`
//! component type through that same registry, so this is the common case
//! for real engine components), each row is *also* decoded into a
//! structured `serde_json::Value` via `RUNTIME_TYPE_REGISTRY`, recursively
//! -- nested structs, enums, and wrapper types (`Vec`/`Option`/etc.) all
//! come through as real nested JSON, not just their top-level bytes. This
//! goes through `ErasedColumn::get_any`, which erases each row to `&dyn
//! Any` typed as the element type itself (unlike `as_any`, which erases to
//! the owning `Column<T>`), so the registry's `TypeId`-keyed dispatch finds
//! the right (monomorphized, macro-generated) serializer without this
//! module ever needing to know `T` at compile time.

use serde::Serialize;

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
    /// Structured, reflection-decoded value per row -- `Some` (one entry
    /// per row, same order as [`ArchetypeSnapshot::entities`]) iff this
    /// column's type is registered with `pulsar_reflection`; `None` for a
    /// type with no reflection registration, in which case `rows_hex` is
    /// the only available view of this column's contents.
    pub rows_reflected: Option<Vec<serde_json::Value>>,
    /// Hex-encoded raw bytes, row-major: `[row0_bytes, row1_bytes, ...]`,
    /// same order as the owning [`ArchetypeSnapshot::entities`]. Only
    /// populated when `rows_reflected` is `None` -- computing both for
    /// every row of every reflectable component (the common case for real
    /// engine components) would silently double the cost of every
    /// snapshot for no reader-visible benefit; nothing consumes raw hex
    /// once a structured decode is available. Empty (not omitted) when
    /// `rows_reflected` is `Some`, so the field always round-trips through
    /// JSON as an array rather than becoming type-dependent on the wire.
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

                        // Reflection is opt-in per type (only types that
                        // registered via `#[derive(Reflectable)]`/
                        // `#[pulsar_type]` show up here), so check once
                        // rather than per row -- every row in a column
                        // shares the same element type -- and branch on it
                        // rather than always doing both encodings (see
                        // `rows_hex`'s doc comment for why that matters).
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
                                    Ok(v) => v,
                                    Err(e) => serde_json::json!({
                                        "__reflection_error__": e.to_string(),
                                    }),
                                };
                                rows_reflected_vec.push(value);
                            }
                        } else {
                            rows_hex.reserve(len);
                            for row in 0..len {
                                // SAFETY: `row < len == col.len()`.
                                let ptr = unsafe { col.get_raw(row) } as *const u8;
                                // SAFETY: `ptr` points at a live element of
                                // this column, valid for `element_size`
                                // bytes.
                                let bytes =
                                    unsafe { std::slice::from_raw_parts(ptr, element_size) };
                                rows_hex.push(hex_encode(bytes));
                            }
                        }

                        let rows_reflected = is_reflectable.then_some(rows_reflected_vec);

                        Some(ArchetypeColumnSnapshot {
                            component_id,
                            type_name: component::type_name(ComponentId(component_id)),
                            element_size,
                            rows_reflected,
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
