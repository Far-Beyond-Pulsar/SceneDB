use std::fmt;

/// A packed 64-bit handle per SceneDB 2.0 spec §3 / CONTRACTS.md C1.
///
/// Bits 0–31: stable slot index. Bits 32–63: generation.
/// Generation 0 is permanently reserved as invalid; live generations start at 1.
/// Unlike row positions, the slot index is stable for the allocation lifetime —
/// the registry's slot→row table absorbs compaction movement.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Handle(u64);

impl pulsar_reflection::Reflectable for Handle {
    fn type_info() -> &'static pulsar_reflection::RuntimeTypeInfo {
        static CELL: std::sync::OnceLock<pulsar_reflection::RuntimeTypeInfo> = std::sync::OnceLock::new();
        CELL.get_or_init(|| {
            let fields: &'static [pulsar_reflection::FieldInfo] = Box::leak(Box::new([
                pulsar_reflection::FieldInfo::new("bits", <u64 as pulsar_reflection::Reflectable>::type_info(), 0),
            ]));
            pulsar_reflection::RuntimeTypeInfo {
                type_id: std::any::TypeId::of::<Handle>(),
                type_name: "Handle",
                size: std::mem::size_of::<Handle>(),
                align: std::mem::align_of::<Handle>(),
                structure: pulsar_reflection::TypeStructure::Struct { fields },
                color: None,
            }
        })
    }

    fn serialize(&self, serializer: &mut dyn pulsar_reflection::TypeSerializer) -> pulsar_reflection::ReflectResult<()> {
        let bits = self.bits();
        serializer.serialize_struct(&[("bits", &bits as &dyn std::any::Any)])
    }

    fn deserialize(deserializer: &mut dyn pulsar_reflection::TypeDeserializer) -> pulsar_reflection::ReflectResult<Self> {
        let fields = deserializer.deserialize_struct(Self::type_info().fields().unwrap())?;
        let bits = fields.get("bits")
            .and_then(|value| value.downcast_ref::<u64>())
            .copied()
            .ok_or_else(|| pulsar_reflection::ReflectError::MissingField { struct_name: "Handle", field_name: "bits" })?;
        Ok(Self(bits))
    }

    fn clone_any(&self) -> Box<dyn std::any::Any> { Box::new(*self) }
}

pulsar_reflection::inventory::submit! {
    pulsar_reflection::RuntimeTypeRegistration {
        type_info: <Handle as pulsar_reflection::Reflectable>::type_info,
        serialize_json: |value: &dyn std::any::Any| {
            let typed = value.downcast_ref::<Handle>().ok_or_else(|| pulsar_reflection::ReflectError::TypeMismatch {
                expected: "Handle",
                found: format!("{:?}", value.type_id()),
            })?;
            let mut serializer = pulsar_reflection::JsonSerializer::new();
            <Handle as pulsar_reflection::Reflectable>::serialize(typed, &mut serializer)?;
            Ok(serializer.into_json())
        },
        deserialize_json: |value: serde_json::Value| {
            let mut deserializer = pulsar_reflection::JsonDeserializer::new(value);
            let typed = <Handle as pulsar_reflection::Reflectable>::deserialize(&mut deserializer)?;
            Ok(Box::new(typed) as Box<dyn std::any::Any>)
        },
    }
}
impl Handle {
    /// The canonical invalid handle (all zero — generation 0).
    pub const INVALID: Handle = Handle(0);

    /// Constructs a handle from a raw slot index and generation.
    ///
    /// Passing `generation = 0` produces an invalid handle. The registry always
    /// starts allocations at generation 1.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | (index as u64))
    }

    /// Stable slot index (bits 0–31). NOT a row offset — resolve through the
    /// registry's slot→row table.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0 as u32
    }

    /// Generation (bits 32–63). 0 = invalid.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns `true` if this handle has a non-zero generation.
    ///
    /// A zero generation is permanently reserved as invalid (spec §3, C1).
    /// Note: slot index 0 is valid; only generation 0 is not.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.generation() != 0
    }

    /// The raw packed value (e.g. for GPU upload).
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index(), self.generation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_and_unpacks() {
        let h = Handle::new(14, 2);
        assert_eq!(h.index(), 14);
        assert_eq!(h.generation(), 2);
    }

    #[test]
    fn generation_zero_is_invalid() {
        assert!(!Handle::new(14, 0).is_valid());
        assert!(Handle::new(14, 1).is_valid());
        assert!(!Handle::INVALID.is_valid());
        assert_eq!(Handle::INVALID.index(), 0);
        assert_eq!(Handle::INVALID.generation(), 0);
    }

    #[test]
    fn max_index_and_generation_roundtrip() {
        let h = Handle::new(u32::MAX, u32::MAX);
        assert_eq!(h.index(), u32::MAX);
        assert_eq!(h.generation(), u32::MAX);
    }

    #[test]
    fn bits_round_trip() {
        let h = Handle::new(14, 2);
        assert_eq!(h.bits(), (2u64 << 32) | 14u64);
        assert_eq!(Handle::INVALID.bits(), 0);
    }

    #[test]
    fn debug_format() {
        assert_eq!(format!("{:?}", Handle::new(14, 2)), "Handle(14v2)");
        assert_eq!(format!("{:?}", Handle::INVALID), "Handle(0v0)");
    }
}
