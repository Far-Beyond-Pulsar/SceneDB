//! # SceneDB Replication Primitives
//!
//! SceneDB is the natural home for replication primitives because it already
//! owns the authoritative state of every scene in the engine — entities,
//! components, spatial cells, liveness, handles, generations, and the frame
//! phase machine. Replication is not a bolt-on service; it is the data layer
//! exposing a controlled, observable, and filterable stream of its own
//! mutations.
//!
//! ## Design tenets
//!
//! 1. **SceneDB owns the data pipeline** — change tracking, delta encoding,
//!    interest management, authority, and condition filtering. Everything up
//!    to encoded byte blobs.
//! 2. **SceneDB does NOT own transport** — networking, encryption, connection
//!    management, and asset streaming live in the engine. SceneDB produces
//!    `Delta` frames and `ReplicatedEvent` payloads; the engine ships them.
//! 3. **SceneDB does NOT own asset payloads** — a `gpu_handle`-mode field
//!    replicates only the handle index (8 bytes), not the vertex data. The
//!    asset system (`engine-fs`, streaming, etc.) independently ensures the
//!    resource exists on the remote peer.
//! 4. **SceneDB does NOT own editor collaboration** — operational transform,
//!    lock servers, undo history, and CRDTs live in the editor. SceneDB
//!    provides `Shared` ownership + deterministic frame-batched conflict
//!    resolution; the editor builds collaboration semantics on top.
//! 5. **Endianness is a non-concern** — every target platform (x86-64, ARM64,
//!    ARM64EC, WebAssembly) is little-endian. The build asserts
//!    `cfg!(target_endian = "little")` and fails fast otherwise.
//!
//! ## Two orthogonal axes
//!
//! Every replicated field declares:
//!
//! - A [`ReplicationEncoding`] — *how* the data is encoded on the wire
//! - A [`ReplicationCondition`] — *who* receives it and *who* owns it
//!
//! ## ReplicationEncoding
//!
//! | Variant | Wire cost | When to use |
//! |---|---|---|
//! | `Pod` | `sizeof(T)` bytes, direct memcpy | Simple value types — transforms, stats, enums. Default for anything implementing `Pod`. Schema negotiated once at handshake. |
//! | `Serialized` | Variable | Reflection-based via `EngineClass`. For blueprint/visual-scripting components. |
//! | `GpuHandle` | `sizeof(Handle)` = 8 B | Mesh references, texture handles, buffer bindings. Only the registry index travels; the GPU resource is loaded independently by the asset system. |
//! | `DeltaCompressed` | Small, variable | Slowly-changing values (health, cooldown, ammo). XOR-diff from the last acknowledged value, then LEB128 or run-length encoded. |
//! | `Event` | 0 in state deltas | One-shot RPC-style delivery. Never appears in frame snapshots or reconciliation state. Delivered on a separate channel. |
//! | `Opaque` | Custom | Escape hatch. The component provides `encode`/`decode` fn pointers at registration time. |
//!
//! ## ReplicationCondition
//!
//! Replication conditions jointly control two things:
//!
//! - **Visibility** — which clients receive this field's value in deltas
//! - **Authority** — which peer is allowed to write it
//!
//! | Condition | Visibility | Authority | Unreal equivalent |
//! |---|---|---|---|
//! | `Always` | All clients | Server | `COND_None` |
//! | `OwnerOnly` | Owning client only | Server | `COND_OwnerOnly` |
//! | `SkipOwner` | All except owner | Server | `COND_SkipOwner` |
//! | `SimulatedOnly` | Non-owning clients only | Server | `COND_SimulatedOnly` |
//! | `AutonomousOnly` | Owning client only | Server | `COND_AutonomousOnly` |
//! | `InitialOnly` | Once, at spawn | Server | `COND_InitialOnly` |
//! | `ServerAuthority` | All clients | **Server** | default server-replicated |
//! | `ClientAuthority` | All clients | **Owning client** | client-replicated movement |
//! | `ServerToClient` | One specific client | Server | `Client` RPC direction |
//! | `ClientToServer` | Server only | Owning client | `Server` RPC direction |
//! | `Multicast` | All except sender | Anyone | `NetMulticast` RPC |
//!
//! `ServerAuthority` is the default for state fields. `ClientAuthority` is
//! used for fields the owning client controls (character movement input,
//! camera look) — the server still validates bounds and rejects violations.
//!
//! ## Events (RPCs)
//!
//! Fields declared with `encoding = ReplicationEncoding::Event` are **not**
//! state. They are one-shot invocations with typed arguments, delivered
//! on a separate reliable-or-unreliable channel:
//!
//! The `ReplicationCondition` on an event field determines direction:
//!
//! | Condition | Direction |
//! |---|---|
//! | `ClientToServer` | Client fires → server receives (Unreal "Server" RPC) |
//! | `ServerToClient` | Server fires → one client receives (Unreal "Client" RPC) |
//! | `Multicast` | Anyone fires → everyone else receives (Unreal "NetMulticast") |
//!
//! Events are queued on the `ChangeTracker` as they fire and flushed once
//! per frame. Reliability is declared on the field, not negotiated per-call.
//!
//! ## Schema registration
//!
//! ```ignore
//! registry.register::<MeshRenderer>()
//!     .field("mesh",            |c| &c.mesh,            |c| &mut c.mesh,            ReplicationEncoding::GpuHandle,   ReplicationCondition::Always)
//!     .field("local_transform", |c| &c.local_transform, |c| &mut c.local_transform, ReplicationEncoding::Pod,         ReplicationCondition::ServerAuthority)
//!     .field("health",          |c| &c.health,          |c| &mut c.health,          ReplicationEncoding::DeltaCompressed, ReplicationCondition::SimulatedOnly)
//!     .event("on_hit",          ReplicationCondition::Multicast,  EventChannel::Unreliable);
//! ```
//!
//! `#[derive(Replicate)]` generates exactly this shape from
//! `#[replicate(...)]` field attributes — see that macro's doc.
//!
//! The registry produces a `ReplicationSchema` — a compact per-component-type
//! descriptor table that the delta encoder walks at runtime. The schema is
//! also shared with remote peers during the initial connection handshake so
//! both sides agree on field layout and encoding.
//!
//! ## Authority model
//!
//! - `Server` is the default. Server authority + client prediction via the
//!   `Reconciler`.
//! - `Client(ClientId)` gives a specific client write permission. Server
//!   still receives the delta, validates bounds, and re-broadcasts.
//! - `Shared` is for multi-user editor sessions. Both peers can write the
//!   same field in the same frame. At the frame boundary, conflicts are
//!   resolved deterministically: the peer with the higher `ClientId` wins.
//!   No locks, no operational transform — optimistic apply with frame-level
//!   rollback.
//!
//! ## The delta pipeline
//!
//! Every frame, at the **SimulateB→Harvest** phase boundary:
//!
//! ```text
//! 1. ChangeTracker.drain()
//!      ↓
//! 2. For each connected client:
//!      a. RelevanceSet.filter(delta, client)
//!           - spatial filter (SpatialCell::query_aabb_in)
//!           - condition filter (ReplicationCondition per field)
//!      b. Encode filtered fields via ReplicationEncoding
//!      c. Append events from the event queue
//!      ↓
//! 3. Emit Delta (state) + EventBatch (RPCs)
//!      ↓
//! 4. Engine transports to remote peer
//! ```
//!
//! Step 2a reuses the existing SIMD-accelerated `SpatialCell::query_aabb_in`
//! with `LivenessSnapshot` — zero allocation, zero serialization for
//! out-of-relevance entities.
//!
//! ## Where SceneDB stops
//!
//! ```text
//! SceneDB owns:
//!   ┌──────────────────────────────────────────────────┐
//!   │ ChangeTracker → Delta/EventBatch                 │
//!   │ RelevanceSet  → per-connection filter            │
//!   │ AuthorityTable → condition + conflict resolution │
//!   │ ReplicationSchema → encoding dispatch            │
//!   │ Reconciler → client prediction + rollback        │
//!   └──────────────────────────────────────────────────┘
//!
//! NOT SceneDB:
//!   - Network transport (TCP, UDP, WebSocket, Steam, EOS, etc.)
//!   - Encryption, authentication, anti-cheat
//!   - Asset streaming (mesh/texture/sound payloads)
//!   - Editor OT/CRDT, lock server, undo history
//!   - Connection lifecycle, NAT punch, relay
//! ```
//!
//! ## Implementation plan
//!
//! ### R1 — Core types and ChangeTracker
//!
//! - Define `ReplicationEncoding`, `ReplicationCondition`, `EventChannel`,
//!   `ClientId`, `Ownership`, `ReplicatedEvent`.
//! - Implement `ChangeTracker` with hooks for `spawn`, `despawn`,
//!   `insert`, `remove`, `set` on `World`.
//! - Wire `World` methods to accept `&mut ChangeTracker`.
//! - Unit tests verifying correct change accumulation across a frame.
//!
//! ### R2 — Schema and delta encoding
//!
//! - Define `ReplicationSchema` and `ReplicationRegistry`.
//! - Derive macro `#[replicate(...)]` for component fields.
//! - Implement `Delta` struct + encoding for all built-in
//!   `ReplicationEncoding` variants (Pod is a direct memcpy, etc.).
//! - Schema handshake message for connection initialization.
//! - Unit tests: round-trip encode/decode for every encoding mode.
//!
//! ### R3 — Relevance and conditions
//!
//! - Implement `RelevanceSet` with spatial filter
//!   (delegates to `SpatialCell::query_aabb_in`).
//! - Implement condition filter (owner check, simulated/autonomous check).
//! - Implement `AuthorityTable` with conflict detection at frame boundary.
//! - Integration test: 10,000 entities, 4 simulated clients, verify each
//!   receives only its relevant subset.
//!
//! ### R4 — Event channel
//!
//! - Wire event fields through `ChangeTracker` with a dedicated event queue.
//! - Define `EventBatch` message (separate from `Delta`).
//! - Implement direction enforcement (Client→Server, Server→Client, Multicast).
//! - Unit tests: event delivery ordering, reliability modes, dropped-event
//!   detection.
//!
//! ### R5 — Snapshot and reconciliation
//!
//! - Implement `Snapshot` (full/partial world state at a frame).
//! - Implement `Reconciler` with history ring buffer + pending input replay.
//! - Support `Shared` ownership with deterministic conflict resolution.
//! - Integration test: client-side prediction with server correction,
//!   verify rollback converges within 3 frames.

use crate::archetype::ArchetypeKey;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::snapshot::LivenessSnapshot;
use crate::spatial::{Aabb, SpatialCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

// ── Error reporting ────────────────────────────────────────────────────────

/// Result code returned by encode/decode operations, especially Opaque mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    Ok,
    /// The output buffer was too small for the encoded data.
    BufferTooSmall { needed: usize },
    /// The input data was malformed or the schema didn't match.
    InvalidData,
    /// Version mismatch between encoder and decoder.
    VersionMismatch,
    /// Catch-all for custom errors in Opaque mode.
    Custom(u32),
}

// ── Client identity ────────────────────────────────────────────────────────

/// Opaque identifier for a connected client/session.
/// Assigned by the engine's connection manager, not by SceneDB.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientId(pub u64);

// ── Encoding ───────────────────────────────────────────────────────────────

/// How a replicated field's value is encoded on the wire.
#[derive(Clone, Debug)]
pub enum ReplicationEncoding {
    /// Direct memcpy of Pod bytes. Schema negotiated at handshake.
    Pod,
    /// Reflection-based via EngineClass (blueprint components).
    Serialized,
    /// Only the registry handle/index (8 bytes). Asset payload is out-of-band.
    GpuHandle,
    /// XOR-diff from last acknowledged value, LEB128/RLE compressed.
    DeltaCompressed,
    /// One-shot RPC. Never included in state deltas.
    Event,
    /// Custom encode/decode closures provided at registration.
    /// `encode_size` returns the exact number of bytes needed to encode
    /// the value at `*const ()`.
    /// `encode` writes into `&mut [u8]` (pre-sized via `encode_size`).
    /// `decode` reads from `&[u8]` into the destination at `*mut ()`.
    Opaque {
        encode_size: fn(*const ()) -> usize,
        encode: fn(*const (), &mut [u8]) -> ErrorCode,
        decode: fn(&[u8], *mut ()) -> ErrorCode,
    },
}

// ── Conditions ─────────────────────────────────────────────────────────────

/// Controls visibility (who receives) and authority (who writes) for a field.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReplicationCondition {
    /// All clients; server writes.
    Always,
    /// Owning client only; server writes.
    OwnerOnly,
    /// All except owner; server writes.
    SkipOwner,
    /// Non-owning clients only; server writes.
    SimulatedOnly,
    /// Owning client only; server writes.
    AutonomousOnly,
    /// Once at entity spawn. Never in subsequent deltas.
    InitialOnly,

    /// All clients; **server** writes (default for state).
    ServerAuthority,
    /// All clients; **owning client** writes (server validates).
    ClientAuthority,

    /// Server → one client (unidirectional).
    ServerToClient,
    /// Client → server (unidirectional).
    ClientToServer,
    /// One sender → all others.
    Multicast,
}

// ── Event channel ──────────────────────────────────────────────────────────

/// Delivery guarantees for the event (RPC) channel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventChannel {
    /// Delivered in order, with retransmission on loss.
    ReliableOrdered,
    /// Fire-and-forget. May be dropped; may arrive out of order.
    Unreliable,
}

/// A one-shot RPC invocation, delivered separately from state deltas.
#[derive(Clone, Debug)]
pub struct ReplicatedEvent {
    pub entity: Entity,
    pub component_type: ComponentId,
    pub event_field: u32,
    pub payload: Vec<u8>,
    pub channel: EventChannel,
    /// For `ServerToClient` direction: the intended recipient.
    /// `None` for `ClientToServer` and `Multicast`.
    pub target_client: Option<ClientId>,
}

// ── Ownership ──────────────────────────────────────────────────────────────

/// Who is allowed to modify an entity or field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Ownership {
    /// Server exclusively (authoritative multiplayer — default).
    Server,
    /// A specific client (client-authoritative movement, etc.).
    Client(ClientId),
    /// Anyone (multi-user editor — optimistic, deterministic tiebreak).
    Shared,
}

// ── Replicable ───────────────────────────────────────────────────────────────

/// Safe, generic wire encode/decode for one replicated *value* — a whole
/// component (for [`SchemaBuilder::whole_field`]) or one named field of it
/// (for [`SchemaBuilder::field`]).
///
/// This is the fix for the fundamental hole a byte-oriented replication
/// design has to close: [`crate::component::Column`] stores real `T` values
/// for ANY `T: Any + Send + Sync + 'static` (a `String`, a `Vec<u8>`, a
/// `Box<dyn Trait>` — anything), but the wire only ever carries bytes.
/// Reinterpreting arbitrary bytes as an arbitrary `T` (the previous
/// approach) is undefined behavior for anything that isn't
/// [`crate::page::Pod`] — a heap pointer read back from garbage bytes is a
/// real memory-safety bug, not a hypothetical one. `Replicable` moves that
/// boundary to a place where it can be checked by the compiler: every type
/// used in a replicated field must implement this trait, and every impl is
/// either the blanket `Pod` fast path below (reusing the crate's existing,
/// already-audited safety marker — no new unsafe surface) or ordinary safe
/// Rust (as the `String`/`Vec`/`Option`/`Box` impls below are).
///
/// Implement this yourself for any other type you want to replicate — the
/// bar is "safely constructible from bytes you produced yourself", which is
/// a much smaller ask than "safe to reinterpret from arbitrary bytes".
pub trait Replicable: Sized {
    /// A placeholder value used to grow a column by one row when an entity
    /// is spawned before its real field values arrive (see
    /// [`crate::World`]'s replication-support methods). Called once per
    /// spawned entity per field — never exposed to a remote peer.
    fn replicate_default() -> Self;

    /// Append this value's wire representation to `buf`. Must be exactly
    /// invertible by [`Self::replicate_decode`] given the bytes appended
    /// (and nothing else — no reliance on out-of-band length information).
    fn replicate_encode(&self, buf: &mut Vec<u8>);

    /// Reconstruct a value from exactly the bytes [`Self::replicate_encode`]
    /// produced for it. Must return [`ErrorCode::InvalidData`] (never
    /// panic, never read out of bounds) for truncated or malformed input —
    /// `bytes` may come from an untrusted remote peer.
    fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode>;
}

/// Fast path: any [`crate::page::Pod`] type is `Replicable` via a direct
/// memcpy. `Pod`'s own safety contract (all-zero / arbitrary bytes are a
/// valid value) is exactly what makes this sound — this reuses that
/// existing, audited marker rather than introducing a new unsafe contract.
impl<T: crate::page::Pod> Replicable for T {
    fn replicate_default() -> Self {
        // SAFETY: `Pod` guarantees all-zero bytes are a valid `T`.
        unsafe { std::mem::zeroed() }
    }

    fn replicate_encode(&self, buf: &mut Vec<u8>) {
        // SAFETY: `T: Pod` guarantees every byte of `T`'s representation is
        // meaningful (no padding-UB) and safe to read.
        let bytes = unsafe {
            std::slice::from_raw_parts(self as *const T as *const u8, std::mem::size_of::<T>())
        };
        buf.extend_from_slice(bytes);
    }

    fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode> {
        if bytes.len() != std::mem::size_of::<T>() {
            return Err(ErrorCode::InvalidData);
        }
        let mut val = std::mem::MaybeUninit::<T>::uninit();
        // SAFETY: `bytes.len()` checked equal to `size_of::<T>()` above;
        // `T: Pod` guarantees any bit pattern (including these bytes) is a
        // valid `T`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), val.as_mut_ptr() as *mut u8, bytes.len());
            Ok(val.assume_init())
        }
    }
}

// NOTE: no blanket `impl<T: Pod, const N: usize> Replicable for [T; N]` —
// same `#[fundamental]`-style coherence conflict as `Box<T>` above, except
// this one is self-inflicted rather than language-mandated: `[f32; 16]`
// already implements `Pod` (the C5 mat4-transform special case in
// `page.rs`), so a generic array blanket would overlap with the `T: Pod`
// blanket above for that exact concrete type. Fixed-size arrays that aren't
// individually `Pod` (like the common `[f32; 3]` position/vector shape) get
// concrete, non-blanket impls instead — no overlap, since `[f32; 3]` etc.
// don't implement `Pod`.
macro_rules! impl_replicable_f32_array {
    ($($n:expr),+ $(,)?) => {
        $(
            impl Replicable for [f32; $n] {
                fn replicate_default() -> Self {
                    [0.0; $n]
                }
                fn replicate_encode(&self, buf: &mut Vec<u8>) {
                    for v in self {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode> {
                    if bytes.len() != $n * 4 {
                        return Err(ErrorCode::InvalidData);
                    }
                    let mut out = [0.0f32; $n];
                    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    Ok(out)
                }
            }
        )+
    };
}
impl_replicable_f32_array!(2, 3, 4);

impl Replicable for String {
    fn replicate_default() -> Self {
        String::new()
    }
    fn replicate_encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.as_bytes());
    }
    fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode> {
        String::from_utf8(bytes.to_vec()).map_err(|_| ErrorCode::InvalidData)
    }
}

/// `Vec<T>` self-frames each element with a `u32` byte-length prefix so it
/// composes recursively (`Vec<String>`, `Vec<Vec<u8>>`, ...) without any
/// type needing to know its own encoded width up front.
impl<T: Replicable> Replicable for Vec<T> {
    fn replicate_default() -> Self {
        Vec::new()
    }
    fn replicate_encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(self.len() as u32).to_le_bytes());
        for item in self {
            let mut elem = Vec::new();
            item.replicate_encode(&mut elem);
            buf.extend_from_slice(&(elem.len() as u32).to_le_bytes());
            buf.extend_from_slice(&elem);
        }
    }
    fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode> {
        if bytes.len() < 4 {
            return Err(ErrorCode::InvalidData);
        }
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut ofs = 4usize;
        let mut out = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            if ofs + 4 > bytes.len() {
                return Err(ErrorCode::InvalidData);
            }
            let elem_len = u32::from_le_bytes([bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3]]) as usize;
            ofs += 4;
            if ofs + elem_len > bytes.len() {
                return Err(ErrorCode::InvalidData);
            }
            out.push(T::replicate_decode(&bytes[ofs..ofs + elem_len])?);
            ofs += elem_len;
        }
        Ok(out)
    }
}

impl<T: Replicable> Replicable for Option<T> {
    fn replicate_default() -> Self {
        None
    }
    fn replicate_encode(&self, buf: &mut Vec<u8>) {
        match self {
            None => buf.push(0),
            Some(v) => {
                buf.push(1);
                v.replicate_encode(buf);
            }
        }
    }
    fn replicate_decode(bytes: &[u8]) -> Result<Self, ErrorCode> {
        match bytes.first() {
            Some(0) => Ok(None),
            Some(1) => Ok(Some(T::replicate_decode(&bytes[1..])?)),
            _ => Err(ErrorCode::InvalidData),
        }
    }
}

// NOTE: no blanket `impl<T: Replicable> Replicable for Box<T>` — `Box` is a
// `#[fundamental]` type, so the compiler must treat it as potentially
// overlapping with the `T: Pod` blanket impl above (a downstream crate could
// legally write `unsafe impl Pod for Box<TheirLocalType>` precisely because
// `Box` is fundamental), and rustc rejects the two blanket impls as
// conflicting even though no such impl actually exists. If you need a boxed
// field, implement `Replicable` directly for your concrete `Box<YourType>`
// (a single non-blanket impl, which does not hit this rule) — it's just
// `(**self).replicate_encode(buf)` / `Box::new(YourType::replicate_decode(bytes)?)`.

// ── Schema ─────────────────────────────────────────────────────────────────

/// Local-only safe encode/decode dispatch for one field, captured
/// generically at [`SchemaBuilder::field`]/[`SchemaBuilder::whole_field`]
/// time (the field's own type `F: Replicable` and the component's type `T`
/// are both known there). `Arc` rather than a plain `fn` pointer because
/// these close over the field's accessor functions.
///
/// `None` on a [`FieldDescriptor`] built from
/// [`ReplicationRegistry::from_handshake`] — wire bytes carry field layout,
/// never Rust types or closures (same limitation
/// `ReplicationEncoding::Opaque`'s fn pointers already have — see
/// `u8_from_encoding`'s doc).
#[derive(Clone)]
pub(crate) struct FieldOps {
    pub encode: Arc<dyn Fn(&dyn crate::component::ErasedColumn, usize, &mut Vec<u8>) -> Result<(), ErrorCode> + Send + Sync>,
    pub decode_into: Arc<dyn Fn(&mut dyn crate::component::ErasedColumn, usize, &[u8]) -> Result<(), ErrorCode> + Send + Sync>,
}

impl std::fmt::Debug for FieldOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FieldOps(..)")
    }
}

/// Local-only row constructors for a component type, captured generically at
/// [`ReplicationRegistry::register`] time (`T: Default` is known there).
/// Plain `fn` pointers (no captures needed) — `Clone`/`Debug` fall out of
/// the derive on [`ReplicationSchema`] for free.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowOps {
    pub new_column: fn() -> Box<dyn crate::component::ErasedColumn>,
    pub push_default: fn(&mut dyn crate::component::ErasedColumn),
}

/// Describes one replicated field on a component type.
#[derive(Clone, Debug)]
pub struct FieldDescriptor {
    pub field_index: u32,
    pub encoding: ReplicationEncoding,
    pub condition: ReplicationCondition,
    pub event_channel: Option<EventChannel>,
    pub(crate) ops: Option<FieldOps>,
}

/// Per-component-type replication schema.
/// Produced by `ReplicationRegistry` and shared at connection handshake.
#[derive(Clone, Debug)]
pub struct ReplicationSchema {
    pub component_type: ComponentId,
    pub fields: Vec<FieldDescriptor>,
    /// `None` for schemas built by [`ReplicationRegistry::from_handshake`]
    /// — see [`RowOps`]'s doc.
    pub(crate) row_ops: Option<RowOps>,
}

// ── Schema builder ─────────────────────────────────────────────────────────

/// Fluent builder for declaring a component type's replication layout.
/// Returned by [`ReplicationRegistry::register`].
pub struct SchemaBuilder<T> {
    component_type: ComponentId,
    fields: Vec<FieldDescriptor>,
    row_ops: RowOps,
    _phantom: PhantomData<T>,
}

impl<T: crate::Component> SchemaBuilder<T> {
    /// Declare a replicated field. `get`/`get_mut` are plain accessors to
    /// the named field on `T` (e.g. `|c: &Self| &c.pos`); `F: Replicable`
    /// is that field's own type — see [`Replicable`]'s doc for what
    /// qualifies (every [`crate::page::Pod`] type already does, plus
    /// `String`/`Vec<T>`/`Option<T>` out of the box). `field_index` is
    /// auto-incremented per call in declaration order.
    pub fn field<F: Replicable + 'static>(
        mut self,
        _name: &str,
        get: fn(&T) -> &F,
        get_mut: fn(&mut T) -> &mut F,
        encoding: ReplicationEncoding,
        condition: ReplicationCondition,
    ) -> Self {
        let field_index = self.fields.len() as u32;
        let encode: Arc<
            dyn Fn(&dyn crate::component::ErasedColumn, usize, &mut Vec<u8>) -> Result<(), ErrorCode> + Send + Sync,
        > = Arc::new(move |col, row, buf| {
            let col = col
                .as_any()
                .downcast_ref::<crate::component::Column<T>>()
                .ok_or(ErrorCode::InvalidData)?;
            let val = col.data.get(row).ok_or(ErrorCode::InvalidData)?;
            get(val).replicate_encode(buf);
            Ok(())
        });
        let decode_into: Arc<
            dyn Fn(&mut dyn crate::component::ErasedColumn, usize, &[u8]) -> Result<(), ErrorCode> + Send + Sync,
        > = Arc::new(move |col, row, bytes| {
            let col = col
                .as_any_mut()
                .downcast_mut::<crate::component::Column<T>>()
                .ok_or(ErrorCode::InvalidData)?;
            let slot = col.data.get_mut(row).ok_or(ErrorCode::InvalidData)?;
            *get_mut(slot) = F::replicate_decode(bytes)?;
            Ok(())
        });
        self.fields.push(FieldDescriptor {
            field_index,
            encoding,
            condition,
            event_channel: None,
            ops: Some(FieldOps { encode, decode_into }),
        });
        self
    }

    /// Convenience for the common case where the WHOLE component `T` is
    /// itself the replicated value (e.g. registering a bare `f32`) — an
    /// identity-accessor wrapper around [`Self::field`].
    pub fn whole_field(self, name: &str, encoding: ReplicationEncoding, condition: ReplicationCondition) -> Self
    where
        T: Replicable + 'static,
    {
        self.field(name, |c: &T| c, |c: &mut T| c, encoding, condition)
    }

    /// Declare an event/RPC field. Its `encoding` is implicitly `Event`.
    pub fn event(mut self, _name: &str, condition: ReplicationCondition, channel: EventChannel) -> Self {
        let field_index = self.fields.len() as u32;
        self.fields.push(FieldDescriptor {
            field_index,
            encoding: ReplicationEncoding::Event,
            condition,
            event_channel: Some(channel),
            ops: None,
        });
        self
    }

    pub(crate) fn build(self) -> ReplicationSchema {
        ReplicationSchema {
            component_type: self.component_type,
            fields: self.fields,
            row_ops: Some(self.row_ops),
        }
    }
}

// ── Registry ───────────────────────────────────────────────────────────────

/// Registry of replication schemas for all component types.
/// Produces handshake messages for connection initialization and is used by
/// the delta encoder to dispatch per-field encoding.
#[derive(Clone, Debug)]
pub struct ReplicationRegistry {
    schemas: HashMap<ComponentId, ReplicationSchema>,
}

impl ReplicationRegistry {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a component type and get a [`SchemaBuilder`] to describe its
    /// replicated fields. Call `.field(...)` / `.whole_field(...)` /
    /// `.event(...)` then pass the builder to [`Self::insert`].
    ///
    /// `T: Default` is used solely to grow a column by one placeholder row
    /// when [`crate::World`] spawns a replicated entity before its real
    /// field values arrive — see [`RowOps`].
    pub fn register<T: crate::Component + Default>(&mut self) -> SchemaBuilder<T> {
        SchemaBuilder {
            component_type: crate::component::component_id::<T>(),
            fields: Vec::new(),
            row_ops: RowOps {
                new_column: || Box::new(crate::component::Column::<T>::new()),
                push_default: |col| {
                    let col = col
                        .as_any_mut()
                        .downcast_mut::<crate::component::Column<T>>()
                        .expect("push_default: column type mismatch — RowOps is generated per-T at register::<T>() time");
                    col.data.push(T::default());
                },
            },
            _phantom: PhantomData,
        }
    }

    /// Local-only row constructors for a registered component type (see
    /// [`RowOps`]). `None` if `cid` isn't registered or was learned only
    /// from a remote handshake.
    pub(crate) fn row_ops(&self, cid: ComponentId) -> Option<&RowOps> {
        self.schemas.get(&cid)?.row_ops.as_ref()
    }

    /// Finalise a builder and insert the resulting schema. Called internally
    /// by the builder when the caller is done chaining.
    pub fn insert<T: crate::Component>(&mut self, builder: SchemaBuilder<T>) {
        let schema = builder.build();
        self.schemas.insert(schema.component_type, schema);
    }

    /// Serialize all registered schemas into a handshake byte buffer.
    ///
    /// Wire format (all values little-endian):
    ///   `schema_count: u32`
    ///   for each schema:
    ///     `component_type: u32`
    ///     `field_count: u32`
    ///     for each field:
    ///       `field_index: u32`
    ///       `encoding: u8`    — 0=Pod,1=Serialized,2=GpuHandle,3=DeltaCompressed,4=Event,5=Opaque
    ///       `condition: u8`   — 0=Always,1=OwnerOnly,2=SkipOwner,3=SimulatedOnly,4=AutonomousOnly,
    ///                           5=InitialOnly,6=ServerAuthority,7=ClientAuthority,8=ServerToClient,
    ///                           9=ClientToServer,10=Multicast
    ///       `event_channel: u8` — 0=None,1=ReliableOrdered,2=Unreliable
    pub fn handshake_message(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.schemas.len() as u32).to_le_bytes());
        for schema in self.schemas.values() {
            buf.extend_from_slice(&schema.component_type.0.to_le_bytes());
            buf.extend_from_slice(&(schema.fields.len() as u32).to_le_bytes());
            for field in &schema.fields {
                buf.extend_from_slice(&field.field_index.to_le_bytes());
                buf.push(encoding_to_u8(&field.encoding));
                buf.push(condition_to_u8(&field.condition));
                buf.push(event_channel_to_u8(&field.event_channel));
            }
        }
        buf
    }

    /// Deserialize a handshake message produced by a remote peer.
    pub fn from_handshake(bytes: &[u8]) -> Result<Self, ErrorCode> {
        let mut ofs = 0;
        if ofs + 4 > bytes.len() {
            return Err(ErrorCode::InvalidData);
        }
        let schema_count = u32::from_le_bytes([bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3]]);
        ofs += 4;
        let mut schemas = HashMap::new();
        for _ in 0..schema_count {
            if ofs + 8 > bytes.len() {
                return Err(ErrorCode::InvalidData);
            }
            let component_type = ComponentId(u32::from_le_bytes([
                bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3],
            ]));
            ofs += 4;
            let field_count = u32::from_le_bytes([
                bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3],
            ]);
            ofs += 4;
            let mut fields = Vec::with_capacity(field_count as usize);
            for _ in 0..field_count {
                // Each field is exactly 7 bytes on the wire: field_index
                // (4) + encoding (1) + condition (1) + event_channel (1).
                // This used to check for 10, which rejected otherwise
                // well-formed handshakes whenever the last field landed
                // exactly at the buffer's end (found by
                // `handshake_round_trip_random_schemas`'s fuzzing).
                if ofs + 7 > bytes.len() {
                    return Err(ErrorCode::InvalidData);
                }
                let field_index = u32::from_le_bytes([
                    bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3],
                ]);
                ofs += 4;
                let encoding = u8_from_encoding(bytes[ofs])?;
                ofs += 1;
                let condition = u8_from_condition(bytes[ofs])?;
                ofs += 1;
                let event_channel = u8_from_event_channel(bytes[ofs])?;
                ofs += 1;
                fields.push(FieldDescriptor {
                    field_index,
                    encoding,
                    condition,
                    event_channel,
                    // Wire bytes never carry Rust types/closures — see
                    // `FieldOps`'s doc.
                    ops: None,
                });
            }
            schemas.insert(
                component_type,
                ReplicationSchema {
                    component_type,
                    fields,
                    // Wire bytes never carry Rust types — see `RowOps`'s doc.
                    row_ops: None,
                },
            );
        }
        Ok(Self { schemas })
    }

    pub fn schema(&self, cid: ComponentId) -> Option<&ReplicationSchema> {
        self.schemas.get(&cid)
    }
}

impl Default for ReplicationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Encoding helpers ───────────────────────────────────────────────────────

fn encoding_to_u8(enc: &ReplicationEncoding) -> u8 {
    match enc {
        ReplicationEncoding::Pod => 0,
        ReplicationEncoding::Serialized => 1,
        ReplicationEncoding::GpuHandle => 2,
        ReplicationEncoding::DeltaCompressed => 3,
        ReplicationEncoding::Event => 4,
        ReplicationEncoding::Opaque { .. } => 5,
    }
}

fn u8_from_encoding(v: u8) -> Result<ReplicationEncoding, ErrorCode> {
    match v {
        0 => Ok(ReplicationEncoding::Pod),
        1 => Ok(ReplicationEncoding::Serialized),
        2 => Ok(ReplicationEncoding::GpuHandle),
        3 => Ok(ReplicationEncoding::DeltaCompressed),
        4 => Ok(ReplicationEncoding::Event),
        // Opaque fn pointers cannot be serialized over the wire.
        // The handshake only communicates *that* a field is Opaque;
        // the actual encode/decode closures must be configured locally
        // on each peer via the schema registration API.
        5 => Ok(ReplicationEncoding::Opaque {
            encode_size: |_| 0,
            encode: |_, _| ErrorCode::InvalidData,
            decode: |_, _| ErrorCode::InvalidData,
        }),
        _ => Err(ErrorCode::InvalidData),
    }
}

fn condition_to_u8(c: &ReplicationCondition) -> u8 {
    match c {
        ReplicationCondition::Always => 0,
        ReplicationCondition::OwnerOnly => 1,
        ReplicationCondition::SkipOwner => 2,
        ReplicationCondition::SimulatedOnly => 3,
        ReplicationCondition::AutonomousOnly => 4,
        ReplicationCondition::InitialOnly => 5,
        ReplicationCondition::ServerAuthority => 6,
        ReplicationCondition::ClientAuthority => 7,
        ReplicationCondition::ServerToClient => 8,
        ReplicationCondition::ClientToServer => 9,
        ReplicationCondition::Multicast => 10,
    }
}

fn u8_from_condition(v: u8) -> Result<ReplicationCondition, ErrorCode> {
    match v {
        0 => Ok(ReplicationCondition::Always),
        1 => Ok(ReplicationCondition::OwnerOnly),
        2 => Ok(ReplicationCondition::SkipOwner),
        3 => Ok(ReplicationCondition::SimulatedOnly),
        4 => Ok(ReplicationCondition::AutonomousOnly),
        5 => Ok(ReplicationCondition::InitialOnly),
        6 => Ok(ReplicationCondition::ServerAuthority),
        7 => Ok(ReplicationCondition::ClientAuthority),
        8 => Ok(ReplicationCondition::ServerToClient),
        9 => Ok(ReplicationCondition::ClientToServer),
        10 => Ok(ReplicationCondition::Multicast),
        _ => Err(ErrorCode::InvalidData),
    }
}

fn event_channel_to_u8(ch: &Option<EventChannel>) -> u8 {
    match ch {
        None => 0,
        Some(EventChannel::ReliableOrdered) => 1,
        Some(EventChannel::Unreliable) => 2,
    }
}

fn u8_from_event_channel(v: u8) -> Result<Option<EventChannel>, ErrorCode> {
    match v {
        0 => Ok(None),
        1 => Ok(Some(EventChannel::ReliableOrdered)),
        2 => Ok(Some(EventChannel::Unreliable)),
        _ => Err(ErrorCode::InvalidData),
    }
}

/// Encode a field value according to its encoding mode into `buf`.
pub fn encode_field_value(encoding: &ReplicationEncoding, value_bytes: &[u8], buf: &mut Vec<u8>) -> ErrorCode {
    match encoding {
        ReplicationEncoding::Pod | ReplicationEncoding::Serialized | ReplicationEncoding::GpuHandle => {
            buf.extend_from_slice(value_bytes);
            ErrorCode::Ok
        }
        ReplicationEncoding::DeltaCompressed => {
            // Simple LEB128 encode of the first 8 bytes as a u64.
            let val = if value_bytes.len() >= 8 {
                u64::from_le_bytes([
                    value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3],
                    value_bytes[4], value_bytes[5], value_bytes[6], value_bytes[7],
                ])
            } else {
                let mut tmp = [0u8; 8];
                tmp[..value_bytes.len()].copy_from_slice(value_bytes);
                u64::from_le_bytes(tmp)
            };
            leb128_encode(val, buf);
            ErrorCode::Ok
        }
        ReplicationEncoding::Event => ErrorCode::Ok,
        ReplicationEncoding::Opaque { encode_size, encode, .. } => {
            let ptr = value_bytes.as_ptr() as *const ();
            let needed = encode_size(ptr);
            let start = buf.len();
            buf.resize(start + needed, 0);
            encode(ptr, &mut buf[start..])
        }
    }
}

/// Encode a Pod field directly into a byte slice — the zero-copy
/// counterpart to [`encode_field_value`]'s `Pod`/`Serialized`/`GpuHandle`
/// branch, which allocates into (and may reallocate/grow) a `Vec<u8>`. For
/// hot paths that already own a destination buffer — e.g. assembling a
/// network packet, or [`ChangeTracker`]'s per-field scratch — this memcpy's
/// straight in with no intermediate allocation.
///
/// Returns the number of bytes written (always `value_bytes.len()`).
///
/// # Safety
/// `buf` must be at least `value_bytes.len()` bytes.
pub unsafe fn encode_pod_raw(value_bytes: &[u8], buf: &mut [u8]) -> usize {
    debug_assert!(buf.len() >= value_bytes.len(), "encode_pod_raw: buf too small");
    std::ptr::copy_nonoverlapping(value_bytes.as_ptr(), buf.as_mut_ptr(), value_bytes.len());
    value_bytes.len()
}

/// Decode a field value from `data` according to its encoding mode into `value_bytes`.
pub fn decode_field_value(encoding: &ReplicationEncoding, data: &[u8], value_bytes: &mut [u8]) -> ErrorCode {
    match encoding {
        ReplicationEncoding::Pod | ReplicationEncoding::Serialized | ReplicationEncoding::GpuHandle => {
            if data.len() < value_bytes.len() {
                return ErrorCode::BufferTooSmall { needed: value_bytes.len() };
            }
            value_bytes.copy_from_slice(&data[..value_bytes.len()]);
            ErrorCode::Ok
        }
        ReplicationEncoding::DeltaCompressed => {
            let (val, _) = match leb128_decode(data) {
                Ok(v) => v,
                Err(e) => return e,
            };
            let bytes = val.to_le_bytes();
            let copy_len = value_bytes.len().min(8);
            value_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);
            ErrorCode::Ok
        }
        ReplicationEncoding::Event => ErrorCode::Ok,
        ReplicationEncoding::Opaque { decode, .. } => {
            let dst = value_bytes.as_mut_ptr() as *mut ();
            decode(data, dst)
        }
    }
}

fn leb128_encode(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

fn leb128_decode(data: &[u8]) -> Result<(u64, usize), ErrorCode> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        i += 1;
    }
    Err(ErrorCode::InvalidData)
}

/// Serialize a set of component ids as an archetype-key blob (wire format:
/// `count: u32` then `count` × `component_id: u32`, little-endian). Used to
/// fill [`Delta::spawned`]'s per-entity blob (see
/// [`ChangeTracker::drain_with_world`]) so [`Delta::apply`] can reconstruct
/// the spawning entity's exact archetype on a remote peer.
pub fn encode_archetype_key(ids: &[ComponentId]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + ids.len() * 4);
    buf.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for id in ids {
        buf.extend_from_slice(&id.0.to_le_bytes());
    }
    buf
}

/// Inverse of [`encode_archetype_key`]. Returns `None` if `bytes` is
/// truncated, malformed, or simply isn't an archetype-key blob (e.g. the
/// placeholder frame-number blob the plain [`ChangeTracker::drain`]
/// produces) — callers should treat that as "unknown archetype" and fall
/// back to the empty archetype.
pub fn decode_archetype_key(bytes: &[u8]) -> Option<Vec<ComponentId>> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if 4 + count * 4 != bytes.len() {
        return None;
    }
    let mut ids = Vec::with_capacity(count);
    for i in 0..count {
        let ofs = 4 + i * 4;
        ids.push(ComponentId(u32::from_le_bytes([
            bytes[ofs], bytes[ofs + 1], bytes[ofs + 2], bytes[ofs + 3],
        ])));
    }
    Some(ids)
}

// ── Delta ──────────────────────────────────────────────────────────────────

/// Frame-consistent set of state changes for one connection.
#[derive(Clone, Debug)]
pub struct Delta {
    pub frame: u64,
    pub base_frame: u64,
    pub spawned: Vec<(Entity, Vec<u8>)>,
    pub despawned: Vec<Entity>,
    pub component_deltas: Vec<ComponentDelta>,
    pub events: Vec<ReplicatedEvent>,
}

impl Delta {
    /// Serialize this Delta into a byte buffer for network transport.
    ///
    /// Wire format (all values little-endian):
    ///
    /// `frame: u64`, `base_frame: u64`
    /// `spawned_count: u32`
    ///   for each: `entity: u64`, `blob_len: u32`, `blob[blob_len]: u8`
    /// `despawned_count: u32`
    ///   for each: `entity: u64`
    /// `cd_count: u32`
    ///   for each: `entity: u64`, `component_type: u32`, `field_count: u32`
    ///     for each field: `field_len: u32`, `field_data[field_len]: u8`
    /// `event_count: u32`
    ///   for each: `entity: u64`, `component_type: u32`, `event_field: u32`,
    ///   `payload_len: u32`, `payload[payload_len]: u8`, `channel: u8`,
    ///   `has_target: u8`, `target_client: u64` (only if has_target != 0)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.frame.to_le_bytes());
        buf.extend_from_slice(&self.base_frame.to_le_bytes());

        buf.extend_from_slice(&(self.spawned.len() as u32).to_le_bytes());
        for (entity, blob) in &self.spawned {
            buf.extend_from_slice(&entity.bits().to_le_bytes());
            buf.extend_from_slice(&(blob.len() as u32).to_le_bytes());
            buf.extend_from_slice(blob);
        }

        buf.extend_from_slice(&(self.despawned.len() as u32).to_le_bytes());
        for entity in &self.despawned {
            buf.extend_from_slice(&entity.bits().to_le_bytes());
        }

        buf.extend_from_slice(&(self.component_deltas.len() as u32).to_le_bytes());
        for cd in &self.component_deltas {
            buf.extend_from_slice(&cd.entity.bits().to_le_bytes());
            buf.extend_from_slice(&cd.component_type.0.to_le_bytes());
            buf.extend_from_slice(&(cd.field_data.len() as u32).to_le_bytes());
            for field in &cd.field_data {
                buf.extend_from_slice(&(field.len() as u32).to_le_bytes());
                buf.extend_from_slice(field);
            }
        }

        buf.extend_from_slice(&(self.events.len() as u32).to_le_bytes());
        for ev in &self.events {
            buf.extend_from_slice(&ev.entity.bits().to_le_bytes());
            buf.extend_from_slice(&ev.component_type.0.to_le_bytes());
            buf.extend_from_slice(&ev.event_field.to_le_bytes());
            buf.extend_from_slice(&(ev.payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&ev.payload);
            buf.push(match ev.channel {
                EventChannel::ReliableOrdered => 0,
                EventChannel::Unreliable => 1,
            });
            match ev.target_client {
                Some(id) => {
                    buf.push(1u8);
                    buf.extend_from_slice(&id.0.to_le_bytes());
                }
                None => buf.push(0u8),
            }
        }
        buf
    }

    /// Deserialize a Delta from bytes produced by [`to_bytes`](Self::to_bytes).
    /// Returns `None` if the input is truncated or malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut ofs = 0;
        macro_rules! read {
            ($n:expr) => {{
                if ofs + $n > bytes.len() { return None; }
                let slice = &bytes[ofs..ofs + $n];
                ofs += $n;
                slice
            }};
        }
        macro_rules! read_u64 {
            () => {{
                let s = read!(8);
                u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
            }};
        }
        macro_rules! read_u32 {
            () => {{
                let s = read!(4);
                u32::from_le_bytes([s[0], s[1], s[2], s[3]])
            }};
        }

        let frame = read_u64!();
        let base_frame = read_u64!();

        let spawned_count = read_u32!();
        let mut spawned = Vec::with_capacity(spawned_count as usize);
        for _ in 0..spawned_count {
            let entity_bits = read_u64!();
            let blob_len = read_u32!() as usize;
            let blob = read!(blob_len).to_vec();
            spawned.push((Entity::from_bits(entity_bits), blob));
        }

        let despawned_count = read_u32!();
        let mut despawned = Vec::with_capacity(despawned_count as usize);
        for _ in 0..despawned_count {
            despawned.push(Entity::from_bits(read_u64!()));
        }

        let cd_count = read_u32!();
        let mut component_deltas = Vec::with_capacity(cd_count as usize);
        for _ in 0..cd_count {
            let entity = Entity::from_bits(read_u64!());
            let component_type = ComponentId(read_u32!());
            let field_count = read_u32!();
            let mut field_data = Vec::with_capacity(field_count as usize);
            for _ in 0..field_count {
                let field_len = read_u32!() as usize;
                field_data.push(read!(field_len).to_vec());
            }
            component_deltas.push(ComponentDelta { entity, component_type, field_data });
        }

        let event_count = read_u32!();
        let mut events = Vec::with_capacity(event_count as usize);
        for _ in 0..event_count {
            let entity = Entity::from_bits(read_u64!());
            let component_type = ComponentId(read_u32!());
            let event_field = read_u32!();
            let payload_len = read_u32!() as usize;
            let payload = read!(payload_len).to_vec();
            let channel = match read!(1)[0] {
                0 => EventChannel::ReliableOrdered,
                _ => EventChannel::Unreliable,
            };
            let has_target = read!(1)[0];
            let target_client = if has_target != 0 {
                Some(ClientId(read_u64!()))
            } else {
                None
            };
            events.push(ReplicatedEvent { entity, component_type, event_field, payload, channel, target_client });
        }

        Some(Delta { frame, base_frame, spawned, despawned, component_deltas, events })
    }

    /// Apply this Delta's changes to a World.
    ///
    /// 1. Despawns removed entities.
    /// 2. Spawns new entities at their EXACT wire `Entity` (index +
    ///    generation) into the archetype encoded in each spawn's blob —
    ///    see [`encode_archetype_key`] / [`ChangeTracker::drain_with_world`].
    ///    A blob that isn't a valid archetype key (truncated, or the plain
    ///    placeholder [`ChangeTracker::drain`] produces) falls back to the
    ///    empty archetype.
    /// 3. Writes each field's bytes back via its own
    ///    [`Replicable::replicate_decode`] (reached through the
    ///    [`FieldOps`] closure captured at
    ///    [`SchemaBuilder::field`]/[`SchemaBuilder::whole_field`] time) —
    ///    `cd.field_data[field_index]` is exactly what
    ///    [`Replicable::replicate_encode`] produced for that field, so
    ///    there is no byte-width guessing: every field is safely,
    ///    generically reconstructed regardless of whether its type is
    ///    `Pod`, `String`, `Vec<T>`, or a user's own `Replicable` impl.
    ///
    /// Entities are placed at the SAME index+generation as the wire value
    /// (via `Entity::bits`/`from_bits`) — SceneDB's replication model
    /// assumes entity handles are shared verbatim between peers (see the
    /// module doc's "Endianness is a non-concern" design tenet). If that
    /// slot is already occupied by a different local entity, the incoming
    /// delta is authoritative and the old occupant is despawned first.
    ///
    /// Spawning an entity whose archetype names a component type `schema`
    /// has no local (non-handshake) registration for fails with
    /// [`ErrorCode::InvalidData`] — see [`RowOps`]. A `ComponentDelta`
    /// targeting a dead entity, or a field with no local `ops` (same
    /// handshake limitation), is silently skipped rather than erroring; a
    /// genuine decode failure (malformed bytes from an untrusted peer) is
    /// propagated as `Err` without panicking.
    pub fn apply(&self, world: &mut crate::World, schema: &ReplicationRegistry) -> Result<(), ErrorCode> {
        let mut scratch = Vec::new();
        self.apply_with_scratch(world, schema, &mut scratch)
    }

    /// Like [`Self::apply`]. `scratch` is accepted for API stability with
    /// earlier callers but is no longer needed internally: field decoding
    /// now goes straight from wire bytes to the field's own owned value via
    /// [`Replicable::replicate_decode`] (a `String`/`Vec<T>` is heap data
    /// either way; a `Pod` field decodes into a stack `MaybeUninit`, not a
    /// `Vec` at all) — there is no intermediate byte buffer left to reuse.
    pub fn apply_with_scratch(
        &self,
        world: &mut crate::World,
        schema: &ReplicationRegistry,
        _scratch: &mut Vec<u8>,
    ) -> Result<(), ErrorCode> {
        for &entity in &self.despawned {
            world.despawn(entity);
        }

        for (entity, blob) in &self.spawned {
            let cids = decode_archetype_key(blob).unwrap_or_default();
            let key = ArchetypeKey::new(cids);
            if world
                .force_spawn_in_archetype(*entity, key, |cid| schema.row_ops(cid).copied())
                .is_none()
            {
                return Err(ErrorCode::InvalidData);
            }
        }

        for cd in &self.component_deltas {
            let Some(s) = schema.schema(cd.component_type) else {
                continue;
            };
            for field in &s.fields {
                if matches!(field.encoding, ReplicationEncoding::Event) {
                    continue;
                }
                let Some(ops) = &field.ops else {
                    continue;
                };
                let idx = field.field_index as usize;
                let Some(bytes) = cd.field_data.get(idx) else {
                    continue;
                };
                if bytes.is_empty() {
                    continue;
                }
                world.write_component_field(cd.entity, cd.component_type, &*ops.decode_into, bytes)?;
            }
        }

        Ok(())
    }
}

/// Sparse component data for one entity within a Delta.
#[derive(Clone, Debug)]
pub struct ComponentDelta {
    pub entity: Entity,
    pub component_type: ComponentId,
    pub field_data: Vec<Vec<u8>>,
}

// ── Change tracker ─────────────────────────────────────────────────────────

/// Accumulates all mutations to a World during a single simulate phase.
/// Reset at the SimulateB→Harvest phase boundary.
#[derive(Clone, Debug)]
pub struct ChangeTracker {
    spawned: Vec<Entity>,
    despawned: Vec<Entity>,
    component_changes: Vec<ComponentDelta>,
    events: Vec<ReplicatedEvent>,
    frame: u64,
}

impl ChangeTracker {
    pub fn new() -> Self {
        Self {
            spawned: Vec::new(),
            despawned: Vec::new(),
            component_changes: Vec::new(),
            events: Vec::new(),
            frame: 0,
        }
    }

    pub fn record_spawn(&mut self, entity: Entity) {
        self.spawned.push(entity);
    }

    pub fn record_despawn(&mut self, entity: Entity) {
        self.despawned.push(entity);
    }

    pub fn record_component_change(
        &mut self,
        entity: Entity,
        component_type: ComponentId,
        field_index: u32,
        field_bytes: Vec<u8>,
    ) {
        // Find existing ComponentDelta for this entity+component, or create one.
        if let Some(existing) = self
            .component_changes
            .iter_mut()
            .find(|cd| cd.entity == entity && cd.component_type == component_type)
        {
            // Ensure field_data is large enough, then insert at field_index.
            let idx = field_index as usize;
            if idx >= existing.field_data.len() {
                existing.field_data.resize(idx + 1, Vec::new());
            }
            existing.field_data[idx] = field_bytes;
        } else {
            let mut field_data = Vec::new();
            let idx = field_index as usize;
            if idx >= field_data.len() {
                field_data.resize(idx + 1, Vec::new());
            }
            field_data[idx] = field_bytes;
            self.component_changes.push(ComponentDelta {
                entity,
                component_type,
                field_data,
            });
        }
    }

    pub fn record_event(&mut self, event: ReplicatedEvent) {
        self.events.push(event);
    }

    pub fn drain(
        &mut self,
        _schema: &ReplicationSchema,
        _client: ClientId,
        _authority: &AuthorityTable,
    ) -> (Delta, Vec<ReplicatedEvent>) {
        let delta = Delta {
            frame: self.frame,
            base_frame: self.frame.wrapping_sub(1),
            spawned: mem::take(&mut self.spawned)
                .into_iter()
                .map(|e| (e, self.frame.to_le_bytes().to_vec()))
                .collect(),
            despawned: mem::take(&mut self.despawned),
            component_deltas: mem::take(&mut self.component_changes),
            events: mem::take(&mut self.events),
        };
        (delta, Vec::new())
    }

    pub fn end_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Like [`drain`](Self::drain) but encodes each spawned entity's
    /// *current* archetype (looked up in `world`) as its blob via
    /// [`encode_archetype_key`], so the resulting [`Delta`] can be fully
    /// reconstructed on a remote peer by [`Delta::apply`]. Plain `drain`
    /// cannot do this — it has no `World` access, so its blob is just a
    /// placeholder frame marker.
    pub fn drain_with_world(&mut self, world: &crate::World) -> Delta {
        let spawned = mem::take(&mut self.spawned)
            .into_iter()
            .map(|e| {
                let blob = if world.is_alive(e) {
                    let arch_id = world.entity_slots[e.index() as usize].archetype;
                    encode_archetype_key(&world.archetypes[arch_id.0 as usize].active_cids)
                } else {
                    Vec::new()
                };
                (e, blob)
            })
            .collect();
        Delta {
            frame: self.frame,
            base_frame: self.frame.wrapping_sub(1),
            spawned,
            despawned: mem::take(&mut self.despawned),
            component_deltas: mem::take(&mut self.component_changes),
            events: mem::take(&mut self.events),
        }
    }
}

// ── Entity ↔ spatial-cell mapping ───────────────────────────────────────────

/// Bi-directional mapping between ECS [`Entity`] handles and `(cell_index,
/// row_token)` pairs in a set of [`SpatialCell`]s.
///
/// [`RelevanceSet::from_frustum`] resolves each hit `(cell_idx, row_token)`
/// to an `Entity` via a caller-supplied closure; this map is the obvious
/// off-the-shelf resolver for the common case where spatial rows and ECS
/// entities are kept in a stable 1:1 correspondence (see
/// [`RelevanceSet::from_frustum_mapped`]).
#[derive(Clone, Debug, Default)]
pub struct EntityCellMap {
    entity_to_cell: HashMap<Entity, (usize, u32)>,
    cell_to_entity: HashMap<(usize, u32), Entity>,
}

impl EntityCellMap {
    pub fn new() -> Self {
        Self {
            entity_to_cell: HashMap::new(),
            cell_to_entity: HashMap::new(),
        }
    }

    /// Record that `entity` lives at `(cell_idx, row)`. Overwrites any prior
    /// mapping for `entity` and for the `(cell_idx, row)` slot, keeping both
    /// directions consistent.
    pub fn insert(&mut self, entity: Entity, cell_idx: usize, row: u32) {
        if let Some(old_key) = self.entity_to_cell.insert(entity, (cell_idx, row)) {
            if old_key != (cell_idx, row) {
                self.cell_to_entity.remove(&old_key);
            }
        }
        if let Some(old_entity) = self.cell_to_entity.insert((cell_idx, row), entity) {
            if old_entity != entity {
                self.entity_to_cell.remove(&old_entity);
            }
        }
    }

    /// Remove `entity` from the map, if present.
    pub fn remove(&mut self, entity: Entity) {
        if let Some(key) = self.entity_to_cell.remove(&entity) {
            self.cell_to_entity.remove(&key);
        }
    }

    /// Resolve a `(cell_idx, row_token)` pair to its mapped entity.
    pub fn entity_at(&self, cell_idx: usize, row: u32) -> Option<Entity> {
        self.cell_to_entity.get(&(cell_idx, row)).copied()
    }

    /// Resolve an entity to its `(cell_idx, row_token)` pair.
    pub fn cell_of(&self, entity: Entity) -> Option<(usize, u32)> {
        self.entity_to_cell.get(&entity).copied()
    }

    /// Number of entities currently mapped.
    pub fn len(&self) -> usize {
        self.entity_to_cell.len()
    }

    /// Returns true if the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.entity_to_cell.is_empty()
    }
}

// ── Interest management ────────────────────────────────────────────────────

/// Per-connection filter, built each frame from spatial queries + conditions.
#[derive(Clone, Debug)]
pub struct RelevanceSet {
    relevant_entities: Vec<Entity>,
}

impl RelevanceSet {
    /// Build an empty relevance set.
    pub fn new() -> Self {
        Self {
            relevant_entities: Vec::new(),
        }
    }

    /// Build from a frustum query against spatial cells.
    ///
    /// For each spatial cell with visible rows, calls `resolve(cell_idx, row_token)`
    /// which should return `Some(entity)` for rows that map to an ECS entity.
    /// Uses `SpatialCell::query_frustum_in` internally — zero alloc after warm-up
    /// when using `Scratchpad` + `LivenessSnapshot`.
    pub fn from_frustum(
        cells: &[SpatialCell],
        frustum: &crate::spatial::Frustum,
        _liveness: &LivenessSnapshot,
        scratch: &mut crate::lease::Scratchpad,
        mut resolve: impl FnMut(usize, u32) -> Option<Entity>,
    ) -> Self {
        let mut set = Self::new();
        for (i, cell) in cells.iter().enumerate() {
            let len = cell.rows_in_use() as usize;
            if len == 0 {
                continue;
            }
            let (out, words) = scratch.get_u32_u64(len, len.div_ceil(64));
            let nw = LivenessSnapshot::capture_words(cell.liveness(), len as u32, words);
            let nh = cell.query_frustum_in(frustum, &words[..nw], out) as usize;
            for row in 0..nh.min(len) {
                let token = out[row];
                if token != crate::registry::NULL_ROW {
                    if let Some(entity) = resolve(i, token) {
                        set.relevant_entities.push(entity);
                    }
                }
            }
        }
        set
    }

    /// Convenience wrapper over [`Self::from_frustum`] that resolves each
    /// hit `(cell_idx, row_token)` through an [`EntityCellMap`] instead of a
    /// caller-supplied closure — the common case where spatial rows and ECS
    /// entities are kept in a stable 1:1 correspondence via `map`.
    pub fn from_frustum_mapped(
        cells: &[SpatialCell],
        frustum: &crate::spatial::Frustum,
        liveness: &LivenessSnapshot,
        scratch: &mut crate::lease::Scratchpad,
        map: &EntityCellMap,
    ) -> Self {
        Self::from_frustum(cells, frustum, liveness, scratch, |cell_idx, row| {
            map.entity_at(cell_idx, row)
        })
    }

    /// Add an entity that should always be relevant (local player, HUD, etc.).
    pub fn add_always_relevant(&mut self, entity: Entity) {
        self.relevant_entities.push(entity);
    }

    /// Add an entity to the relevance set.
    pub fn add(&mut self, entity: Entity) {
        self.relevant_entities.push(entity);
    }

    /// Check whether a single entity is in the relevance set.
    pub fn contains(&self, entity: Entity) -> bool {
        self.relevant_entities.contains(&entity)
    }

    /// Number of entities in this relevance set.
    pub fn len(&self) -> usize {
        self.relevant_entities.len()
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.relevant_entities.is_empty()
    }

    /// Filter a Delta to only the changes relevant to `client`.
    ///
    /// Applies two layers of filtering:
    /// 1. Entity allowlist — only entities in this set are considered.
    /// 2. Per-field condition check — each `ComponentDelta`'s component schema
    ///    is checked against `ReplicationCondition` using the `AuthorityTable`.
    pub fn filter<'a>(
        &self,
        delta: &'a Delta,
        authority: &AuthorityTable,
        schema_registry: &ReplicationRegistry,
        client: ClientId,
    ) -> DeltaView<'a> {
        let component_deltas = delta
            .component_deltas
            .iter()
            .filter(|cd| {
                if !self.relevant_entities.contains(&cd.entity) {
                    return false;
                }
                let owner = authority.owner_of(cd.entity);
                if let Some(schema) = schema_registry.schema(cd.component_type) {
                    for field in &schema.fields {
                        if !condition_passes(&field.condition, &owner, &client) {
                            return false;
                        }
                    }
                }
                true
            })
            .collect();

        let events = delta
            .events
            .iter()
            .filter(|ev| self.relevant_entities.contains(&ev.entity))
            .collect();

        DeltaView {
            spawned: &delta.spawned,
            despawned: &delta.despawned,
            component_deltas,
            events,
        }
    }
}

impl Default for RelevanceSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if `client` passes the replication condition given the
/// entity's ownership.
pub fn condition_passes(condition: &ReplicationCondition, owner: &Ownership, client: &ClientId) -> bool {
    match condition {
        ReplicationCondition::Always => true,
        ReplicationCondition::ServerAuthority => true,
        ReplicationCondition::ClientAuthority => true,
        ReplicationCondition::OwnerOnly => match owner {
            Ownership::Client(id) => id == client,
            Ownership::Server => false,
            Ownership::Shared => true,
        },
        ReplicationCondition::SkipOwner => match owner {
            Ownership::Client(id) => id != client,
            Ownership::Server => true,
            Ownership::Shared => true,
        },
        ReplicationCondition::SimulatedOnly => match owner {
            Ownership::Client(id) => id != client,
            Ownership::Server => true,
            Ownership::Shared => true,
        },
        ReplicationCondition::AutonomousOnly => match owner {
            Ownership::Client(id) => id == client,
            Ownership::Server => false,
            Ownership::Shared => true,
        },
        ReplicationCondition::InitialOnly => false,
        ReplicationCondition::ServerToClient => false,
        ReplicationCondition::ClientToServer => false,
        ReplicationCondition::Multicast => true,
    }
}

/// Borrowed sub-slices of a Delta, filtered by relevance + conditions.
#[derive(Clone, Debug)]
pub struct DeltaView<'a> {
    pub spawned: &'a [(Entity, Vec<u8>)],
    pub despawned: &'a [Entity],
    pub component_deltas: Vec<&'a ComponentDelta>,
    pub events: Vec<&'a ReplicatedEvent>,
}

// ── Event / RPC channel ────────────────────────────────────────────────────

/// A batch of events destined for one connection, sent separately from Deltas.
/// The engine transports this as a distinct message type (reliable or
/// unreliable depending on each event's [`EventChannel`]).
#[derive(Clone, Debug)]
pub struct EventBatch {
    pub frame: u64,
    pub events: Vec<ReplicatedEvent>,
}

/// Check whether an event may be sent from `sender` to `recipient` based on
/// the event field's declared direction (encoded in the field's
/// [`ReplicationCondition`] within the schema).
///
/// The caller must look up the field's condition from the schema and pass it
/// here. Invalid events (e.g. a client trying to multicast) return `false`
/// and should be silently dropped.
///
/// # Panics
///
/// Panics if `direction` is not one of the three RPC direction conditions
/// (`ClientToServer`, `ServerToClient`, `Multicast`).
pub fn can_send_event(direction: &ReplicationCondition, sender: ClientId, recipient: ClientId) -> bool {
    match direction {
        ReplicationCondition::ClientToServer => true,
        ReplicationCondition::ServerToClient => sender == recipient,
        ReplicationCondition::Multicast => sender != recipient,
        other => {
            // Other conditions don't make sense for events — treat as always
            // sendable (the state path handles them).
            true
        }
    }
}

/// Extract events from a filtered `DeltaView` into an `EventBatch`, applying
/// direction enforcement. Returns `None` if no events pass the filter.
///
/// The `schema_registry` is used to look up each event field's
/// `ReplicationCondition` to determine direction. Events whose direction
/// rejects `(sender, recipient)` are silently dropped.
pub fn events_to_batch(
    view: &DeltaView,
    frame: u64,
    schema_registry: &ReplicationRegistry,
    sender: ClientId,
    recipient: ClientId,
) -> Option<EventBatch> {
    let events: Vec<ReplicatedEvent> = view
        .events
        .iter()
        .filter(|ev| {
            if let Some(schema) = schema_registry.schema(ev.component_type) {
                if let Some(field) = schema.fields.iter().find(|f| f.field_index == ev.event_field) {
                    return can_send_event(&field.condition, sender, recipient);
                }
            }
            true
        })
        .map(|ev| (*ev).clone())
        .collect();

    if events.is_empty() {
        None
    } else {
        Some(EventBatch { frame, events })
    }
}

// ── Authority table ────────────────────────────────────────────────────────

/// Tracks ownership for entities and per-field overrides.
#[derive(Clone, Debug)]
pub struct AuthorityTable {
    entity_owners: Vec<(Entity, Ownership)>,
    field_owners: Vec<(Entity, ComponentId, u32, Ownership)>,
}

impl AuthorityTable {
    pub fn new() -> Self {
        Self {
            entity_owners: Vec::new(),
            field_owners: Vec::new(),
        }
    }

    pub fn set_entity_owner(&mut self, entity: Entity, owner: Ownership) {
        if let Some(slot) = self.entity_owners.iter_mut().find(|(e, _)| *e == entity) {
            slot.1 = owner;
        } else {
            self.entity_owners.push((entity, owner));
        }
    }

    pub fn set_field_owner(
        &mut self,
        entity: Entity,
        component: ComponentId,
        field: u32,
        owner: Ownership,
    ) {
        if let Some(slot) = self
            .field_owners
            .iter_mut()
            .find(|(e, c, f, _)| *e == entity && *c == component && *f == field)
        {
            slot.3 = owner;
        } else {
            self.field_owners
                .push((entity, component, field, owner));
        }
    }

    pub fn owner_of(&self, entity: Entity) -> Ownership {
        // Per-field override is not returned here — this is entity-level.
        self.entity_owners
            .iter()
            .find(|(e, _)| *e == entity)
            .map(|(_, o)| *o)
            .unwrap_or(Ownership::Server)
    }

    pub fn can_write(
        &self,
        entity: Entity,
        component: ComponentId,
        field: u32,
        client: ClientId,
    ) -> bool {
        // Per-field override takes precedence, then entity-level.
        if let Some((_, _, _, owner)) = self
            .field_owners
            .iter()
            .find(|(e, c, f, _)| *e == entity && *c == component && *f == field)
        {
            return match owner {
                Ownership::Server => false,
                Ownership::Client(id) => *id == client,
                Ownership::Shared => true,
            };
        }
        if let Some((_, owner)) = self.entity_owners.iter().find(|(e, _)| *e == entity) {
            return match owner {
                Ownership::Server => false,
                Ownership::Client(id) => *id == client,
                Ownership::Shared => true,
            };
        }
        false
    }

    /// Resolve conflicting writes from two deltas (Shared / multi-user editor).
    ///
    /// For each entity+component in both deltas, the field values from the
    /// delta whose client has the higher `ClientId` are kept. Tiebreak is
    /// deterministic: higher `ClientId` wins. Spawns and despawns from both
    /// deltas are merged (deduplicated by Entity).
    pub fn resolve_conflict(
        _authority: &Self,
        a: &Delta,
        client_a: ClientId,
        b: &Delta,
        client_b: ClientId,
    ) -> Delta {
        let winner = if client_a > client_b { a } else { b };
        let loser = if client_a > client_b { b } else { a };

        // Start with the winner's data.
        let mut spawned = winner.spawned.clone();
        let mut despawned = winner.despawned.clone();
        let mut component_deltas = winner.component_deltas.clone();
        let mut events = winner.events.clone();

        // Merge the loser's spawns that don't conflict with the winner.
        for s in &loser.spawned {
            if !spawned.iter().any(|(e, _)| e == &s.0) {
                spawned.push(s.clone());
            }
        }

        // Merge the loser's despawns.
        for d in &loser.despawned {
            if !despawned.contains(d) {
                despawned.push(*d);
            }
        }

        // Merge the loser's component deltas for entities the winner didn't touch.
        for cd in &loser.component_deltas {
            if !component_deltas.iter().any(|c| c.entity == cd.entity && c.component_type == cd.component_type) {
                component_deltas.push(cd.clone());
            }
        }

        // Merge events.
        for ev in &loser.events {
            events.push(ev.clone());
        }

        Delta {
            frame: winner.frame.max(loser.frame),
            base_frame: winner.base_frame.min(loser.base_frame),
            spawned,
            despawned,
            component_deltas,
            events,
        }
    }
}

impl Default for AuthorityTable {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot & reconciliation ──────────────────────────────────────────────

/// A point-in-time capture of a world state at a specific frame.
/// Used as the basis for client correction in the reconciler.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub frame: u64,
    pub entities: Vec<EntitySnapshot>,
    /// Rows captured from [`SpatialCell`]s via [`Snapshot::capture_cells`].
    /// Empty for an archetype-`World` snapshot ([`Snapshot::capture_full`]/
    /// [`Snapshot::capture_relevant`]) — the two capture paths are
    /// independent and can be combined on one `Snapshot` when a scene mixes
    /// ECS entities and standalone spatial cells.
    pub cell_rows: Vec<CellRowSnapshot>,
}

/// A single entity's component data within a snapshot.
/// Each component's field data is encoded per its [`ReplicationEncoding`].
#[derive(Clone, Debug)]
pub struct EntitySnapshot {
    pub entity: Entity,
    pub components: Vec<(ComponentId, Vec<Vec<u8>>)>,
}

/// A single [`SpatialCell`] row's component data within a snapshot, captured
/// by [`Snapshot::capture_cells`]. Rows aren't ECS entities, so — unlike
/// [`EntitySnapshot`] — identity is just positional: `cell_index` plus this
/// entry's position among the cell's other captured rows (live-row order at
/// capture time).
#[derive(Clone, Debug)]
pub struct CellRowSnapshot {
    pub cell_index: usize,
    pub components: Vec<(ComponentId, Vec<Vec<u8>>)>,
}

impl Snapshot {
    /// Capture the full world state at `frame`.
    ///
    /// Walks every archetype, every entity, and encodes every component's
    /// field data using the schema registry. Returns a complete snapshot
    /// suitable for initial replication or full state recovery.
    pub fn capture_full(
        world: &crate::World,
        schema: &ReplicationRegistry,
        frame: u64,
    ) -> Self {
        let mut entities = Vec::new();
        for arch in &world.archetypes {
            for row in 0..arch.entities.len() {
                let entity = arch.entities[row];
                let components = Self::capture_row_components(arch, row, schema);
                entities.push(EntitySnapshot { entity, components });
            }
        }
        Self { frame, entities, cell_rows: Vec::new() }
    }

    /// Capture only entities relevant to a specific client.
    /// Avoids serializing off-relevance entities — an optimization for
    /// initial spawn or recovery from packet loss.
    pub fn capture_relevant(
        world: &crate::World,
        schema: &ReplicationRegistry,
        relevance: &RelevanceSet,
        frame: u64,
    ) -> Self {
        let mut entities = Vec::new();
        for arch in &world.archetypes {
            for row in 0..arch.entities.len() {
                let entity = arch.entities[row];
                if !relevance.contains(entity) {
                    continue;
                }
                let components = Self::capture_row_components(arch, row, schema);
                entities.push(EntitySnapshot { entity, components });
            }
        }
        Self { frame, entities, cell_rows: Vec::new() }
    }

    /// Shared row-capture body for [`Self::capture_full`]/
    /// [`Self::capture_relevant`]: for every component the archetype
    /// carries, encode every schema field via its own
    /// [`FieldOps::encode`] closure (reached through
    /// [`Replicable::replicate_encode`]) — safe, and genuinely per-field
    /// (unlike the previous raw-byte approach, which re-read the whole
    /// column's bytes once per field, duplicating them across any
    /// multi-field schema).
    fn capture_row_components(
        arch: &crate::archetype::Archetype,
        row: usize,
        schema: &ReplicationRegistry,
    ) -> Vec<(ComponentId, Vec<Vec<u8>>)> {
        let mut components = Vec::new();
        for &cid in &arch.active_cids {
            let mut field_data: Vec<Vec<u8>> = Vec::new();
            if let Some(s) = schema.schema(cid) {
                if let Some(col) = arch.get_erased(cid) {
                    for field in &s.fields {
                        let idx = field.field_index as usize;
                        if idx >= field_data.len() {
                            field_data.resize(idx + 1, Vec::new());
                        }
                        if matches!(field.encoding, ReplicationEncoding::Event) {
                            continue;
                        }
                        let Some(ops) = &field.ops else { continue };
                        let mut buf = Vec::new();
                        if (ops.encode)(col, row, &mut buf).is_ok() {
                            field_data[idx] = buf;
                        }
                    }
                }
            }
            components.push((cid, field_data));
        }
        components
    }

    /// Capture all live rows from a set of [`SpatialCell`]s.
    ///
    /// For each cell, for each of `schema`'s registered component types that
    /// the cell exposes as a raw Pod column (`CellStorage::column_raw_bytes`
    /// — i.e. columns registered via `register_token_column`/`from_cell_type`,
    /// such as `SpatialCell::with_transform`'s transform/instance-info
    /// columns), encodes that row's bytes per schema field, indexed by
    /// `field.field_index` (matching [`Delta::apply`]'s convention so
    /// [`Self::restore_to_cells`] can invert this losslessly for Pod-style
    /// fields). Dead rows are skipped.
    pub fn capture_cells(cells: &[SpatialCell], schema: &ReplicationRegistry, frame: u64) -> Self {
        let mut cell_rows = Vec::new();
        for (cell_index, cell) in cells.iter().enumerate() {
            let storage = cell.storage();
            let rows = storage.rows_in_use();
            if rows == 0 {
                continue;
            }
            let liveness = LivenessSnapshot::capture(storage.liveness(), rows);
            for row in 0..rows {
                if !liveness.is_live(row) {
                    continue;
                }
                let mut components = Vec::new();
                for s in schema.schemas.values() {
                    let Some(col_bytes) = storage.column_raw_bytes(s.component_type) else {
                        continue;
                    };
                    let elem_size = col_bytes.len() / rows as usize;
                    if elem_size == 0 {
                        continue;
                    }
                    let start = row as usize * elem_size;
                    let value = &col_bytes[start..start + elem_size];

                    let mut field_data: Vec<Vec<u8>> = Vec::new();
                    let mut any = false;
                    for field in &s.fields {
                        let idx = field.field_index as usize;
                        if idx >= field_data.len() {
                            field_data.resize(idx + 1, Vec::new());
                        }
                        if matches!(field.encoding, ReplicationEncoding::Event) {
                            continue;
                        }
                        let mut buf = Vec::new();
                        encode_field_value(&field.encoding, value, &mut buf);
                        field_data[idx] = buf;
                        any = true;
                    }
                    if any {
                        components.push((s.component_type, field_data));
                    }
                }
                cell_rows.push(CellRowSnapshot { cell_index, components });
            }
        }
        Self { frame, entities: Vec::new(), cell_rows }
    }

    /// Restore a snapshot captured by [`Self::capture_cells`] into a set of
    /// [`SpatialCell`]s. Allocates one fresh handle per captured row (in
    /// capture order — the snapshot doesn't carry original handle bits,
    /// since a row's identity is transient across cell compaction anyway)
    /// and writes each component's bytes back, packing fields sequentially
    /// by byte offset like [`Delta::apply`].
    ///
    /// Returns [`ErrorCode::InvalidData`] if a row names a `cell_index`
    /// outside `cells`, or if a cell is full and can't allocate.
    ///
    /// # Safety-adjacent caveat
    ///
    /// Like `Delta::apply`, this writes raw bytes into Pod columns — sound
    /// only for the Pod-like column types `SpatialCell`'s token-registered
    /// columns actually hold (`[f32; 16]`, `InstanceInfo`, plain `f32`
    /// bounds, etc.).
    pub fn restore_to_cells(&self, cells: &mut [SpatialCell], schema: &ReplicationRegistry) -> Result<(), ErrorCode> {
        // Reused across every field of every row instead of a fresh `Vec`
        // per field — same rationale as `Delta::apply_with_scratch`.
        let mut scratch: Vec<u8> = Vec::new();
        for row_snap in &self.cell_rows {
            let Some(cell) = cells.get_mut(row_snap.cell_index) else {
                return Err(ErrorCode::InvalidData);
            };
            let handle = cell
                .alloc(Aabb { min: [0.0; 3], max: [0.0; 3] })
                .ok_or(ErrorCode::InvalidData)?;
            let row = cell.row_of(handle).ok_or(ErrorCode::InvalidData)?;

            for (cid, field_data) in &row_snap.components {
                let Some(s) = schema.schema(*cid) else {
                    continue;
                };
                let storage = cell.storage_mut();
                let rows_now = storage.rows_in_use();
                let Some(col_bytes) = storage.column_raw_bytes_mut(*cid) else {
                    continue;
                };
                let elem_size = col_bytes.len() / rows_now as usize;
                if elem_size == 0 {
                    continue;
                }
                let start = row as usize * elem_size;
                let dst = &mut col_bytes[start..start + elem_size];

                let mut offset = 0usize;
                for field in &s.fields {
                    if offset >= elem_size {
                        break;
                    }
                    if matches!(field.encoding, ReplicationEncoding::Event) {
                        continue;
                    }
                    let idx = field.field_index as usize;
                    let Some(raw) = field_data.get(idx) else {
                        continue;
                    };
                    if raw.is_empty() {
                        continue;
                    }
                    let remaining = elem_size - offset;
                    let width = match &field.encoding {
                        ReplicationEncoding::Pod | ReplicationEncoding::Serialized | ReplicationEncoding::GpuHandle => {
                            raw.len().min(remaining)
                        }
                        ReplicationEncoding::DeltaCompressed => remaining.min(8),
                        ReplicationEncoding::Opaque { .. } => remaining,
                        ReplicationEncoding::Event => 0,
                    };
                    if width == 0 {
                        continue;
                    }
                    scratch.clear();
                    scratch.resize(width, 0);
                    let code = decode_field_value(&field.encoding, raw, &mut scratch);
                    if code != ErrorCode::Ok {
                        return Err(code);
                    }
                    dst[offset..offset + width].copy_from_slice(&scratch);
                    offset += width;
                }
            }
        }
        Ok(())
    }
}

/// A predicted local write to be replayed after server correction.
#[derive(Clone, Debug)]
pub struct ClientInput {
    pub frame: u64,
    pub entity: Entity,
    pub component: ComponentId,
    pub field_data: Vec<(u32, Vec<u8>)>,
}

/// Client-side prediction reconciler.
///
/// Maintains a history ring buffer of server snapshots and a queue of
/// unacknowledged local inputs. When a server delta arrives,
/// [`reconcile`](Self::reconcile) applies it, rolls back to the matching
/// server snapshot, and replays pending local inputs on top.
#[derive(Clone, Debug)]
pub struct Reconciler {
    snapshots: Vec<Snapshot>,
    pending_inputs: Vec<ClientInput>,
    server_frame: u64,
    local_frame: u64,
}

impl Reconciler {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            pending_inputs: Vec::new(),
            server_frame: 0,
            local_frame: 0,
        }
    }

    /// Push a server snapshot into the history ring buffer.
    /// Drops the oldest snapshot if the buffer exceeds 64 entries.
    pub fn push_snapshot(&mut self, snapshot: Snapshot) {
        self.server_frame = snapshot.frame;
        if self.snapshots.len() >= 64 {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    /// Record a local predicted write before the server responds.
    pub fn push_input(&mut self, input: ClientInput) {
        self.local_frame = input.frame;
        self.pending_inputs.push(input);
    }

    pub fn pending_inputs(&self) -> &[ClientInput] {
        &self.pending_inputs
    }

    /// Forget inputs and snapshots older than `frame`.
    pub fn clear_before(&mut self, frame: u64) {
        self.pending_inputs.retain(|i| i.frame > frame);
        self.snapshots.retain(|s| s.frame > frame);
    }

    /// The most recent server frame the reconciler knows about.
    pub fn server_frame(&self) -> u64 {
        self.server_frame
    }

    /// The most recent local frame with pending inputs.
    pub fn local_frame(&self) -> u64 {
        self.local_frame
    }

    /// Apply a server delta and reconcile pending inputs on top.
    ///
    /// Steps:
    /// 1. Discard acknowledged inputs (frame <= delta.base_frame).
    /// 2. Discard snapshots older than delta.base_frame.
    /// 3. Call `replay(world, input)` for each remaining pending input
    ///    in frame order, re-applying predicted local writes on top of
    ///    the now-corrected world.
    ///
    /// # Usage
    ///
    /// The caller must apply `server_delta` to `world` *before* calling
    /// this method. The reconciler only handles replay of pending inputs:
    ///
    /// ```ignore
    /// apply_delta_to_world(&server_delta, &mut world, &schema);
    /// reconciler.reconcile(&server_delta, &mut world, |w, input| {
    ///     // write input.field_data back to w[input.entity][input.component]
    /// });
    /// ```
    pub fn reconcile(
        &mut self,
        server_delta: &Delta,
        world: &mut crate::World,
        mut replay: impl FnMut(&mut crate::World, &ClientInput),
    ) {
        let base = server_delta.base_frame;
        self.clear_before(base);

        // Re-play pending inputs in frame order.
        let to_replay: Vec<ClientInput> = self.pending_inputs.clone();
        for input in &to_replay {
            replay(world, input);
        }
    }
}

impl Default for Reconciler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Frame phase machine integration ────────────────────────────────────────

/// Non-GPU witness for the simulate phase. Always available (no `gpu`
/// feature required) — the C0 counterpart to `gpu::phase::SimulateWitness`,
/// for callers that drive their own frame loop without the GPU-resident
/// scene store and its compile-time phase machine.
///
/// Holding this witness carries no compile-time ordering guarantee (unlike
/// `gpu::phase`'s sealed, consuming witness chain) — it exists purely to
/// give [`ChangeTracker::drain_with_world`] + [`ChangeTracker::end_frame`]
/// a named call site that mirrors the GPU path's `SimulateWitness::run_tracked`.
pub struct CpuSimulateWitness;

impl CpuSimulateWitness {
    pub fn new() -> Self {
        Self
    }

    /// Run a simulate phase with change tracking.
    ///
    /// Runs `systems` (which mutates `world` and records changes into
    /// `tracker`), then drains the tracker at the Simulate→Harvest boundary
    /// via [`ChangeTracker::drain_with_world`] (so spawns carry a real
    /// archetype-key blob — see [`encode_archetype_key`]) and advances the
    /// tracker's frame counter. Returns the resulting [`Delta`].
    pub fn run_tracked<F>(&self, world: &mut crate::World, tracker: &mut ChangeTracker, systems: F) -> Delta
    where
        F: FnOnce(&mut crate::World, &mut ChangeTracker),
    {
        systems(world, tracker);
        let delta = tracker.drain_with_world(world);
        tracker.end_frame();
        delta
    }
}

impl Default for CpuSimulateWitness {
    fn default() -> Self {
        Self::new()
    }
}

// ── Assert endianness ──────────────────────────────────────────────────────

const _: () = assert!(
    cfg!(target_endian = "little"),
    "SceneDB replication requires a little-endian target"
);

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn make_entity(index: u32, gen: u32) -> Entity {
        Entity::new(index, gen)
    }

    /// Empty-fields schema for tests that only exercise `ChangeTracker::drain`
    /// (which ignores its `_schema` argument entirely).
    fn test_schema(component_type: ComponentId) -> ReplicationSchema {
        ReplicationSchema { component_type, fields: vec![], row_ops: None }
    }

    /// `RowOps` for hand-built `SchemaBuilder::<f32>` test literals — mirrors
    /// exactly what `ReplicationRegistry::register::<f32>()` would capture.
    fn test_row_ops_f32() -> RowOps {
        RowOps {
            new_column: || Box::new(crate::component::Column::<f32>::new()),
            push_default: |col| {
                col.as_any_mut()
                    .downcast_mut::<crate::component::Column<f32>>()
                    .unwrap()
                    .data
                    .push(0.0);
            },
        }
    }

    // ── Fuzz / property-test helpers ─────────────────────────────────

    /// Random `ReplicationEncoding`, excluding `Opaque` — its fn pointers
    /// don't round-trip through the wire handshake format (see
    /// `u8_from_encoding`'s doc), so a generator feeding handshake/wire
    /// round-trip tests must not produce it.
    fn random_encoding(rng: &mut impl Rng) -> ReplicationEncoding {
        match rng.gen_range(0..5) {
            0 => ReplicationEncoding::Pod,
            1 => ReplicationEncoding::Serialized,
            2 => ReplicationEncoding::GpuHandle,
            3 => ReplicationEncoding::DeltaCompressed,
            _ => ReplicationEncoding::Event,
        }
    }

    fn random_condition(rng: &mut impl Rng) -> ReplicationCondition {
        match rng.gen_range(0..11) {
            0 => ReplicationCondition::Always,
            1 => ReplicationCondition::OwnerOnly,
            2 => ReplicationCondition::SkipOwner,
            3 => ReplicationCondition::SimulatedOnly,
            4 => ReplicationCondition::AutonomousOnly,
            5 => ReplicationCondition::InitialOnly,
            6 => ReplicationCondition::ServerAuthority,
            7 => ReplicationCondition::ClientAuthority,
            8 => ReplicationCondition::ServerToClient,
            9 => ReplicationCondition::ClientToServer,
            _ => ReplicationCondition::Multicast,
        }
    }

    fn random_bytes(rng: &mut impl Rng, max_len: usize) -> Vec<u8> {
        let len = rng.gen_range(0..=max_len);
        (0..len).map(|_| rng.gen()).collect()
    }

    fn random_entity(rng: &mut impl Rng) -> Entity {
        make_entity(rng.gen_range(0..1000), rng.gen_range(0..10))
    }

    fn random_delta(rng: &mut impl Rng) -> Delta {
        let spawned = (0..rng.gen_range(0..5))
            .map(|_| (random_entity(rng), random_bytes(rng, 20)))
            .collect();
        let despawned = (0..rng.gen_range(0..5)).map(|_| random_entity(rng)).collect();
        let component_deltas = (0..rng.gen_range(0..5))
            .map(|_| ComponentDelta {
                entity: random_entity(rng),
                component_type: ComponentId(rng.gen_range(0..50)),
                field_data: (0..rng.gen_range(0..4)).map(|_| random_bytes(rng, 16)).collect(),
            })
            .collect();
        let events = (0..rng.gen_range(0..5))
            .map(|_| ReplicatedEvent {
                entity: random_entity(rng),
                component_type: ComponentId(rng.gen_range(0..50)),
                event_field: rng.gen_range(0..10),
                payload: random_bytes(rng, 20),
                channel: if rng.gen_bool(0.5) { EventChannel::ReliableOrdered } else { EventChannel::Unreliable },
                target_client: if rng.gen_bool(0.5) { Some(ClientId(rng.gen_range(0..20))) } else { None },
            })
            .collect();

        Delta {
            frame: rng.gen(),
            base_frame: rng.gen(),
            spawned,
            despawned,
            component_deltas,
            events,
        }
    }

    fn assert_deltas_equal(a: &Delta, b: &Delta) {
        assert_eq!(a.frame, b.frame);
        assert_eq!(a.base_frame, b.base_frame);
        assert_eq!(a.spawned, b.spawned);
        assert_eq!(a.despawned, b.despawned);
        assert_eq!(a.component_deltas.len(), b.component_deltas.len());
        for (x, y) in a.component_deltas.iter().zip(b.component_deltas.iter()) {
            assert_eq!(x.entity, y.entity);
            assert_eq!(x.component_type, y.component_type);
            assert_eq!(x.field_data, y.field_data);
        }
        assert_eq!(a.events.len(), b.events.len());
        for (x, y) in a.events.iter().zip(b.events.iter()) {
            assert_eq!(x.entity, y.entity);
            assert_eq!(x.component_type, y.component_type);
            assert_eq!(x.event_field, y.event_field);
            assert_eq!(x.payload, y.payload);
            assert_eq!(x.channel, y.channel);
            assert_eq!(x.target_client, y.target_client);
        }
    }

    #[test]
    fn tracker_records_spawns() {
        let mut t = ChangeTracker::new();
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        t.record_spawn(e1);
        t.record_spawn(e2);
        let (delta, _events) = t.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.spawned.len(), 2);
        assert!(delta.spawned.iter().any(|(e, _)| *e == e1));
        assert!(delta.spawned.iter().any(|(e, _)| *e == e2));
        assert!(delta.despawned.is_empty());
        assert!(delta.component_deltas.is_empty());
        assert!(_events.is_empty());
    }

    #[test]
    fn tracker_records_despawns() {
        let mut t = ChangeTracker::new();
        let e = make_entity(0, 1);
        t.record_despawn(e);
        let (delta, _events) = t.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.despawned.len(), 1);
        assert_eq!(delta.despawned[0], e);
    }

    #[test]
    fn tracker_records_component_changes() {
        let mut t = ChangeTracker::new();
        let e = make_entity(0, 1);
        let cid = ComponentId(3);
        let data = vec![1u8, 2, 3, 4];
        t.record_component_change(e, cid, 0, data.clone());
        let (delta, _) = t.drain(&test_schema(cid), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.component_deltas.len(), 1);
        assert_eq!(delta.component_deltas[0].entity, e);
        assert_eq!(delta.component_deltas[0].component_type, cid);
        assert_eq!(delta.component_deltas[0].field_data[0], data);
    }

    #[test]
    fn tracker_accrues_multiple_fields_on_same_component() {
        let mut t = ChangeTracker::new();
        let e = make_entity(0, 1);
        let cid = ComponentId(7);
        t.record_component_change(e, cid, 1, vec![10u8]);
        t.record_component_change(e, cid, 0, vec![20u8]);
        let (delta, _) = t.drain(&test_schema(cid), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.component_deltas.len(), 1);
        assert_eq!(delta.component_deltas[0].field_data[0], vec![20u8]);
        assert_eq!(delta.component_deltas[0].field_data[1], vec![10u8]);
    }

    #[test]
    fn tracker_records_events() {
        let mut t = ChangeTracker::new();
        let e = make_entity(2, 5);
        let ev = ReplicatedEvent {
            entity: e,
            component_type: ComponentId(1),
            event_field: 0,
            payload: vec![99u8],
            channel: EventChannel::ReliableOrdered,
            target_client: None,
        };
        t.record_event(ev.clone());
        let (delta, _) = t.drain(&test_schema(ComponentId(1)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.events.len(), 1);
        assert_eq!(delta.events[0].entity, e);
        assert_eq!(delta.events[0].payload, vec![99u8]);
    }

    #[test]
    fn drain_clears_tracker() {
        let mut t = ChangeTracker::new();
        t.record_spawn(make_entity(0, 1));
        let (delta, _) = t.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.spawned.len(), 1);
        let (delta2, _) = t.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert!(delta2.spawned.is_empty());
        assert!(delta2.despawned.is_empty());
        assert!(delta2.component_deltas.is_empty());
    }

    #[test]
    fn end_frame_increments_counter() {
        let mut t = ChangeTracker::new();
        assert_eq!(t.frame, 0);
        t.end_frame();
        assert_eq!(t.frame, 1);
        t.end_frame();
        assert_eq!(t.frame, 2);
    }

    #[test]
    fn drain_includes_frame_number() {
        let mut t = ChangeTracker::new();
        t.end_frame();
        let (delta, _) = t.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.frame, 1);
    }

    #[test]
    fn world_spawn_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn_tracked(&mut tracker);
        let (delta, _) = tracker.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.spawned.len(), 1);
        assert_eq!(delta.spawned[0].0, e);
    }

    #[test]
    fn world_despawn_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn();
        assert!(world.despawn_tracked(e, &mut tracker));
        let (delta, _) = tracker.drain(&test_schema(ComponentId(0)), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.despawned.len(), 1);
        assert_eq!(delta.despawned[0], e);
    }

    #[test]
    fn world_insert_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn();
        // spawn is untracked here — only insert is tracked
        let cid = crate::component::component_id::<f32>();
        world.insert_tracked(e, 42.0f32, &mut tracker);
        let (delta, _) = tracker.drain(&test_schema(cid), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.component_deltas.len(), 1);
        assert_eq!(delta.component_deltas[0].entity, e);
    }

    #[test]
    fn world_remove_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn();
        world.insert(e, 10.0f32);
        let removed = world.remove_tracked::<f32>(e, &mut tracker);
        assert_eq!(removed, Some(10.0f32));
        let (delta, _) = tracker.drain(&test_schema(crate::component::component_id::<f32>()), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.component_deltas.len(), 1);
        assert_eq!(delta.component_deltas[0].entity, e);
    }

    // ── R2: Schema builder and registry ─────────────────────────────────

    #[test]
    fn schema_builder_produces_correct_fields() {
        let cid = ComponentId(42);
        let schema = SchemaBuilder::<f32> {
            component_type: cid,
            fields: Vec::new(),
            row_ops: test_row_ops_f32(),
            _phantom: PhantomData,
        }
        .field("x", |c: &f32| c, |c: &mut f32| c, ReplicationEncoding::Pod, ReplicationCondition::Always)
        .field("y", |c: &f32| c, |c: &mut f32| c, ReplicationEncoding::Pod, ReplicationCondition::SimulatedOnly)
        .build();

        assert_eq!(schema.component_type, cid);
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[0].field_index, 0);
        assert_eq!(schema.fields[1].field_index, 1);
        assert_eq!(schema.fields[1].condition, ReplicationCondition::SimulatedOnly);
    }

    #[test]
    fn schema_builder_event_field() {
        let schema = SchemaBuilder::<f32> {
            component_type: ComponentId(1),
            fields: Vec::new(),
            row_ops: test_row_ops_f32(),
            _phantom: PhantomData,
        }
        .event("on_foo", ReplicationCondition::Multicast, EventChannel::Unreliable)
        .build();

        assert_eq!(schema.fields.len(), 1);
        match &schema.fields[0].encoding {
            ReplicationEncoding::Event => {}
            _ => panic!("expected Event encoding"),
        }
        assert_eq!(schema.fields[0].event_channel, Some(EventChannel::Unreliable));
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder);
        assert!(reg.schema(cid).is_some());
        assert!(reg.schema(ComponentId(999)).is_none());
    }

    #[test]
    fn registry_handshake_round_trip() {
        let mut reg = ReplicationRegistry::new();

        let b1 = reg.register::<f32>();
        reg.insert(b1);

        let msg = reg.handshake_message();
        let reg2 = ReplicationRegistry::from_handshake(&msg).unwrap();

        assert_eq!(reg.schemas.len(), reg2.schemas.len());
        for (cid, s1) in &reg.schemas {
            let s2 = reg2.schema(*cid).unwrap();
            assert_eq!(s1.fields.len(), s2.fields.len());
        }
    }

    #[test]
    fn registry_handshake_empty() {
        let reg = ReplicationRegistry::new();
        let msg = reg.handshake_message();
        let reg2 = ReplicationRegistry::from_handshake(&msg).unwrap();
        assert_eq!(reg2.schemas.len(), 0);
    }

    #[test]
    fn registry_handshake_invalid_truncated() {
        let result = ReplicationRegistry::from_handshake(&[0x01, 0x00, 0x00, 0x00]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ErrorCode::InvalidData);
    }

    /// Regression test for a fuzz-discovered off-by-3 in `from_handshake`'s
    /// per-field truncation check: it required 10 bytes remaining per field
    /// but only ever consumes 7 (field_index:4 + encoding:1 + condition:1 +
    /// event_channel:1), so a well-formed handshake whose LAST field landed
    /// exactly at the buffer's end was rejected as truncated.
    #[test]
    fn registry_handshake_last_field_at_exact_buffer_end() {
        let mut reg = ReplicationRegistry::new();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let msg = reg.handshake_message();
        // The single field here is the last bytes of the buffer — exactly
        // the shape that used to trip the bug.
        let reg2 = ReplicationRegistry::from_handshake(&msg).expect("well-formed handshake must parse");
        assert_eq!(reg2.schemas.len(), 1);
    }

    #[test]
    fn encode_decode_pod_round_trip() {
        let original = vec![1u8, 2, 3, 4, 5];
        let mut buf = Vec::new();
        assert_eq!(encode_field_value(&ReplicationEncoding::Pod, &original, &mut buf), ErrorCode::Ok);
        let mut decoded = vec![0u8; 5];
        assert_eq!(decode_field_value(&ReplicationEncoding::Pod, &buf, &mut decoded), ErrorCode::Ok);
        assert_eq!(original, decoded);
    }

    #[test]
    fn encode_pod_raw_writes_bytes_directly() {
        let value = 0xDEAD_BEEFu32.to_le_bytes();
        let mut buf = [0u8; 8];
        let n = unsafe { encode_pod_raw(&value, &mut buf) };
        assert_eq!(n, 4);
        assert_eq!(&buf[..4], &value);
        assert_eq!(&buf[4..], &[0u8; 4]);
    }

    #[test]
    fn encode_pod_raw_into_exact_size_buffer() {
        let value = [1u8, 2, 3];
        let mut buf = [0u8; 3];
        let n = unsafe { encode_pod_raw(&value, &mut buf) };
        assert_eq!(n, 3);
        assert_eq!(buf, value);
    }

    #[test]
    fn apply_with_scratch_reuses_buffer_across_calls() {
        let mut world = crate::World::new();
        let e = world.spawn();
        world.insert(e, 1.0f32);

        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let mut scratch = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            let delta = Delta {
                frame: 0, base_frame: 0,
                spawned: vec![], despawned: vec![],
                component_deltas: vec![ComponentDelta {
                    entity: e, component_type: cid,
                    field_data: vec![v.to_le_bytes().to_vec()],
                }],
                events: vec![],
            };
            assert_eq!(delta.apply_with_scratch(&mut world, &reg, &mut scratch), Ok(()));
            assert_eq!(*world.get::<f32>(e).unwrap(), v);
        }
    }

    #[test]
    fn encode_decode_gpu_handle_round_trip() {
        let original = 42u64.to_le_bytes().to_vec();
        let mut buf = Vec::new();
        assert_eq!(encode_field_value(&ReplicationEncoding::GpuHandle, &original, &mut buf), ErrorCode::Ok);
        let mut decoded = vec![0u8; 8];
        assert_eq!(decode_field_value(&ReplicationEncoding::GpuHandle, &buf, &mut decoded), ErrorCode::Ok);
        assert_eq!(original, decoded);
    }

    #[test]
    fn encode_decode_delta_compressed_round_trip() {
        let original = 123456789u64.to_le_bytes().to_vec();
        let mut buf = Vec::new();
        assert_eq!(encode_field_value(&ReplicationEncoding::DeltaCompressed, &original, &mut buf), ErrorCode::Ok);
        let mut decoded = vec![0u8; 8];
        assert_eq!(decode_field_value(&ReplicationEncoding::DeltaCompressed, &buf, &mut decoded), ErrorCode::Ok);
        assert_eq!(original, decoded);
    }

    #[test]
    fn encode_decode_opaque_round_trip() {
        let opaque = ReplicationEncoding::Opaque {
            encode_size: |_ptr| {
                std::mem::size_of::<u64>()
            },
            encode: |ptr, buf| {
                let val = unsafe { *(ptr as *const u64) };
                buf.copy_from_slice(&val.to_le_bytes());
                ErrorCode::Ok
            },
            decode: |data, dst| {
                let val = u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]);
                unsafe { *(dst as *mut u64) = val; }
                ErrorCode::Ok
            },
        };

        let original = 0xDEAD_BEEFu64;
        let bytes = original.to_le_bytes();
        let mut buf = Vec::new();
        assert_eq!(encode_field_value(&opaque, &bytes, &mut buf), ErrorCode::Ok);
        let mut decoded = vec![0u8; 8];
        assert_eq!(decode_field_value(&opaque, &buf, &mut decoded), ErrorCode::Ok);
        let result = u64::from_le_bytes(decoded.try_into().unwrap());
        assert_eq!(result, original);
    }

    // ── R3: Relevance, conditions, authority ────────────────────────

    #[test]
    fn condition_passes_always() {
        assert!(condition_passes(&ReplicationCondition::Always, &Ownership::Server, &ClientId(0)));
        assert!(condition_passes(&ReplicationCondition::Always, &Ownership::Client(ClientId(1)), &ClientId(0)));
    }

    #[test]
    fn condition_passes_owner_only() {
        let owner = Ownership::Client(ClientId(42));
        assert!(condition_passes(&ReplicationCondition::OwnerOnly, &owner, &ClientId(42)));
        assert!(!condition_passes(&ReplicationCondition::OwnerOnly, &owner, &ClientId(7)));
        assert!(!condition_passes(&ReplicationCondition::OwnerOnly, &Ownership::Server, &ClientId(0)));
    }

    #[test]
    fn condition_passes_skip_owner() {
        let owner = Ownership::Client(ClientId(42));
        assert!(!condition_passes(&ReplicationCondition::SkipOwner, &owner, &ClientId(42)));
        assert!(condition_passes(&ReplicationCondition::SkipOwner, &owner, &ClientId(7)));
    }

    #[test]
    fn condition_passes_simulated() {
        let owner = Ownership::Client(ClientId(42));
        assert!(!condition_passes(&ReplicationCondition::SimulatedOnly, &owner, &ClientId(42)));
        assert!(condition_passes(&ReplicationCondition::SimulatedOnly, &owner, &ClientId(7)));
        assert!(condition_passes(&ReplicationCondition::SimulatedOnly, &Ownership::Server, &ClientId(0)));
    }

    #[test]
    fn condition_passes_autonomous() {
        let owner = Ownership::Client(ClientId(42));
        assert!(condition_passes(&ReplicationCondition::AutonomousOnly, &owner, &ClientId(42)));
        assert!(!condition_passes(&ReplicationCondition::AutonomousOnly, &owner, &ClientId(7)));
        assert!(!condition_passes(&ReplicationCondition::AutonomousOnly, &Ownership::Server, &ClientId(0)));
    }

    #[test]
    fn condition_passes_authority_flags() {
        assert!(condition_passes(&ReplicationCondition::ServerAuthority, &Ownership::Server, &ClientId(0)));
        assert!(condition_passes(&ReplicationCondition::ClientAuthority, &Ownership::Server, &ClientId(0)));
        assert!(condition_passes(&ReplicationCondition::Multicast, &Ownership::Server, &ClientId(0)));
    }

    #[test]
    fn condition_passes_initial_and_rpc_always_false() {
        assert!(!condition_passes(&ReplicationCondition::InitialOnly, &Ownership::Server, &ClientId(0)));
        assert!(!condition_passes(&ReplicationCondition::ServerToClient, &Ownership::Server, &ClientId(0)));
        assert!(!condition_passes(&ReplicationCondition::ClientToServer, &Ownership::Server, &ClientId(0)));
    }

    #[test]
    fn authority_table_owner_of() {
        let mut at = AuthorityTable::new();
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        assert_eq!(at.owner_of(e1), Ownership::Server); // default
        at.set_entity_owner(e1, Ownership::Client(ClientId(7)));
        assert_eq!(at.owner_of(e1), Ownership::Client(ClientId(7)));
        assert_eq!(at.owner_of(e2), Ownership::Server); // unset
    }

    #[test]
    fn authority_table_can_write_respects_entity_owner() {
        let mut at = AuthorityTable::new();
        let e = make_entity(0, 1);
        at.set_entity_owner(e, Ownership::Client(ClientId(5)));
        assert!(at.can_write(e, ComponentId(1), 0, ClientId(5)));
        assert!(!at.can_write(e, ComponentId(1), 0, ClientId(9)));
        // Server cannot write to a client-owned entity.
        assert!(!at.can_write(e, ComponentId(1), 0, ClientId(0)));
    }

    #[test]
    fn authority_table_can_write_field_override_takes_precedence() {
        let mut at = AuthorityTable::new();
        let e = make_entity(0, 1);
        at.set_entity_owner(e, Ownership::Client(ClientId(5)));
        at.set_field_owner(e, ComponentId(1), 0, Ownership::Shared);
        // Field override: anyone can write field 0.
        assert!(at.can_write(e, ComponentId(1), 0, ClientId(99)));
        // Different field still uses entity-level.
        assert!(!at.can_write(e, ComponentId(1), 1, ClientId(99)));
    }

    #[test]
    fn resolve_conflict_picks_higher_client_id() {
        let mut at = AuthorityTable::new();
        let e = make_entity(0, 1);
        let cid = ComponentId(1);

        let delta_a = Delta {
            frame: 10, base_frame: 9,
            spawned: vec![(e, vec![1])],
            despawned: vec![],
            component_deltas: vec![ComponentDelta { entity: e, component_type: cid, field_data: vec![vec![1]] }],
            events: vec![],
        };
        let delta_b = Delta {
            frame: 10, base_frame: 9,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![ComponentDelta { entity: e, component_type: cid, field_data: vec![vec![2]] }],
            events: vec![],
        };

        // client_b (higher ID) wins.
        let merged = AuthorityTable::resolve_conflict(&at, &delta_a, ClientId(1), &delta_b, ClientId(2));
        assert_eq!(merged.component_deltas[0].field_data[0], vec![2]);
        assert_eq!(merged.spawned.len(), 1);
    }

    #[test]
    fn resolve_conflict_merges_spawns_from_both() {
        let at = AuthorityTable::new();
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        let delta_a = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(e1, vec![])],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        let delta_b = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(e2, vec![])],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        let merged = AuthorityTable::resolve_conflict(&at, &delta_a, ClientId(1), &delta_b, ClientId(2));
        assert_eq!(merged.spawned.len(), 2);
    }

    #[test]
    fn relevance_set_condition_filter() {
        let mut set = RelevanceSet::new();
        let e = make_entity(0, 1);
        set.add(e);

        let mut authority = AuthorityTable::new();
        authority.set_entity_owner(e, Ownership::Client(ClientId(42)));

        let mut reg = ReplicationRegistry::new();
        let f32_cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        let builder = builder.whole_field("test", ReplicationEncoding::Pod, ReplicationCondition::SimulatedOnly);
        reg.insert(builder);

        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![ComponentDelta { entity: e, component_type: f32_cid, field_data: vec![vec![1]] }],
            events: vec![],
        };

        // Different client (not owner) — SimulatedOnly passes because client != owner.
        let view = set.filter(&delta, &authority, &reg, ClientId(7));
        assert_eq!(view.component_deltas.len(), 1);

        // Owner client — SimulatedOnly fails because client == owner.
        let view = set.filter(&delta, &authority, &reg, ClientId(42));
        assert_eq!(view.component_deltas.len(), 0);
    }

    #[test]
    fn relevance_set_spawns_always_pass() {
        let set = RelevanceSet::new(); // empty
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(make_entity(0, 1), vec![])],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        let view = set.filter(&delta, &AuthorityTable::new(), &ReplicationRegistry::new(), ClientId(0));
        // Spawns pass through even with empty relevance set.
        assert_eq!(view.spawned.len(), 1);
    }

    #[test]
    fn multi_client_relevance_filter() {
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        let e3 = make_entity(2, 1);
        let cid = crate::component::component_id::<f32>();

        let mut authority = AuthorityTable::new();
        authority.set_entity_owner(e1, Ownership::Client(ClientId(100)));
        authority.set_entity_owner(e2, Ownership::Client(ClientId(200)));
        // e3 has no owner (defaults to Server)

        // Client 100: only e1 and e3 are relevant.
        let mut set_a = RelevanceSet::new();
        set_a.add(e1);
        set_a.add(e3);

        // Client 200: only e2 and e3 are relevant.
        let mut set_b = RelevanceSet::new();
        set_b.add(e2);
        set_b.add(e3);

        let delta = Delta {
            frame: 5, base_frame: 4,
            spawned: vec![(e1, vec![]), (e2, vec![]), (e3, vec![])],
            despawned: vec![],
            component_deltas: vec![
                ComponentDelta { entity: e1, component_type: cid, field_data: vec![vec![10]] },
                ComponentDelta { entity: e2, component_type: cid, field_data: vec![vec![20]] },
                ComponentDelta { entity: e3, component_type: cid, field_data: vec![vec![30]] },
            ],
            events: vec![],
        };

        let reg = ReplicationRegistry::new(); // no schema means conditions always pass (empty fields)

        let view_a = set_a.filter(&delta, &authority, &reg, ClientId(100));
        assert_eq!(view_a.component_deltas.len(), 2); // e1 + e3
        assert!(view_a.component_deltas.iter().any(|cd| cd.entity == e1));
        assert!(view_a.component_deltas.iter().any(|cd| cd.entity == e3));
        assert!(!view_a.component_deltas.iter().any(|cd| cd.entity == e2));

        let view_b = set_b.filter(&delta, &authority, &reg, ClientId(200));
        assert_eq!(view_b.component_deltas.len(), 2); // e2 + e3
        assert!(view_b.component_deltas.iter().any(|cd| cd.entity == e2));
        assert!(view_b.component_deltas.iter().any(|cd| cd.entity == e3));
        assert!(!view_b.component_deltas.iter().any(|cd| cd.entity == e1));
    }

    #[test]
    fn shared_conflict_resolution_two_writes() {
        let at = AuthorityTable::new();
        let e = make_entity(0, 1);
        let cid = ComponentId(99);

        let delta_a = Delta {
            frame: 10, base_frame: 9,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![ComponentDelta { entity: e, component_type: cid, field_data: vec![vec![10]] }],
            events: vec![],
        };
        let delta_b = Delta {
            frame: 10, base_frame: 9,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![ComponentDelta { entity: e, component_type: cid, field_data: vec![vec![20]] }],
            events: vec![],
        };

        // Higher ClientId wins: client_b (200) beats client_a (100).
        let merged = AuthorityTable::resolve_conflict(&at, &delta_a, ClientId(100), &delta_b, ClientId(200));
        assert_eq!(merged.component_deltas[0].field_data[0], vec![20]);

        // Reversed: client_a (200) now beats client_b (100).
        let merged = AuthorityTable::resolve_conflict(&at, &delta_b, ClientId(100), &delta_a, ClientId(200));
        assert_eq!(merged.component_deltas[0].field_data[0], vec![10]);
    }

    // ── EntityCellMap ───────────────────────────────────────────────

    #[test]
    fn entity_cell_map_round_trips() {
        let mut map = EntityCellMap::new();
        let e = make_entity(3, 1);
        map.insert(e, 2, 5);
        assert_eq!(map.entity_at(2, 5), Some(e));
        assert_eq!(map.cell_of(e), Some((2, 5)));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn entity_cell_map_remove_clears_both_directions() {
        let mut map = EntityCellMap::new();
        let e = make_entity(0, 1);
        map.insert(e, 1, 1);
        map.remove(e);
        assert_eq!(map.entity_at(1, 1), None);
        assert_eq!(map.cell_of(e), None);
        assert!(map.is_empty());
    }

    #[test]
    fn entity_cell_map_reinsert_moves_entity_and_frees_old_slot() {
        let mut map = EntityCellMap::new();
        let e = make_entity(0, 1);
        map.insert(e, 0, 0);
        map.insert(e, 3, 7); // entity moved to a new cell/row
        assert_eq!(map.entity_at(0, 0), None, "old slot vacated");
        assert_eq!(map.entity_at(3, 7), Some(e));
        assert_eq!(map.cell_of(e), Some((3, 7)));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn entity_cell_map_reinsert_slot_evicts_old_entity() {
        let mut map = EntityCellMap::new();
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        map.insert(e1, 0, 0);
        map.insert(e2, 0, 0); // e2 takes over e1's slot
        assert_eq!(map.entity_at(0, 0), Some(e2));
        assert_eq!(map.cell_of(e1), None, "e1 evicted from the map entirely");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn from_frustum_mapped_resolves_entities_via_map() {
        let mut cell = SpatialCell::new(64).unwrap();
        let h = cell
            .alloc(crate::spatial::Aabb { min: [0.0; 3], max: [1.0; 3] })
            .unwrap();
        let row = cell.row_of(h).unwrap();

        let mut map = EntityCellMap::new();
        let e = make_entity(9, 1);
        map.insert(e, 0, row);

        let liveness = LivenessSnapshot::capture(cell.liveness(), cell.rows_in_use());
        let frustum = crate::spatial::Frustum {
            planes: [
                [1.0, 0.0, 0.0, 10.0], [-1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 10.0], [0.0, -1.0, 0.0, 10.0],
                [0.0, 0.0, 1.0, 10.0], [0.0, 0.0, -1.0, 10.0],
            ],
        };
        let mut scratch = crate::lease::Scratchpad::new();
        let set = RelevanceSet::from_frustum_mapped(&[cell], &frustum, &liveness, &mut scratch, &map);
        assert!(set.contains(e));
        assert_eq!(set.len(), 1);
    }

    // ── R4: Event / RPC channel ────────────────────────────────────────

    #[test]
    fn event_batch_struct() {
        let batch = EventBatch {
            frame: 42,
            events: vec![],
        };
        assert_eq!(batch.frame, 42);
        assert!(batch.events.is_empty());
    }

    #[test]
    fn can_send_event_client_to_server() {
        // ClientToServer: always sendable (server will receive it).
        assert!(can_send_event(&ReplicationCondition::ClientToServer, ClientId(1), ClientId(0)));
        assert!(can_send_event(&ReplicationCondition::ClientToServer, ClientId(1), ClientId(2)));
    }

    #[test]
    fn can_send_event_server_to_client() {
        // ServerToClient: only send if sender == recipient (server targeting itself
        // isn't useful, but the primitive just checks equality).
        assert!(can_send_event(&ReplicationCondition::ServerToClient, ClientId(5), ClientId(5)));
        assert!(!can_send_event(&ReplicationCondition::ServerToClient, ClientId(5), ClientId(7)));
    }

    #[test]
    fn can_send_event_multicast() {
        // Multicast: send to everyone except sender.
        assert!(can_send_event(&ReplicationCondition::Multicast, ClientId(1), ClientId(2)));
        assert!(!can_send_event(&ReplicationCondition::Multicast, ClientId(1), ClientId(1)));
    }

    #[test]
    fn events_to_batch_filters_by_direction() {
        let e1 = make_entity(0, 1);
        let f32_cid = crate::component::component_id::<f32>();

        let mut reg = ReplicationRegistry::new();
        let builder = reg.register::<f32>();
        let builder = builder
            .event("rpc", ReplicationCondition::ClientToServer, EventChannel::ReliableOrdered);
        reg.insert(builder);

        let ev = ReplicatedEvent {
            entity: e1,
            component_type: f32_cid,
            event_field: 0,
            payload: vec![1, 2, 3],
            channel: EventChannel::ReliableOrdered,
            target_client: None,
        };

        let delta = Delta {
            frame: 5, base_frame: 4,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![ev],
        };

        // Build DeltaView directly (no relevance filtering needed for direction tests).
        let view = DeltaView {
            spawned: &delta.spawned,
            despawned: &delta.despawned,
            component_deltas: delta.component_deltas.iter().collect(),
            events: delta.events.iter().collect(),
        };

        // A client sending to the server (client 1 → server 0) with ClientToServer should pass.
        let batch = events_to_batch(&view, 5, &reg, ClientId(1), ClientId(0));
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().events.len(), 1);
    }

    #[test]
    fn events_to_batch_filters_out_multicast_to_self() {
        let f32_cid = crate::component::component_id::<f32>();

        let mut reg = ReplicationRegistry::new();
        let builder = reg.register::<f32>();
        let builder = builder
            .event("rpc", ReplicationCondition::Multicast, EventChannel::Unreliable);
        reg.insert(builder);

        let ev = ReplicatedEvent {
            entity: make_entity(0, 1),
            component_type: f32_cid,
            event_field: 0,
            payload: vec![],
            channel: EventChannel::Unreliable,
            target_client: None,
        };

        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![ev],
        };

        let view = DeltaView {
            spawned: &delta.spawned,
            despawned: &delta.despawned,
            component_deltas: delta.component_deltas.iter().collect(),
            events: delta.events.iter().collect(),
        };

        // Sending to self: Multicast should be filtered out.
        let batch = events_to_batch(&view, 1, &reg, ClientId(5), ClientId(5));
        assert!(batch.is_none());

        // Sending to another client: should pass.
        let batch = events_to_batch(&view, 1, &reg, ClientId(5), ClientId(7));
        assert!(batch.is_some());
    }

    #[test]
    fn events_order_is_preserved_within_frame() {
        let e = make_entity(0, 1);
        let cid = ComponentId(20);

        let mut tracker = ChangeTracker::new();
        for i in 0..5u8 {
            tracker.record_event(ReplicatedEvent {
                entity: e,
                component_type: cid,
                event_field: i as u32,
                payload: vec![i],
                channel: EventChannel::ReliableOrdered,
                target_client: None,
            });
        }

        let mut reg = ReplicationRegistry::new();
        let builder = reg.register::<f32>();
        let builder = builder
            .event("ev0", ReplicationCondition::Multicast, EventChannel::ReliableOrdered)
            .event("ev1", ReplicationCondition::Multicast, EventChannel::ReliableOrdered)
            .event("ev2", ReplicationCondition::Multicast, EventChannel::ReliableOrdered)
            .event("ev3", ReplicationCondition::Multicast, EventChannel::ReliableOrdered)
            .event("ev4", ReplicationCondition::Multicast, EventChannel::ReliableOrdered);
        reg.insert(builder);

        let (delta, _) = tracker.drain(&test_schema(cid), ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.events.len(), 5);
        for (i, ev) in delta.events.iter().enumerate() {
            assert_eq!(ev.event_field, i as u32);
            assert_eq!(ev.payload, vec![i as u8]);
        }
    }

    #[test]
    fn event_encoding_excluded_from_state_deltas() {
        // Event encoding should produce no bytes in the buffer.
        let mut buf = Vec::new();
        assert_eq!(encode_field_value(&ReplicationEncoding::Event, &[1, 2, 3], &mut buf), ErrorCode::Ok);
        assert!(buf.is_empty());
    }

    #[test]
    fn event_channel_marking_is_preserved() {
        let ev = ReplicatedEvent {
            entity: make_entity(0, 1),
            component_type: ComponentId(1),
            event_field: 0,
            payload: vec![],
            channel: EventChannel::Unreliable,
            target_client: None,
        };
        assert_eq!(ev.channel, EventChannel::Unreliable);

        let ev2 = ReplicatedEvent {
            channel: EventChannel::ReliableOrdered,
            ..ev.clone()
        };
        assert_eq!(ev2.channel, EventChannel::ReliableOrdered);
    }

    // ── R5: Snapshot + reconciliation ────────────────────────────────

    #[test]
    fn snapshot_capture_full_includes_all_entities() {
        let mut world = crate::World::new();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, 1.0f32);
        world.insert(e2, 2.0f32);

        let reg = ReplicationRegistry::new();
        let snap = Snapshot::capture_full(&world, &reg, 7);
        assert_eq!(snap.frame, 7);
        assert_eq!(snap.entities.len(), 2);
    }

    #[test]
    fn snapshot_capture_relevant_filters_entities() {
        let mut world = crate::World::new();
        let e1 = world.spawn();
        let e2 = world.spawn();

        let mut relevance = RelevanceSet::new();
        relevance.add(e1); // only e1 is relevant

        let reg = ReplicationRegistry::new();
        let snap = Snapshot::capture_relevant(&world, &reg, &relevance, 3);
        assert_eq!(snap.frame, 3);
        assert_eq!(snap.entities.len(), 1);
        assert_eq!(snap.entities[0].entity, e1);
    }

    #[test]
    fn reconciler_push_and_clear() {
        let mut r = Reconciler::new();
        let snap = Snapshot { frame: 10, entities: vec![], cell_rows: vec![] };
        r.push_snapshot(snap);
        r.push_input(ClientInput {
            frame: 11, entity: make_entity(0, 1), component: ComponentId(1), field_data: vec![],
        });
        assert_eq!(r.server_frame(), 10);
        assert_eq!(r.local_frame(), 11);
        assert_eq!(r.pending_inputs().len(), 1);

        r.clear_before(10);
        assert_eq!(r.pending_inputs().len(), 1); // input at frame 11 > 10

        r.clear_before(12);
        assert!(r.pending_inputs().is_empty());
    }

    #[test]
    fn reconciler_ring_buffer_drops_oldest() {
        let mut r = Reconciler::new();
        for i in 0..70u64 {
            r.push_snapshot(Snapshot { frame: i, entities: vec![], cell_rows: vec![] });
        }
        // Ring buffer capped at 64 — oldest 6 frames dropped.
        assert!(r.snapshots.len() <= 64);
        assert_eq!(r.snapshots.first().unwrap().frame, 6);
        assert_eq!(r.snapshots.last().unwrap().frame, 69);
    }

    #[test]
    fn reconciler_reconcile_discards_old_inputs_and_replays() {
        let mut r = Reconciler::new();
        let e = make_entity(0, 1);

        // Three inputs at frames 1, 2, 3.
        for i in 1..=3u64 {
            r.push_input(ClientInput {
                frame: i,
                entity: e,
                component: ComponentId(1),
                field_data: vec![(0, vec![i as u8])],
            });
        }

        let server_delta = Delta {
            frame: 5, base_frame: 2,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };

        let mut replayed: Vec<u64> = Vec::new();
        let mut world = crate::World::new();

        r.reconcile(&server_delta, &mut world, |_, input| {
            replayed.push(input.frame);
        });

        // Only input at frame 3 should be replayed (frame > base_frame 2).
        assert_eq!(replayed, vec![3]);
        // Input at frames 1, 2 should be discarded.
        assert_eq!(r.pending_inputs().len(), 1);
        assert_eq!(r.pending_inputs()[0].frame, 3);
    }

    #[test]
    fn reconciler_three_frame_prediction_cycle() {
        let mut r = Reconciler::new();
        let e = make_entity(0, 1);

        // Simulate 3 frames of client prediction.
        for i in 1..=3u64 {
            r.push_input(ClientInput {
                frame: i,
                entity: e,
                component: ComponentId(1),
                field_data: vec![(0, vec![i as u8])],
            });
        }

        // Server delta at frame 4, base frame 1 → acknowledges frames ≤1.
        let server_delta = Delta {
            frame: 4, base_frame: 1,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };

        let mut replayed: Vec<u64> = Vec::new();
        let mut world = crate::World::new();
        r.reconcile(&server_delta, &mut world, |_, input| {
            replayed.push(input.frame);
        });

        // Frames 2 and 3 should be replayed.
        assert_eq!(replayed, vec![2, 3]);
        assert_eq!(r.pending_inputs().len(), 2);
    }

    // ── Delta serialization ──────────────────────────────────────────

    #[test]
    fn delta_to_bytes_round_trip_empty() {
        let d = Delta {
            frame: 0, base_frame: 0,
            spawned: vec![], despawned: vec![],
            component_deltas: vec![], events: vec![],
        };
        let bytes = d.to_bytes();
        let d2 = Delta::from_bytes(&bytes).unwrap();
        assert_eq!(d2.frame, 0);
        assert!(d2.spawned.is_empty());
        assert!(d2.despawned.is_empty());
    }

    #[test]
    fn delta_to_bytes_round_trip_full() {
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        let cid = ComponentId(5);

        let d = Delta {
            frame: 42, base_frame: 40,
            spawned: vec![(e1, vec![1, 2, 3])],
            despawned: vec![e2],
            component_deltas: vec![ComponentDelta {
                entity: e1, component_type: cid,
                field_data: vec![vec![10], vec![20, 30]],
            }],
            events: vec![ReplicatedEvent {
                entity: e1, component_type: cid,
                event_field: 0, payload: vec![99],
                channel: EventChannel::Unreliable,
                target_client: None,
            }],
        };
        let bytes = d.to_bytes();
        let d2 = Delta::from_bytes(&bytes).unwrap();

        assert_eq!(d2.frame, 42);
        assert_eq!(d2.base_frame, 40);
        assert_eq!(d2.spawned.len(), 1);
        assert_eq!(d2.spawned[0].0, e1);
        assert_eq!(d2.spawned[0].1, vec![1, 2, 3]);
        assert_eq!(d2.despawned, vec![e2]);
        assert_eq!(d2.component_deltas.len(), 1);
        assert_eq!(d2.component_deltas[0].field_data[0], vec![10]);
        assert_eq!(d2.component_deltas[0].field_data[1], vec![20, 30]);
        assert_eq!(d2.events.len(), 1);
        assert_eq!(d2.events[0].payload, vec![99]);
    }

    #[test]
    fn delta_from_bytes_truncated_returns_none() {
        assert!(Delta::from_bytes(&[]).is_none());
        assert!(Delta::from_bytes(&[0u8; 7]).is_none());
    }

    // ── Snapshot field bytes ──────────────────────────────────────────

    #[test]
    fn snapshot_capture_reads_field_bytes() {
        let mut world = crate::World::new();
        let e = world.spawn();
        world.insert(e, 42.0f32);

        let mut reg = ReplicationRegistry::new();
        let f32_cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let snap = Snapshot::capture_full(&world, &reg, 1);
        assert_eq!(snap.entities.len(), 1);
        assert_eq!(snap.entities[0].entity, e);

        let has_bytes = snap.entities[0]
            .components
            .iter()
            .any(|(cid, data)| *cid == f32_cid && data.iter().any(|d| !d.is_empty()));
        assert!(has_bytes, "snapshot should contain non-empty field data for f32 component");
    }

    // ── Snapshot capture/restore for SpatialCell ────────────────────

    #[test]
    fn capture_and_restore_cells_round_trip() {
        let mut cell = crate::spatial::SpatialCell::with_transform(64).unwrap();
        let h = cell.alloc(Aabb { min: [0.0; 3], max: [1.0; 3] }).unwrap();
        let row = cell.row_of(h).unwrap() as usize;
        cell.storage_mut().column_for_mut::<crate::spatial::InstanceInfo>().unwrap()[row] =
            crate::spatial::InstanceInfo { mesh_index: 42, flags: 1 };

        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<crate::spatial::InstanceInfo>();
        let builder = reg.register::<crate::spatial::InstanceInfo>();
        reg.insert(builder.whole_field("info", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let snap = Snapshot::capture_cells(&[cell], &reg, 1);
        assert_eq!(snap.frame, 1);
        assert_eq!(snap.cell_rows.len(), 1);
        assert_eq!(snap.cell_rows[0].cell_index, 0);
        assert!(snap.cell_rows[0].components.iter().any(|(c, _)| *c == cid));

        let mut restored = vec![crate::spatial::SpatialCell::with_transform(64).unwrap()];
        assert_eq!(snap.restore_to_cells(&mut restored, &reg), Ok(()));
        assert_eq!(restored[0].rows_in_use(), 1);
        let info = restored[0].storage().column_for::<crate::spatial::InstanceInfo>().unwrap()[0];
        assert_eq!(info, crate::spatial::InstanceInfo { mesh_index: 42, flags: 1 });
    }

    #[test]
    fn capture_cells_skips_dead_rows() {
        let mut cell = crate::spatial::SpatialCell::with_transform(64).unwrap();
        let ha = cell.alloc(Aabb { min: [0.0; 3], max: [1.0; 3] }).unwrap();
        let _hb = cell.alloc(Aabb { min: [0.0; 3], max: [1.0; 3] }).unwrap();
        cell.free(ha);

        let mut reg = ReplicationRegistry::new();
        let builder = reg.register::<crate::spatial::InstanceInfo>();
        reg.insert(builder.whole_field("info", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let snap = Snapshot::capture_cells(&[cell], &reg, 1);
        assert_eq!(snap.cell_rows.len(), 1, "dead row excluded");
    }

    #[test]
    fn restore_to_cells_rejects_out_of_range_cell_index() {
        let snap = Snapshot {
            frame: 0,
            entities: vec![],
            cell_rows: vec![CellRowSnapshot { cell_index: 5, components: vec![] }],
        };
        let reg = ReplicationRegistry::new();
        let mut cells: Vec<crate::spatial::SpatialCell> = vec![];
        assert_eq!(snap.restore_to_cells(&mut cells, &reg), Err(ErrorCode::InvalidData));
    }

    // ── Frame phase machine integration ─────────────────────────────

    #[test]
    fn cpu_simulate_witness_runs_tracked_and_advances_frame() {
        let witness = CpuSimulateWitness::new();
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();

        let delta = witness.run_tracked(&mut world, &mut tracker, |w, t| {
            w.spawn_tracked(t);
        });

        assert_eq!(delta.frame, 0);
        assert_eq!(delta.spawned.len(), 1);
        // A second run_tracked call should observe the advanced frame
        // counter and an emptied tracker.
        let delta2 = witness.run_tracked(&mut world, &mut tracker, |_, _| {});
        assert_eq!(delta2.frame, 1);
        assert!(delta2.spawned.is_empty());
    }

    // ── Delta::apply ─────────────────────────────────────────────────

    #[test]
    fn archetype_key_round_trips() {
        let ids = vec![ComponentId(3), ComponentId(7), ComponentId(1)];
        let bytes = encode_archetype_key(&ids);
        let decoded = decode_archetype_key(&bytes).unwrap();
        assert_eq!(decoded, ids);
    }

    #[test]
    fn archetype_key_decode_rejects_garbage() {
        assert!(decode_archetype_key(&[]).is_none());
        assert!(decode_archetype_key(&[1, 0, 0, 0]).is_none()); // claims 1 id, has 0
        // Plain `drain`'s placeholder blob (a bare frame u64) isn't valid.
        assert!(decode_archetype_key(&7u64.to_le_bytes()).is_none());
    }

    #[test]
    fn apply_writes_component_update_to_existing_entity() {
        let mut world = crate::World::new();
        let e = world.spawn();
        world.insert(e, 1.0f32);

        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![ComponentDelta {
                entity: e,
                component_type: cid,
                field_data: vec![9.5f32.to_le_bytes().to_vec()],
            }],
            events: vec![],
        };

        assert_eq!(delta.apply(&mut world, &reg), Ok(()));
        assert_eq!(*world.get::<f32>(e).unwrap(), 9.5f32);
    }

    #[test]
    fn apply_despawns_entities() {
        let mut world = crate::World::new();
        let e = world.spawn();
        assert!(world.is_alive(e));

        let reg = ReplicationRegistry::new();
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![],
            despawned: vec![e],
            component_deltas: vec![],
            events: vec![],
        };
        assert_eq!(delta.apply(&mut world, &reg), Ok(()));
        assert!(!world.is_alive(e));
    }

    #[test]
    fn apply_spawns_entity_at_exact_wire_handle_with_archetype() {
        let mut world = crate::World::new();

        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        let wire_entity = make_entity(41, 3);
        let blob = encode_archetype_key(&[cid]);
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(wire_entity, blob)],
            despawned: vec![],
            component_deltas: vec![ComponentDelta {
                entity: wire_entity,
                component_type: cid,
                field_data: vec![3.25f32.to_le_bytes().to_vec()],
            }],
            events: vec![],
        };

        assert_eq!(delta.apply(&mut world, &reg), Ok(()));
        assert!(world.is_alive(wire_entity), "entity lands at the exact wire index+generation");
        assert_eq!(*world.get::<f32>(wire_entity).unwrap(), 3.25f32);
    }

    #[test]
    fn apply_spawn_with_unknown_component_fails() {
        let mut world = crate::World::new();
        // Registry has no locally-registered factory for this component —
        // only a bare ComponentId with nothing behind it.
        let reg = ReplicationRegistry::new();
        let blob = encode_archetype_key(&[ComponentId(999)]);
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(make_entity(0, 0), blob)],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        assert_eq!(delta.apply(&mut world, &reg), Err(ErrorCode::InvalidData));
    }

    #[test]
    fn apply_spawn_with_malformed_blob_falls_back_to_empty_archetype() {
        let mut world = crate::World::new();
        let reg = ReplicationRegistry::new();
        let e = make_entity(5, 0);
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(e, vec![1, 2, 3])], // not a valid archetype-key blob
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        assert_eq!(delta.apply(&mut world, &reg), Ok(()));
        assert!(world.is_alive(e));
    }

    #[test]
    fn apply_respawn_evicts_old_occupant_at_same_slot() {
        let mut world = crate::World::new();
        let reg = ReplicationRegistry::new();

        let old = world.spawn(); // claims index 0, generation 0
        assert!(world.is_alive(old));

        // A wire entity at the same index but generation 5 — a later
        // delta's spawn is authoritative over whatever locally occupies
        // that slot.
        let new_wire = make_entity(old.index(), 5);
        let delta = Delta {
            frame: 1, base_frame: 0,
            spawned: vec![(new_wire, Vec::new())],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };
        assert_eq!(delta.apply(&mut world, &reg), Ok(()));
        assert!(!world.is_alive(old), "old occupant evicted");
        assert!(world.is_alive(new_wire), "new wire entity now owns the slot");
    }

    #[test]
    fn drain_with_world_encodes_real_archetype_key() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn_tracked(&mut tracker);
        let cid = crate::component::component_id::<f32>();
        world.insert(e, 1.0f32); // not tracked — spawn's archetype is looked up at drain time

        let delta = tracker.drain_with_world(&world);
        assert_eq!(delta.spawned.len(), 1);
        let (spawned_entity, blob) = &delta.spawned[0];
        assert_eq!(*spawned_entity, e);
        let cids = decode_archetype_key(blob).expect("valid archetype-key blob");
        assert_eq!(cids, vec![cid]);
    }

    #[test]
    fn full_round_trip_drain_wire_apply() {
        // Server-side world: spawn + insert, drain with real archetype info.
        let mut server = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = server.spawn_tracked(&mut tracker);

        let mut reg = ReplicationRegistry::new();
        let cid = crate::component::component_id::<f32>();
        let builder = reg.register::<f32>();
        reg.insert(builder.whole_field("val", ReplicationEncoding::Pod, ReplicationCondition::Always));

        server.insert(e, 7.0f32);
        let mut server_delta = tracker.drain_with_world(&server);
        // insert_inner recorded the change with empty field data (the value
        // had already moved into the column) — patch it in like a real
        // caller would after reading the value back out.
        server_delta.component_deltas.push(ComponentDelta {
            entity: e,
            component_type: cid,
            field_data: vec![server.get::<f32>(e).unwrap().to_le_bytes().to_vec()],
        });

        // Wire round trip.
        let bytes = server_delta.to_bytes();
        let wire_delta = Delta::from_bytes(&bytes).unwrap();

        // Client-side world: starts empty, apply reconstructs the entity.
        let mut client = crate::World::new();
        assert_eq!(wire_delta.apply(&mut client, &reg), Ok(()));
        assert!(client.is_alive(e));
        assert_eq!(*client.get::<f32>(e).unwrap(), 7.0f32);
    }

    // ── Fuzz / property tests ────────────────────────────────────────

    #[test]
    fn delta_round_trip_random() {
        let mut rng = StdRng::seed_from_u64(0xD317A);
        for _ in 0..100 {
            let d = random_delta(&mut rng);
            let bytes = d.to_bytes();
            let d2 = Delta::from_bytes(&bytes).expect("round trip should decode");
            assert_deltas_equal(&d, &d2);
        }
    }

    #[test]
    fn handshake_round_trip_random_schemas() {
        // component_id() can't mint an arbitrary ComponentId out of thin
        // air, so schemas are built directly (registry internals are
        // visible within this module) with random ComponentIds and random
        // field descriptors — the actual variability this test wants to
        // exercise is the field-level encode/decode, not the id allocator.
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        for _ in 0..30 {
            let mut reg = ReplicationRegistry::new();
            let n_schemas = rng.gen_range(1..6);
            let mut expected: Vec<ReplicationSchema> = Vec::new();
            for _ in 0..n_schemas {
                let component_type = ComponentId(rng.gen_range(1..1000));
                let n_fields = rng.gen_range(0..8);
                let fields = (0..n_fields)
                    .map(|field_index| {
                        let encoding = random_encoding(&mut rng);
                        let condition = random_condition(&mut rng);
                        let event_channel = if matches!(encoding, ReplicationEncoding::Event) {
                            Some(if rng.gen_bool(0.5) { EventChannel::ReliableOrdered } else { EventChannel::Unreliable })
                        } else {
                            None
                        };
                        FieldDescriptor { field_index, encoding, condition, event_channel, ops: None }
                    })
                    .collect();
                let schema = ReplicationSchema { component_type, fields, row_ops: None };
                reg.schemas.insert(component_type, schema.clone());
                expected.push(schema);
            }

            let msg = reg.handshake_message();
            let reg2 = ReplicationRegistry::from_handshake(&msg).expect("valid handshake");

            assert_eq!(reg2.schemas.len(), expected.len());
            for schema in &expected {
                let s2 = reg2.schema(schema.component_type).expect("schema present after round trip");
                assert_eq!(s2.fields.len(), schema.fields.len());
                for (f1, f2) in schema.fields.iter().zip(s2.fields.iter()) {
                    assert_eq!(f1.field_index, f2.field_index);
                    assert_eq!(encoding_to_u8(&f1.encoding), encoding_to_u8(&f2.encoding));
                    assert_eq!(f1.condition, f2.condition);
                    assert_eq!(f1.event_channel, f2.event_channel);
                }
            }
        }
    }

    #[test]
    fn encode_decode_round_trip_random_bytes() {
        let mut rng = StdRng::seed_from_u64(0xFEED);
        for _ in 0..200 {
            let len = rng.gen_range(1..=16);
            let original: Vec<u8> = (0..len).map(|_| rng.gen()).collect();

            for encoding in [ReplicationEncoding::Pod, ReplicationEncoding::Serialized, ReplicationEncoding::GpuHandle] {
                let mut buf = Vec::new();
                assert_eq!(encode_field_value(&encoding, &original, &mut buf), ErrorCode::Ok);
                let mut decoded = vec![0u8; original.len()];
                assert_eq!(decode_field_value(&encoding, &buf, &mut decoded), ErrorCode::Ok);
                assert_eq!(decoded, original, "encoding {encoding:?} round trip");
            }

            // DeltaCompressed only preserves the first 8 bytes (see
            // `encode_field_value`'s doc) — compare against that window.
            let mut buf = Vec::new();
            assert_eq!(
                encode_field_value(&ReplicationEncoding::DeltaCompressed, &original, &mut buf),
                ErrorCode::Ok
            );
            let width = original.len().min(8);
            let mut decoded = vec![0u8; width];
            assert_eq!(decode_field_value(&ReplicationEncoding::DeltaCompressed, &buf, &mut decoded), ErrorCode::Ok);
            if original.len() >= 8 {
                assert_eq!(decoded, original[..8]);
            } else {
                assert_eq!(decoded, original);
            }

            // Opaque: `decode` here is a raw memcpy from `data`, so it
            // round-trips any random payload regardless of length —
            // `encode`/`encode_size` aren't exercised on this path since
            // `decode_field_value` only calls `decode`.
            let opaque = ReplicationEncoding::Opaque {
                encode_size: |_| 0,
                encode: |_, _| ErrorCode::Ok,
                decode: |data, dst| unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len());
                    ErrorCode::Ok
                },
            };
            let mut decoded_opaque = vec![0u8; original.len()];
            let code = decode_field_value(&opaque, &original, &mut decoded_opaque);
            assert_eq!(code, ErrorCode::Ok);
            assert_eq!(decoded_opaque, original, "Opaque decode is a raw memcpy from `data`");
        }
    }

    #[test]
    fn relevance_set_filter_matches_manual_condition_evaluation_random() {
        let mut rng = StdRng::seed_from_u64(0xABCD);
        for _ in 0..200 {
            let entity = random_entity(&mut rng);
            let in_set = rng.gen_bool(0.5);
            let mut set = RelevanceSet::new();
            if in_set {
                set.add(entity);
            }

            let owner = match rng.gen_range(0..3) {
                0 => Ownership::Server,
                1 => Ownership::Client(ClientId(rng.gen_range(0..5))),
                _ => Ownership::Shared,
            };
            let mut authority = AuthorityTable::new();
            authority.set_entity_owner(entity, owner);

            let client = ClientId(rng.gen_range(0..5));

            let cid = ComponentId(1);
            let n_fields = rng.gen_range(0..5);
            let conditions: Vec<ReplicationCondition> = (0..n_fields).map(|_| random_condition(&mut rng)).collect();
            let schema_fields = conditions
                .iter()
                .enumerate()
                .map(|(i, &condition)| FieldDescriptor {
                    field_index: i as u32,
                    encoding: ReplicationEncoding::Pod,
                    condition,
                    event_channel: None,
                    ops: None,
                })
                .collect();

            let mut reg = ReplicationRegistry::new();
            reg.schemas.insert(cid, ReplicationSchema { component_type: cid, fields: schema_fields, row_ops: None });

            let delta = Delta {
                frame: 0, base_frame: 0,
                spawned: vec![], despawned: vec![],
                component_deltas: vec![ComponentDelta {
                    entity,
                    component_type: cid,
                    field_data: vec![vec![1]; n_fields],
                }],
                events: vec![],
            };

            let view = set.filter(&delta, &authority, &reg, client);
            let expected = in_set && conditions.iter().all(|c| condition_passes(c, &owner, &client));
            assert_eq!(
                !view.component_deltas.is_empty(),
                expected,
                "in_set={in_set} owner={owner:?} client={client:?} conditions={conditions:?}"
            );
        }
    }

    #[test]
    fn resolve_conflict_is_deterministic_random() {
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let at = AuthorityTable::new();
        for _ in 0..50 {
            let a = random_delta(&mut rng);
            let b = random_delta(&mut rng);
            let ca = ClientId(rng.gen_range(0..10));
            let cb = ClientId(rng.gen_range(0..10));

            let r1 = AuthorityTable::resolve_conflict(&at, &a, ca, &b, cb);
            let r2 = AuthorityTable::resolve_conflict(&at, &a, ca, &b, cb);
            assert_deltas_equal(&r1, &r2);
        }
    }

    // ── Adversarial / boundary fuzzing ──────────────────────────────
    //
    // The tests above feed well-formed random *content*. These feed
    // well-formed messages truncated at every possible byte offset — the
    // shape a hostile or simply out-of-sync peer's traffic actually takes.
    // Every decoder here must return a clean `Err`/`None` for a truncated
    // or malformed buffer; a panic (or, before this hardening pass, actual
    // undefined behavior for non-`Pod` fields) is a test failure.

    #[test]
    fn delta_from_bytes_truncated_at_every_offset_never_panics() {
        let mut rng = StdRng::seed_from_u64(0xBEEF01);
        for _ in 0..20 {
            let d = random_delta(&mut rng);
            let bytes = d.to_bytes();
            for cut in 0..bytes.len() {
                // Must not panic; a `Some` result (rare — only when `cut`
                // happens to land on a valid message boundary) must itself
                // be internally consistent, not garbage.
                let _ = Delta::from_bytes(&bytes[..cut]);
            }
            // The untruncated buffer must still decode correctly.
            assert!(Delta::from_bytes(&bytes).is_some());
        }
    }

    #[test]
    fn handshake_truncated_at_every_offset_never_panics() {
        let mut rng = StdRng::seed_from_u64(0xBEEF02);
        for _ in 0..20 {
            let mut reg = ReplicationRegistry::new();
            let n_schemas = rng.gen_range(1..4);
            for _ in 0..n_schemas {
                let component_type = ComponentId(rng.gen_range(1..1000));
                let n_fields = rng.gen_range(0..5);
                let fields = (0..n_fields)
                    .map(|field_index| FieldDescriptor {
                        field_index,
                        encoding: random_encoding(&mut rng),
                        condition: random_condition(&mut rng),
                        event_channel: None,
                        ops: None,
                    })
                    .collect();
                reg.schemas.insert(component_type, ReplicationSchema { component_type, fields, row_ops: None });
            }
            let msg = reg.handshake_message();
            for cut in 0..msg.len() {
                let _ = ReplicationRegistry::from_handshake(&msg[..cut]);
            }
            assert!(ReplicationRegistry::from_handshake(&msg).is_ok());
        }
    }

    #[test]
    fn string_replicate_decode_never_panics_on_random_bytes() {
        let mut rng = StdRng::seed_from_u64(0xBEEF03);
        for _ in 0..500 {
            let bytes = random_bytes(&mut rng, 32);
            // Either a valid UTF-8 string comes back, or a clean error —
            // never a panic, regardless of what garbage bytes arrive.
            let _ = String::replicate_decode(&bytes);
        }
    }

    #[test]
    fn vec_replicate_decode_rejects_truncated_length_prefix() {
        // Declares 100 elements but supplies none — must error, not read
        // out of bounds or loop forever.
        let mut bytes = 100u32.to_le_bytes().to_vec();
        assert_eq!(Vec::<u32>::replicate_decode(&bytes), Err(ErrorCode::InvalidData));

        // Declares one element with a byte-length far larger than what's
        // actually supplied.
        bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1_000_000u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);
        assert_eq!(Vec::<u32>::replicate_decode(&bytes), Err(ErrorCode::InvalidData));

        // Fuzz: random short buffers must never panic, even though they
        // often start with a `count` prefix implying far more data than
        // is actually present.
        let mut rng = StdRng::seed_from_u64(0xBEEF04);
        for _ in 0..500 {
            let bytes = random_bytes(&mut rng, 16);
            let _ = Vec::<u32>::replicate_decode(&bytes);
            let _ = Vec::<String>::replicate_decode(&bytes);
        }
    }

    #[test]
    fn option_replicate_decode_rejects_invalid_tag() {
        assert_eq!(Option::<u32>::replicate_decode(&[]), Err(ErrorCode::InvalidData));
        assert_eq!(Option::<u32>::replicate_decode(&[2]), Err(ErrorCode::InvalidData));
        assert_eq!(Option::<u32>::replicate_decode(&[0]), Ok(None));
        // Tag says `Some` but the payload is truncated for a 4-byte u32.
        assert_eq!(Option::<u32>::replicate_decode(&[1, 0, 0]), Err(ErrorCode::InvalidData));
    }

    #[test]
    fn pod_replicate_decode_rejects_wrong_length_instead_of_ub() {
        assert_eq!(f32::replicate_decode(&[1, 2, 3]), Err(ErrorCode::InvalidData));
        assert_eq!(f32::replicate_decode(&[1, 2, 3, 4, 5]), Err(ErrorCode::InvalidData));
        assert_eq!(u64::replicate_decode(&[0u8; 7]), Err(ErrorCode::InvalidData));
        assert!(f32::replicate_decode(&1.5f32.to_le_bytes()).is_ok());
    }

    #[test]
    fn f32_array_replicate_decode_rejects_wrong_length() {
        assert_eq!(<[f32; 3]>::replicate_decode(&[0u8; 11]), Err(ErrorCode::InvalidData));
        assert_eq!(<[f32; 3]>::replicate_decode(&[0u8; 13]), Err(ErrorCode::InvalidData));
        let encoded = {
            let mut buf = Vec::new();
            [1.0f32, 2.0, 3.0].replicate_encode(&mut buf);
            buf
        };
        assert_eq!(<[f32; 3]>::replicate_decode(&encoded), Ok([1.0, 2.0, 3.0]));
    }
}
