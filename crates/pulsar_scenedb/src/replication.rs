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
//!     .field("mesh",            ReplicationEncoding::GpuHandle,   ReplicationCondition::Always)
//!     .field("local_transform", ReplicationEncoding::Pod,         ReplicationCondition::ServerAuthority)
//!     .field("health",          ReplicationEncoding::DeltaCompressed, ReplicationCondition::SimulatedOnly)
//!     .event("on_hit",          ReplicationCondition::Multicast,  EventChannel::Unreliable);
//! ```
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

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::snapshot::LivenessSnapshot;
use crate::spatial::SpatialCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;

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

// ── Schema ─────────────────────────────────────────────────────────────────

/// Describes one replicated field on a component type.
#[derive(Clone, Debug)]
pub(crate) struct FieldDescriptor {
    field_index: u32,
    encoding: ReplicationEncoding,
    condition: ReplicationCondition,
    /// For Event fields: the delivery channel.
    event_channel: Option<EventChannel>,
}

/// Per-component-type replication schema.
/// Produced by `ReplicationRegistry` and shared at connection handshake.
#[derive(Clone, Debug)]
pub struct ReplicationSchema {
    pub component_type: ComponentId,
    pub fields: Vec<FieldDescriptor>,
}

// ── Schema builder ─────────────────────────────────────────────────────────

/// Fluent builder for declaring a component type's replication layout.
/// Returned by [`ReplicationRegistry::register`].
#[derive(Clone, Debug)]
pub struct SchemaBuilder<T> {
    component_type: ComponentId,
    fields: Vec<FieldDescriptor>,
    _phantom: PhantomData<T>,
}

impl<T: crate::Component> SchemaBuilder<T> {
    /// Declare a state field with its encoding and replication condition.
    /// `field_index` is auto-incremented per call in declaration order.
    pub fn field(mut self, _name: &str, encoding: ReplicationEncoding, condition: ReplicationCondition) -> Self {
        let field_index = self.fields.len() as u32;
        self.fields.push(FieldDescriptor {
            field_index,
            encoding,
            condition,
            event_channel: None,
        });
        self
    }

    /// Declare an event/RPC field. Its `encoding` is implicitly `Event`.
    pub fn event(mut self, _name: &str, condition: ReplicationCondition, channel: EventChannel) -> Self {
        let field_index = self.fields.len() as u32;
        self.fields.push(FieldDescriptor {
            field_index,
            encoding: ReplicationEncoding::Event,
            condition,
            event_channel: Some(channel),
        });
        self
    }

    pub(crate) fn build(self) -> ReplicationSchema {
        ReplicationSchema {
            component_type: self.component_type,
            fields: self.fields,
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
    /// replicated fields. Call `.field(...)` / `.event(...)` and the schema
    /// is inserted on drop.
    pub fn register<T: crate::Component>(&mut self) -> SchemaBuilder<T> {
        SchemaBuilder {
            component_type: crate::component::component_id::<T>(),
            fields: Vec::new(),
            _phantom: PhantomData,
        }
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
                if ofs + 10 > bytes.len() {
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
                });
            }
            schemas.insert(
                component_type,
                ReplicationSchema {
                    component_type,
                    fields,
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
        liveness: &LivenessSnapshot,
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
        authority: &Self,
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

/// A full or partial world state at a specific frame.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub frame: u64,
    pub entities: Vec<EntitySnapshot>,
}

#[derive(Clone, Debug)]
pub struct EntitySnapshot {
    pub entity: Entity,
    pub components: Vec<(ComponentId, Vec<Vec<u8>>)>,
}

/// Client-side prediction reconciler.
#[derive(Clone, Debug)]
pub struct Reconciler {
    snapshots: Vec<Snapshot>,
    pending_inputs: Vec<ClientInput>,
}

impl Reconciler {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            pending_inputs: Vec::new(),
        }
    }

    pub fn push_snapshot(&mut self, snapshot: Snapshot) {
        if self.snapshots.len() >= 64 {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    pub fn push_input(&mut self, input: ClientInput) {
        self.pending_inputs.push(input);
    }

    pub fn pending_inputs(&self) -> &[ClientInput] {
        &self.pending_inputs
    }

    pub fn clear_acknowledged(&mut self, frame: u64) {
        self.pending_inputs.retain(|i| i.frame > frame);
        self.snapshots.retain(|s| s.frame > frame);
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

    fn make_entity(index: u32, gen: u32) -> Entity {
        Entity::new(index, gen)
    }

    #[test]
    fn tracker_records_spawns() {
        let mut t = ChangeTracker::new();
        let e1 = make_entity(0, 1);
        let e2 = make_entity(1, 1);
        t.record_spawn(e1);
        t.record_spawn(e2);
        let (delta, _events) = t.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _events) = t.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = t.drain(&ReplicationSchema { component_type: cid, fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = t.drain(&ReplicationSchema { component_type: cid, fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = t.drain(&ReplicationSchema { component_type: ComponentId(1), fields: vec![] }, ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.events.len(), 1);
        assert_eq!(delta.events[0].entity, e);
        assert_eq!(delta.events[0].payload, vec![99u8]);
    }

    #[test]
    fn drain_clears_tracker() {
        let mut t = ChangeTracker::new();
        t.record_spawn(make_entity(0, 1));
        let (delta, _) = t.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.spawned.len(), 1);
        let (delta2, _) = t.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = t.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.frame, 1);
    }

    #[test]
    fn world_spawn_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn_tracked(&mut tracker);
        let (delta, _) = tracker.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
        assert_eq!(delta.spawned.len(), 1);
        assert_eq!(delta.spawned[0].0, e);
    }

    #[test]
    fn world_despawn_tracked_records_change() {
        let mut world = crate::World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn();
        assert!(world.despawn_tracked(e, &mut tracker));
        let (delta, _) = tracker.drain(&ReplicationSchema { component_type: ComponentId(0), fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = tracker.drain(&ReplicationSchema { component_type: cid, fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
        let (delta, _) = tracker.drain(&ReplicationSchema { component_type: crate::component::component_id::<f32>(), fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
            _phantom: PhantomData,
        }
        .field("x", ReplicationEncoding::Pod, ReplicationCondition::Always)
        .field("y", ReplicationEncoding::Pod, ReplicationCondition::SimulatedOnly)
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
        let builder = builder.field("test", ReplicationEncoding::Pod, ReplicationCondition::SimulatedOnly);
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

        let (delta, _) = tracker.drain(&ReplicationSchema { component_type: cid, fields: vec![] }, ClientId(0), &AuthorityTable::new());
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
}
