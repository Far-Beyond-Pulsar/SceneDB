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

// ── Delta ──────────────────────────────────────────────────────────────────

/// Frame-consistent set of state changes for one connection.
#[derive(Clone, Debug)]
pub struct Delta {
    pub frame: u64,
    pub base_frame: u64,
    pub spawned: Vec<(Entity, Vec<u8>)>,
    pub despawned: Vec<Entity>,
    pub component_deltas: Vec<ComponentDelta>,
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
        _field_index: u32,
        _field_bytes: Vec<u8>,
    ) {
        self.component_changes.push(ComponentDelta {
            entity,
            component_type,
            field_data: Vec::new(),
        });
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
                .map(|e| (e, Vec::new()))
                .collect(),
            despawned: mem::take(&mut self.despawned),
            component_deltas: mem::take(&mut self.component_changes),
        };
        let events = mem::take(&mut self.events);
        (delta, events)
    }

    pub fn end_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }
}

// ── Interest management ────────────────────────────────────────────────────

/// Per-connection filter, built each frame from spatial queries + conditions.
#[derive(Clone, Debug)]
pub struct RelevanceSet {
    /// Placeholder: raw entity allowlist.
    relevant_entities: Vec<Entity>,
}

impl RelevanceSet {
    /// Build from spatial queries. Delegates to `SpatialCell::query_aabb_in`.
    pub fn new() -> Self {
        Self {
            relevant_entities: Vec::new(),
        }
    }

    /// Add an entity to the relevance set.
    pub fn add(&mut self, entity: Entity) {
        self.relevant_entities.push(entity);
    }

    /// Filter a Delta to only the changes relevant to `client`.
    pub fn filter<'a>(
        &self,
        delta: &'a Delta,
        _authority: &AuthorityTable,
        _client: ClientId,
    ) -> DeltaView<'a> {
        DeltaView {
            spawned: &delta.spawned,
            despawned: &delta.despawned,
            component_deltas: delta
                .component_deltas
                .iter()
                .filter(|cd| self.relevant_entities.contains(&cd.entity))
                .collect(),
            events: Vec::new(),
        }
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
