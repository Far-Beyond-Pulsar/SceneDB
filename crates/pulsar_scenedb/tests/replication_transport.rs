//! Real-socket transport integration tests (HARDENING plan Phase C).
//!
//! Every other replication test exchanges bytes in-process — this file is
//! the one place that actually puts `Delta`/handshake bytes through a real
//! `std::net::TcpStream` loopback connection across two threads, subject to
//! TCP's genuine partial-read/partial-write and connection-teardown
//! behavior, which an in-process `Vec<u8>` round trip cannot exercise.
//!
//! Run with: `cargo test -p pulsar_scenedb --test replication_transport`.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::thread;
use std::time::Duration;

use pulsar_scenedb::*;
use pulsar_scenedb_derive::Replicate;

#[derive(Replicate, Default)]
struct Position {
    #[replicate(encoding = Pod, condition = Always)]
    xyz: [f32; 3],
}

// ── Length-prefixed framing over a raw stream ───────────────────────────
//
// TCP is a byte stream, not a message stream — a real transport has to
// impose its own framing. This is the simplest possible one (u32 LE
// length prefix + payload), used here purely to exercise partial reads.

fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

/// Like `Read::read_exact`, but distinguishes a clean EOF *before* any byte
/// of the read is available (`Ok(false)` — the peer hung up between
/// frames, expected) from an EOF *partway through* (`Err` — a truncated
/// frame, never silently accepted as a short read).
fn read_exact_or_eof(stream: &mut TcpStream, buf: &mut [u8]) -> io::Result<bool> {
    let mut read = 0;
    while read < buf.len() {
        match stream.read(&mut buf[read..]) {
            Ok(0) if read == 0 => return Ok(false),
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated frame")),
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn read_frame(stream: &mut TcpStream) -> io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    if !read_exact_or_eof(stream, &mut len_buf)? {
        return Ok(None);
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    // A real transport would cap this against a max-frame-size constant to
    // avoid a hostile length prefix triggering a multi-GB allocation; kept
    // simple here since the hostile-peer test below targets truncation, not
    // an oversized length claim.
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(Some(buf))
}

// ── Happy path: handshake + Delta round trip over a real socket ────────

#[test]
fn handshake_and_delta_round_trip_over_real_tcp_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || -> io::Result<()> {
        let (mut stream, _) = listener.accept()?;

        let mut registry = ReplicationRegistry::new();
        Position::register_replication(&mut registry);
        write_frame(&mut stream, &registry.handshake_message())?;

        // Build one real Delta the way an actual server would: spawn an
        // entity, insert a component, drain with real archetype info.
        let mut world = World::new();
        let mut tracker = ChangeTracker::new();
        let e = world.spawn_tracked(&mut tracker);
        world.insert(e, Position { xyz: [1.0, 2.0, 3.0] });
        let mut delta = tracker.drain_with_world(&world);
        // `insert` (untracked-value capture path) doesn't record field
        // bytes for a freshly-migrated column — see `World::insert`'s doc —
        // so patch the real value in exactly like a real caller would after
        // reading it back, mirroring `full_round_trip_drain_wire_apply` in
        // replication.rs's own test suite.
        let cid = component_id::<Position>();
        let mut xyz_bytes = Vec::new();
        world.get::<Position>(e).unwrap().xyz.replicate_encode(&mut xyz_bytes);
        delta.component_deltas.push(ComponentDelta {
            entity: e,
            component_type: cid,
            field_data: vec![xyz_bytes],
        });

        write_frame(&mut stream, &delta.to_bytes())?;
        Ok(())
    });

    let mut client_stream = TcpStream::connect(addr).expect("connect to loopback listener");

    // The handshake bytes only carry field *layout* (encoding/condition),
    // never Rust types or closures — see `RowOps`'s doc — so a real peer
    // uses them to check protocol compatibility, not to reconstruct a
    // working registry. The client registers `Position` locally too (both
    // peers compile the same component types); that local registration is
    // what actually applies the incoming delta below.
    let handshake_bytes = read_frame(&mut client_stream)
        .expect("read handshake frame")
        .expect("server sent a handshake");
    let handshake_registry = ReplicationRegistry::from_handshake(&handshake_bytes).expect("valid handshake");
    let mut registry = ReplicationRegistry::new();
    Position::register_replication(&mut registry);
    assert_eq!(
        handshake_registry.schema(component_id::<Position>()).unwrap().fields.len(),
        registry.schema(component_id::<Position>()).unwrap().fields.len(),
        "handshake and local schema must agree on field layout",
    );

    let delta_bytes = read_frame(&mut client_stream)
        .expect("read delta frame")
        .expect("server sent a delta");
    let delta = Delta::from_bytes(&delta_bytes).expect("valid delta");

    let mut client_world = World::new();
    assert_eq!(delta.apply(&mut client_world, &registry), Ok(()));

    assert_eq!(delta.spawned.len(), 1);
    let e = delta.spawned[0].0;
    assert!(client_world.is_alive(e));
    assert_eq!(client_world.get::<Position>(e).unwrap().xyz, [1.0, 2.0, 3.0]);

    server.join().expect("server thread panicked").expect("server I/O failed");
}

// ── Hostile peer: truncated frames must error cleanly, never hang/panic ─

#[test]
fn truncated_frame_over_real_socket_errors_cleanly_instead_of_hanging() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Claim a 1000-byte frame, then send only 10 bytes and disconnect —
        // a corrupted/hostile or simply crashed peer's traffic shape.
        stream.write_all(&1000u32.to_le_bytes()).unwrap();
        stream.write_all(&[0u8; 10]).unwrap();
        stream.flush().unwrap();
        // Dropping `stream` here closes the connection, delivering EOF to
        // the client mid-frame.
    });

    let mut client_stream = TcpStream::connect(addr).expect("connect to loopback listener");
    let result = read_frame(&mut client_stream);
    assert!(result.is_err(), "a truncated frame must be a clean I/O error, not a hang or a short read accepted as valid");

    server.join().expect("server thread panicked");
}

#[test]
fn garbage_handshake_bytes_over_real_socket_are_rejected_not_panicked() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // A well-formed frame, but its payload is NOT a valid handshake
        // message (random bytes) — the framing layer succeeds; the
        // application-level parser must be the one to reject it.
        write_frame(&mut stream, &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]).unwrap();
    });

    let mut client_stream = TcpStream::connect(addr).expect("connect to loopback listener");
    let bytes = read_frame(&mut client_stream).unwrap().expect("frame delivered");
    let result = ReplicationRegistry::from_handshake(&bytes);
    assert!(result.is_err(), "garbage handshake bytes must be rejected, not panic or produce a bogus registry");

    server.join().expect("server thread panicked");
}

// ── Inverse of the happy path's compatibility check ─────────────────────
//
// `handshake_and_delta_round_trip_over_real_tcp_socket` asserts the
// handshake-derived schema and the locally-registered schema agree on
// field count. An `assert_eq!` that always happens to see equal values
// proves nothing about whether it would actually catch a real mismatch —
// this test builds a genuine version-skewed pair (same `ComponentId`, a
// different number of declared fields) and confirms the disagreement is
// visible, not silently absorbed.

#[test]
fn mismatched_schema_field_count_is_detected() {
    let cid = component_id::<Position>();

    // "Remote" peer: the real, derive-generated `Position` schema, round
    // tripped through the wire handshake format exactly like the happy path.
    let mut remote = ReplicationRegistry::new();
    Position::register_replication(&mut remote);
    let handshake_bytes = remote.handshake_message();
    let remote_view = ReplicationRegistry::from_handshake(&handshake_bytes).expect("valid handshake");
    let remote_field_count = remote_view.schema(cid).unwrap().fields.len();

    // "Local" peer: a hand-built schema for the SAME `ComponentId` that
    // declares an extra field — standing in for a locally-compiled
    // `Position` that drifted from what the remote binary was built with
    // (e.g. a field added on one side of a rolling deploy).
    let mut local = ReplicationRegistry::new();
    let builder = local.register::<Position>();
    let builder = builder
        .field(
            "xyz",
            |c: &Position| &c.xyz,
            |c: &mut Position| &mut c.xyz,
            ReplicationEncoding::Pod,
            ReplicationCondition::Always,
        )
        .field(
            "xyz_again",
            |c: &Position| &c.xyz,
            |c: &mut Position| &mut c.xyz,
            ReplicationEncoding::Pod,
            ReplicationCondition::Always,
        );
    local.insert(builder);
    let local_field_count = local.schema(cid).unwrap().fields.len();

    assert_ne!(
        remote_field_count, local_field_count,
        "this test's own setup must produce a genuine schema mismatch, or it isn't proving the \
         compatibility check in the happy-path test can ever fail",
    );
}

// ── Shared setup ──────────────────────────────────────────────────────

fn position_registry() -> ReplicationRegistry {
    let mut registry = ReplicationRegistry::new();
    Position::register_replication(&mut registry);
    registry
}

/// Spawns one entity with `Position { xyz }` on a fresh `World` and returns
/// a `Delta` describing that spawn — real archetype-key blob, real encoded
/// field bytes, exactly what `ChangeTracker::drain_with_world` plus a
/// `Delta::apply`-compatible field patch produces.
fn one_entity_spawn_delta(xyz: [f32; 3]) -> (World, Entity, Delta) {
    let mut world = World::new();
    let mut tracker = ChangeTracker::new();
    let e = world.spawn_tracked(&mut tracker);
    world.insert(e, Position { xyz });
    let mut delta = tracker.drain_with_world(&world);
    let cid = component_id::<Position>();
    let mut bytes = Vec::new();
    world.get::<Position>(e).unwrap().xyz.replicate_encode(&mut bytes);
    delta.component_deltas.push(ComponentDelta { entity: e, component_type: cid, field_data: vec![bytes] });
    (world, e, delta)
}

// ── UDP: real games replicate over UDP, not TCP ─────────────────────────

#[test]
fn delta_round_trips_over_real_udp_datagram() {
    // Unlike TCP, UDP preserves message boundaries per `send_to`/`recv_from`
    // call — no length-prefix framing needed, but also no automatic
    // retransmission or ordering, which is exactly why real game netcode
    // that picks UDP has to build those guarantees itself (again: SceneDB
    // does not own transport).
    let server_socket = UdpSocket::bind("127.0.0.1:0").expect("bind server UDP socket");
    let server_addr = server_socket.local_addr().unwrap();
    let client_socket = UdpSocket::bind("127.0.0.1:0").expect("bind client UDP socket");

    let (_world, e, delta) = one_entity_spawn_delta([4.0, 5.0, 6.0]);
    let payload = delta.to_bytes();
    assert!(payload.len() < 1400, "keep this comfortably under a typical UDP MTU for the test's own sake");

    client_socket.send_to(&payload, server_addr).expect("send datagram");

    let mut buf = [0u8; 2048];
    let (n, _from) = server_socket.recv_from(&mut buf).expect("receive datagram");
    let received = Delta::from_bytes(&buf[..n]).expect("valid delta");

    let registry = position_registry();
    let mut receiver_world = World::new();
    assert_eq!(received.apply(&mut receiver_world, &registry), Ok(()));
    assert!(receiver_world.is_alive(e));
    assert_eq!(receiver_world.get::<Position>(e).unwrap().xyz, [4.0, 5.0, 6.0]);
}

// ── Multiple concurrent clients ──────────────────────────────────────────

#[test]
fn server_handles_multiple_concurrent_clients_over_tcp() {
    const CLIENT_COUNT: usize = 8;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || {
        for i in 0..CLIENT_COUNT {
            let (mut stream, _) = listener.accept().expect("accept a client");
            // Each client gets its OWN entity/value — proves the server
            // isn't accidentally sharing state across connections handled
            // concurrently.
            let (_world, _e, delta) = one_entity_spawn_delta([i as f32, 0.0, 0.0]);
            write_frame(&mut stream, &delta.to_bytes()).expect("write delta to client");
        }
    });

    let registry = position_registry();
    let clients: Vec<_> = (0..CLIENT_COUNT)
        .map(|i| {
            let registry = position_registry();
            thread::spawn(move || {
                // Small stagger so connections don't all race the single
                // `accept()` loop identically every run — real clients
                // don't connect in perfect lockstep either.
                thread::sleep(Duration::from_millis(i as u64 % 3));
                let mut stream = TcpStream::connect(addr).expect("connect to loopback listener");
                let bytes = read_frame(&mut stream).unwrap().expect("delta frame delivered");
                let delta = Delta::from_bytes(&bytes).expect("valid delta");
                let mut world = World::new();
                assert_eq!(delta.apply(&mut world, &registry), Ok(()));
                let e = delta.spawned[0].0;
                assert!(world.is_alive(e));
                world.get::<Position>(e).unwrap().xyz[0]
            })
        })
        .collect();

    let mut received_values: Vec<f32> = clients.into_iter().map(|c| c.join().expect("client thread panicked")).collect();
    received_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f32> = (0..CLIENT_COUNT).map(|i| i as f32).collect();
    assert_eq!(received_values, expected, "every client independently received and applied its own distinct entity");

    server.join().expect("server thread panicked");
    let _ = registry; // kept alive for clarity of intent; each client built its own copy above
}

// ── Reconnect + resync via Snapshot, over a real socket ─────────────────

#[test]
fn client_reconnect_and_resync_via_snapshot_over_real_socket() {
    // Models the scenario `Snapshot::restore_to_world`'s doc describes:
    // a client that drops its connection and misses however many Deltas
    // the server produced in the meantime cannot recover that missing
    // state from later Deltas alone (each one only carries its own frame's
    // changes) — it has to request a fresh Snapshot on reconnect instead.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || -> io::Result<()> {
        // First connection: client gets one entity, then "disconnects"
        // (this thread's handling of that connection just ends).
        let (mut first, _) = listener.accept()?;
        let (mut server_world, e1, delta) = one_entity_spawn_delta([1.0, 0.0, 0.0]);
        write_frame(&mut first, &delta.to_bytes())?;
        drop(first);

        // While the client is "away", the server keeps advancing state the
        // client never sees a Delta for — exactly the gap a resync has to
        // recover from.
        let mut tracker = ChangeTracker::new();
        let e2 = server_world.spawn_tracked(&mut tracker);
        server_world.insert(e2, Position { xyz: [2.0, 0.0, 0.0] });
        let e3 = server_world.spawn_tracked(&mut tracker);
        server_world.insert(e3, Position { xyz: [3.0, 0.0, 0.0] });
        let _dropped_delta = tracker.drain_with_world(&server_world); // never sent — simulates loss

        // Second connection: the same client reconnecting. Instead of
        // resuming Deltas (which would skip straight from frame 0 to
        // whatever's next, silently missing e2/e3), the server sends a
        // full resync Snapshot.
        let (mut second, _) = listener.accept()?;
        let registry = position_registry();
        let snapshot = Snapshot::capture_full(&server_world, &registry, 99);
        write_frame(&mut second, &encode_snapshot(&snapshot))?;

        let _ = (e1, e2, e3); // kept alive for readability of intent above
        Ok(())
    });

    let registry = position_registry();

    // First connection: apply the one Delta the client actually receives.
    let mut client_world = World::new();
    let mut stream = TcpStream::connect(addr).expect("first connect");
    let first_bytes = read_frame(&mut stream).unwrap().expect("first delta frame");
    let first_delta = Delta::from_bytes(&first_bytes).expect("valid delta");
    assert_eq!(first_delta.apply(&mut client_world, &registry), Ok(()));
    drop(stream);

    // Reconnect and resync from a full snapshot instead of continuing to
    // wait for Deltas the connection gap already made unrecoverable.
    let mut stream = TcpStream::connect(addr).expect("reconnect");
    let snapshot_bytes = read_frame(&mut stream).unwrap().expect("snapshot frame");
    let snapshot = decode_snapshot(&snapshot_bytes);
    assert_eq!(snapshot.restore_to_world(&mut client_world, &registry), Ok(()));

    // All three entities (including the two the client never got a Delta
    // for) are present and correct after the snapshot resync.
    assert_eq!(client_world.query::<()>().count(), 3, "resync recovers every entity, not just the ones seen via Delta");

    server.join().expect("server thread panicked").expect("server I/O failed");
}

/// Minimal ad-hoc wire encoding for a `Snapshot` — `Snapshot` itself has no
/// `to_bytes`/`from_bytes` (unlike `Delta`), so this test's server/client
/// halves agree on a small bincode-free format just to move one across the
/// socket: entity count, then per entity its bits, component count, then
/// per component its id, field count, then per field a length-prefixed blob.
fn encode_snapshot(snapshot: &Snapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(snapshot.entities.len() as u32).to_le_bytes());
    for es in &snapshot.entities {
        buf.extend_from_slice(&es.entity.bits().to_le_bytes());
        buf.extend_from_slice(&(es.components.len() as u32).to_le_bytes());
        for (cid, field_data) in &es.components {
            buf.extend_from_slice(&cid.0.to_le_bytes());
            buf.extend_from_slice(&(field_data.len() as u32).to_le_bytes());
            for field in field_data {
                buf.extend_from_slice(&(field.len() as u32).to_le_bytes());
                buf.extend_from_slice(field);
            }
        }
    }
    buf
}

fn decode_snapshot(bytes: &[u8]) -> Snapshot {
    let mut ofs = 0usize;
    let mut read_u32 = || {
        let v = u32::from_le_bytes(bytes[ofs..ofs + 4].try_into().unwrap());
        ofs += 4;
        v
    };
    let entity_count = read_u32();
    let mut entities = Vec::new();
    for _ in 0..entity_count {
        let bits = u64::from_le_bytes(bytes[ofs..ofs + 8].try_into().unwrap());
        ofs += 8;
        let entity = Entity::from_bits(bits);
        let component_count = read_u32();
        let mut components = Vec::new();
        for _ in 0..component_count {
            let cid = ComponentId(read_u32());
            let field_count = read_u32();
            let mut field_data = Vec::new();
            for _ in 0..field_count {
                let field_len = read_u32() as usize;
                field_data.push(bytes[ofs..ofs + field_len].to_vec());
                ofs += field_len;
            }
            components.push((cid, field_data));
        }
        entities.push(EntitySnapshot { entity, components });
    }
    Snapshot { frame: 0, entities, cell_rows: vec![] }
}
