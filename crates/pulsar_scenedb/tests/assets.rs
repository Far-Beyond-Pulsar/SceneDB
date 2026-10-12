//! Asset references owned by the world (SceneDB#84): a component names an
//! asset by [`AssetId`], and [`World::pump_assets`] loads it off the write
//! path and binds the result into every component that references it.
//!
//! The loader and binding registries are process-wide and these tests run
//! in parallel, so each test declares its own asset kind and sources.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use pulsar_scenedb::{
    register_binding, register_loader, AssetId, AssetPayload, AssetState, AssetView, Entity,
    SceneStore, World,
};

/// A mesh-like component: it names its asset, and its binding fills in
/// data derived from the asset (here a digest of its vertices).
#[derive(Clone, Copy, Debug, SceneStore)]
struct Mesh {
    asset: AssetId,
    tint: u32,
    /// The bound vertices' sum and count.
    sum: u32,
    len: u32,
    /// 0 unresolved, 1 ready, 2 failed.
    status: u32,
    version: u64,
    binds: u64,
}

impl Mesh {
    fn of(asset: AssetId) -> Self {
        Self {
            asset,
            tint: 0,
            sum: 0,
            len: 0,
            status: 0,
            version: 0,
            binds: 0,
        }
    }
}

fn bind_mesh(mesh: &mut Mesh, view: &AssetView) {
    assert_eq!(
        view.id, mesh.asset,
        "bound to an asset the mesh does not name"
    );
    match view.state {
        AssetState::Ready => {
            let vertices = view.payload::<Vec<u32>>().unwrap();
            mesh.sum = vertices.iter().sum();
            mesh.len = vertices.len() as u32;
            mesh.status = 1;
        }
        AssetState::Failed(_) => {
            mesh.sum = 0;
            mesh.len = 0;
            mesh.status = 2;
        }
        other => panic!("bound while {other:?}"),
    }
    mesh.version = view.version;
    mesh.binds += 1;
}

/// What each source currently holds: geometry, or an error message.
type Contents = Mutex<HashMap<String, Result<Vec<u32>, String>>>;

fn contents() -> &'static Contents {
    static CONTENTS: OnceLock<Contents> = OnceLock::new();
    CONTENTS.get_or_init(Default::default)
}

fn set_content(source: &str, content: Result<Vec<u32>, String>) {
    contents()
        .lock()
        .unwrap()
        .insert(source.to_owned(), content);
}

/// Holds loads until opened, and counts them.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    opened: Condvar,
    calls: AtomicUsize,
}

impl Gate {
    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }

    fn close(&self) {
        *self.open.lock().unwrap() = false;
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// Register `kind`'s loader (reading [`contents`], waiting on the returned
/// gate, which starts open) and its binding to [`Mesh`].
fn kind(kind: &'static str) -> Arc<Gate> {
    let gate = Arc::new(Gate::default());
    gate.open();
    let loader_gate = gate.clone();
    register_loader(kind, move |source| -> Result<AssetPayload, String> {
        loader_gate.calls.fetch_add(1, Ordering::SeqCst);
        let mut open = loader_gate.open.lock().unwrap();
        while !*open {
            open = loader_gate.opened.wait(open).unwrap();
        }
        drop(open);
        let content = contents().lock().unwrap().get(source).cloned();
        match content {
            Some(Ok(vertices)) => Ok(Arc::new(vertices)),
            Some(Err(message)) if message == "panic" => panic!("corrupt asset"),
            Some(Err(message)) => Err(message),
            None => Err(format!("{source}: not found")),
        }
    });
    register_binding::<Mesh>(kind, bind_mesh);
    gate
}

const WAIT: Duration = Duration::from_secs(10);

fn resolve(world: &mut World) {
    assert!(
        world.resolve_assets_blocking(WAIT),
        "assets did not resolve"
    );
}

fn mesh(world: &World, entity: Entity) -> Mesh {
    *world.get::<Mesh>(entity).unwrap()
}

/// What a mesh bound to `vertices` holds.
fn geometry(vertices: &[u32]) -> (u32, u32) {
    (vertices.iter().sum(), vertices.len() as u32)
}

impl Mesh {
    fn geometry(&self) -> (u32, u32) {
        (self.sum, self.len)
    }
}

/// The world's incremental reference counts agree with a full rescan.
fn assert_counts_agree(world: &World) {
    for (id, count) in world.handle_audit() {
        assert_eq!(world.handle_ref_count(id), count, "{id:?}");
    }
}

#[test]
fn a_referenced_asset_loads_and_binds_without_the_caller_loading_it() {
    let _gate = kind("t-basic");
    set_content("basic/cube", Ok(vec![1, 2, 3]));
    let id = AssetId::declare("t-basic", "basic/cube");
    assert_eq!(id, AssetId::of("t-basic", "basic/cube"));
    assert_eq!(id.source().unwrap().1.as_ref(), "basic/cube");

    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    assert_eq!(world.asset_state(id), AssetState::Unresolved);
    assert_eq!(world.asset_ref_count(id), 1);

    resolve(&mut world);
    assert_eq!(world.asset_state(id), AssetState::Ready);
    let m = mesh(&world, e);
    assert_eq!(
        (m.geometry(), m.status, m.version, m.binds),
        (geometry(&[1, 2, 3]), 1, 1, 1)
    );
    assert_eq!(*world.asset_payload::<Vec<u32>>(id).unwrap(), vec![1, 2, 3]);
    assert_counts_agree(&world);
}

#[test]
fn missing_and_corrupt_assets_fail_and_their_components_see_it() {
    let _gate = kind("t-fail");
    set_content("fail/corrupt", Err("panic".into()));
    let missing = AssetId::declare("t-fail", "fail/missing");
    let corrupt = AssetId::declare("t-fail", "fail/corrupt");

    let mut world = World::new();
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, Mesh::of(missing));
    world.insert(b, Mesh::of(corrupt));
    resolve(&mut world);

    assert_eq!(
        world.asset_state(missing),
        AssetState::Failed("fail/missing: not found".into())
    );
    assert_eq!(
        world.asset_state(corrupt),
        AssetState::Failed("the loader panicked".into())
    );
    assert_eq!(mesh(&world, a).status, 2);
    assert_eq!(mesh(&world, b).status, 2);
    assert!(world.asset_payload::<Vec<u32>>(missing).is_none());

    // A fixed source and a reload recover it.
    set_content("fail/missing", Ok(vec![7]));
    world.reload_asset(missing);
    resolve(&mut world);
    assert_eq!(world.asset_state(missing), AssetState::Ready);
    let m = mesh(&world, a);
    assert_eq!((m.geometry(), m.status, m.version), (geometry(&[7]), 1, 2));
}

#[test]
fn a_kind_without_a_loader_fails_and_binds_the_failure() {
    register_binding::<Mesh>("t-noloader", bind_mesh);
    let id = AssetId::declare("t-noloader", "noloader/a");
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    let report = world.pump_assets();
    assert_eq!((report.started, report.bound), (0, 1));
    assert!(matches!(world.asset_state(id), AssetState::Failed(_)));
    assert_eq!(mesh(&world, e).status, 2);
}

#[test]
fn an_undeclared_id_is_not_loaded() {
    let gate = kind("t-undeclared");
    let id = AssetId::of("t-undeclared", "undeclared/a");
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    resolve(&mut world);
    assert_eq!(world.asset_state(id), AssetState::Unresolved);
    assert_eq!(gate.calls(), 0);
    assert_eq!(mesh(&world, e).binds, 0);
}

#[test]
fn entities_sharing_an_asset_share_one_load() {
    let gate = kind("t-shared");
    set_content("shared/rock", Ok(vec![4, 5]));
    let id = AssetId::declare("t-shared", "shared/rock");

    let mut world = World::new();
    let entities: Vec<Entity> = (0..8)
        .map(|_| {
            let e = world.spawn();
            world.insert(e, Mesh::of(id));
            e
        })
        .collect();
    resolve(&mut world);
    assert_eq!(gate.calls(), 1);
    assert_eq!(world.asset_ref_count(id), 8);
    for &e in &entities {
        let m = mesh(&world, e);
        assert_eq!((m.geometry(), m.binds), (geometry(&[4, 5]), 1));
    }

    // A late reference binds from the loaded payload.
    let late = world.spawn();
    world.insert(late, Mesh::of(id));
    let report = world.pump_assets();
    assert_eq!(report.started, 0);
    assert_eq!(gate.calls(), 1);
    assert_eq!(mesh(&world, late).geometry(), geometry(&[4, 5]));
    assert_counts_agree(&world);
}

#[test]
fn editing_another_property_neither_reloads_nor_rebinds() {
    let gate = kind("t-edit");
    set_content("edit/a", Ok(vec![1]));
    let id = AssetId::declare("t-edit", "edit/a");
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    resolve(&mut world);

    world.get_mut::<Mesh>(e).unwrap().tint = 9;
    let mut overwritten = mesh(&world, e);
    overwritten.tint = 10;
    world.insert(e, overwritten);
    let report = world.pump_assets();
    assert_eq!((report.started, report.bound), (0, 0));
    assert_eq!(gate.calls(), 1);
    let m = mesh(&world, e);
    assert_eq!((m.tint, m.binds), (10, 1));
}

#[test]
fn replacing_an_asset_while_it_loads_never_binds_the_old_one() {
    let gate = kind("t-replace");
    set_content("replace/a", Ok(vec![1]));
    set_content("replace/b", Ok(vec![2]));
    let a = AssetId::declare("t-replace", "replace/a");
    let b = AssetId::declare("t-replace", "replace/b");

    let mut world = World::new();
    let e = world.spawn();
    gate.close();
    world.insert(e, Mesh::of(a));
    assert_eq!(world.pump_assets().started, 1);
    assert_eq!(world.asset_state(a), AssetState::Loading);

    // The component now names B; A is still loading.
    world.get_mut::<Mesh>(e).unwrap().asset = b;
    assert_eq!(world.asset_ref_count(a), 0);
    gate.open();
    resolve(&mut world);

    // A finished too, but nothing references it, so nothing was bound to it
    // (`bind_mesh` asserts the id it is bound to is the one it names).
    assert_eq!(world.asset_state(a), AssetState::Ready);
    let m = mesh(&world, e);
    assert_eq!((m.asset, m.geometry(), m.binds), (b, geometry(&[2]), 1));
    assert_counts_agree(&world);
}

#[test]
fn removing_a_component_while_its_asset_loads_skips_it_and_re_adding_reuses_the_load() {
    let gate = kind("t-remove");
    set_content("remove/a", Ok(vec![3, 3]));
    let id = AssetId::declare("t-remove", "remove/a");

    let mut world = World::new();
    let e = world.spawn();
    gate.close();
    world.insert(e, Mesh::of(id));
    world.pump_assets();
    world.remove::<Mesh>(e).unwrap();
    gate.open();
    resolve(&mut world);
    assert_eq!(world.asset_state(id), AssetState::Ready);
    assert!(world.get::<Mesh>(e).is_none());

    world.insert(e, Mesh::of(id));
    let report = world.pump_assets();
    assert_eq!((report.started, report.bound), (0, 1));
    assert_eq!(gate.calls(), 1);
    assert_eq!(mesh(&world, e).geometry(), geometry(&[3, 3]));
}

#[test]
fn a_reference_released_before_the_pump_starts_no_load() {
    let gate = kind("t-transient");
    let id = AssetId::declare("t-transient", "transient/a");
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    world.despawn(e);
    assert_eq!(world.pump_assets().started, 0);
    assert_eq!(gate.calls(), 0);
}

#[test]
fn a_reload_supersedes_the_load_in_flight_and_rebinds_every_dependent() {
    let gate = kind("t-reload");
    set_content("reload/a", Ok(vec![1]));
    let id = AssetId::declare("t-reload", "reload/a");

    let mut world = World::new();
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, Mesh::of(id));
    world.insert(b, Mesh::of(id));
    gate.close();
    world.pump_assets();

    // The source changes while the first load is in flight.
    set_content("reload/a", Ok(vec![2, 2]));
    assert_eq!(world.asset_source_changed("reload/a"), 1);
    assert_eq!(world.pump_assets().started, 1);
    gate.open();

    let mut stale = 0;
    let deadline = std::time::Instant::now() + WAIT;
    while world.asset_state(id) != AssetState::Ready || stale == 0 {
        assert!(std::time::Instant::now() < deadline, "loads did not finish");
        stale += world.pump_assets().stale;
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(stale, 1, "the superseded load is dropped");
    assert_eq!(gate.calls(), 2);
    assert_eq!(world.asset_version(id), 1, "only the current load applied");
    for e in [a, b] {
        let m = mesh(&world, e);
        assert_eq!((m.geometry(), m.binds), (geometry(&[2, 2]), 1));
    }

    // A later edit to the source reloads and rebinds both.
    set_content("reload/a", Ok(vec![5]));
    assert_eq!(world.asset_source_changed("reload/a"), 1);
    resolve(&mut world);
    for e in [a, b] {
        let m = mesh(&world, e);
        assert_eq!((m.geometry(), m.version, m.binds), (geometry(&[5]), 2, 2));
    }
    assert_eq!(world.asset_source_changed("reload/unknown"), 0);
}

#[test]
fn eviction_drops_only_unreferenced_assets() {
    let gate = kind("t-evict");
    set_content("evict/kept", Ok(vec![1]));
    set_content("evict/dropped", Ok(vec![2]));
    let kept = AssetId::declare("t-evict", "evict/kept");
    let dropped = AssetId::declare("t-evict", "evict/dropped");

    let mut world = World::new();
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, Mesh::of(kept));
    world.insert(b, Mesh::of(dropped));
    resolve(&mut world);
    world.despawn(b);

    assert_eq!(world.evict_unreferenced_assets(), 1);
    assert_eq!(world.asset_state(kept), AssetState::Ready);
    assert_eq!(world.asset_state(dropped), AssetState::Unresolved);
    assert!(world.asset_payload::<Vec<u32>>(dropped).is_none());

    // Referencing it again loads it again.
    let c = world.spawn();
    world.insert(c, Mesh::of(dropped));
    resolve(&mut world);
    assert_eq!(gate.calls(), 3);
    assert_eq!(mesh(&world, c).geometry(), geometry(&[2]));
    assert_counts_agree(&world);
}

#[test]
fn asset_ids_are_stable_and_never_zero_for_a_real_source() {
    assert_eq!(AssetId::of("mesh", ""), AssetId::ZERO);
    assert!(AssetId::ZERO.is_zero());
    let id = AssetId::of("mesh", "a.glb");
    assert!(!id.is_zero());
    assert_eq!(id, AssetId::of("mesh", "a.glb"));
    assert_ne!(id, AssetId::of("texture", "a.glb"));
    assert_ne!(id, AssetId::of("mesh", "b.glb"));
    // Kind and source are separated, so moving bytes between them changes
    // the id.
    assert_ne!(AssetId::of("ab", "c"), AssetId::of("a", "bc"));
}

#[test]
fn asset_work_pending_tells_a_frame_loop_when_to_pump() {
    let gate = kind("t-pending");
    set_content("pending/a", Ok(vec![1]));
    let id = AssetId::declare("t-pending", "pending/a");
    let mut world = World::new();
    assert!(!world.asset_work_pending());

    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    assert!(world.asset_work_pending(), "a new reference");

    gate.close();
    world.pump_assets();
    assert!(world.asset_work_pending(), "a load in flight");
    gate.open();
    resolve(&mut world);
    assert!(!world.asset_work_pending());

    // A superseded load still counts until its result is received.
    gate.close();
    world.reload_asset(id);
    assert!(world.asset_work_pending(), "a requested reload");
    world.pump_assets();
    world.reload_asset(id);
    world.pump_assets();
    gate.open();
    resolve(&mut world);
    assert!(!world.asset_work_pending());
    assert_eq!(mesh(&world, e).geometry(), geometry(&[1]));
}

#[test]
fn registering_a_binding_again_replaces_it() {
    let _gate = kind("t-rebind");
    // A second copy of the defining crate registers the same binding.
    register_binding::<Mesh>("t-rebind", bind_mesh);
    set_content("rebind/a", Ok(vec![2]));
    let id = AssetId::declare("t-rebind", "rebind/a");
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Mesh::of(id));
    resolve(&mut world);
    assert_eq!(
        mesh(&world, e).binds,
        1,
        "bound once, not once per registration"
    );
}
