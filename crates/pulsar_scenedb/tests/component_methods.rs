//! Component methods: reflected (`&self`/`&mut self`) and world-receiving
//! methods registered with `#[component_methods]`, looked up by
//! `ComponentId` and called through `World`/`ComponentRef`.

use pulsar_scenedb::component_methods::{component_methods, find_component_method};
use pulsar_scenedb::pulsar_reflection::methods::CallError;
use pulsar_scenedb::{
    component_id, component_methods, ChangeRead, ComponentCallError, ComponentChangeKind,
    ComponentMethod, ComponentRef, Entity, World,
};

#[derive(Debug, PartialEq)]
struct Health {
    value: f32,
}

#[derive(Debug, PartialEq)]
struct Unscripted;

#[component_methods]
impl Health {
    #[reflect_method(pure)]
    fn current(&self) -> f32 {
        self.value
    }

    #[reflect_method]
    fn damage(&mut self, amount: f32) {
        self.value -= amount;
    }

    /// Associated functions are not callable on a component reference.
    #[reflect_method]
    fn full() -> Health {
        Health { value: 100.0 }
    }

    /// Move `amount` health from this entity to `to`.
    #[world_method(category = "Combat")]
    fn transfer(world: &mut World, entity: Entity, to: Entity, amount: f32) -> Result<(), String> {
        if world.get::<Health>(to).is_none() {
            return Err(format!("{to:?} has no health"));
        }
        world.get_mut::<Health>(entity).unwrap().value -= amount;
        world.get_mut::<Health>(to).unwrap().value += amount;
        Ok(())
    }

    #[world_method(pure)]
    fn total(world: &World, _entity: Entity) -> f32 {
        world.query::<&Health>().map(|(_, h)| h.value).sum()
    }
}

fn spawn(world: &mut World, value: f32) -> Entity {
    let e = world.spawn();
    world.insert(e, Health { value });
    e
}

#[test]
fn lists_both_kinds_by_component_id() {
    let health = component_id::<Health>();
    let mut names: Vec<_> = component_methods(health).iter().map(ComponentMethod::name).collect();
    names.sort_unstable();
    assert_eq!(names, ["current", "damage", "total", "transfer"]);
    assert!(component_methods(component_id::<Unscripted>()).is_empty());

    let transfer = find_component_method(health, "transfer").unwrap();
    assert!(matches!(transfer, ComponentMethod::World(_)));
    assert_eq!(transfer.info().attr("category"), Some("Combat"));
    assert_eq!(transfer.info().doc, "Move `amount` health from this entity to `to`.");
    // The world and entity are context, not script parameters.
    let params: Vec<_> = transfer.info().params.iter().map(|p| p.name).collect();
    assert_eq!(params, ["to", "amount"]);
    assert!(transfer.mutates());

    let current = find_component_method(health, "current").unwrap();
    assert!(matches!(current, ComponentMethod::Reflected(_)));
    assert!(!current.mutates());
    assert!(!find_component_method(health, "total").unwrap().mutates());
}

#[test]
fn calls_reflected_methods_on_the_live_component() {
    let mut world = World::new();
    let e = spawn(&mut world, 10.0);
    let health = component_id::<Health>();

    world.call_component_method(e, health, "damage", &mut [Box::new(4.0f32)]).unwrap();
    assert_eq!(world.get::<Health>(e).unwrap().value, 6.0);

    let out = ComponentRef::of::<Health>(e).call(&mut world, "current", &mut []).unwrap();
    assert_eq!(*out.unwrap().downcast::<f32>().unwrap(), 6.0);
}

#[test]
fn calls_world_methods_with_the_entity() {
    let mut world = World::new();
    let a = spawn(&mut world, 10.0);
    let b = spawn(&mut world, 1.0);
    let health = component_id::<Health>();

    world
        .call_component_method(a, health, "transfer", &mut [Box::new(b), Box::new(3.0f32)])
        .unwrap();
    assert_eq!(world.get::<Health>(a).unwrap().value, 7.0);
    assert_eq!(world.get::<Health>(b).unwrap().value, 4.0);

    let out = world.call_component_method(a, health, "total", &mut []).unwrap();
    assert_eq!(*out.unwrap().downcast::<f32>().unwrap(), 11.0);

    let nobody = world.spawn();
    let err = world
        .call_component_method(a, health, "transfer", &mut [Box::new(nobody), Box::new(1.0f32)])
        .unwrap_err();
    assert!(matches!(err, ComponentCallError::Call(CallError::Failed(_))));
}

#[test]
fn reports_errors() {
    let mut world = World::new();
    let e = spawn(&mut world, 10.0);
    let bare = world.spawn();
    let health = component_id::<Health>();

    let err = world.call_component_method(e, health, "full", &mut []).unwrap_err();
    assert!(matches!(err, ComponentCallError::NoSuchMethod { .. }));

    for method in ["damage", "current", "total"] {
        let err = world.call_component_method(bare, health, method, &mut []).unwrap_err();
        assert!(matches!(err, ComponentCallError::MissingComponent { .. }), "{method}");
    }

    let err = world.call_component_method(e, health, "damage", &mut [Box::new(1u8)]).unwrap_err();
    assert!(matches!(err, ComponentCallError::Call(CallError::ArgType { index: 0, .. })));
    assert_eq!(world.get::<Health>(e).unwrap().value, 10.0);
}

#[test]
fn mutating_methods_are_observed_and_reads_are_not() {
    let mut world = World::new();
    let e = spawn(&mut world, 10.0);
    world.subscribe::<Health>(e).unwrap();
    let mut journal = world.open_change_cursor::<Health>();
    let health = component_id::<Health>();

    world.call_component_method(e, health, "current", &mut []).unwrap();
    assert!(world.take_component_change_events().is_empty());

    world.call_component_method(e, health, "damage", &mut [Box::new(1.0f32)]).unwrap();
    let events = world.take_component_change_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ComponentChangeKind::Mutated);
    let mut changes = Vec::new();
    assert_eq!(world.read_changes(&mut journal, &mut changes), ChangeRead::Complete);
    assert_eq!(changes.len(), 1);
}
