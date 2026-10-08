//! Methods callable on a component of an entity, keyed by [`ComponentId`].
//!
//! Two kinds of method make up a component's method set:
//!
//! - **Reflected methods** of the component type with a `&self` or
//!   `&mut self` receiver, registered with `#[reflect_method]` (see
//!   `pulsar_reflection::methods`). They see only the component value.
//! - **World methods**, registered with `#[world_method]` inside a
//!   [`component_methods`](crate::component_methods) block. They receive
//!   `(&mut World, Entity)` and can touch other components and entities
//!   (e.g. "look at another entity"). They can only be called while the
//!   entity has the component.
//!
//! Both present a [`MethodInfo`], so a scripting frontend lists "methods
//! callable on a reference to component X" without caring which kind a
//! method is. [`World::call_component_method`] calls either kind by name;
//! [`World::invoke_component_method`] calls a method resolved in advance
//! (what a script linker does once, instead of a name lookup per call).
//!
//! Writes go through the same change hooks as [`World::get_mut`]: a
//! `&mut self` reflected method borrows the component through a
//! [`crate::MutDyn`] guard, and a `&self` one through a shared borrow that
//! reports nothing.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

use pulsar_reflection::methods::{
    CallError, MethodInfo, Receiver, ReceiverKind, ReflectedMethod, TypeRef, METHOD_REGISTRY,
};

use crate::component::{type_name, type_of, ComponentId};
use crate::entity::Entity;
use crate::world::World;

/// Shim for a world method: validates `args`, then calls the method with
/// the world and the entity. Same argument conventions as
/// `pulsar_reflection::methods::InvokeFn`.
pub type WorldInvokeFn =
    fn(&mut World, Entity, &mut [Box<dyn Any>]) -> Result<Option<Box<dyn Any>>, CallError>;

/// A component method that receives `(&mut World, Entity)`.
pub struct WorldMethod {
    pub info: MethodInfo,
    pub invoke: WorldInvokeFn,
}

impl fmt::Debug for WorldMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldMethod")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// One `#[component_methods]` block's world methods.
pub struct WorldMethodRegistration {
    pub component: TypeRef,
    pub methods: &'static [WorldMethod],
}

pulsar_reflection::inventory::collect!(WorldMethodRegistration);

/// A method callable on a component reference.
#[derive(Clone, Copy, Debug)]
pub enum ComponentMethod {
    Reflected(&'static ReflectedMethod),
    World(&'static WorldMethod),
}

impl ComponentMethod {
    pub fn info(&self) -> &'static MethodInfo {
        match self {
            Self::Reflected(method) => &method.info,
            Self::World(method) => &method.info,
        }
    }

    pub fn name(&self) -> &'static str {
        self.info().name
    }

    /// Whether calling this can change the component (or, for a world
    /// method, anything in the world).
    pub fn mutates(&self) -> bool {
        match self {
            Self::Reflected(method) => method.receiver == ReceiverKind::Mut,
            Self::World(method) => !method.info.flags.side_effect_free,
        }
    }
}

/// Why [`World::call_component_method`] failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComponentCallError {
    /// The component type has no method with this name.
    NoSuchMethod {
        component: &'static str,
        method: String,
    },
    /// The entity is dead or does not have the component.
    MissingComponent {
        entity: Entity,
        component: &'static str,
    },
    /// The call was rejected (bad arguments) or the method returned `Err`.
    Call(CallError),
}

impl fmt::Display for ComponentCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchMethod { component, method } => {
                write!(f, "{component} has no method `{method}`")
            }
            Self::MissingComponent { entity, component } => {
                write!(f, "{entity:?} has no {component}")
            }
            Self::Call(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for ComponentCallError {}

impl From<CallError> for ComponentCallError {
    fn from(err: CallError) -> Self {
        Self::Call(err)
    }
}

/// Every `#[component_methods]` block's world methods: this copy's, and
/// every attached copy's (see `crate::runtime`).
static WORLD_METHODS: crate::runtime::AppendList<WorldMethodRegistration> =
    crate::runtime::AppendList::new(|| {
        pulsar_reflection::inventory::iter::<WorldMethodRegistration>
            .into_iter()
            .collect()
    });

/// Component methods by component `TypeId`, built from [`METHOD_REGISTRY`]
/// and [`WORLD_METHODS`], and built again when an attached copy adds to
/// either (the old table is leaked).
static BY_TYPE: AtomicPtr<HashMap<TypeId, Vec<ComponentMethod>>> =
    AtomicPtr::new(std::ptr::null_mut());
static BY_TYPE_WRITE: Mutex<()> = Mutex::new(());

/// Every type's reflected methods with a receiver, then its world methods. A
/// world method that shares a reflected method's name is dropped (names are
/// how scripts bind).
fn build_by_type() -> HashMap<TypeId, Vec<ComponentMethod>> {
    let mut by_type: HashMap<TypeId, Vec<ComponentMethod>> = HashMap::new();
    for ty in METHOD_REGISTRY.types() {
        let methods: Vec<_> = ty
            .methods
            .iter()
            .filter(|m| m.receiver != ReceiverKind::None)
            .map(|m| ComponentMethod::Reflected(m))
            .collect();
        if !methods.is_empty() {
            by_type.insert(ty.ty.type_id(), methods);
        }
    }
    for registration in WORLD_METHODS.get() {
        let methods = by_type.entry(registration.component.type_id()).or_default();
        for method in registration.methods {
            if methods.iter().any(|m| m.name() == method.info.name) {
                tracing::error!(
                    "duplicate component method {}::{}; keeping the first",
                    registration.component.type_name(),
                    method.info.name
                );
                continue;
            }
            methods.push(ComponentMethod::World(method));
        }
    }
    by_type
}

fn by_type_locked() -> &'static HashMap<TypeId, Vec<ComponentMethod>> {
    let table = BY_TYPE.load(Ordering::Acquire);
    if !table.is_null() {
        // SAFETY: tables are leaked, never freed.
        return unsafe { &*table };
    }
    let built = Box::leak(Box::new(build_by_type()));
    BY_TYPE.store(built, Ordering::Release);
    built
}

/// This copy's table (the runtime's function, for the copy that owns it).
pub(crate) fn own_methods_of_type(ty: TypeId) -> &'static [ComponentMethod] {
    let table = BY_TYPE.load(Ordering::Acquire);
    let table = if table.is_null() {
        let _write = BY_TYPE_WRITE.lock().unwrap_or_else(|e| e.into_inner());
        by_type_locked()
    } else {
        // SAFETY: tables are leaked, never freed.
        unsafe { &*table }
    };
    table.get(&ty).map_or(&[], Vec::as_slice)
}

/// Add an attached copy's world methods, and pick up the reflected methods
/// its reflection registrations added.
pub(crate) fn extend(registrations: &[&'static WorldMethodRegistration]) {
    let _write = BY_TYPE_WRITE.lock().unwrap_or_else(|e| e.into_inner());
    WORLD_METHODS.extend(registrations);
    BY_TYPE.store(Box::leak(Box::new(build_by_type())), Ordering::Release);
}

/// Every method callable on a component with `TypeId` `ty`.
pub fn component_methods_of_type(ty: TypeId) -> &'static [ComponentMethod] {
    (crate::runtime::runtime().component_methods)(ty)
}

/// Every method callable on component `cid`.
pub fn component_methods(cid: ComponentId) -> &'static [ComponentMethod] {
    component_methods_of_type(type_of(cid))
}

/// The method `name` on component `cid`.
pub fn find_component_method(cid: ComponentId, name: &str) -> Option<ComponentMethod> {
    component_methods(cid)
        .iter()
        .copied()
        .find(|m| m.name() == name)
}

impl World {
    /// Call method `name` on `entity`'s component `cid`.
    pub fn call_component_method(
        &mut self,
        entity: Entity,
        cid: ComponentId,
        name: &str,
        args: &mut [Box<dyn Any>],
    ) -> Result<Option<Box<dyn Any>>, ComponentCallError> {
        let method =
            find_component_method(cid, name).ok_or_else(|| ComponentCallError::NoSuchMethod {
                component: type_name(cid),
                method: name.to_owned(),
            })?;
        self.invoke_component_method(entity, cid, method, args)
    }

    /// Call a method already resolved for component `cid` (by
    /// [`find_component_method`] or [`component_methods`]).
    pub fn invoke_component_method(
        &mut self,
        entity: Entity,
        cid: ComponentId,
        method: ComponentMethod,
        args: &mut [Box<dyn Any>],
    ) -> Result<Option<Box<dyn Any>>, ComponentCallError> {
        let missing = || ComponentCallError::MissingComponent {
            entity,
            component: type_name(cid),
        };
        let result = match method {
            ComponentMethod::Reflected(method) if method.receiver == ReceiverKind::Mut => {
                if !self.has_component(entity, cid) {
                    return Err(missing());
                }
                validate_reflected_args(method, args).map_err(ComponentCallError::Call)?;
                let mut guard = self.get_dyn_mut(entity, cid).ok_or_else(missing)?;
                method.call(Receiver::Mut(&mut *guard), args)
            }
            ComponentMethod::Reflected(method) => {
                let value = self.get_dyn(entity, cid).ok_or_else(missing)?;
                method.call(Receiver::Ref(value), args)
            }
            ComponentMethod::World(method) => {
                if !self.has_component(entity, cid) {
                    return Err(missing());
                }
                (method.invoke)(self, entity, args)
            }
        };
        result.map_err(ComponentCallError::Call)
    }
}

/// Validate the script-facing arguments before `invoke_component_method`
/// obtains a `MutDyn` guard. The reflected shim validates these again when
/// it invokes the method; doing the metadata check here keeps rejected calls
/// from firing mutable-borrow hooks (journals or GPU uploads).
fn validate_reflected_args(
    method: &ReflectedMethod,
    args: &[Box<dyn Any>],
) -> Result<(), CallError> {
    if args.len() != method.info.params.len() {
        return Err(CallError::ArgCount {
            expected: method.info.params.len(),
            found: args.len(),
        });
    }

    for (index, (arg, param)) in args.iter().zip(method.info.params).enumerate() {
        if arg.as_ref().type_id() != param.ty.type_id() {
            return Err(CallError::ArgType {
                index,
                expected: param.ty.type_name(),
            });
        }
    }
    Ok(())
}
