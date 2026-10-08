//! Process-wide state shared by every statically linked copy of this crate
//! (Pulsar-Native#1083).
//!
//! The editor links this crate statically, and so does every plugin library
//! it loads, so each `static` here exists once per binary. A plugin's copy
//! would allocate its own component ids, count its own journals and look up
//! GPU dispatch in tables the editor never reads. Instead, one copy (the
//! editor's) owns that state, and every other copy is [`attach`]ed to the
//! owner's [`Runtime`] when its library loads: from then on it reaches the
//! owner's state through the owner's own functions, which run with the
//! owner's statics and allocator. Attaching also hands the owner the
//! registrations this copy's `inventory` collected, so the owner's dispatch
//! tables cover the plugin's component types.
//!
//! This is the model WGPUI uses for gpui (`shared_runtime`). Every copy is
//! built by the same compiler from the same sources, which [`Runtime::abi`]
//! checks; types still cross between copies only where their layouts agree.
//! `TypeId`s do not agree between copies built by different cargo
//! invocations, so component identity is keyed by type name and layout, and
//! each copy maps its own `TypeId`s onto the shared ids.

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::component::{ComponentId, ErasedComponentInfo};
use crate::handle_ledger::{CollectHandlesFn, HandleLedgerRegistration};

/// Version of [`Runtime`]'s ABI. Bump it on any change to `Runtime`'s fields
/// or to a type passed through them.
pub const ABI_VERSION: u64 = 1;

/// Folded into [`Runtime::abi`] so that a layout change which forgot to bump
/// [`ABI_VERSION`], or a copy built with other features, is still refused.
const FINGERPRINT: u64 = {
    let parts = [
        ABI_VERSION as usize,
        size_of::<Runtime>(),
        size_of::<ComponentId>(),
        size_of::<TypeId>(),
        size_of::<ErasedComponentInfo>(),
        size_of::<crate::World>(),
        align_of::<crate::World>(),
        size_of::<Registrations>(),
        cfg!(feature = "gpu") as usize,
    ];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0;
    while index < parts.len() {
        hash ^= parts[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
};

/// The process-wide state of this crate, as the functions of the copy that
/// owns it. See the module doc.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Runtime {
    /// [`ABI_VERSION`] and the layout fingerprint of the copy that built
    /// this runtime; [`attach`] refuses a runtime whose `abi` differs.
    pub abi: u64,
    pub(crate) component_id: fn(TypeId, &'static str, usize, usize) -> ComponentId,
    pub(crate) resolve_type: fn(TypeId) -> Option<ComponentId>,
    pub(crate) type_of: fn(ComponentId) -> Option<TypeId>,
    pub(crate) type_name: fn(ComponentId) -> Option<&'static str>,
    pub(crate) component_count: fn() -> u32,
    pub(crate) register_erased: fn(TypeId, ErasedComponentInfo),
    pub(crate) erased_info: fn(TypeId) -> Option<ErasedComponentInfo>,
    pub(crate) next_journals_id: fn() -> u64,
    pub(crate) next_subscription_id: fn() -> u64,
    pub(crate) collect_handles: fn(ComponentId) -> Option<CollectHandlesFn>,
    pub(crate) component_methods:
        fn(TypeId) -> &'static [crate::component_methods::ComponentMethod],
    #[cfg(feature = "gpu")]
    pub(crate) gpu: crate::gpu::world_mirror::GpuDispatchRuntime,
    pub(crate) add_registrations: fn(&Registrations),
}

/// The registrations one copy's `inventory` collected, handed to the owner
/// by [`attach`].
pub struct Registrations {
    pub(crate) handle_ledger: Vec<&'static HandleLedgerRegistration>,
    pub(crate) world_methods: Vec<&'static crate::component_methods::WorldMethodRegistration>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu: crate::gpu::world_mirror::GpuRegistrations,
}

impl Registrations {
    fn collected() -> Self {
        Self {
            handle_ledger: pulsar_reflection::inventory::iter::<HandleLedgerRegistration>()
                .collect(),
            world_methods: pulsar_reflection::inventory::iter::<
                crate::component_methods::WorldMethodRegistration,
            >()
            .collect(),
            #[cfg(feature = "gpu")]
            gpu: crate::gpu::world_mirror::GpuRegistrations::collected(),
        }
    }
}

/// This copy's own runtime: the state this copy owns, through its own
/// functions.
static OWN: Runtime = Runtime {
    abi: FINGERPRINT,
    component_id: own::component_id,
    resolve_type: own::resolve_type,
    type_of: own::type_of,
    type_name: own::type_name,
    component_count: own::component_count,
    register_erased: own::register_erased,
    erased_info: own::erased_info,
    next_journals_id: own::next_journals_id,
    next_subscription_id: own::next_subscription_id,
    collect_handles: own::collect_handles,
    component_methods: crate::component_methods::own_methods_of_type,
    #[cfg(feature = "gpu")]
    gpu: crate::gpu::world_mirror::GpuDispatchRuntime::OWN,
    add_registrations: own::add_registrations,
};

/// The owner's runtime once this copy is attached; null while it owns its
/// own state.
static ATTACHED: AtomicPtr<Runtime> = AtomicPtr::new(std::ptr::null_mut());

/// The runtime this copy uses: the one it is attached to, or its own.
#[inline]
pub(crate) fn runtime() -> &'static Runtime {
    let attached = ATTACHED.load(Ordering::Acquire);
    if attached.is_null() {
        &OWN
    } else {
        // SAFETY: `attach` only stores a pointer to a `Runtime` that lives
        // for the process (the owner's static), checked for this ABI.
        unsafe { &*attached }
    }
}

/// The runtime to hand a plugin library's copy of this crate: the one this
/// copy uses.
pub fn shared() -> &'static Runtime {
    runtime()
}

/// Why [`attach`] (this crate's, or another world crate's runtime attach
/// built on the same rules) refused a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachError {
    /// The pointer was null.
    Null,
    /// The runtime was built by a copy of another ABI: another version of
    /// the crate, other features, or another compiler.
    Abi { expected: u64, found: u64 },
    /// This copy already used state of its own (here, allocated component
    /// ids), which the owner's would contradict.
    AlreadyInUse,
    /// This copy is already attached to another runtime.
    AlreadyAttached,
}

impl fmt::Display for AttachError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("no runtime given"),
            Self::Abi { expected, found } => write!(
                f,
                "runtime ABI mismatch (expected {expected:#x}, found {found:#x}): \
                 the library was built from other sources, features or compiler"
            ),
            Self::AlreadyInUse => f.write_str("this copy already used state of its own"),
            Self::AlreadyAttached => f.write_str("this copy is attached to another runtime"),
        }
    }
}

impl std::error::Error for AttachError {}

/// Attach this copy to `owner`, the runtime of the copy that owns the
/// process's SceneDB state (see [`shared`]), and hand it the registrations
/// this copy collected. Call it before this copy does anything else.
/// Attaching a copy to its own runtime, or again to the same one, does
/// nothing.
///
/// # Safety
///
/// `owner` is null or points to a [`Runtime`] that lives for the rest of
/// the process, and whose functions stay loaded that long.
pub unsafe fn attach(owner: *const Runtime) -> Result<(), AttachError> {
    // SAFETY: the caller's contract.
    let owner = unsafe { owner.as_ref() }.ok_or(AttachError::Null)?;
    if owner.abi != OWN.abi {
        return Err(AttachError::Abi {
            expected: OWN.abi,
            found: owner.abi,
        });
    }
    if std::ptr::eq(owner, &OWN) {
        return Ok(());
    }
    if own::component_count() > 0 {
        return Err(AttachError::AlreadyInUse);
    }
    let owner_ptr = owner as *const Runtime as *mut Runtime;
    match ATTACHED.compare_exchange(
        std::ptr::null_mut(),
        owner_ptr,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {}
        Err(current) if current == owner_ptr => return Ok(()),
        Err(_) => return Err(AttachError::AlreadyAttached),
    }
    (owner.add_registrations)(&Registrations::collected());
    Ok(())
}

/// A `ComponentId`-keyed table built from this copy's `inventory` and
/// extended by attached copies' registrations. Readers load one pointer;
/// extending replaces the table and leaks the old one (once per plugin
/// load, so bounded).
pub(crate) struct AppendTable<F: Copy + 'static> {
    table: AtomicPtr<HashMap<ComponentId, Vec<F>>>,
    write: Mutex<()>,
    build: fn() -> HashMap<ComponentId, Vec<F>>,
}

impl<F: Copy + 'static> AppendTable<F> {
    pub(crate) const fn new(build: fn() -> HashMap<ComponentId, Vec<F>>) -> Self {
        Self {
            table: AtomicPtr::new(std::ptr::null_mut()),
            write: Mutex::new(()),
            build,
        }
    }

    fn map(&'static self) -> &'static HashMap<ComponentId, Vec<F>> {
        let table = self.table.load(Ordering::Acquire);
        if !table.is_null() {
            // SAFETY: tables are leaked, never freed.
            return unsafe { &*table };
        }
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        self.map_locked()
    }

    fn map_locked(&'static self) -> &'static HashMap<ComponentId, Vec<F>> {
        let table = self.table.load(Ordering::Acquire);
        if !table.is_null() {
            // SAFETY: tables are leaked, never freed.
            return unsafe { &*table };
        }
        let built = Box::leak(Box::new((self.build)()));
        self.table.store(built, Ordering::Release);
        built
    }

    /// Every entry for `id`, in registration order.
    pub(crate) fn get(&'static self, id: ComponentId) -> Option<&'static [F]> {
        self.map().get(&id).map(Vec::as_slice)
    }

    /// Append `entries` after the existing ones.
    pub(crate) fn extend(&'static self, entries: impl IntoIterator<Item = (ComponentId, F)>) {
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let mut next = self.map_locked().clone();
        for (id, entry) in entries {
            next.entry(id).or_default().push(entry);
        }
        self.table
            .store(Box::leak(Box::new(next)), Ordering::Release);
    }
}

/// A list of `'static` registrations: one copy's `inventory` collection,
/// extended by attached copies' (see the module doc). For the world crates
/// above SceneDB, whose runtimes keep their registries this way. Readers
/// load one pointer; extending replaces the list and leaks the old one
/// (once per plugin load, so bounded).
pub struct AppendList<T: 'static> {
    list: AtomicPtr<Vec<&'static T>>,
    write: Mutex<()>,
    build: fn() -> Vec<&'static T>,
}

impl<T: 'static> AppendList<T> {
    /// A list that starts as `build()` (typically `inventory::iter` of this
    /// copy) on first use.
    pub const fn new(build: fn() -> Vec<&'static T>) -> Self {
        Self {
            list: AtomicPtr::new(std::ptr::null_mut()),
            write: Mutex::new(()),
            build,
        }
    }

    fn get_locked(&'static self) -> &'static [&'static T] {
        let list = self.list.load(Ordering::Acquire);
        if !list.is_null() {
            // SAFETY: lists are leaked, never freed.
            return unsafe { &*list };
        }
        let built = Box::leak(Box::new((self.build)()));
        self.list.store(built, Ordering::Release);
        built
    }

    /// Every registration, in link order, then in the order copies attached.
    pub fn get(&'static self) -> &'static [&'static T] {
        let list = self.list.load(Ordering::Acquire);
        if !list.is_null() {
            // SAFETY: lists are leaked, never freed.
            return unsafe { &*list };
        }
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        self.get_locked()
    }

    /// Append `entries` after the existing ones.
    pub fn extend(&'static self, entries: &[&'static T]) {
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let mut next = self.get_locked().to_vec();
        next.extend_from_slice(entries);
        self.list
            .store(Box::leak(Box::new(next)), Ordering::Release);
    }
}

/// The state this copy owns, and the functions its [`Runtime`] carries.
/// Only reached through [`runtime`].
mod own {
    use super::*;

    struct ComponentEntry {
        name: &'static str,
        size: usize,
        align: usize,
        type_id: TypeId,
    }

    #[derive(Default)]
    struct Components {
        /// Indexed by `ComponentId - 1`.
        entries: Vec<ComponentEntry>,
        /// Every copy's `TypeId` for each component.
        by_type: HashMap<TypeId, ComponentId>,
    }

    static COMPONENTS: Mutex<Option<Components>> = Mutex::new(None);

    fn components<R>(f: impl FnOnce(&mut Components) -> R) -> R {
        let mut components = COMPONENTS.lock().expect("ComponentId registry lock");
        f(components.get_or_insert_with(Default::default))
    }

    pub(super) fn component_id(
        type_id: TypeId,
        name: &'static str,
        size: usize,
        align: usize,
    ) -> ComponentId {
        components(|c| {
            if let Some(&id) = c.by_type.get(&type_id) {
                return id;
            }
            // The same type seen by another copy has another `TypeId`, but
            // the same name and layout.
            let id = match c
                .entries
                .iter()
                .position(|e| e.name == name && e.size == size && e.align == align)
            {
                Some(index) => ComponentId(index as u32 + 1),
                None => {
                    c.entries.push(ComponentEntry {
                        name,
                        size,
                        align,
                        type_id,
                    });
                    ComponentId(c.entries.len() as u32)
                }
            };
            c.by_type.insert(type_id, id);
            id
        })
    }

    pub(super) fn resolve_type(type_id: TypeId) -> Option<ComponentId> {
        components(|c| c.by_type.get(&type_id).copied())
    }

    pub(super) fn type_of(id: ComponentId) -> Option<TypeId> {
        components(|c| {
            c.entries
                .get((id.0 as usize).checked_sub(1)?)
                .map(|e| e.type_id)
        })
    }

    pub(super) fn type_name(id: ComponentId) -> Option<&'static str> {
        components(|c| {
            c.entries
                .get((id.0 as usize).checked_sub(1)?)
                .map(|e| e.name)
        })
    }

    pub(super) fn component_count() -> u32 {
        components(|c| c.entries.len() as u32)
    }

    static ERASED: Mutex<Option<HashMap<TypeId, ErasedComponentInfo>>> = Mutex::new(None);

    pub(super) fn register_erased(type_id: TypeId, info: ErasedComponentInfo) {
        ERASED
            .lock()
            .expect("erased component registry lock")
            .get_or_insert_with(HashMap::new)
            .entry(type_id)
            .or_insert(info);
    }

    pub(super) fn erased_info(type_id: TypeId) -> Option<ErasedComponentInfo> {
        ERASED
            .lock()
            .expect("erased component registry lock")
            .as_ref()?
            .get(&type_id)
            .copied()
    }

    static NEXT_JOURNALS_ID: AtomicU64 = AtomicU64::new(1);
    static NEXT_SUBSCRIPTION: AtomicU64 = AtomicU64::new(1);

    pub(super) fn next_journals_id() -> u64 {
        NEXT_JOURNALS_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub(super) fn next_subscription_id() -> u64 {
        NEXT_SUBSCRIPTION.fetch_add(1, Ordering::Relaxed)
    }

    pub(super) fn collect_handles(id: ComponentId) -> Option<CollectHandlesFn> {
        crate::handle_ledger::own_collect_fn_for(id)
    }

    pub(super) fn add_registrations(registrations: &Registrations) {
        crate::handle_ledger::extend(&registrations.handle_ledger);
        crate::component_methods::extend(&registrations.world_methods);
        #[cfg(feature = "gpu")]
        crate::gpu::world_mirror::extend(&registrations.gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_ids_follow_type_name_and_layout_not_type_id() {
        // Two copies of one type: different `TypeId`s, same name and layout.
        struct CopyA;
        struct CopyB;
        let id = |type_id| (runtime().component_id)(type_id, "plugin::SharedWidget", 8, 4);
        let a = id(TypeId::of::<CopyA>());
        assert_eq!(id(TypeId::of::<CopyB>()), a);
        assert_eq!((runtime().resolve_type)(TypeId::of::<CopyB>()), Some(a));
        // Another layout under the same name is another component.
        let other = (runtime().component_id)(TypeId::of::<u64>(), "plugin::SharedWidget", 16, 8);
        assert_ne!(other, a);
    }

    #[test]
    fn attach_accepts_its_own_runtime_and_refuses_another_abi() {
        // SAFETY: `OWN` is a static.
        assert_eq!(unsafe { attach(&OWN) }, Ok(()));
        let mut other = OWN;
        other.abi ^= 1;
        let other: &'static Runtime = Box::leak(Box::new(other));
        assert_eq!(
            // SAFETY: leaked, so it lives for the process.
            unsafe { attach(other) },
            Err(AttachError::Abi {
                expected: OWN.abi,
                found: other.abi
            })
        );
        // SAFETY: null is allowed.
        assert_eq!(unsafe { attach(std::ptr::null()) }, Err(AttachError::Null));
        assert!(ATTACHED.load(Ordering::Acquire).is_null());
    }
}
