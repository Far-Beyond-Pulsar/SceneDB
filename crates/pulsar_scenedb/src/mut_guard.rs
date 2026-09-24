//! Write guards for component access: [`Mut`] (typed, from
//! [`crate::World::get_mut`]) and [`MutDyn`] (type-erased, from
//! [`crate::World::get_dyn_mut`]).
//!
//! Both guards own the same [`MutHooks`]: the GPU-mirror dispatch, the
//! replication change record, subscription and change-journal events, and
//! handle-ledger accounting that a component write must trigger. The hooks
//! are resolved once when the guard is created and fired when it drops (or
//! immediately, from `into_inner`), so erased and typed writes are observed
//! identically.

use std::any::Any;
use std::ops::{Deref, DerefMut};

use crate::component::ComponentId;
use crate::entity::Entity;

/// A mutable borrow of component `T` on some entity, returned by
/// [`crate::World::get_mut`]. `Deref`/`DerefMut` to `T`, so every existing
/// call site (`*guard = value`, `guard.field += 1`, method calls) keeps
/// working unchanged; the only observable difference from a plain `&mut T`
/// is that dropping this guard, not the mutation itself, is when the write
/// reaches the GPU mirror, change tracker, subscriptions, change journals
/// and handle ledger (see [`MutHooks`]).
pub struct Mut<'a, T> {
    value: &'a mut T,
    /// Set by `DerefMut`: whether the caller actually wrote through the
    /// guard. Subscription, journal and handle hooks only fire when set; a
    /// borrow-only `get_mut` is not a mutation.
    mutated_via_deref_mut: bool,
    hooks: MutHooks,
}

/// Type-erased counterpart to [`Mut`], returned by
/// [`crate::World::get_dyn_mut`]: a mutable borrow of whichever component a
/// `ComponentId` names, as `dyn Any` of the component's own type (so
/// `downcast_mut::<T>()` works). Fires exactly the hooks a [`Mut`] for the
/// same component would.
pub struct MutDyn<'a> {
    value: &'a mut dyn Any,
    mutated_via_deref_mut: bool,
    hooks: MutHooks,
}

/// Everything a component write must notify, resolved when a guard is
/// created. Each hook is `None` when its subsystem is absent (no mirror, no
/// tracker, no subscriptions, no journals, no handle fields), so an
/// unobserved write costs a few `Option` checks.
pub(crate) struct MutHooks {
    #[cfg(feature = "gpu")]
    gpu: Option<GpuHook>,
    change: Option<ChangeHook>,
    subscription: Option<SubscriptionHook>,
    journal: Option<JournalHook>,
    /// Captures the OLD handle values at guard creation (afterwards they are
    /// unrecoverable) so a write can be reported as a release/acquire swap.
    handle: Option<HandleHook>,
}

/// The `World` fields hooks are built from, borrowed individually so a guard
/// can be built while the component column itself is mutably borrowed.
pub(crate) struct HookSources<'w> {
    #[cfg(feature = "gpu")]
    pub gpu_mirror: Option<&'w crate::gpu::GpuMirrorHandle>,
    pub change_tracker: Option<&'w crate::replication::SharedChangeTracker>,
    pub subscriptions: Option<&'w crate::subscriptions::SubscriptionRegistryHandle>,
    pub change_journals: &'w std::sync::OnceLock<crate::change_journal::ChangeJournalHandle>,
    pub handle_counts: &'w std::sync::Arc<std::sync::Mutex<crate::handle_ledger::HandleCounts>>,
}

#[cfg(feature = "gpu")]
struct GpuHook {
    mirror: crate::gpu::GpuMirrorHandle,
    row: u32,
    dispatch: crate::gpu::world_mirror::DispatchFn,
}

struct ChangeHook {
    tracker: crate::replication::SharedChangeTracker,
    entity: Entity,
    component_id: ComponentId,
}

struct SubscriptionHook {
    registry: crate::subscriptions::SubscriptionRegistryHandle,
    entity: Entity,
    component_id: ComponentId,
}

struct JournalHook {
    journals: crate::change_journal::ChangeJournalHandle,
    entity: Entity,
    component_id: ComponentId,
}

struct HandleHook {
    counts: std::sync::Arc<std::sync::Mutex<crate::handle_ledger::HandleCounts>>,
    /// Old values, field-declaration order (the collector's contract).
    captured: Vec<crate::handle_ledger::HandleId>,
    collect: crate::handle_ledger::CollectHandlesFn,
}

impl MutHooks {
    /// Resolve every hook for a write to component `component_id` of
    /// `entity`, whose current value is at `value` (a pointer to the
    /// component's own type, used to capture old handle values).
    pub(crate) fn new(
        sources: HookSources<'_>,
        entity: Entity,
        component_id: ComponentId,
        value: *const (),
    ) -> Self {
        let handle = crate::handle_ledger::collect_fn_for(component_id).map(|collect| {
            let mut captured = Vec::new();
            collect(value, &mut captured);
            HandleHook {
                counts: std::sync::Arc::clone(sources.handle_counts),
                captured,
                collect,
            }
        });
        Self {
            #[cfg(feature = "gpu")]
            gpu: sources.gpu_mirror.and_then(|mirror| {
                crate::gpu::world_mirror::dispatch_for(component_id).map(|dispatch| GpuHook {
                    mirror: mirror.clone(),
                    row: entity.index(),
                    dispatch,
                })
            }),
            change: sources.change_tracker.map(|tracker| ChangeHook {
                tracker: tracker.clone(),
                entity,
                component_id,
            }),
            subscription: sources.subscriptions.map(|registry| SubscriptionHook {
                registry: std::sync::Arc::clone(registry),
                entity,
                component_id,
            }),
            journal: sources.change_journals.get().map(|journals| JournalHook {
                journals: std::sync::Arc::clone(journals),
                entity,
                component_id,
            }),
            handle,
        }
    }

    /// Guard drop. The GPU dispatch and change record run on every drop
    /// (shipped behavior: an explicit `get_mut` re-uploads, including `Once`
    /// fields); handle, subscription and journal hooks only on a real write.
    fn fire_on_drop(&self, value: *const (), mutated: bool) {
        if mutated {
            if let Some(hook) = &self.handle {
                hook.fire(value);
            }
        }
        #[cfg(feature = "gpu")]
        if let Some(hook) = &self.gpu {
            (hook.dispatch)(&hook.mirror, hook.row, value, true);
        }
        if let Some(hook) = &self.change {
            hook.tracker
                .record_component_change(hook.entity, hook.component_id, 0, Vec::new());
        }
        if mutated {
            if let Some(hook) = &self.subscription {
                hook.fire();
            }
            if let Some(hook) = &self.journal {
                hook.fire();
            }
        }
    }

    /// `into_inner`: handing the unique reference out ends automatic
    /// observation, so every hook fires now, unconditionally, and is taken
    /// so the forgotten guard owns nothing.
    fn fire_now(&mut self, value: *const ()) {
        #[cfg(feature = "gpu")]
        if let Some(hook) = self.gpu.take() {
            (hook.dispatch)(&hook.mirror, hook.row, value, true);
        }
        if let Some(hook) = self.handle.take() {
            hook.fire(value);
        }
        if let Some(hook) = self.change.take() {
            hook.tracker
                .record_component_change(hook.entity, hook.component_id, 0, Vec::new());
        }
        if let Some(hook) = self.subscription.take() {
            hook.fire();
        }
        if let Some(hook) = self.journal.take() {
            hook.fire();
        }
    }
}

impl HandleHook {
    fn fire(&self, current_value: *const ()) {
        let mut counts = self.counts.lock().expect("World handle_counts: mutex poisoned");
        crate::handle_ledger::report_captured_swap(
            &mut counts,
            self.collect,
            &self.captured,
            current_value,
        );
    }
}

impl SubscriptionHook {
    fn fire(&self) {
        crate::subscriptions::lock(&self.registry).record(
            self.entity,
            self.component_id,
            crate::subscriptions::ComponentChangeKind::Mutated,
        );
    }
}

impl JournalHook {
    fn fire(&self) {
        crate::change_journal::lock(&self.journals).record(
            self.entity,
            self.component_id,
            crate::subscriptions::ComponentChangeKind::Mutated,
        );
    }
}

impl<'a, T> Mut<'a, T> {
    pub(crate) fn new(value: &'a mut T, hooks: MutHooks) -> Self {
        Self { value, mutated_via_deref_mut: false, hooks }
    }

    /// Escape hatch back to a bare `&'a mut T`, for callers that must hand
    /// the reference across an API boundary with no room for the guard.
    /// Fires every hook immediately for the value as it is now; **further
    /// mutation through the returned reference is not observed**.
    pub fn into_inner(mut self) -> &'a mut T {
        self.hooks.fire_now(self.value as *const T as *const ());
        let ptr: *mut T = self.value as *mut T;
        // SAFETY: `ptr` is `self.value`, a `&'a mut T` this guard uniquely
        // owned. `fire_now` took every hook, and `mem::forget` means the
        // guard's `Drop` never runs, so handing the same unique borrow back
        // under its original lifetime aliases nothing.
        std::mem::forget(self);
        unsafe { &mut *ptr }
    }
}

impl<'a, T> Deref for Mut<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T> DerefMut for Mut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.mutated_via_deref_mut = true;
        self.value
    }
}

impl<'a, T> Drop for Mut<'a, T> {
    fn drop(&mut self) {
        self.hooks
            .fire_on_drop(self.value as *const T as *const (), self.mutated_via_deref_mut);
    }
}

impl<'a> MutDyn<'a> {
    pub(crate) fn new(value: &'a mut dyn Any, hooks: MutHooks) -> Self {
        Self { value, mutated_via_deref_mut: false, hooks }
    }

    /// Typed view of the guarded component, if it is a `T`. Returns a
    /// reference into the guard, so the write is still observed on drop.
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.mutated_via_deref_mut = true;
        self.value.downcast_mut::<T>()
    }

    /// Erased counterpart to [`Mut::into_inner`]; the same caveat applies.
    pub fn into_inner(mut self) -> &'a mut dyn Any {
        let value: *mut dyn Any = self.value;
        self.hooks.fire_now(value as *const ());
        // SAFETY: same argument as `Mut::into_inner`.
        std::mem::forget(self);
        unsafe { &mut *value }
    }
}

impl<'a> Deref for MutDyn<'a> {
    type Target = dyn Any;
    #[inline]
    fn deref(&self) -> &dyn Any {
        self.value
    }
}

impl<'a> DerefMut for MutDyn<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut dyn Any {
        self.mutated_via_deref_mut = true;
        self.value
    }
}

impl<'a> Drop for MutDyn<'a> {
    fn drop(&mut self) {
        let value: *const dyn Any = self.value;
        self.hooks.fire_on_drop(value as *const (), self.mutated_via_deref_mut);
    }
}
