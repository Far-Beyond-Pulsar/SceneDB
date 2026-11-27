//! How much of SceneDB's video memory holds live data.
//!
//! A device's allocator report says what every GPU allocation is called and
//! how large it is; only SceneDB knows how much of a pool it has handed out.
//! Every variable-length pool a store creates joins this process-wide list
//! (a weak reference, so a dropped pool leaves it), and [`vram_use`] reports
//! each one's capacity and live bytes under the label its buffer is allocated
//! with. Nothing runs until a tool asks.

use std::sync::{Mutex, OnceLock, Weak};

/// Opens the host's owner scope for SceneDB's GPU allocations: what it
/// returns is held while SceneDB creates a buffer or texture, and dropped
/// after. The engine installs one that names SceneDB as the owner in the
/// device's allocator report (wgpu's allocation owners); without one nothing
/// happens. SceneDB itself depends on no particular wgpu build for this.
pub type OwnerHook = fn() -> Box<dyn std::any::Any>;

static OWNER_HOOK: OnceLock<OwnerHook> = OnceLock::new();

/// Install the owner hook (the first call wins).
pub fn set_owner_hook(hook: OwnerHook) {
    let _ = OWNER_HOOK.set(hook);
}

/// The owner scope every SceneDB allocation is made in: hold the guard
/// across the `create_buffer`/`create_texture` call.
pub(crate) fn owner() -> Option<Box<dyn std::any::Any>> {
    OWNER_HOOK.get().map(|hook| hook())
}

/// One SceneDB GPU allocation: the label its buffer carries (the allocator
/// report's name for it), its size and how many of its bytes are live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VramUse {
    pub label: String,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
}

/// Something SceneDB allocated on the GPU that can say how full it is.
pub(crate) trait VramUser: Send + Sync {
    fn vram_use(&self) -> VramUse;
}

static USERS: Mutex<Vec<Weak<dyn VramUser>>> = Mutex::new(Vec::new());

/// Add `user` to what [`vram_use`] reports while it lives.
pub(crate) fn track(user: Weak<dyn VramUser>) {
    if let Ok(mut users) = USERS.lock() {
        users.retain(|user| user.strong_count() > 0);
        users.push(user);
    }
}

/// Every live SceneDB pool's capacity and live bytes. Takes each pool's read
/// lock briefly; call it at a tool's pace, not per frame.
pub fn vram_use() -> Vec<VramUse> {
    let users: Vec<_> = match USERS.lock() {
        Ok(mut users) => {
            users.retain(|user| user.strong_count() > 0);
            users.clone()
        }
        Err(_) => return Vec::new(),
    };
    users
        .iter()
        .filter_map(Weak::upgrade)
        .map(|user| user.vram_use())
        .collect()
}
