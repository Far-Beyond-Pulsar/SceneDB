//! SceneDB as one shared library.
//!
//! This crate contains `pulsar_scenedb` and every crate it links that no other shared library
//! holds. A binary or plugin library that links it (`use scenedb_dylib as _;`) gets SceneDB from
//! this library, so its process-wide state (component ids, counters, registries) exists once per
//! process. `wgpu` comes from `wgpui_dylib`, reflection from `reflection_dylib` and the profiler
//! from `profiling_dylib`.

// Linked, not just depended on: a crate this library shares with another shared library must
// come from that library, or it would be in both (rustc: "only shows up once").
use profiling_dylib as _;
use reflection_dylib as _;
use wgpui_dylib as _;

pub use pulsar_scenedb;
