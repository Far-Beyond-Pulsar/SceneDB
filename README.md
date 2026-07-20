<p align="center">
  <img width="300" height="300" alt="Gemini_Generated_Image_r9d18er9d18er9d1" src="https://github.com/user-attachments/assets/06f129f1-a6b0-4885-a6f1-f0d2c7b6a569" />
</p>

# SceneDB

GPU-native ECS and spatial database for game engines.

SceneDB keeps entity data in cache-friendly SoA pages on the CPU side, syncs
only what changed to GPU buffers each frame, and gives you stable handles that
don't dangle when things get compacted. AVX2/NEON spatial queries, a streaming
grid for world cells, and a compile-time frame phase machine that prevents you
from mutating stuff during the readback phase.

Two crates in this workspace:

- **pulsar_scenedb** — the core library. Archetype ECS, paged storage layer,
  SIMD culling, GPU mirroring (feature-gated), streaming grid, asset stores.
- **pulsar_scenedb_derive** — `#[derive(SceneStore)]` that generates Pod impls,
  column set descriptors, and GPU column write dispatch. Saves a lot of
  boilerplate.

Still actively being built out. Milestone M3-β is next. Tests live in
`tests/`, benchmarks in `benches/`, and there's a TUI stress test dashboard
at `examples/stress_tui.rs` if you want to poke at it interactively.

Licensed under MIT OR Apache-2.0.
