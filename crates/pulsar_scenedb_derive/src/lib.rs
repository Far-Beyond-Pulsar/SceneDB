use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod cell;
mod gpu;
mod replicate;
mod scene_store;

/// Derive `HasTypeToken`, `Pod`, `SceneColumnSet`, and `GpuColumnSet` for a
/// SceneDB component struct.
///
/// # Attributes
///
/// - `#[gpu]` — mark a field as GPU-mirrored (requires the `gpu` feature on
///   `pulsar_scenedb`).
/// - `#[gpu(mirror = Once)]` — GPU-mirrored field uploaded once at registration.
/// - `#[gpu(mirror = DirtyTracked)]` — GPU-mirrored field synced every frame
///   (default for bare `#[gpu]`).
#[proc_macro_derive(SceneStore, attributes(gpu))]
pub fn derive_scene_store(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    scene_store::expand(input)
        .unwrap_or_else(|err| err.to_compile_error().into())
        .into()
}

/// Derive a `register_replication` method that builds a
/// [`ReplicationSchema`](pulsar_scenedb::ReplicationSchema) from
/// `#[replicate(encoding = ..., condition = ...)]` field attributes.
///
/// # Example
///
/// ```ignore
/// #[derive(Replicate)]
/// struct Health {
///     #[replicate(encoding = DeltaCompressed, condition = SimulatedOnly)]
///     value: f32,
/// }
///
/// let mut registry = ReplicationRegistry::new();
/// Health::register_replication(&mut registry);
/// ```
#[proc_macro_derive(Replicate, attributes(replicate))]
pub fn derive_replicate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    replicate::expand(input)
        .unwrap_or_else(|err| err.to_compile_error().into())
        .into()
}
