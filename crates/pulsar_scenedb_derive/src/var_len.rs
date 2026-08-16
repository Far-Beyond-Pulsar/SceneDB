//! Codegen for a `#[derive(SceneStore)]` struct with at least one
//! `Vec<T>`-typed `#[gpu]` field — a completely separate path from
//! `crate::gpu`'s (the "classic" scalar-`#[gpu]`-fields-only codegen),
//! because `GpuColumnSet`/`SceneColumnSet` (`pulsar_scenedb`'s own trait
//! definitions) both require `Self: Pod`, and a `Vec<T>` field can
//! structurally never make that true (a heap pointer + length + capacity is
//! not a memcpy-safe byte pattern — this is a real soundness boundary, not
//! a gap to paper over).
//!
//! What this generates instead, for a var-len-bearing struct:
//!
//! - NO `unsafe impl Pod for #name` — the struct genuinely isn't `Pod`.
//! - NO `SceneColumnSet`/`GpuColumnSet` impl — the struct is not usable
//!   through the cell-mirrored (`CellStorage`/`Handle`) path at all, only
//!   through `World`. This matches `#[gpu(heavy)]`'s already-established
//!   scoping ("heavy fields skip the fixed/cell-mirrored registration path
//!   entirely -- they're a World-mirror-only concept") extended to the
//!   whole struct, not just one field.
//! - `impl #name { pub fn register_gpu_columns_growable(...) }` — same
//!   signature and call-site contract as the classic path's version (so
//!   `World::insert`'s auto-registration-on-first-use call site needs no
//!   changes), but registers EVERY `#[gpu]` field itself (scalar fields via
//!   the same `SceneGpuStore::register_dirty_tracked_gpu_buffer` the classic
//!   path already uses; `Vec<T>` fields via
//!   `SceneGpuStore::register_var_len_gpu_pool` + a `VarLenHandle`-typed
//!   handle-table registration through that same, already-proven call).
//! - The World-mirror dispatch fn (`inventory::submit!`'d
//!   [`pulsar_scenedb::gpu::GpuMirrorRegistration`], exactly like the
//!   classic path) — writes each field directly: a scalar field's bytes go
//!   straight to `SceneGpuStore::mark_gpu_row_dirty` (the same call
//!   `write_gpu_columns_at_row` makes for a `DirtyTracked`/`Once` column,
//!   inlined here per-field since there's no `GpuColumnSet::gpu_columns()`
//!   list to walk generically without `Self: Pod`); a `Vec<T>` field goes
//!   through `pulsar_scenedb::gpu::write_var_len_field_at_row`.
//!
//! No `#[gpu(layout = packed)]` support (rejected with a clear
//! `compile_error!` one level up, in `scene_store.rs`) — packed layout is a
//! fixed-size-interleaved-record concept with no meaning for a field whose
//! length varies per row.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::scene_store::{FieldInfo, MirrorModeAttr};

fn mirror_mode_tokens(mode: MirrorModeAttr) -> TokenStream {
    match mode {
        MirrorModeAttr::DirtyTracked => quote! { ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked },
        MirrorModeAttr::Once => quote! { ::pulsar_scenedb::gpu::MirrorMode::Once },
    }
}

pub fn generate_var_len_bearing_type(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    field_infos: &[FieldInfo],
) -> syn::Result<TokenStream> {
    let gpu_fields: Vec<&FieldInfo> = field_infos.iter().filter(|f| f.is_gpu).collect();

    // Every #[gpu] field's own wrapper type, same uniqueness reasoning as
    // the classic path (`FieldInfo::gpu_wrapper`'s doc) -- but a var-len
    // field's wrapper wraps `VarLenHandle` (the handle TABLE's element),
    // never the field's own `Vec<T>` type, which can never be `Pod`.
    let gpu_wrapper_defs: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            if f.is_var_len {
                quote! {
                    #[doc(hidden)]
                    #[allow(non_camel_case_types)]
                    #[repr(transparent)]
                    #[derive(Clone, Copy)]
                    pub struct #wrapper(pub ::pulsar_scenedb::gpu::VarLenHandle);
                    unsafe impl ::pulsar_scenedb::page::Pod for #wrapper {}
                }
            } else {
                let ty = &f.ty;
                quote! {
                    #[doc(hidden)]
                    #[allow(non_camel_case_types)]
                    #[repr(transparent)]
                    #[derive(Clone, Copy)]
                    pub struct #wrapper(pub #ty);
                    unsafe impl ::pulsar_scenedb::page::Pod for #wrapper {}
                }
            }
        })
        .collect();

    // `register_gpu_columns_growable`'s body -- one call per #[gpu] field.
    let register_growable_calls: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
            if f.is_var_len {
                let elem_ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                // Handle table gets its own key, distinct from the pool's --
                // two different buffers (one Vec<VarLenHandle>-shaped, one
                // Vec<ElemTy>-shaped), so they can never share a BufferKey
                // even when the field itself declares one (that declared
                // key names the POOL -- the payload data other fields might
                // also want to share -- not the per-field handle table,
                // which is never a sharing target).
                let handle_key = format!("{key}::handles");
                quote! {
                    store.register_var_len_gpu_pool::<#elem_ty>(
                        ::pulsar_scenedb::gpu::BufferKey::of(#key),
                        initial_capacity,
                        device,
                    );
                    store.register_dirty_tracked_gpu_buffer::<#wrapper, ::pulsar_scenedb::gpu::VarLenHandle>(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#handle_key),
                        ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked,
                    );
                }
            } else {
                let ty = &f.ty;
                let mirror_mode = mirror_mode_tokens(f.mirror_mode);
                quote! {
                    store.register_dirty_tracked_gpu_buffer::<#wrapper, #ty>(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#key),
                        #mirror_mode,
                    );
                }
            }
        })
        .collect();

    // World-mirror dispatch fn body -- one arm per #[gpu] field, mirroring
    // `write_gpu_columns_at_row`'s per-column logic (inlined per-field,
    // since there is no `GpuColumnSet::gpu_columns()` list to walk
    // generically without `Self: Pod`).
    let write_arms: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            if f.is_var_len {
                let elem_ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                let field_name = f.ident.to_string();
                let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
                quote! {
                    {
                        let handle_id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                        ::pulsar_scenedb::gpu::write_var_len_field_at_row::<#elem_ty>(
                            store,
                            queue,
                            ::pulsar_scenedb::gpu::BufferKey::of(#key),
                            handle_id,
                            row,
                            &data.#field_ident,
                        );
                    }
                }
            } else {
                let ty = &f.ty;
                let body = quote! {
                    let field_ptr = unsafe {
                        (data as *const #name #ty_generics as *const u8)
                            .add(::std::mem::offset_of!(#name #ty_generics, #field_ident))
                    };
                    // SAFETY: `field_ptr` is in-bounds of `data` (computed
                    // via `offset_of!` on `data`'s own type) and correctly
                    // aligned for `#ty`; `#ty: Pod` (enforced by this
                    // field's own wrapper's `unsafe impl Pod` above sharing
                    // its size) guarantees every byte in range is a valid
                    // read.
                    let bytes: &[u8] = unsafe {
                        ::std::slice::from_raw_parts(field_ptr, ::std::mem::size_of::<#ty>())
                    };
                    let id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                    store.mark_gpu_row_dirty(id, row, bytes);
                };
                match f.mirror_mode {
                    MirrorModeAttr::Once => quote! {
                        if is_new_insert {
                            #body
                        }
                    },
                    MirrorModeAttr::DirtyTracked => body,
                }
            }
        })
        .collect();

    let mirror_dispatch_fn_name = quote::format_ident!("__scenedb_gpu_mirror_dispatch_{}", name);

    // Auto-registration-on-first-use gate: reuses `SceneGpuStore::
    // buffer_key_for` (the same primitive `SceneGpuStore::is_registered`
    // itself is built on) against the FIRST `#[gpu]` field's own wrapper
    // type as a proxy for "is this type registered at all" -- correct for
    // the identical reason `is_registered`'s own doc gives: registration
    // always covers every field together, in one
    // `register_gpu_columns_growable` call, never partially. Can't call
    // `is_registered::<Self>()` directly here (it requires `Self:
    // GpuColumnSet`, which a var-len-bearing struct never implements — see
    // this module's top doc) — `buffer_key_for` needs only a `ComponentId`,
    // which the first field's wrapper already gives us without that bound.
    let first_wrapper = gpu_fields
        .first()
        .expect("var-len-bearing type must have at least one #[gpu] field (it's why this codegen path was chosen)")
        .gpu_wrapper
        .as_ref()
        .expect("gpu field has a wrapper ident");

    // Element-type bounds for `register_gpu_columns_growable`'s generated
    // signature -- scalar fields need `Ty: Pod + HasTypeToken`; var-len
    // fields need their ELEMENT type to satisfy the same (the field's own
    // `Vec<T>` type itself is never bounded this way -- it's never used as
    // a column element, only `T` is).
    let register_field_ty_bounds: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let ty = if f.is_var_len {
                f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty")
            } else {
                &f.ty
            };
            quote! { #ty: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken }
        })
        .collect();

    Ok(quote! {
        #[cfg(feature = "gpu")]
        const _: () = {
            #(#gpu_wrapper_defs)*

            impl #impl_generics #name #ty_generics #where_clause {
                /// See the classic path's identically-named method for the
                /// full doc -- same contract, same call-site shape
                /// (`World::insert`'s auto-registration-on-first-use calls
                /// this exact signature regardless of which codegen path
                /// produced it). This struct has no `register_gpu_columns`
                /// (fixed-capacity) counterpart -- a `Vec<T>` field only
                /// ever makes sense World-mirrored/growable, never
                /// cell-mirrored/fixed, so there is nothing for a fixed
                /// variant to do that would differ from this one.
                pub fn register_gpu_columns_growable(
                    store: &::pulsar_scenedb::gpu::SceneGpuStore,
                    initial_capacity: u32,
                    device: &::std::sync::Arc<::wgpu::Device>,
                ) where
                    #(#register_field_ty_bounds),*
                {
                    #(#register_growable_calls)*
                }
            }

            #[doc(hidden)]
            #[allow(non_snake_case, unused_variables)]
            fn #mirror_dispatch_fn_name(
                mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
                row: u32,
                data: *const (),
                is_new_insert: bool,
            ) {
                // SAFETY: the sole caller, `World::insert_inner`, only
                // reaches this function by looking it up under `#name`'s
                // own `ComponentId`, and passes `&value as *const T as
                // *const ()` for that exact `T` -- so `data` is guaranteed
                // to point at a live, correctly-aligned `#name`.
                let data = unsafe { &*(data as *const #name #ty_generics) };
                let store = mirror.store();
                let queue = mirror.queue();
                // Auto-registration on first use (issue #41's "removes the
                // rest" of the manual setup burden, same as the classic
                // path) -- see this fn's construction site above for why
                // `buffer_key_for` against the first field's wrapper is the
                // right proxy check here.
                if store
                    .buffer_key_for(::pulsar_scenedb::component::component_id::<#first_wrapper>())
                    .is_none()
                {
                    <#name #ty_generics>::register_gpu_columns_growable(
                        store,
                        ::pulsar_scenedb::gpu::world_mirror::DEFAULT_AUTO_REGISTER_CAPACITY,
                        &store.device_arc(),
                    );
                }
                #(#write_arms)*
            }

            ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
                ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                    component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                    dispatch: #mirror_dispatch_fn_name,
                }
            }
        };
    })
}
