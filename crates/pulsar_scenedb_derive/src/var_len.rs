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
use syn::{Ident, Type};

use crate::scene_store::{FieldInfo, MirrorModeAttr};

fn mirror_mode_tokens(mode: MirrorModeAttr) -> TokenStream {
    match mode {
        MirrorModeAttr::DirtyTracked => quote! { ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked },
        MirrorModeAttr::Once => quote! { ::pulsar_scenedb::gpu::MirrorMode::Once },
    }
}

/// Resolves a `#[gpu(content_id = "sibling")]` field's named sibling
/// against the struct's full field list -- errors (macro-expansion time,
/// not a runtime panic) if no field with that ident exists. Returns the
/// sibling's own `Type` for splicing into the generated
/// `<SiblingTy as ContentAddressed>::content_id(&data.sibling)`-shaped
/// call -- the compiler enforces the trait bound on THAT call, so a sibling
/// whose type doesn't implement `ContentAddressed` is still a clear
/// compile error, just one message later than this check.
fn resolve_content_id_sibling<'a>(
    field: &FieldInfo,
    sibling_ident: &Ident,
    field_infos: &'a [FieldInfo],
) -> syn::Result<&'a Type> {
    field_infos
        .iter()
        .find(|f| &f.ident == sibling_ident)
        .map(|f| &f.ty)
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &field.ident,
                format!(
                    "#[gpu(content_id = \"{sibling_ident}\")] names a field that doesn't exist on this struct"
                ),
            )
        })
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
    // A HeavyEither field gets THREE wrappers (base/_ok/_err): its three
    // handle tables are looked up by ComponentId alone, so they cannot
    // share one wrapper type.
    let gpu_wrapper_defs: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            if f.is_var_len {
                if matches!(f.var_len_shape, crate::scene_store::VarLenShape::HeavyEither) {
                    let ok_wrapper = quote::format_ident!("{}_ok", wrapper);
                    let err_wrapper = quote::format_ident!("{}_err", wrapper);
                    return quote! {
                        #[doc(hidden)]
                        #[allow(non_camel_case_types)]
                        #[repr(transparent)]
                        #[derive(Clone, Copy)]
                        pub struct #wrapper(pub ::pulsar_scenedb::gpu::VarLenHandle);
                        unsafe impl ::pulsar_scenedb::page::Pod for #wrapper {}
                        #[doc(hidden)]
                        #[allow(non_camel_case_types)]
                        #[repr(transparent)]
                        #[derive(Clone, Copy)]
                        pub struct #ok_wrapper(pub ::pulsar_scenedb::gpu::VarLenHandle);
                        unsafe impl ::pulsar_scenedb::page::Pod for #ok_wrapper {}
                        #[doc(hidden)]
                        #[allow(non_camel_case_types)]
                        #[repr(transparent)]
                        #[derive(Clone, Copy)]
                        pub struct #err_wrapper(pub ::pulsar_scenedb::gpu::VarLenHandle);
                        unsafe impl ::pulsar_scenedb::page::Pod for #err_wrapper {}
                    };
                }
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
    // Var-len fields branch on their lowering SHAPE (`FieldInfo::
    // var_len_shape`): Plain is today's path verbatim; the Heavy placements
    // lower the pool element through the boundary (`GpuHeavy<H>` -> `H`)
    // and ALWAYS route through the INTERNED pool -- a Heavy boundary IS a
    // dedup declaration, so sharing is the default even without an explicit
    // `content_id` sibling (identity then comes from
    // `gpu::structural_content_id` over the row's own references). The
    // Either placement fans out into THREE pools (tag/ok/err) with a handle
    // table each -- three independent chain families per row.
    let register_growable_calls: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
            if matches!(f.var_len_shape, crate::scene_store::VarLenShape::HeavyEither) {
                // Three pools, three handle tables -- one chain family per
                // side of the discriminant plus the tag stream itself. The
                // tag stream is always a PLAIN pool (a u32 discriminant has
                // nothing to dedup); ok/err are always INTERNED (a Heavy
                // boundary is a dedup declaration).
                let tag_key = format!("{key}::tag");
                let ok_key = format!("{key}::ok");
                let err_key = format!("{key}::err");
                let ok_elem = f.either_ok_ty.as_ref().expect("HeavyEither implies ok elem ty");
                let err_elem = f.either_err_ty.as_ref().expect("HeavyEither implies err elem ty");
                let tag_table_key = format!("{tag_key}::handles");
                let ok_table_key = format!("{ok_key}::handles");
                let err_table_key = format!("{err_key}::handles");
                let ok_wrapper = quote::format_ident!("{}_ok", wrapper);
                let err_wrapper = quote::format_ident!("{}_err", wrapper);
                quote! {
                    store.register_var_len_gpu_pool::<u32>(
                        ::pulsar_scenedb::gpu::BufferKey::of(#tag_key),
                        initial_capacity,
                        device,
                    );
                    store.register_interned_var_len_gpu_pool::<#ok_elem>(
                        ::pulsar_scenedb::gpu::BufferKey::of(#ok_key),
                        initial_capacity,
                        device,
                    );
                    store.register_interned_var_len_gpu_pool::<#err_elem>(
                        ::pulsar_scenedb::gpu::BufferKey::of(#err_key),
                        initial_capacity,
                        device,
                    );
                    store.register_dirty_tracked_gpu_buffer::<#wrapper, ::pulsar_scenedb::gpu::VarLenHandle>(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#tag_table_key),
                        ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked,
                    );
                    store.register_dirty_tracked_gpu_buffer::<#ok_wrapper, ::pulsar_scenedb::gpu::VarLenHandle>(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#ok_table_key),
                        ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked,
                    );
                    store.register_dirty_tracked_gpu_buffer::<#err_wrapper, ::pulsar_scenedb::gpu::VarLenHandle>(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#err_table_key),
                        ::pulsar_scenedb::gpu::MirrorMode::DirtyTracked,
                    );
                }
            } else if f.is_var_len {
                let elem_ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                // Handle table gets its own key, distinct from the pool's --
                // two different buffers (one Vec<VarLenHandle>-shaped, one
                // Vec<ElemTy>-shaped), so they can never share a BufferKey
                // even when the field itself declares one (that declared
                // key names the POOL -- the payload data other fields might
                // also want to share -- not the per-field handle table,
                // which is never a sharing target).
                let handle_key = format!("{key}::handles");
                // Pool routing: Plain keeps today's exact behavior (plain
                // unless content_id); Heavy placements are ALWAYS interned.
                let pool_register = match f.var_len_shape {
                    crate::scene_store::VarLenShape::Plain => {
                        if f.content_id_field.is_some() {
                            quote! {
                                store.register_interned_var_len_gpu_pool::<#elem_ty>(
                                    ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                    initial_capacity,
                                    device,
                                );
                            }
                        } else {
                            quote! {
                                store.register_var_len_gpu_pool::<#elem_ty>(
                                    ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                    initial_capacity,
                                    device,
                                );
                            }
                        }
                    }
                    crate::scene_store::VarLenShape::HeavyHandle
                    | crate::scene_store::VarLenShape::HeavyOptionHandle => quote! {
                        store.register_interned_var_len_gpu_pool::<#elem_ty>(
                            ::pulsar_scenedb::gpu::BufferKey::of(#key),
                            initial_capacity,
                            device,
                        );
                    },
                    crate::scene_store::VarLenShape::HeavyEither => unreachable!(),
                };
                quote! {
                    #pool_register
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
    //
    // Var-len fields branch on their lowering SHAPE first: Heavy placements
    // lower the slice through the boundary BEFORE any write and return
    // early; Plain falls through to today's arms verbatim. `Vec<GpuHeavy<H>>`
    // reinterprets in place (zero allocation -- `repr(transparent)` + `H:
    // Pod` make the bytes identical); Option/Result positions build a small
    // mapped buffer (one allocation per edit-driven write; NULL/discriminant
    // handling makes an in-place view impossible). Identity: sibling content
    // id when declared, else `gpu::structural_content_id` over the lowered
    // chain.
    let write_arms: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| -> syn::Result<TokenStream> {
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            if f.is_var_len {
                let elem_ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                let field_name = f.ident.to_string();
                let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
                let handle_id_decl = quote! {
                    let handle_id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                };

                match f.var_len_shape {
                    crate::scene_store::VarLenShape::HeavyEither => {
                        let ok_elem = f.either_ok_ty.as_ref().expect("HeavyEither implies ok elem ty");
                        let err_elem = f.either_err_ty.as_ref().expect("HeavyEither implies err elem ty");
                        let tag_key = format!("{key}::tag");
                        let ok_key = format!("{key}::ok");
                        let err_key = format!("{key}::err");
                        let (ok_id_expr, err_id_expr) = if let Some(sibling_ident) = &f.content_id_field {
                            let sibling_ty = resolve_content_id_sibling(f, sibling_ident, field_infos)?;
                            (
                                quote! { ::pulsar_scenedb::handle_ledger::ContentAddressed::content_id(&data.#sibling_ident) },
                                quote! { ::pulsar_scenedb::handle_ledger::ContentAddressed::content_id(&data.#sibling_ident) },
                            )
                        } else {
                            (
                                quote! { ::pulsar_scenedb::gpu::structural_content_id(&__ok) },
                                quote! { ::pulsar_scenedb::gpu::structural_content_id(&__err) },
                            )
                        };
                        let ok_wrapper = quote::format_ident!("{}_ok", wrapper);
                        let err_wrapper = quote::format_ident!("{}_err", wrapper);
                        return Ok(quote! {
                            {
                                #handle_id_decl
                                let mut __tags: ::std::vec::Vec<u32> =
                                    ::std::vec::Vec::with_capacity(data.#field_ident.len());
                                let mut __ok: ::std::vec::Vec<#ok_elem> =
                                    ::std::vec::Vec::with_capacity(data.#field_ident.len());
                                let mut __err: ::std::vec::Vec<#err_elem> =
                                    ::std::vec::Vec::with_capacity(data.#field_ident.len());
                                for slot in &data.#field_ident {
                                    match slot {
                                        ::std::result::Result::Ok(v) => {
                                            __tags.push(0);
                                            __ok.push(v.0);
                                        }
                                        ::std::result::Result::Err(e) => {
                                            __tags.push(1);
                                            __err.push(e.0);
                                        }
                                    }
                                }
                                // Per-row invariant (see FieldInfo::
                                // var_len_shape): __ok.len() + __err.len() ==
                                // data.#field_ident.len().
                                ::pulsar_scenedb::gpu::write_var_len_field_at_row::<u32>(
                                    store,
                                    queue,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#tag_key),
                                    handle_id,
                                    row,
                                    &__tags,
                                );
                                let ok_handle_id =
                                    ::pulsar_scenedb::component::component_id::<#ok_wrapper>();
                                let err_handle_id =
                                    ::pulsar_scenedb::component::component_id::<#err_wrapper>();
                                let __id_ok = #ok_id_expr;
                                ::pulsar_scenedb::gpu::write_interned_var_len_field_with_id_at_row::<#ok_elem>(
                                    store,
                                    queue,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#ok_key),
                                    ok_handle_id,
                                    row,
                                    __id_ok,
                                    &__ok,
                                );
                                let __id_err = #err_id_expr;
                                ::pulsar_scenedb::gpu::write_interned_var_len_field_with_id_at_row::<#err_elem>(
                                    store,
                                    queue,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#err_key),
                                    err_handle_id,
                                    row,
                                    __id_err,
                                    &__err,
                                );
                            }
                        });
                    }
                    crate::scene_store::VarLenShape::HeavyHandle => {
                        let id_expr = if let Some(sibling_ident) = &f.content_id_field {
                            let sibling_ty = resolve_content_id_sibling(f, sibling_ident, field_infos)?;
                            quote! { ::pulsar_scenedb::handle_ledger::ContentAddressed::content_id(&data.#sibling_ident) }
                        } else {
                            quote! { ::pulsar_scenedb::gpu::structural_content_id(&__handles) }
                        };
                        return Ok(quote! {
                            {
                                #handle_id_decl
                                // ZERO allocation by construction:
                                // `GpuHeavy<H>` is `#[repr(transparent)]` over
                                // a Pod `H`, so this Vec's bytes ARE an `[H]`.
                                let __handles: &[#elem_ty] = unsafe {
                                    ::std::slice::from_raw_parts(
                                        data.#field_ident.as_ptr() as *const #elem_ty,
                                        data.#field_ident.len(),
                                    )
                                };
                                let __id = #id_expr;
                                ::pulsar_scenedb::gpu::write_interned_var_len_field_with_id_at_row::<#elem_ty>(
                                    store,
                                    queue,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                    handle_id,
                                    row,
                                    __id,
                                    __handles,
                                );
                            }
                        });
                    }
                    crate::scene_store::VarLenShape::HeavyOptionHandle => {
                        let id_expr = if let Some(sibling_ident) = &f.content_id_field {
                            let sibling_ty = resolve_content_id_sibling(f, sibling_ident, field_infos)?;
                            quote! { ::pulsar_scenedb::handle_ledger::ContentAddressed::content_id(&data.#sibling_ident) }
                        } else {
                            quote! { ::pulsar_scenedb::gpu::structural_content_id(&__handles) }
                        };
                        return Ok(quote! {
                            {
                                #handle_id_decl
                                let mut __handles: ::std::vec::Vec<#elem_ty> =
                                    ::std::vec::Vec::with_capacity(data.#field_ident.len());
                                for slot in &data.#field_ident {
                                    match slot {
                                        ::std::option::Option::Some(h) => __handles.push(h.0),
                                        ::std::option::Option::None => {
                                            __handles.push(<#elem_ty as ::pulsar_scenedb::gpu::GpuRef>::NULL)
                                        }
                                    }
                                }
                                let __id = #id_expr;
                                ::pulsar_scenedb::gpu::write_interned_var_len_field_with_id_at_row::<#elem_ty>(
                                    store,
                                    queue,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                    handle_id,
                                    row,
                                    __id,
                                    &__handles,
                                );
                            }
                        });
                    }
                    crate::scene_store::VarLenShape::Plain => {}
                }

                if let Some(sibling_ident) = &f.content_id_field {
                    let sibling_ty = resolve_content_id_sibling(f, sibling_ident, field_infos)?;
                    return Ok(quote! {
                        {
                            let handle_id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                            ::pulsar_scenedb::gpu::write_interned_var_len_field_at_row::<#elem_ty, #sibling_ty>(
                                store,
                                queue,
                                ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                handle_id,
                                row,
                                &data.#sibling_ident,
                                &data.#field_ident,
                            );
                        }
                    });
                }
                return Ok(quote! {
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
                });
            }
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
            Ok(match f.mirror_mode {
                MirrorModeAttr::Once => quote! {
                    if is_new_insert {
                        #body
                    }
                },
                MirrorModeAttr::DirtyTracked => body,
            })
        })
        .collect::<syn::Result<Vec<TokenStream>>>()?;

    // Despawn/remove release dispatch -- one arm per var-len field (plain
    // or interned; scalar `#[gpu]` fields have no pool allocation to free,
    // so they contribute nothing here). Emitted only when at least one
    // var-len field exists, which is always true on this codegen path (it's
    // why this path was chosen at all) -- so this registration is
    // unconditional for every var-len-bearing struct, matching
    // `VarLenReleaseRegistration`'s "absent means the type never submitted
    // one" cost contract exactly (a struct with zero var-len fields never
    // reaches `generate_var_len_bearing_type` in the first place).
    let release_arms: Vec<TokenStream> = gpu_fields
        .iter()
        .filter(|f| f.is_var_len)
        .flat_map(|f| {
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let elem_ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
            let field_name = f.ident.to_string();
            let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
            match f.var_len_shape {
                crate::scene_store::VarLenShape::HeavyEither => {
                    // Three chains to release: tag (plain u32) plus ok and
                    // err (both interned -- a Heavy boundary is a dedup
                    // declaration). Three handle TABLES means three
                    // ComponentIds, matching the three registrations.
                    let ok_wrapper = quote::format_ident!("{}_ok", wrapper);
                    let err_wrapper = quote::format_ident!("{}_err", wrapper);
                    let ok_elem = f.either_ok_ty.as_ref().expect("HeavyEither implies ok elem ty");
                    let err_elem = f.either_err_ty.as_ref().expect("HeavyEither implies err elem ty");
                    let tag_key = format!("{key}::tag");
                    let ok_key = format!("{key}::ok");
                    let err_key = format!("{key}::err");
                    vec![
                        quote! {
                            {
                                let handle_id =
                                    ::pulsar_scenedb::component::component_id::<#wrapper>();
                                ::pulsar_scenedb::gpu::free_var_len_field_at_row::<u32>(
                                    store,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#tag_key),
                                    handle_id,
                                    row,
                                );
                            }
                        },
                        quote! {
                            {
                                let handle_id =
                                    ::pulsar_scenedb::component::component_id::<#ok_wrapper>();
                                ::pulsar_scenedb::gpu::free_interned_var_len_field_at_row::<#ok_elem>(
                                    store,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#ok_key),
                                    handle_id,
                                    row,
                                );
                            }
                        },
                        quote! {
                            {
                                let handle_id =
                                    ::pulsar_scenedb::component::component_id::<#err_wrapper>();
                                ::pulsar_scenedb::gpu::free_interned_var_len_field_at_row::<#err_elem>(
                                    store,
                                    ::pulsar_scenedb::gpu::BufferKey::of(#err_key),
                                    handle_id,
                                    row,
                                );
                            }
                        },
                    ]
                }
                crate::scene_store::VarLenShape::HeavyHandle
                | crate::scene_store::VarLenShape::HeavyOptionHandle => {
                    // Always interned -- a Heavy boundary is a dedup
                    // declaration.
                    vec![quote! {
                        {
                            let handle_id =
                                ::pulsar_scenedb::component::component_id::<#wrapper>();
                            ::pulsar_scenedb::gpu::free_interned_var_len_field_at_row::<#elem_ty>(
                                store,
                                ::pulsar_scenedb::gpu::BufferKey::of(#key),
                                handle_id,
                                row,
                            );
                        }
                    }]
                }
                crate::scene_store::VarLenShape::Plain => {
                    let free_fn = if f.content_id_field.is_some() {
                        quote! { ::pulsar_scenedb::gpu::free_interned_var_len_field_at_row::<#elem_ty> }
                    } else {
                        quote! { ::pulsar_scenedb::gpu::free_var_len_field_at_row::<#elem_ty> }
                    };
                    vec![quote! {
                        {
                            let handle_id =
                                ::pulsar_scenedb::component::component_id::<#wrapper>();
                            #free_fn(store, ::pulsar_scenedb::gpu::BufferKey::of(#key), handle_id, row);
                        }
                    }]
                }
            }
        })
        .collect();

    let mirror_dispatch_fn_name = quote::format_ident!("__scenedb_gpu_mirror_dispatch_{}", name);
    let release_dispatch_fn_name = quote::format_ident!("__scenedb_gpu_release_dispatch_{}", name);

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
    // fields need their LOWERED element type to satisfy the same (the
    // field's own `Vec<T>` type itself is never bounded this way -- it's
    // never used as a column element, only the lowered element is).
    // Heavy placements add their extra requirements on top: optional slots
    // need a NULL sentinel (`GpuRef`), and Either needs bounds for BOTH
    // sides (hence flat_map, two entries there).
    let register_field_ty_bounds: Vec<TokenStream> = gpu_fields
        .iter()
        .flat_map(|f| {
            if f.is_var_len {
                match f.var_len_shape {
                    crate::scene_store::VarLenShape::Plain => {
                        let ty = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                        vec![quote! { #ty: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken }]
                    }
                    crate::scene_store::VarLenShape::HeavyHandle
                    | crate::scene_store::VarLenShape::HeavyOptionHandle => {
                        let h = f.var_len_elem_ty.as_ref().expect("is_var_len implies var_len_elem_ty");
                        let gpu_ref_bound = if matches!(
                            f.var_len_shape,
                            crate::scene_store::VarLenShape::HeavyOptionHandle
                        ) {
                            quote! { , #h: ::pulsar_scenedb::gpu::GpuRef }
                        } else {
                            quote! {}
                        };
                        vec![quote! { #h: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken #gpu_ref_bound }]
                    }
                    crate::scene_store::VarLenShape::HeavyEither => {
                        let ok = f.either_ok_ty.as_ref().expect("HeavyEither implies ok elem ty");
                        let err = f.either_err_ty.as_ref().expect("HeavyEither implies err elem ty");
                        vec![
                            quote! { #ok: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken },
                            quote! { #err: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken },
                        ]
                    }
                }
            } else {
                let ty = &f.ty;
                vec![quote! { #ty: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken }]
            }
        })
        .collect();

    // One `pub fn #{field}_gpu_handle(store, row) -> Option<VarLenHandle>`
    // per var-len field -- lets a consumer that already has `row` (e.g.
    // `entity.index()`) find WHERE this entity's data landed in the shared
    // pool (offset/count), without needing to know the field's own hidden
    // wrapper type or its `ComponentId`. The motivating case (Pulsar-
    // Native#561 Phase E): a renderer wiring a `MeshId`/draw call up to
    // point at an entity's `#[gpu] Vec<T>` field's data directly -- it
    // needs the OFFSET (to build a slice into the shared pool), not a copy
    // of the data itself, which is already GPU-resident via the mirror
    // dispatch. `read_dirty_tracked_row_bytes` is a CPU-shadow-only read
    // (see its own doc) -- cheap enough to call once per entity per frame,
    // not a GPU readback.
    //
    // A HeavyEither field has THREE chains (tag/ok/err), so it gets three
    // accessors suffixed `_tag`/`_ok`/`_err`, each reading its own handle
    // table under its own wrapper ComponentId.
    let handle_accessors: Vec<TokenStream> = gpu_fields
        .iter()
        .filter(|f| f.is_var_len)
        .flat_map(|f| {
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let accessor_body = |accessor_name: proc_macro2::Ident,
                                 table_wrapper: &proc_macro2::Ident| {
                quote! {
                    /// Returns this entity's current `VarLenHandle` (offset/count
                    /// into the shared pool) for this chain, as of the last
                    /// `World::insert`/mirror dispatch. `None` only when `row`
                    /// is out of the registered column's capacity (never a
                    /// valid row at all) -- a row that's simply never been
                    /// written yet comes back `Some(VarLenHandle::default())`
                    /// (`count == 0`, the same "no allocation" sentinel an
                    /// explicitly-emptied field also produces; the CPU shadow
                    /// is zero-initialized at registration, not sparse). Reads
                    /// a CPU shadow, not the GPU buffer itself -- cheap enough
                    /// to call every frame.
                    pub fn #accessor_name(
                        store: &::pulsar_scenedb::gpu::SceneGpuStore,
                        row: u32,
                    ) -> ::std::option::Option<::pulsar_scenedb::gpu::VarLenHandle> {
                        let id = ::pulsar_scenedb::component::component_id::<#table_wrapper>();
                        let bytes = store.read_dirty_tracked_row_bytes(id, row)?;
                        if bytes.len() != ::std::mem::size_of::<::pulsar_scenedb::gpu::VarLenHandle>() {
                            return None;
                        }
                        let mut handle = ::pulsar_scenedb::gpu::VarLenHandle::default();
                        unsafe {
                            ::std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                &mut handle as *mut ::pulsar_scenedb::gpu::VarLenHandle as *mut u8,
                                bytes.len(),
                            );
                        }
                        ::std::option::Option::Some(handle)
                    }
                }
            };
            if matches!(f.var_len_shape, crate::scene_store::VarLenShape::HeavyEither) {
                let tag_name = quote::format_ident!("{}_gpu_handle_tag", field_ident);
                let ok_name = quote::format_ident!("{}_gpu_handle_ok", field_ident);
                let err_name = quote::format_ident!("{}_gpu_handle_err", field_ident);
                let ok_wrapper = quote::format_ident!("{}_ok", wrapper);
                let err_wrapper = quote::format_ident!("{}_err", wrapper);
                vec![
                    accessor_body(tag_name, wrapper),
                    accessor_body(ok_name, &ok_wrapper),
                    accessor_body(err_name, &err_wrapper),
                ]
            } else {
                let accessor_name = quote::format_ident!("{}_gpu_handle", field_ident);
                vec![accessor_body(accessor_name, wrapper)]
            }
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

                #(#handle_accessors)*
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

            #[doc(hidden)]
            #[allow(non_snake_case, unused_variables)]
            fn #release_dispatch_fn_name(mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
                let store = mirror.store();
                #(#release_arms)*
            }

            ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
                ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                    component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                    dispatch: #mirror_dispatch_fn_name,
                }
            }

            ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
                ::pulsar_scenedb::gpu::VarLenReleaseRegistration {
                    component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                    release: #release_dispatch_fn_name,
                }
            }
        };
    })
}