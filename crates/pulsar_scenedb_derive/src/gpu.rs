use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::scene_store::{FieldInfo, MirrorModeAttr};

pub fn generate_gpu_column_set(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    gpu_fields: &[&FieldInfo],
    is_packed: bool,
    packed_buffer_key: Option<&str>,
) -> TokenStream {
    if gpu_fields.is_empty() {
        return quote! {
            impl #impl_generics ::pulsar_scenedb::GpuColumnSet for #name #ty_generics #where_clause {
                fn gpu_columns() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
                    Vec::new()
                }
                fn write_gpu(
                    _store: &::pulsar_scenedb::gpu::SceneGpuStore,
                    _id: ::pulsar_scenedb::gpu::CellId,
                    _cell: &mut ::pulsar_scenedb::cell::CellStorage,
                    _handle: ::pulsar_scenedb::handle::Handle,
                    _data: &Self,
                    _phase: &impl ::pulsar_scenedb::gpu::SimulateWitness,
                ) {
                }
            }

            impl #impl_generics #name #ty_generics #where_clause {
                /// No `#[gpu]` fields on this type -- nothing to register.
                pub fn register_gpu_columns(
                    _store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                    _capacity: u32,
                    _device: &::wgpu::Device,
                ) {
                }

                /// No `#[gpu]` fields on this type -- nothing to register.
                pub fn register_gpu_columns_growable(
                    _store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                    _initial_capacity: u32,
                    _device: &::std::sync::Arc<::wgpu::Device>,
                ) {
                }
            }
        };
    }

    // Packed layout writes the whole record as one unit, so every `#[gpu]`
    // field in a packed struct must share one mirror mode -- there's no
    // sensible meaning for "some of this one packed write is Once and some
    // is DirtyTracked." Validated here, at macro-expansion time, as a real
    // diagnostic (not a macro-execution-time panic).
    if is_packed {
        let all_once = gpu_fields.iter().all(|f| f.mirror_mode == MirrorModeAttr::Once);
        let all_dirty_tracked = gpu_fields.iter().all(|f| f.mirror_mode == MirrorModeAttr::DirtyTracked);
        if !all_once && !all_dirty_tracked {
            return quote! {
                compile_error!(
                    "#[gpu(layout = packed)] requires every #[gpu] field to share the same \
                     mirror mode -- either all plain #[gpu] (DirtyTracked, the default) or all \
                     #[gpu(mirror = Once)]. Mixed modes have no sensible meaning for a packed \
                     column, since the whole record is written as one unit."
                );
            };
        }
    }
    let packed_is_once = is_packed && gpu_fields.iter().all(|f| f.mirror_mode == MirrorModeAttr::Once);

    // `heavy` (the CPU handle / heavy GPU element split, `GpuUploadSource`)
    // is only meaningful for `Once` fields -- `DirtyTracked` fields keep a
    // full CPU shadow of whatever type they declare by design (delta-sync
    // needs something to diff against), which a handle/heavy split has no
    // useful meaning against. Also not supported inside a packed struct: a
    // packed buffer's element type is the struct's own generated packed
    // view, not any one field's `GpuUploadSource::Element`, and mixing the
    // two concepts (a per-field heavy mapper feeding one interleaved record)
    // isn't a shape this macro builds -- kept out of scope, not silently
    // wrong.
    if let Some(bad) = gpu_fields.iter().find(|f| f.heavy && f.mirror_mode != MirrorModeAttr::Once) {
        let field_name = bad.ident.to_string();
        return quote! {
            compile_error!(concat!(
                "#[gpu(heavy)] on field `", #field_name, "` requires #[gpu(mirror = Once, heavy)] -- \
                 the handle/heavy-element split (GpuUploadSource) only applies to Once-mode fields."
            ));
        };
    }
    if is_packed {
        if let Some(bad) = gpu_fields.iter().find(|f| f.heavy) {
            let field_name = bad.ident.to_string();
            return quote! {
                compile_error!(concat!(
                    "#[gpu(heavy)] on field `", #field_name, "` is not supported inside \
                     #[gpu(layout = packed)] -- a packed buffer's element is the struct's own \
                     interleaved record, not any one field's GpuUploadSource::Element."
                ));
            };
        }
    }

    // Element-type bounds for the generated registration methods: the raw
    // field type (not the per-field wrapper) is what a shared buffer's
    // element identity is validated against, so `register_gpu_columns(_growable)`
    // requires each `#[gpu]` field's type to be Pod + HasTypeToken -- the
    // same requirements the struct's own generated `Pod` impl carries, so
    // this is never the binding constraint for a real (Pod) component.
    let register_field_ty_bounds: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let ty = &f.ty;
            if f.heavy {
                // Heavy fields additionally need `GpuUploadSource` (used by
                // `register_gpu_columns_growable`'s body) and its `Element`
                // to itself satisfy the same requirements every GPU buffer
                // element type needs.
                quote! {
                    #ty: ::pulsar_scenedb::page::Pod
                        + ::pulsar_scenedb::token::HasTypeToken
                        + ::pulsar_scenedb::gpu::GpuUploadSource,
                    <#ty as ::pulsar_scenedb::gpu::GpuUploadSource>::Element:
                        ::pulsar_scenedb::page::Pod
                        + ::pulsar_scenedb::token::HasTypeToken
                        + Send + Sync + 'static
                }
            } else {
                quote! { #ty: ::pulsar_scenedb::page::Pod + ::pulsar_scenedb::token::HasTypeToken }
            }
        })
        .collect();

    // Every `#[gpu]` field is stored (and registered as a GPU buffer) under
    // its own generated wrapper type, not its raw field type -- see
    // `FieldInfo::gpu_wrapper`'s doc for why: `TypeToken`/`ComponentId` are
    // TypeId-keyed globally, so two different structs' same-shaped `#[gpu]`
    // fields would otherwise silently alias one GPU buffer.
    let column_descs: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let mirror_mode = match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => {
                    quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
                }
                MirrorModeAttr::Once => {
                    quote! { ::pulsar_scenedb::MirrorMode::Once }
                }
            };
            let upload_expr = if f.heavy {
                let ty = &f.ty;
                quote! {
                    Some(|handle_ptr: *const ()| {
                        // SAFETY: `handle_ptr` is `write_gpu_columns_at_row`'s
                        // `(data as *const T as *const u8).add(col.field_offset)`
                        // for this exact field -- a live, correctly-aligned
                        // `#ty` (the handle), per that function's own contract.
                        let handle = unsafe { &*(handle_ptr as *const #ty) };
                        let element = <#ty as ::pulsar_scenedb::gpu::GpuUploadSource>::upload_element(handle);
                        let bytes: &[u8] = unsafe {
                            ::std::slice::from_raw_parts(
                                &element as *const _ as *const u8,
                                ::std::mem::size_of_val(&element),
                            )
                        };
                        bytes.to_vec()
                    })
                }
            } else {
                quote! { None }
            };
            quote! {
                ::pulsar_scenedb::GpuColumnDesc {
                    field_token: ::pulsar_scenedb::token::TypeToken::of::<#wrapper>(),
                    field_offset: ::std::mem::offset_of!(#name, #field_ident),
                    mode: #mirror_mode,
                    buffer_name: #field_name,
                    upload: #upload_expr,
                }
            }
        })
        .collect();

    // `heavy` fields are World-mirror-only (see the `heavy` validation
    // above and `register_calls`'s doc below) -- excluded from the
    // cell-mirrored `write_gpu` match arms, so a heavy field's `desc` just
    // falls through `write_gpu`'s `_ => {}` catch-all: a harmless no-op,
    // never a panic or a write into a buffer that was never registered on
    // the fixed/cell-mirrored path.
    let write_arms: Vec<_> = gpu_fields
        .iter()
        .filter(|f| !f.heavy)
        .map(|f| {
            let field_name = f.ident.to_string();
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            quote! {
                #field_name => {
                    let row = cell.row_of(handle).unwrap_or_else(|| {
                        panic!("write_gpu: handle {:?} not found in cell", handle);
                    }) as usize;
                    if let Some(col) = cell.column_for_mut::<#wrapper>() {
                        col[row] = #wrapper(data.#field_ident);
                    }
                    let comp_id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                    store.mark_column_dirty(id, comp_id, row as u32);
                }
            }
        })
        .collect();

    // `heavy` fields skip the fixed/cell-mirrored registration path entirely
    // -- they're a World-mirror-only concept (the spec's own motivating use
    // is asset handles attached via `World::insert`, not per-cell instance
    // data). A struct mixing heavy and non-heavy `#[gpu]` fields still
    // registers its non-heavy fields here normally; only the heavy ones are
    // absent, matching `write_arms`'s exclusion above.
    let register_calls: Vec<_> = gpu_fields
        .iter()
        .filter(|f| !f.heavy)
        .map(|f| {
            let field_name = f.ident.to_string();
            let field_ident = &f.ident;
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let ty = &f.ty;
            // Effective buffer key: the declared `#[gpu(buffer = "...")]`
            // key if present, else a key unique to this (struct, field)
            // pair (identical to the old per-field buffer label). Two
            // fields that declare the SAME key share one physical buffer —
            // validated (element type, stride, access, mirror mode) by
            // `SceneGpuStore::register_gpu_buffer` at registration.
            let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
            let mirror_mode = match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => {
                    quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
                }
                MirrorModeAttr::Once => {
                    quote! { ::pulsar_scenedb::MirrorMode::Once }
                }
            };
            quote! {
                store.register_gpu_buffer::<#wrapper, #ty>(
                    capacity,
                    device,
                    ::pulsar_scenedb::gpu::BufferKey::of(#key),
                    #mirror_mode,
                );
            }
        })
        .collect();

    // Per field, registered through the dirty-tracked path regardless of
    // its declared `#[gpu(mirror = ...)]` mode (SceneDB#39): `Once` and
    // `DirtyTracked` fields both need `World::flush_gpu_mirror` to actually
    // have something registered to flush -- they differ only in HOW OFTEN
    // `write_gpu_columns_at_row` marks a row dirty at all (`Once`: just the
    // first insert; `DirtyTracked`: every insert), not in which map they
    // live in or how the flush itself works. Reusing one proven,
    // read-lock-first write path for both, instead of a separate immediate
    // or hand-rolled-pending-queue mechanism for `Once`, is deliberate --
    // see `GenerationMirror`'s doc (`gpu::world_mirror`, SceneDB crate) for
    // why an earlier, more "obviously cheap" alternative measured worse in
    // practice. Never sets a `max_capacity` ceiling -- see
    // `register_gpu_columns_growable`'s own doc for why that's deliberate
    // for World-mirrored columns.
    let register_growable_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let ty = &f.ty;
            let key = f.buffer_key.clone().unwrap_or_else(|| format!("{name}::{field_name}"));
            // A `heavy` field registers its buffer sized to
            // `GpuUploadSource::Element` (the heavy type), not to `#ty`
            // itself (the handle) -- `register_dirty_tracked_gpu_buffer_heavy`
            // is the variant that builds the buffer from `Element` instead of
            // asserting `#wrapper`/`#ty` are byte-identical. Always `Once`
            // (validated above), so no `mode` argument to pass.
            if f.heavy {
                quote! {
                    store.register_dirty_tracked_gpu_buffer_heavy::<
                        #wrapper,
                        <#ty as ::pulsar_scenedb::gpu::GpuUploadSource>::Element,
                    >(
                        initial_capacity,
                        device,
                        ::pulsar_scenedb::gpu::BufferKey::of(#key),
                    );
                }
            } else {
            let mirror_mode = match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => {
                    quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
                }
                MirrorModeAttr::Once => {
                    quote! { ::pulsar_scenedb::MirrorMode::Once }
                }
            };
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

    // World-mirror dispatch registration (`pulsar_scenedb::gpu::world_mirror`):
    // a non-generic function with `#name` already concrete, submitted via
    // `inventory` so `World::insert` can find it by `ComponentId` without
    // needing compile-time specialization on the caller's generic `T` --
    // see that module's doc for why the specialization approach doesn't
    // work through a generic `insert_inner<T>` body. Only emitted here (in
    // the non-empty-`gpu_fields` branch) -- a type with no `#[gpu]` fields
    // submits no registration at all, so `World::insert` for it is exactly
    // one `HashMap` miss when a mirror is attached, nothing when it isn't.
    let mirror_dispatch_fn_name = quote::format_ident!("__scenedb_gpu_mirror_dispatch_{}", name);

    // Packed layout (`#[gpu(layout = packed)]`, struct-level): ONE GPU
    // buffer for every `#[gpu]` field combined, instead of the default
    // one-per-field split -- see `struct_is_packed`'s doc for the full
    // rationale and why this is scoped to ONLY `register_gpu_columns_growable`
    // + the World-mirror dispatch fn (`gpu_columns()`/`write_gpu`/the fixed
    // `register_gpu_columns`, generated above, are completely unaffected --
    // still per-field, for the cell-mirrored path, regardless of this flag).
    let packed_view_ident = quote::format_ident!("__ScenedbGpuPacked_{}", name);
    let packed_field_defs: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let ident = &f.ident;
            let ty = &f.ty;
            quote! { pub #ident: #ty }
        })
        .collect();
    let packed_field_idents: Vec<_> = gpu_fields.iter().map(|f| f.ident.clone()).collect();
    let packed_view_def = quote! {
        // Field-for-field copy of every #[gpu] field on #name, in
        // declaration order -- NOT a repr(transparent) single-field wrapper
        // like the per-field #[gpu] wrappers above; this is the actual
        // interleaved GPU-side record a shader reads as one struct. Unique
        // by construction (its name embeds #name), so unlike the per-field
        // wrappers it doesn't need the (struct, field) disambiguation
        // trick -- there's exactly one of these per packed struct.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct #packed_view_ident {
            #(#packed_field_defs),*
        }
        unsafe impl ::pulsar_scenedb::page::Pod for #packed_view_ident {}
    };

    // SceneDB#39: registered through the dirty-tracked path regardless of
    // `packed_is_once` -- see the non-packed `register_growable_calls`'s
    // comment above for why both modes share one registration + write path
    // now.
    let register_growable_calls_packed = {
        let default_packed_key = format!("{}::packed", name);
        let packed_key = packed_buffer_key.unwrap_or(&default_packed_key);
        // The registered ELEMENT type is the packed view itself
        // (`#packed_view_ident`), NOT `#name`: the buffer's element size is
        // the packed record's size (what the dispatch fn assembles and
        // `mark_gpu_row_dirty` writes), and `#name` may carry non-`#[gpu]`
        // fields that aren't part of that record. Two packed structs sharing
        // one buffer key would have to be the exact same type -- and since
        // each struct's packed view has a unique name, element-type-identity
        // naturally forbids cross-struct sharing; document, don't special-case.
        let packed_mirror_mode = if packed_is_once {
            quote! { ::pulsar_scenedb::MirrorMode::Once }
        } else {
            quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
        };
        quote! {
            store.register_dirty_tracked_gpu_buffer::<#packed_view_ident, #packed_view_ident>(
                initial_capacity,
                device,
                ::pulsar_scenedb::gpu::BufferKey::of(#packed_key),
                #packed_mirror_mode,
            );
        }
    };

    // Packed write body: `Once` marks dirty only on the first insert (skips
    // entirely on a later update, matching the non-packed Once behavior
    // exactly) and never again; `DirtyTracked` marks dirty on every insert.
    // Both then flush identically via `World::flush_gpu_mirror` --
    // `packed_is_once` was validated uniform across every #[gpu] field on
    // this struct above.
    let packed_write_body = if packed_is_once {
        quote! {
            if !is_new_insert {
                return;
            }
            store.mark_gpu_row_dirty(id, row, bytes);
        }
    } else {
        quote! {
            let _ = is_new_insert; // DirtyTracked packed columns re-mark on every insert, first or not
            store.mark_gpu_row_dirty(id, row, bytes);
        }
    };

    let world_mirror_registration_packed = quote! {
        #packed_view_def

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_dispatch_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
            data: *const (),
            is_new_insert: bool,
        ) {
            // SAFETY: same contract as the non-packed dispatch fn below --
            // `data` is guaranteed to point at a live, correctly-aligned
            // `#name` (the sole caller, `World::insert_inner`, only reaches
            // this via `#name`'s own `ComponentId`).
            let data = unsafe { &*(data as *const #name #ty_generics) };
            // Assembled via ordinary field access, NOT a raw byte-offset
            // read from `&#name` -- #name's own field layout is compiler-
            // chosen (no repr(C) forced on it) and may interleave #[gpu]
            // and non-#[gpu] fields in any order, so the packed fields are
            // not generally contiguous within #name itself. Building a
            // fresh #packed_view_ident value by name is what makes packed
            // layout correct regardless of #name's actual layout.
            let packed = #packed_view_ident {
                #(#packed_field_idents: data.#packed_field_idents),*
            };
            let bytes: &[u8] = unsafe {
                ::std::slice::from_raw_parts(
                    &packed as *const #packed_view_ident as *const u8,
                    ::std::mem::size_of::<#packed_view_ident>(),
                )
            };
            let id = ::pulsar_scenedb::component::component_id::<#packed_view_ident>();
            let store = mirror.store();
            #packed_write_body
        }

        ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
            ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                dispatch: #mirror_dispatch_fn_name,
            }
        }
    };

    let world_mirror_registration_default = quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_dispatch_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
            data: *const (),
            is_new_insert: bool,
        ) {
            // SAFETY: the sole caller, `World::insert_inner`, only reaches
            // this function by looking it up under `#name`'s own
            // `ComponentId` (via `crate::component::component_id::<#name>`,
            // the same key this registration is submitted under below), and
            // passes `&value as *const T as *const ()` for that exact `T` --
            // so `data` is guaranteed to point at a live, correctly-aligned
            // `#name`.
            let data = unsafe { &*(data as *const #name #ty_generics) };
            ::pulsar_scenedb::gpu::world_mirror::write_gpu_columns_at_row(
                mirror.store(),
                mirror.queue(),
                row,
                data,
                is_new_insert,
            );
        }

        ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
            ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                dispatch: #mirror_dispatch_fn_name,
            }
        }
    };

    let (register_growable_calls, world_mirror_registration) = if is_packed {
        (vec![register_growable_calls_packed], world_mirror_registration_packed)
    } else {
        (register_growable_calls, world_mirror_registration_default)
    };

    // Only emitted for packed types: the packed view struct is intentionally
    // unnameable from outside this macro's generated code (same reasoning as
    // the per-field `#[gpu]` wrapper types -- see `FieldInfo::gpu_wrapper`'s
    // doc), so this is the supported way to reach its `ComponentId` (and,
    // through `SceneGpuStore::with_growable_buffer_for_id`/`buffer_for_id`,
    // its actual `wgpu::Buffer`) from outside — needed by anything binding
    // this buffer into a shader (the real motivating use, e.g. Helio).
    let packed_accessor = if is_packed {
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                /// `ComponentId` of this type's packed GPU buffer
                /// (registered via [`Self::register_gpu_columns_growable`]).
                pub fn packed_gpu_component_id() -> ::pulsar_scenedb::ComponentId {
                    ::pulsar_scenedb::component::component_id::<#packed_view_ident>()
                }
            }
        }
    } else {
        quote! {}
    };

    quote! {
        impl #impl_generics ::pulsar_scenedb::GpuColumnSet for #name #ty_generics #where_clause {
            fn gpu_columns() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
                vec![
                    #(#column_descs),*
                ]
            }
            fn write_gpu(
                store: &::pulsar_scenedb::gpu::SceneGpuStore,
                id: ::pulsar_scenedb::gpu::CellId,
                cell: &mut ::pulsar_scenedb::cell::CellStorage,
                handle: ::pulsar_scenedb::handle::Handle,
                data: &Self,
                _phase: &impl ::pulsar_scenedb::gpu::SimulateWitness,
            ) {
                let descs = Self::gpu_columns();
                for desc in &descs {
                    match desc.buffer_name {
                        #(#write_arms)*
                        _ => {}
                    }
                }
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            /// Registers this type's `#[gpu]` fields as GPU buffers on
            /// `store` -- one call per field, using the same
            /// disambiguated wrapper types [`Self::write_gpu`] writes
            /// through, so `write_gpu`'s `mark_column_dirty` always finds
            /// a matching buffer instead of silently no-op'ing (the gap
            /// this method exists to close: previously nothing called
            /// `register_gpu_buffer` for derive-generated `#[gpu]` fields
            /// at all).
            ///
            /// Call once per type, at `SceneGpuStore` construction time,
            /// with the same `capacity` every other column on the store
            /// uses (the row-region-partitioned row count -- see
            /// `SceneGpuStore::new`'s own `register_gpu_buffer` calls for
            /// its two built-ins, which this mirrors).
            ///
            /// A field that declares `#[gpu(buffer = "key")]` shares `key`'s
            /// physical buffer with every other field/component that declared
            /// the same key (compat-validated at registration, see
            /// `SceneGpuStore::register_gpu_buffer`); fields without the
            /// attribute each get their own buffer under a key unique to this
            /// (struct, field) pair.
            ///
            /// The `#gpu_ty: ...` bounds are the registration target's
            /// element type requirements (shared-buffer compatibility is
            /// keyed by the field's raw type, not the per-field wrapper) --
            /// only satisfiable when this struct is itself `Pod`, which is
            /// what `#[derive(SceneStore)]`'s generated `Pod` impl requires
            /// anyway.
            pub fn register_gpu_columns(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                capacity: u32,
                device: &::wgpu::Device,
            ) where
                #(#register_field_ty_bounds),*
            {
                #(#register_calls)*
            }

            /// Growable counterpart to [`Self::register_gpu_columns`] --
            /// for World-mirrored use (`World::attach_gpu_mirror`), where
            /// the eventual entity count isn't known ahead of time.
            /// `initial_capacity` only needs to be cheap, not sized for the
            /// eventual world; buffers grow transparently on writes past
            /// their current capacity (`SceneGpuStore::write_row_bytes_growing`,
            /// called automatically by `World::insert`'s dispatch path).
            /// Never sets a `max_capacity` ceiling -- see
            /// `SceneGpuStore::register_growable_gpu_buffer`'s doc for why
            /// that's deliberate for World-mirrored columns specifically.
            /// Same shared-key semantics as [`Self::register_gpu_columns`],
            /// applied through the dirty-tracked World-mirror path.
            pub fn register_gpu_columns_growable(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                initial_capacity: u32,
                device: &::std::sync::Arc<::wgpu::Device>,
            ) where
                #(#register_field_ty_bounds),*
            {
                #(#register_growable_calls)*
            }
        }

        #packed_accessor

        #world_mirror_registration
    }
}
