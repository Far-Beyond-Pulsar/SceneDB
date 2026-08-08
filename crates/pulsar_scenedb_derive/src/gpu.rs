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
            quote! {
                ::pulsar_scenedb::GpuColumnDesc {
                    field_token: ::pulsar_scenedb::token::TypeToken::of::<#wrapper>(),
                    field_offset: ::std::mem::offset_of!(#name, #field_ident),
                    mode: #mirror_mode,
                    buffer_name: #field_name,
                }
            }
        })
        .collect();

    let write_arms: Vec<_> = gpu_fields
        .iter()
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

    let register_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let buffer_label = format!("{}::{}", name, field_name);
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            quote! {
                store.register_gpu_buffer::<#wrapper>(capacity, device, #buffer_label);
            }
        })
        .collect();

    // Per field, registered through whichever of the two growable paths
    // matches its declared `#[gpu(mirror = ...)]` mode: `Once` fields go
    // through the plain growable registration (write-once-ish data, no
    // benefit from dirty-tracking); `DirtyTracked` fields (the default)
    // through the dirty-tracked registration, so `World::flush_gpu_mirror`
    // actually has something registered to flush. Both never set a
    // `max_capacity` ceiling -- see `register_gpu_columns_growable`'s own
    // doc for why that's deliberate for World-mirrored columns.
    let register_growable_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let buffer_label = format!("{}::{}", name, field_name);
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            match f.mirror_mode {
                MirrorModeAttr::Once => quote! {
                    store.register_growable_gpu_buffer::<#wrapper>(initial_capacity, None, device, #buffer_label);
                },
                MirrorModeAttr::DirtyTracked => quote! {
                    store.register_dirty_tracked_gpu_buffer::<#wrapper>(initial_capacity, device, #buffer_label);
                },
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

    let register_growable_calls_packed = {
        let buffer_label = format!("{}::packed", name);
        if packed_is_once {
            quote! {
                store.register_growable_gpu_buffer::<#packed_view_ident>(initial_capacity, None, device, #buffer_label);
            }
        } else {
            quote! {
                store.register_dirty_tracked_gpu_buffer::<#packed_view_ident>(initial_capacity, device, #buffer_label);
            }
        }
    };

    // Packed write body: either immediate (Once -- skip entirely on a later
    // update, matching the non-packed Once behavior exactly) or
    // mark-dirty-and-defer (DirtyTracked -- picked up by the next
    // `World::flush_gpu_mirror`), never both -- `packed_is_once` was
    // validated uniform across every #[gpu] field on this struct above.
    let packed_write_body = if packed_is_once {
        quote! {
            if !is_new_insert {
                return;
            }
            if store.write_row_bytes(id, queue, bytes, row) {
                return;
            }
            match store.write_row_bytes_growing(id, queue, bytes, row) {
                None => {} // not registered at all -- bring-up, not an error
                Some(Ok(())) => {}
                Some(Err(cap_err)) => panic!(
                    "World-mirrored packed GPU column for {} hit its configured \
                     max_capacity ({}) at row {row} (requested {}) -- packed columns \
                     registered via register_gpu_columns_growable never set a ceiling, \
                     so this should be unreachable unless this type's packed buffer was \
                     registered by hand with an explicit max_capacity",
                    stringify!(#name), cap_err.max, cap_err.requested,
                ),
            }
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
            let queue = mirror.queue();
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
            pub fn register_gpu_columns(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                capacity: u32,
                device: &::wgpu::Device,
            ) {
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
            pub fn register_gpu_columns_growable(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                initial_capacity: u32,
                device: &::std::sync::Arc<::wgpu::Device>,
            ) {
                #(#register_growable_calls)*
            }
        }

        #packed_accessor

        #world_mirror_registration
    }
}
