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
            }
        };
    }

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
        }
    }
}
