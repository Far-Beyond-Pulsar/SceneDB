use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::Parse, Data, DeriveInput, Fields, Ident, Type,
};

use crate::cell::generate_scene_column_set;
use crate::gpu::generate_gpu_column_set;

// ── #[gpu] attribute parsing ──────────────────────────────────────────────

pub struct GpuAttr {
    pub mirror_mode: Option<MirrorModeAttr>,
    /// The buffer key a `#[gpu(buffer = "...")]` field declares; `None` when
    /// the field uses the default one-buffer-per-field split (the derive then
    /// derives a key unique to this (struct, field) pair).
    pub buffer_key: Option<String>,
    /// `#[gpu(mirror = Once, heavy)]` -- a bare flag (no `= value`). Declares
    /// that this field's type implements `GpuUploadSource`: the CPU column
    /// stores the field's own (lightweight handle) type, but the GPU buffer's
    /// element is `<FieldTy as GpuUploadSource>::Element` instead, populated
    /// via the trait's `upload_element`. Only valid alongside `mirror = Once`
    /// -- checked at macro-expansion time, not here (this struct doesn't see
    /// the field's default mirror mode when `mirror` is omitted).
    pub heavy: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MirrorModeAttr {
    DirtyTracked,
    Once,
}

impl Parse for GpuAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut mirror_mode = None;
        let mut buffer_key = None;
        let mut heavy = false;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "mirror" => {
                    let _: syn::Token![=] = input.parse()?;
                    let mode: Ident = input.parse()?;
                    let mode = match mode.to_string().as_str() {
                        "DirtyTracked" => MirrorModeAttr::DirtyTracked,
                        "Once" => MirrorModeAttr::Once,
                        _ => {
                            return Err(syn::Error::new(
                                mode.span(),
                                "expected DirtyTracked or Once",
                            ))
                        }
                    };
                    mirror_mode = Some(mode);
                }
                "buffer" => {
                    let _: syn::Token![=] = input.parse()?;
                    let lit: syn::LitStr = input.parse()?;
                    buffer_key = Some(lit.value());
                }
                "heavy" => {
                    // Bare flag -- no `= value`, unlike `mirror`/`buffer`.
                    heavy = true;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[gpu] option `{other}` (expected `mirror`, `buffer`, or `heavy`)"),
                    ))
                }
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            } else {
                break;
            }
        }
        Ok(GpuAttr { mirror_mode, buffer_key, heavy })
    }
}

// ── Struct-level `#[gpu(layout = packed)]` attribute parsing ───────────────
//
// A separate, struct-level use of the same `gpu` attribute name as the
// per-field one above -- no ambiguity, since `syn`/the derive macro reads
// struct attrs (`DeriveInput::attrs`) and field attrs (`Field::attrs`)
// through entirely separate code paths.

pub struct StructGpuAttr {
    pub layout_packed: bool,
    /// Optional struct-level `buffer = "..."` key for packed layout — the
    /// key under which the single interleaved packed buffer is registered
    /// (defaults to `{Type}::packed`). Only meaningful alongside
    /// `layout = packed`.
    pub buffer_key: Option<String>,
}

impl Parse for StructGpuAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut layout_packed = false;
        let mut buffer_key = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: syn::Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "layout" => {
                    let value: Ident = input.parse()?;
                    if value != "packed" {
                        return Err(syn::Error::new(value.span(), "expected `packed` -- the only supported layout today"));
                    }
                    layout_packed = true;
                }
                "buffer" => {
                    let lit: syn::LitStr = input.parse()?;
                    buffer_key = Some(lit.value());
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[gpu] option `{other}` (expected `layout` or `buffer`)"),
                    ))
                }
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            } else {
                break;
            }
        }
        Ok(StructGpuAttr { layout_packed, buffer_key })
    }
}

/// Scans a struct's own attributes (not its fields') for `#[gpu(...)]`,
/// returning the parsed [`StructGpuAttr`].
///
/// Packed layout groups every `#[gpu]` field into ONE GPU buffer (a single
/// interleaved record per row) instead of the default one-buffer-per-field
/// split -- for structs, like a renderer's per-instance GPU record, whose
/// `#[gpu]` fields are always read together and were never independent
/// columns to begin with. Deliberately scoped to the World-mirror path only
/// (`register_gpu_columns_growable` + `World::insert`'s automatic dispatch):
/// it does NOT change `gpu_columns()`, `write_gpu`, or the fixed
/// `register_gpu_columns` at all -- those stay exactly as they are for
/// EVERY `#[derive(SceneStore)]` type, packed or not, because the
/// cell-mirrored path's dirty-tracked boundary sync reads FROM CellStorage's
/// own per-field SoA columns, which packing has no relationship to (packing
/// only changes what shape of buffer the data is written *into* on the GPU
/// side, not how it's stored on the CPU side). Requires at least one
/// `#[gpu]` field to have any effect -- a packed struct with none behaves
/// identically to one without the attribute at all (nothing to pack).
///
/// A struct-level `buffer = "..."` names the packed buffer's shared key;
/// without it the packed buffer registers under `{Type}::packed`. Sharing a
/// packed key across two different structs is only accepted when their
/// element types agree, and the registered element type for a packed record
/// is the struct itself (`#name`), so in practice a packed key is shared
/// only by the exact same struct type -- document this if you rely on it.
///
/// Lenient on parse failure (a bare `#[gpu]` with no `(...)` at all, or
/// unrecognized content), matching the existing per-field `#[gpu]`
/// parsing's own tolerance (`scene_store::expand`'s field loop: `if let
/// Ok(...) = attr.parse_args()`) rather than hard-erroring -- consistent
/// behavior for the same attribute name used at two different syntactic
/// positions in this macro.
pub fn struct_gpu_attr(attrs: &[syn::Attribute]) -> StructGpuAttr {
    attrs.iter().find(|attr| attr.path().is_ident("gpu")).and_then(|attr| attr.parse_args::<StructGpuAttr>().ok()).unwrap_or(StructGpuAttr { layout_packed: false, buffer_key: None })
}

// ── Per-field metadata ────────────────────────────────────────────────────

pub struct FieldInfo {
    pub ident: Ident,
    pub ty: Type,
    pub is_gpu: bool,
    pub mirror_mode: MirrorModeAttr,
    /// The declared `buffer = "..."` key from a `#[gpu(buffer = "...")]`
    /// field, or `None` for the default one-buffer-per-field split. See the
    /// `GpuBufferRegistry` doc for what sharing a key means and the
    /// compatibility rules that apply to it.
    pub buffer_key: Option<String>,
    /// `#[gpu(mirror = Once, heavy)]` -- see [`GpuAttr::heavy`]'s doc.
    /// Validated (macro-expansion-time `compile_error!`, not a runtime
    /// panic) to only appear alongside `mirror_mode == Once` in
    /// `generate_gpu_column_set`, since `FieldInfo` itself doesn't know
    /// whether `mirror_mode` came from an explicit `mirror = ...` or the
    /// default.
    pub heavy: bool,
    /// Present iff `is_gpu`. `ComponentId`/`TypeToken` (this crate's GPU
    /// buffer + CPU-column keys) are derived from a Rust `TypeId`, globally
    /// — keyed by the field's own raw type, they carry no notion of which
    /// *struct* the field belongs to. Two different `#[derive(SceneStore)]`
    /// types both having, say, an `f32` field marked `#[gpu]` would
    /// otherwise resolve to the exact same `ComponentId`, and the second
    /// type's `register_gpu_buffer::<f32>()` call would silently replace
    /// the first's GPU buffer outright (`SceneGpuStore::register_gpu_buffer`
    /// does a plain `HashMap::insert`, no collision check) — not a data
    /// corruption in the row-range sense (each cell's rows are disjoint,
    /// per `RegionPool`), but a semantic one: "the roughness buffer" and
    /// "the intensity buffer" would silently be the same physical buffer,
    /// interleaved by row region, which is never what marking two
    /// unrelated fields `#[gpu]` is asking for.
    ///
    /// Fixed by generating one `#[repr(transparent)]` newtype wrapper per
    /// `#[gpu]` field (`__ScenedbGpuCol_<Struct>_<Field>`, byte-identical
    /// to the field's own type) and using *that* — not the raw field type
    /// — as the column's registered type everywhere: `GpuColumnDesc::
    /// field_token`, the `write_gpu`-generated `component_id::<_>()` call,
    /// and (when the `gpu` feature is on) the `SceneColumnSet`-generated
    /// `CellType` column token. A wrapper's own `TypeId` is unique to its
    /// (struct, field) pair by construction, so two `#[gpu] f32` fields on
    /// different structs get two distinct, collision-free `ComponentId`s
    /// even though their underlying data is the same shape.
    pub gpu_wrapper: Option<Ident>,
}

// ── Entry point ───────────────────────────────────────────────────────────

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(ds) => match &ds.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "SceneStore requires named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "SceneStore only supports structs",
            ))
        }
    };

    let mut field_infos: Vec<FieldInfo> = Vec::new();
    for field in fields {
        let ident = field.ident.as_ref().unwrap().clone();
        let ty = field.ty.clone();
        let mut is_gpu = false;
        let mut mirror_mode = MirrorModeAttr::DirtyTracked;
        let mut buffer_key: Option<String> = None;
        let mut heavy = false;

        for attr in &field.attrs {
            if attr.path().is_ident("gpu") {
                is_gpu = true;
                if let Ok(gpu_attr) = attr.parse_args::<GpuAttr>() {
                    if let Some(mode) = gpu_attr.mirror_mode {
                        mirror_mode = mode;
                    }
                    if let Some(key) = gpu_attr.buffer_key {
                        buffer_key = Some(key);
                    }
                    heavy = gpu_attr.heavy;
                }
            }
        }

        // Not generics-aware (see doc on `gpu_wrapper`'s uniqueness
        // reasoning): a generic `SceneStore` struct instantiated at two
        // different type parameters would generate the SAME wrapper ident
        // for both instantiations. Named `#[derive(SceneStore)]` structs
        // in practice are concrete GPU-data structs (this crate's own
        // built-ins included), not generic over their `#[gpu]` fields'
        // types, so this covers the real cases; a future fix for the
        // generic case would fold `ty_generics` into the wrapper name.
        let gpu_wrapper = is_gpu.then(|| {
            Ident::new(
                &format!("__ScenedbGpuCol_{}_{}", name, ident),
                ident.span(),
            )
        });

        field_infos.push(FieldInfo {
            ident,
            ty,
            is_gpu,
            mirror_mode,
            buffer_key,
            heavy,
            gpu_wrapper,
        });
    }

    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "SceneStore requires at least one field",
        ));
    }

    let field_types: Vec<&Type> = field_infos.iter().map(|f| &f.ty).collect();
    let gpu_fields: Vec<&FieldInfo> = field_infos.iter().filter(|f| f.is_gpu).collect();

    let pod_impl = generate_pod_impl(name, &impl_generics, &ty_generics, where_clause, &field_types);

    // Two `SceneColumnSet` impls, `cfg`-split on the `gpu` feature: with it
    // on, `#[gpu]` fields' CellType column tokens must match the wrapper
    // types `write_gpu`/`GpuColumnDesc` use (see `gpu_wrapper`'s doc) or
    // `cell.column_for_mut::<Wrapper>()` would find no column; with it off
    // there is no GPU column concept at all, so every field (including ones
    // marked `#[gpu]`, which is a no-op without the feature) keeps its own
    // natural type -- unchanged from before this fix.
    let scene_column_set_gpu = generate_scene_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &field_infos,
        true,
    );
    let scene_column_set_no_gpu = generate_scene_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &field_infos,
        false,
    );

    let gpu_wrapper_defs: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            let ty = &f.ty;
            quote! {
                // Byte-identical to #ty (repr(transparent), single field) --
                // exists solely to give this field's GPU column a TypeId
                // unique to (this struct, this field). See `FieldInfo::
                // gpu_wrapper`'s doc for why that's load-bearing.
                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                #[repr(transparent)]
                #[derive(Clone, Copy)]
                pub struct #wrapper(pub #ty);
                unsafe impl ::pulsar_scenedb::page::Pod for #wrapper {}
            }
        })
        .collect();

    let struct_gpu_attr = struct_gpu_attr(&input.attrs);
    let is_packed = struct_gpu_attr.layout_packed;
    let gpu_column_set = generate_gpu_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &gpu_fields,
        is_packed,
        struct_gpu_attr.buffer_key.as_deref(),
    );
    // NOTE: HasTypeToken is NOT generated here — the blanket impl in
    // `pulsar_scenedb::token` covers `T: Pod + 'static`, which our Pod impl
    // satisfies.  An explicit impl would conflict.

    Ok(quote! {
        #pod_impl

        #[cfg(feature = "gpu")]
        const _: () = {
            #(#gpu_wrapper_defs)*
            #scene_column_set_gpu
            #gpu_column_set
        };
        #[cfg(not(feature = "gpu"))]
        #scene_column_set_no_gpu
    })
}

// ── Pod impl ──────────────────────────────────────────────────────────────

fn generate_pod_impl(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    field_types: &[&Type],
) -> TokenStream {
    let pod_bounds: Vec<_> = field_types
        .iter()
        .map(|ty| {
            quote! { #ty: ::pulsar_scenedb::page::Pod }
        })
        .collect();

    let mut wc: syn::WhereClause = where_clause
        .cloned()
        .unwrap_or_else(|| syn::WhereClause {
            where_token: Default::default(),
            predicates: syn::punctuated::Punctuated::new(),
        });

    for bound in &pod_bounds {
        let pred: syn::WherePredicate = syn::parse_quote! { #bound };
        wc.predicates.push(pred);
    }

    quote! {
        unsafe impl #impl_generics ::pulsar_scenedb::page::Pod for #name #ty_generics #wc {}
    }
}
