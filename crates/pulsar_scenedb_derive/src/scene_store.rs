use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::Parse, Data, DeriveInput, Fields, Ident, Type,
};

use crate::cell::generate_scene_column_set;
use crate::gpu::generate_gpu_column_set;
use crate::ty_shape::{self, TyShape};

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
    /// `#[gpu(content_id = "sibling_field")]` on a `Vec<T>` field -- names a
    /// SIBLING field (by ident, on the same struct) whose type implements
    /// `pulsar_scenedb::handle_ledger::ContentAddressed`. Routes this
    /// field's var-len GPU allocation through the content-id-interned pool
    /// (`gpu::interned_pool`) instead of the plain per-row one: rows sharing
    /// the sibling's content id share ONE allocation, refcounted, freed at
    /// zero. Only valid on a `Vec<T>` `#[gpu]` field -- checked in
    /// `var_len.rs` (which is where `is_var_len` is known), not here.
    pub content_id: Option<syn::Ident>,
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
        let mut content_id = None;
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
                "content_id" => {
                    let _: syn::Token![=] = input.parse()?;
                    let lit: syn::LitStr = input.parse()?;
                    content_id = Some(Ident::new(&lit.value(), lit.span()));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[gpu] option `{other}` (expected `mirror`, `buffer`, `heavy`, or `content_id`)"
                        ),
                    ))
                }
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            } else {
                break;
            }
        }
        Ok(GpuAttr { mirror_mode, buffer_key, heavy, content_id })
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
    ///
    /// For a variable-length field ([`Self::is_var_len`]), this wraps
    /// [`crate::var_len::VAR_LEN_HANDLE_PATH`]'s type (`VarLenHandle`), not
    /// the field's own `Vec<T>` type — see that module's doc for why the
    /// wrapper always has to be something `Pod`, and a `Vec<T>` itself never
    /// is.
    pub gpu_wrapper: Option<Ident>,
    /// `true` iff the field's declared type is syntactically `Vec<_>` (any
    /// path ending in a `Vec` segment with exactly one angle-bracketed type
    /// argument — not resolved against the real type, this crate has no
    /// access to that at macro-expansion time, same limitation every other
    /// syntactic check in this file already has). Only meaningful when
    /// [`Self::is_gpu`] is also `true` — a plain (non-`#[gpu]`) `Vec<T>`
    /// field is just an ordinary CPU-only field, nothing this crate cares
    /// about. See `crate::var_len` for what routing a field through this
    /// flag actually generates.
    pub is_var_len: bool,
    /// `true` iff the field's declared type is syntactically `HandleId`
    /// (any path ending in a bare `HandleId` segment with no type
    /// arguments). Same detection philosophy as [`Self::is_var_len`] below
    /// -- a last-segment name match, because macro expansion has no real
    /// type information to resolve against; see `as_handle_id_field`'s doc
    /// for why that's the established trade here too. Unlike var-len,
    /// handle fields are perfectly ordinary `Pod` data: detection does NOT
    /// reroute the struct onto another codegen path, it only adds ONE extra
    /// link-time registration (the ledger event collector) alongside every
    /// classic/var-len artifact the struct already gets.
    pub is_handle: bool,
    /// The `T` in `Vec<T>`, present iff [`Self::is_var_len`]. For a
    /// Heavy-placement field this is the LOWERED pool element (the handle
    /// type behind `GpuHeavy`), not the field's declared element.
    pub var_len_elem_ty: Option<Type>,
    /// Which lowering this var-len field uses — see [`VarLenShape`].
    pub var_len_shape: VarLenShape,
    /// The `A` behind `Vec<Result<GpuHeavy<A>, _>>`, present only for
    /// [`VarLenShape::HeavyEither`].
    pub either_ok_ty: Option<Type>,
    /// The `B` behind `Vec<Result<_, GpuHeavy<B>>>`, same condition.
    pub either_err_ty: Option<Type>,
    /// See [`GpuAttr::content_id`]'s doc -- carried through unchanged, plus
    /// the RESOLVED sibling field's own type (looked up by ident against
    /// the struct's other fields once every field has been scanned; `None`
    /// until `var_len.rs`'s validation pass fills it in, alongside checking
    /// the sibling actually exists and this field is `is_var_len`).
    pub content_id_field: Option<Ident>,
}

/// HOW a var-len field's element type lowers into its GPU pool element —
/// the per-shape branch every registration/write/release/accessor site in
/// `var_len.rs` switches on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VarLenShape {
    /// `Vec<T>` with a Heavy-free element: pool element IS `T`, written by
    /// passing the field's own slice straight through (today's path,
    /// byte-identical output).
    Plain,
    /// `Vec<GpuHeavy<H>>`: pool element is `H`; the write reinterprets the
    /// field's slice in place (`repr(transparent)` + `H: Pod` makes the
    /// bytes identical — zero allocation).
    HeavyHandle,
    /// `Vec<Option<GpuHeavy<H>>>`: pool element is `H` with `H: GpuRef`;
    /// the write maps `None -> H::NULL` into a small per-write buffer.
    HeavyOptionHandle,
    /// `Vec<Result<GpuHeavy<A>, GpuHeavy<B>>>`: THREE pools — `{key}::tag`
    /// (`u32`, 0=Ok/1=Err), `{key}::ok` (A), `{key}::err` (B). Per-row
    /// invariant: `len(ok) + len(err) == vec.len()`.
    HeavyEither,
}

/// Returns `Some(T)` if `ty` is syntactically `Vec<T>` (any path whose last
/// segment is literally named `Vec`, with exactly one angle-bracketed type
/// argument) — a syntactic check, not a real type-resolution one (macro
/// expansion has no type information to resolve against). A field typed
/// `some_other_crate::NotAVec<T>` that happens to also be named `Vec` would
/// be (incorrectly) detected here; not a real-world concern in practice
/// (nothing in this crate's own field types, or any consumer seen so far,
/// shadows the name), and the alternative (requiring the literal path
/// `std::vec::Vec` or `alloc::vec::Vec`) would reject the overwhelmingly
/// common bare `Vec<T>` spelling most callers actually write.
pub(crate) fn as_vec_elem_type(ty: &Type) -> Option<Type> {
    let Type::Path(type_path) = ty else { return None };
    let last = type_path.path.segments.last()?;
    if last.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else { return None };
    if args.args.len() != 1 {
        return None;
    }
    match args.args.first()? {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    }
}

/// The macro-expansion-time rejection for a `#[gpu] Vec<..>` element whose
/// Heavy placement composes deeper than the var-len lowering implements.
/// Enumerates the supported set verbatim so the error is self-documenting.
fn unsupported_heavy_composition_error(field: &Ident) -> syn::Error {
    syn::Error::new_spanned(
        field,
        "unsupported #[gpu] Heavy composition. Supported placements: \
         GpuHeavy<H>, Vec<GpuHeavy<H>>, Vec<Option<GpuHeavy<H>>>, \
         Vec<Result<GpuHeavy<A>, GpuHeavy<B>>> -- structural nesting beyond \
         one container over a Heavy boundary is deliberately not implemented",
    )
}

/// Whether `ty` is syntactically `HandleId` (a path whose last segment is
/// literally named `HandleId`, with NO angle-bracketed arguments -- the
/// type is a plain non-generic newtype, so any generic arguments at all
/// mean it isn't the handle type). The same syntactic trade
/// `as_vec_elem_type` documents: a user type shadowing the name would be
/// misdetected, which nothing in practice does, and requiring the full
/// `pulsar_scenedb::handle_ledger::HandleId` spelling would reject the bare
/// import every real caller writes.
fn as_handle_id_field(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else { return false };
    let Some(last) = type_path.path.segments.last() else { return false };
    if last.ident != "HandleId" {
        return false;
    }
    matches!(last.arguments, syn::PathArguments::None)
}

/// Generates the link-time handle-ledger registration for a struct with at
/// least one `HandleId`-typed field: one pure collector fn (copies each
/// handle field's current value onto the caller's buffer, declaration
/// order) plus an `inventory::submit!`'d
/// [`pulsar_scenedb::handle_ledger::HandleLedgerRegistration`] under the
/// struct's own `ComponentId`. Mirrors `gpu.rs`'s world-mirror dispatch
/// registration shape exactly -- same inventory mechanism, same
/// non-generic-fn-with-concrete-T reasoning (see `gpu::world_mirror`'s
/// module doc for why that's required inside `World::insert_inner`'s
/// generic body).
///
/// Deliberately NOT gated behind the `gpu` feature: handles are a
/// domain-neutral concept with no wgpu dependency, and the registration is
/// pure link-time metadata -- a type with handle fields in a
/// `--no-default-features` build still gets its collector, and it simply
/// never fires unless someone attaches a ledger. Empty input (no handle
/// fields) produces an empty stream: such structs submit NOTHING and are
/// indistinguishable, cost-wise, from types the derive never saw.
fn generate_handle_registration(
    name: &Ident,
    ty_generics: &syn::TypeGenerics,
    field_infos: &[FieldInfo],
) -> TokenStream {
    let handle_fields: Vec<&Ident> = field_infos
        .iter()
        .filter(|f| f.is_handle)
        .map(|f| &f.ident)
        .collect();
    if handle_fields.is_empty() {
        return quote! {};
    }

    let collect_fn_name = quote::format_ident!("__scenedb_handle_collect_{}", name);
    quote! {
        const _: () = {
            #[doc(hidden)]
            #[allow(non_snake_case)]
            fn #collect_fn_name(
                value: *const (),
                out: &mut ::std::vec::Vec<::pulsar_scenedb::handle_ledger::HandleId>,
            ) {
                // SAFETY: `CollectHandlesFn`'s own contract -- the sole
                // caller (`World`'s four handle-reporting sites) only ever
                // passes a pointer obtained from a live, correctly-aligned
                // `#name` (column element, moved-out removal value, or the
                // guard target), reached via this exact registration's
                // `component_id`.
                let data = unsafe { &*(value as *const #name #ty_generics) };
                #( out.push(::pulsar_scenedb::handle_ledger::HandleId(data.#handle_fields.0)); )*
            }

            ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
                ::pulsar_scenedb::handle_ledger::HandleLedgerRegistration {
                    component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                    collect_from_value: #collect_fn_name,
                }
            }
        };
    }
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
        let mut content_id_field: Option<Ident> = None;

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
                    content_id_field = gpu_attr.content_id;
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

        let mut var_len_elem_ty = is_gpu.then(|| as_vec_elem_type(&ty)).flatten();
        let is_var_len = var_len_elem_ty.is_some();

        // Heavy-placement lowering (Phase 3): classify HOW this var-len
        // field's element reaches its GPU pool. A Heavy-free element keeps
        // today's exact behavior — pool element IS the declared element,
        // whatever it is (including its existing compile errors when that
        // element was never Pod to begin with). A Heavy-containing element
        // must match one of the four supported placements exactly;
        // anything deeper is a compile error naming the supported set, not
        // a quiet miscompile.
        let mut var_len_shape = VarLenShape::Plain;
        let mut either_ok_ty: Option<Type> = None;
        let mut either_err_ty: Option<Type> = None;
        if is_gpu && is_var_len {
            let shape = ty_shape::analyze(var_len_elem_ty.as_ref().expect("is_var_len implies elem"));
            if shape.contains_heavy() {
                match shape {
                    TyShape::Heavy(handle) => {
                        var_len_shape = VarLenShape::HeavyHandle;
                        var_len_elem_ty = Some(handle);
                    }
                    TyShape::Optional(inner) => match *inner {
                        TyShape::Heavy(handle) => {
                            var_len_shape = VarLenShape::HeavyOptionHandle;
                            var_len_elem_ty = Some(handle);
                        }
                        _ => return Err(unsupported_heavy_composition_error(&ident)),
                    },
                    TyShape::Either(ok, err) => match (*ok, *err) {
                        (TyShape::Heavy(a), TyShape::Heavy(b)) => {
                            var_len_shape = VarLenShape::HeavyEither;
                            either_ok_ty = Some(a.clone());
                            either_err_ty = Some(b);
                            // The tag pool's element is u32; the ok/err
                            // pools' elements are `a`/`b` — tracked in the
                            // pair fields above, so the shared elem slot
                            // carries `a`.
                            var_len_elem_ty = Some(a);
                        }
                        _ => return Err(unsupported_heavy_composition_error(&ident)),
                    },
                    TyShape::List(_) => return Err(unsupported_heavy_composition_error(&ident)),
                    TyShape::Leaf(_) => unreachable!("contains_heavy true on a Leaf"),
                }
            }
        }

        // Handle detection is independent of `#[gpu]` and of the var-len
        // fork: a `HandleId` field is ordinary Pod data wherever it
        // appears. It only ever ADDS the ledger registration below.
        let is_handle = as_handle_id_field(&ty);

        field_infos.push(FieldInfo {
            ident,
            ty,
            is_gpu,
            mirror_mode,
            buffer_key,
            heavy,
            gpu_wrapper,
            is_var_len,
            is_handle,
            var_len_elem_ty,
            content_id_field,
            var_len_shape,
            either_ok_ty,
            either_err_ty,
        });
    }

    // `content_id = "..."` only has a meaning on a `Vec<T>` `#[gpu]` field
    // (see `GpuAttr::content_id`'s doc) -- reject it elsewhere at
    // macro-expansion time rather than silently ignoring it (a field typo'd
    // onto a scalar column would otherwise look accepted and just never do
    // anything, the worst kind of silent no-op).
    for f in &field_infos {
        if f.content_id_field.is_some() && !f.is_var_len {
            let field_name = &f.ident;
            return Err(syn::Error::new_spanned(
                field_name,
                "#[gpu(content_id = \"...\")] is only valid on a Vec<T> #[gpu] field -- \
                 it names the content-identity source for that field's interned var-len GPU pool",
            ));
        }
    }

    // Emitted for BOTH codegen paths below (classic and var-len-bearing):
    // the ledger collector + link-time registration, or nothing at all for
    // structs without handle fields. Computed before the var-len early
    // return so both branches share it -- one source, no drift.
    let handle_registration = generate_handle_registration(name, &ty_generics, &field_infos);

    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "SceneStore requires at least one field",
        ));
    }

    // A struct with any `Vec<T>`-typed `#[gpu]` field forks onto a
    // completely separate codegen path (`crate::var_len`) -- see that
    // module's doc for why: `GpuColumnSet`/`SceneColumnSet` both require
    // `Self: Pod` (`gpu/scene_store.rs`'s and `cell_type.rs`'s own trait
    // bounds), and a `Vec<T>` field structurally can never be `Pod` (a
    // heap pointer + length + capacity is not a memcpy-safe byte pattern),
    // so the WHOLE-STRUCT `Pod`/`SceneColumnSet`/`GpuColumnSet` impls this
    // function generates below are simply not implementable for such a
    // struct -- not a limitation to work around, a real soundness
    // boundary. The var-len path generates a smaller, World-mirror-only
    // surface instead (no cell-mirrored/`CellStorage` support at all for a
    // struct with a `Vec<T>` field -- that field only ever makes sense
    // World-mirrored, same scoping already established for `heavy`
    // fields).
    if field_infos.iter().any(|f| f.is_var_len) {
        let struct_gpu_attr = struct_gpu_attr(&input.attrs);
        if struct_gpu_attr.layout_packed {
            return Err(syn::Error::new_spanned(
                name,
                "#[gpu(layout = packed)] is not supported on a struct with a Vec<T> #[gpu] field -- \
                 packed layout is a fixed-size-record concept, which a variable-length field has no \
                 meaningful interpretation under",
            ));
        }
        return crate::var_len::generate_var_len_bearing_type(
            name,
            &impl_generics,
            &ty_generics,
            where_clause,
            &field_infos,
        )
        .map(|tokens| {
            // The var-len struct's handle registration rides on the same
            // output -- see `generate_handle_registration`'s doc for why
            // this is unconditional (not gpu-gated) like the collector's
            // runtime it targets.
            quote! { #tokens #handle_registration }
        });
    }

    let field_types: Vec<&Type> = field_infos.iter().map(|f| &f.ty).collect();
    let gpu_fields: Vec<&FieldInfo> = field_infos.iter().filter(|f| f.is_gpu).collect();

    let pod_impl = generate_pod_impl(name, &impl_generics, &ty_generics, where_clause, &field_types);

    // A `SceneStore` derive expands in its *consumer* crate, so it cannot
    // inspect SceneDB's dependency features with `cfg(feature = "gpu")`:
    // that predicate checks the consumer's feature set. A type with `#[gpu]`
    // fields must always generate its wrappers, packed view, and mirror
    // registration together. Types without GPU fields retain the lean
    // CPU-only column-set implementation.
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

    let scene_column_set = if gpu_fields.is_empty() {
        scene_column_set_no_gpu
    } else {
        scene_column_set_gpu
    };

    let gpu_expansion = if gpu_fields.is_empty() {
        quote! {}
    } else {
        quote! {
            #(#gpu_wrapper_defs)*
            #gpu_column_set
        }
    };

    Ok(quote! {
        #pod_impl

        #handle_registration

        #scene_column_set

        #gpu_expansion
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
