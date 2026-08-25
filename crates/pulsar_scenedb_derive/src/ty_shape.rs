//! Recursive syntactic shape analysis for `#[gpu]` field types — the
//! vocabulary that lets [`crate::var_len`]'s lowering rules key off WHERE a
//! cache/dedup boundary sits inside a type signature, not just whether the
//! whole field happens to be one of a fixed set of shapes.
//!
//! `GpuHeavy<T>` marks such a boundary: the real resource lives behind an
//! interned store keyed by content identity, and only the lightweight
//! reference occupies this position. Containers compose structurally and are
//! never silently flattened (`Vec<Vec<..>>` lowers to a chain of
//! chain-handles, since `VarLenHandle` is itself `Pod`) — compositions the
//! var-len lowering does not yet implement are rejected with a compile
//! error naming the supported set, not quietly miscompiled.
//!
//! Syntactic-only, exactly like every other type check in this crate (see
//! `as_vec_elem_type`'s doc in `scene_store.rs`): last-segment ident
//! comparison, no real type resolution available at macro-expansion time.

use syn::Type;

pub(crate) enum TyShape {
    /// `GpuHeavy<handle>` — `inner` is the lightweight Pod reference type,
    /// unwrapped.
    Heavy(Type),
    /// `Vec<inner>` — a variable-length chain of lowered elements.
    List(Box<TyShape>),
    /// `Option<inner>` — a reference slot with a NULL sentinel
    /// ([`super::ty_shape`] consumers require `inner`'s handle to implement
    /// `pulsar_scenedb::gpu::GpuRef`).
    Optional(Box<TyShape>),
    /// `Result<ok, err>` — a discriminant plus both sides lowered.
    Either(Box<TyShape>, Box<TyShape>),
    /// Everything else — an opaque Pod leaf (scalar or plain handle).
    Leaf(Type),
}

pub(crate) fn analyze(ty: &Type) -> TyShape {
    if let Some(handle) = as_heavy_inner_type(ty) {
        return TyShape::Heavy(handle);
    }
    if let Some(elem) = as_vec_elem_type_pub(ty) {
        return TyShape::List(Box::new(analyze(&elem)));
    }
    if let Some(inner) = as_option_inner_type(ty) {
        return TyShape::Optional(Box::new(analyze(&inner)));
    }
    if let Some((ok, err)) = as_result_tys(ty) {
        return TyShape::Either(Box::new(analyze(&ok)), Box::new(analyze(&err)));
    }
    TyShape::Leaf(ty.clone())
}

impl TyShape {
    /// Whether any `GpuHeavy` boundary sits anywhere inside this signature.
    pub(crate) fn contains_heavy(&self) -> bool {
        match self {
            TyShape::Heavy(_) => true,
            TyShape::List(inner) | TyShape::Optional(inner) => inner.contains_heavy(),
            TyShape::Either(ok, err) => ok.contains_heavy() || err.contains_heavy(),
            TyShape::Leaf(_) => false,
        }
    }

    /// The `H` behind a top-level `Heavy`, when this shape is exactly
    /// `Heavy(H)` / `Optional(Heavy(H))` / one side of an `Either` pair of
    /// heavies — the three placements the var-len lowering implements.
    pub(crate) fn heavy_handle(&self) -> Option<Type> {
        match self {
            TyShape::Heavy(h) => Some(h.clone()),
            TyShape::Optional(inner) => inner.heavy_handle(),
            _ => None,
        }
    }
}

/// `scene_store.rs` already owns the canonical `Vec<T>` syntactic check;
/// re-exported here under the analyzer's own naming so shape code reads
/// uniformly.
pub(crate) fn as_vec_elem_type_pub(ty: &Type) -> Option<Type> {
    super::scene_store::as_vec_elem_type(ty)
}

fn as_heavy_inner_type(ty: &Type) -> Option<Type> {
    let Type::Path(type_path) = ty else { return None };
    let last = type_path.path.segments.last()?;
    if last.ident != "GpuHeavy" {
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

fn as_option_inner_type(ty: &Type) -> Option<Type> {
    let Type::Path(type_path) = ty else { return None };
    let last = type_path.path.segments.last()?;
    if last.ident != "Option" {
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

fn as_result_tys(ty: &Type) -> Option<(Type, Type)> {
    let Type::Path(type_path) = ty else { return None };
    let last = type_path.path.segments.last()?;
    if last.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else { return None };
    if args.args.len() != 2 {
        return None;
    }
    let mut tys = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    });
    Some((tys.next()?, tys.next()?))
}
