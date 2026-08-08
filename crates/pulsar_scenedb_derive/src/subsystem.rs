//! `#[scenedb_subsystem]` / `#[subsystem_method]`: generate
//! `pulsar_reflection::DynMethodRegistration` entries (link-time
//! `inventory::submit!`) for a `Subsystem` impl block, wiring subsystem
//! methods into Pulsar's real reflection database
//! (`pulsar_reflection::DYN_METHOD_REGISTRY`) instead of a bespoke one.
//!
//! Mirrors `engine_class_derive::component_methods`'s shape closely on
//! purpose — same "walk an `ItemImpl`, pull `MethodMetadata` out of
//! `#[method]`-marked fns, `inventory::submit!` a registration" structure —
//! with one deliberate correction: `#[subsystem_method]` is registered as
//! a real (identity) `#[proc_macro_attribute]` (see `lib.rs`), not left as
//! a bare marker. `component_methods` re-quotes the parsed `ItemImpl`
//! (including each method's original attributes) verbatim into its output;
//! if `#[method]` were not a real attribute anywhere in scope, that output
//! would fail to compile with "cannot find attribute `method` in this
//! scope" the first time anyone actually used it (nothing in this
//! workspace currently does — `rg '#\[component_methods\]'` has no
//! consumers). Registering `subsystem_method` as a real no-op passthrough
//! sidesteps that: it's already a valid attribute, whether or not it also
//! gets stripped.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr, ExprLit, FnArg, ImplItem, ItemImpl, Lit, Meta, Pat, PatType, ReturnType, Type,
    parse::Parser, punctuated::Punctuated,
};

/// `#[scenedb_subsystem(name = "physics")]` on an `impl Foo { .. }` block.
pub fn expand(attr: TokenStream, impl_block: ItemImpl) -> syn::Result<TokenStream> {
    let type_name = match &*impl_block.self_ty {
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .map(|seg| seg.ident.clone())
            .ok_or_else(|| syn::Error::new_spanned(&impl_block.self_ty, "expected a type path"))?,
        other => {
            return Err(syn::Error::new_spanned(other, "expected a type path"));
        }
    };

    let subsystem_name = subsystem_name_from_attr(&attr, &type_name)?;

    let mut method_metadata_items = Vec::new();
    for item in &impl_block.items {
        let ImplItem::Fn(method) = item else {
            continue;
        };
        let Some(attr) = method
            .attrs
            .iter()
            .find(|a| a.path().is_ident("subsystem_method"))
        else {
            continue;
        };

        let method_ident = &method.sig.ident;
        let method_name_str = method_ident.to_string();
        let display_name = capitalize_first(&method_name_str.replace('_', " "));
        let category = category_from_attr(attr)?;
        let category_expr = match category {
            Some(cat) => quote! { Some(#cat) },
            None => quote! { None },
        };

        let mut params = Vec::new();
        for input in &method.sig.inputs {
            if let FnArg::Typed(PatType { pat, ty, .. }) = input {
                if let Pat::Ident(pat_ident) = &**pat {
                    params.push((pat_ident.ident.to_string(), (**ty).clone()));
                }
            }
        }

        let return_type = match &method.sig.output {
            ReturnType::Default => None,
            ReturnType::Type(_, ty) => Some((**ty).clone()),
        };

        let param_metadata: Vec<_> = params
            .iter()
            .map(|(name, ty)| {
                quote! {
                    pulsar_reflection::MethodParameter {
                        name: #name,
                        type_info: <#ty as pulsar_reflection::Reflectable>::type_info(),
                    }
                }
            })
            .collect();

        let return_metadata = if let Some(ret_ty) = &return_type {
            quote! {
                Some(pulsar_reflection::MethodReturnType {
                    type_info: <#ret_ty as pulsar_reflection::Reflectable>::type_info(),
                })
            }
        } else {
            quote! { None }
        };

        // Receiver is always `&mut dyn Any` (`DynMethodCaller`'s shape) —
        // downcast to `&mut #type_name` unconditionally and call through it
        // regardless of whether the method itself takes `&self` or
        // `&mut self`; a `&mut T` can call either.
        let param_reads: Vec<_> = params
            .iter()
            .enumerate()
            .map(|(i, (_, ty))| {
                quote! {
                    {
                        let boxed = __pulsar_args
                            .next()
                            .unwrap_or_else(|| panic!("missing argument at index {}", #i));
                        match boxed.downcast::<#ty>() {
                            Ok(value) => *value,
                            Err(_) => panic!("wrong argument type at index {}", #i),
                        }
                    }
                }
            })
            .collect();

        let result_conversion = if return_type.is_some() {
            quote! { Some(Box::new(result) as Box<dyn std::any::Any>) }
        } else {
            quote! { let _ = result; None }
        };

        let caller = quote! {
            Box::new(|target: &mut dyn std::any::Any, args: pulsar_reflection::DynMethodArgs| {
                let concrete = target.downcast_mut::<#type_name>().expect(concat!(
                    "DynMethodCaller for '",
                    #subsystem_name,
                    "' invoked on the wrong concrete type"
                ));
                let mut __pulsar_args = args.into_iter();
                let result = concrete.#method_ident(#(#param_reads),*);
                #result_conversion
            })
        };

        method_metadata_items.push(quote! {
            pulsar_reflection::DynMethodMetadata {
                name: #method_name_str,
                display_name: #display_name.to_string(),
                category: #category_expr,
                params: vec![#(#param_metadata),*],
                return_type: #return_metadata,
                method_type: pulsar_reflection::MethodType::Fn,
                caller: #caller,
            }
        });
    }

    let registration = if method_metadata_items.is_empty() {
        quote! {}
    } else {
        quote! {
            pulsar_reflection::inventory::submit! {
                pulsar_reflection::DynMethodRegistration {
                    receiver_name: #subsystem_name,
                    methods: || vec![#(#method_metadata_items),*],
                }
            }
        }
    };

    Ok(quote! {
        #impl_block
        #registration
    })
}

fn subsystem_name_from_attr(attr: &TokenStream, type_name: &syn::Ident) -> syn::Result<String> {
    if attr.is_empty() {
        return Ok(type_name.to_string());
    }
    let metas =
        Punctuated::<Meta, syn::Token![,]>::parse_terminated.parse2(attr.clone())?;
    for meta in metas {
        if let Meta::NameValue(nv) = &meta {
            if nv.path.is_ident("name") {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(lit_str),
                    ..
                }) = &nv.value
                {
                    return Ok(lit_str.value());
                }
            }
        }
    }
    Ok(type_name.to_string())
}

fn category_from_attr(attr: &syn::Attribute) -> syn::Result<Option<String>> {
    if matches!(attr.meta, Meta::Path(_)) {
        return Ok(None);
    }
    let mut category = None;
    attr.parse_nested_meta(|nested| {
        if nested.path.is_ident("category") {
            let value = nested.value()?;
            let lit: syn::LitStr = value.parse()?;
            category = Some(lit.value());
        }
        Ok(())
    })?;
    Ok(category)
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
