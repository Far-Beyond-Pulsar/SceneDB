//! `#[component_methods]`: reflected and world-receiving methods of a
//! SceneDB component type. Signature handling is shared with
//! `pulsar_reflection`'s `#[reflect_methods]` via
//! `pulsar_reflection_codegen`.

use proc_macro2::TokenStream;
use pulsar_reflection_codegen::{
    reflected_registration, take_marked, type_ref, MethodSpec, SelfKind,
};
use quote::{format_ident, quote};
use syn::ItemImpl;

pub fn expand(attr: TokenStream, mut item: ItemImpl) -> syn::Result<TokenStream> {
    if !attr.is_empty() {
        return Err(syn::Error::new_spanned(
            attr,
            "#[component_methods] takes no arguments",
        ));
    }
    if let Some((_, path, _)) = &item.trait_ {
        return Err(syn::Error::new_spanned(
            path,
            "#[component_methods] goes on an inherent impl block, not a trait impl",
        ));
    }
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "#[component_methods] needs a concrete component type",
        ));
    }

    let reflected = take_marked(&mut item.items, "reflect_method")
        .into_iter()
        .map(|(func, marker)| MethodSpec::parse(func, &marker, 0))
        .collect::<syn::Result<Vec<_>>>()?;
    let world = take_marked(&mut item.items, "world_method")
        .into_iter()
        .map(|(func, marker)| {
            let spec = MethodSpec::parse(func, &marker, 2)?;
            if spec.self_kind != SelfKind::None {
                return Err(syn::Error::new_spanned(
                    &func.sig,
                    "#[world_method]s take `(world, entity, ..)`, not `self`",
                ));
            }
            Ok(spec)
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let s = quote!(::pulsar_scenedb);
    let r = quote!(#s::pulsar_reflection);
    let self_ty = &item.self_ty;

    let reflected = if reflected.is_empty() {
        quote!()
    } else {
        reflected_registration(&r, self_ty, &reflected)
    };

    let world = if world.is_empty() {
        quote!()
    } else {
        let const_name = format_ident!("__PULSAR_WORLD_METHODS_{}", world[0].ident);
        let args = format_ident!("args");
        let shims = world.iter().map(|spec| {
            let shim = format_ident!("__pulsar_world_invoke_{}", spec.ident);
            let ident = &spec.ident;
            let extract = spec.extract_args(&r, &args);
            let call = spec.call(&r, quote!(Self::#ident), &[quote!(world), quote!(entity)]);
            quote! {
                #[doc(hidden)]
                fn #shim(
                    world: &mut #s::World,
                    entity: #s::Entity,
                    #args: &mut [::std::boxed::Box<dyn ::std::any::Any>],
                ) -> ::std::result::Result<
                    ::std::option::Option<::std::boxed::Box<dyn ::std::any::Any>>,
                    #r::methods::CallError,
                > {
                    #extract
                    #call
                }
            }
        });
        let entries = world.iter().map(|spec| {
            let shim = format_ident!("__pulsar_world_invoke_{}", spec.ident);
            let info = spec.info(&r);
            quote!(#s::component_methods::WorldMethod { info: #info, invoke: Self::#shim })
        });
        let ty = type_ref(&r, self_ty);
        quote! {
            #[doc(hidden)]
            #[allow(non_snake_case, non_upper_case_globals, clippy::needless_borrow, clippy::unit_arg)]
            impl #self_ty {
                #(#shims)*

                #[doc(hidden)]
                const #const_name: &'static [#s::component_methods::WorldMethod] = &[#(#entries),*];
            }

            #r::inventory::submit! {
                #s::component_methods::WorldMethodRegistration {
                    component: #ty,
                    methods: <#self_ty>::#const_name,
                }
            }
        }
    };

    Ok(quote! {
        #item
        #reflected
        #world
    })
}
