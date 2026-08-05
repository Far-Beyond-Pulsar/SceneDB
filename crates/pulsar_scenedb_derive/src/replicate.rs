use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    DeriveInput, Ident, Token,
};

/// Parsed `#[replicate(encoding = Pod, condition = Always)]` on a field.
struct ReplicateAttr {
    encoding: String,
    condition: String,
    is_event: bool,
}

impl Parse for ReplicateAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut encoding = String::from("Pod");
        let mut condition = String::from("Always");
        let mut is_event = false;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let value: Ident = input.parse()?;
            let val_str = value.to_string();

            match key.to_string().as_str() {
                "encoding" => {
                    if val_str == "Event" {
                        is_event = true;
                    }
                    encoding = val_str;
                }
                "condition" => condition = val_str,
                _ => {
                    return Err(syn::Error::new(key.span(), "expected encoding or condition"));
                }
            }

            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(ReplicateAttr { encoding, condition, is_event })
    }
}

struct FieldInfo {
    ident: Ident,
    encoding: String,
    condition: String,
    is_event: bool,
}

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        syn::Data::Struct(ds) => match &ds.fields {
            syn::Fields::Named(named) => &named.named,
            _ => return Err(syn::Error::new_spanned(name, "Replicate requires named fields")),
        },
        _ => return Err(syn::Error::new_spanned(name, "Replicate only supports structs")),
    };

    let mut field_infos: Vec<FieldInfo> = Vec::new();
    for field in fields {
        let ident = match &field.ident {
            Some(i) => i.clone(),
            None => continue,
        };
        let mut encoding = String::from("Pod");
        let mut condition = String::from("Always");
        let mut is_event = false;

        for attr in &field.attrs {
            if attr.path().is_ident("replicate") {
                let parsed: ReplicateAttr = attr.parse_args()?;
                encoding = parsed.encoding;
                condition = parsed.condition;
                is_event = parsed.is_event;
            }
        }

        field_infos.push(FieldInfo { ident, encoding, condition, is_event });
    }

    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(name, "Replicate requires at least one field with #[replicate]"));
    }

    let builder_calls: Vec<TokenStream> = field_infos
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let encoding_ident = Ident::new(&f.encoding, f.ident.span());
            let condition_ident = Ident::new(&f.condition, f.ident.span());

            if f.is_event {
                quote! {
                    .event(#field_name, ::pulsar_scenedb::ReplicationCondition::#condition_ident, ::pulsar_scenedb::EventChannel::ReliableOrdered)
                }
            } else {
                quote! {
                    .field(#field_name, ::pulsar_scenedb::ReplicationEncoding::#encoding_ident, ::pulsar_scenedb::ReplicationCondition::#condition_ident)
                }
            }
        })
        .collect();

    Ok(quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Register this component's replication schema with a registry.
            pub fn register_replication(registry: &mut ::pulsar_scenedb::ReplicationRegistry) {
                let builder = registry.register::<Self>();
                registry.insert(
                    builder #(#builder_calls)*
                );
            }
        }
    })
}
