use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    DeriveInput, Ident, Token,
};

/// Known `ReplicationEncoding` variant names the derive can emit directly.
/// `Opaque` is deliberately excluded — see [`ReplicateAttr::parse`]'s check.
const VALID_ENCODINGS: &[&str] = &["Pod", "Serialized", "GpuHandle", "DeltaCompressed", "Event"];

/// Known `ReplicationCondition` variant names.
const VALID_CONDITIONS: &[&str] = &[
    "Always", "OwnerOnly", "SkipOwner", "SimulatedOnly", "AutonomousOnly", "InitialOnly",
    "ServerAuthority", "ClientAuthority", "ServerToClient", "ClientToServer", "Multicast",
];

/// Known `EventChannel` variant names.
const VALID_EVENT_CHANNELS: &[&str] = &["ReliableOrdered", "Unreliable"];

fn unknown_value_error(span: proc_macro2::Span, kind: &str, value: &Ident, valid: &[&str]) -> syn::Error {
    syn::Error::new(
        span,
        format!(
            "unknown replication {kind} `{value}` — expected one of: {}",
            valid.join(", "),
        ),
    )
}

/// Parsed `#[replicate(encoding = Pod, condition = Always)]` on a field.
struct ReplicateAttr {
    encoding: Ident,
    condition: Ident,
    is_event: bool,
    /// `#[replicate(event_channel = Unreliable)]` — only meaningful (and
    /// only accepted) alongside `encoding = Event`; overrides the default
    /// `ReliableOrdered`.
    event_channel: Option<Ident>,
}

impl Parse for ReplicateAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut encoding: Option<Ident> = None;
        let mut condition: Option<Ident> = None;
        let mut event_channel: Option<Ident> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let value: Ident = input.parse()?;
            let val_str = value.to_string();

            match key.to_string().as_str() {
                "encoding" => {
                    if val_str == "Opaque" {
                        return Err(syn::Error::new(
                            value.span(),
                            "`Opaque` encoding cannot be derived — it requires custom encode/decode \
                             fn pointers supplied at registration time. Register this field manually \
                             via `ReplicationRegistry::register::<T>()` and `SchemaBuilder::field(..., \
                             ReplicationEncoding::Opaque { encode_size, encode, decode }, ...)` instead \
                             of `#[derive(Replicate)]`.",
                        ));
                    }
                    if !VALID_ENCODINGS.contains(&val_str.as_str()) {
                        return Err(unknown_value_error(value.span(), "encoding", &value, VALID_ENCODINGS));
                    }
                    encoding = Some(value);
                }
                "condition" => {
                    if !VALID_CONDITIONS.contains(&val_str.as_str()) {
                        return Err(unknown_value_error(value.span(), "condition", &value, VALID_CONDITIONS));
                    }
                    condition = Some(value);
                }
                "event_channel" => {
                    if !VALID_EVENT_CHANNELS.contains(&val_str.as_str()) {
                        return Err(unknown_value_error(value.span(), "event channel", &value, VALID_EVENT_CHANNELS));
                    }
                    event_channel = Some(value);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `#[replicate(...)]` key `{other}` — expected `encoding`, \
                             `condition`, or `event_channel`",
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
            }
        }

        let encoding = encoding.unwrap_or_else(|| Ident::new("Pod", proc_macro2::Span::call_site()));
        let condition = condition.unwrap_or_else(|| Ident::new("Always", proc_macro2::Span::call_site()));
        let is_event = encoding == "Event";

        if !is_event {
            if let Some(ch) = &event_channel {
                return Err(syn::Error::new(
                    ch.span(),
                    "`event_channel` is only valid alongside `encoding = Event`",
                ));
            }
        }

        Ok(ReplicateAttr { encoding, condition, is_event, event_channel })
    }
}

struct FieldInfo {
    ident: Ident,
    encoding: Ident,
    condition: Ident,
    is_event: bool,
    event_channel: Option<Ident>,
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

        // Fields without an explicit `#[replicate(...)]` attribute are not
        // replicated — annotation is opt-in, so e.g. a `PhantomData<T>`
        // marker field or a purely-local cache field doesn't silently pick
        // up `Pod`/`Always` defaults just for existing on the struct.
        let mut parsed: Option<ReplicateAttr> = None;
        for attr in &field.attrs {
            if attr.path().is_ident("replicate") {
                parsed = Some(attr.parse_args()?);
            }
        }
        let Some(parsed) = parsed else { continue };

        field_infos.push(FieldInfo {
            ident,
            encoding: parsed.encoding,
            condition: parsed.condition,
            is_event: parsed.is_event,
            event_channel: parsed.event_channel,
        });
    }

    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(name, "Replicate requires at least one field with #[replicate]"));
    }

    let builder_calls: Vec<TokenStream> = field_infos
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let encoding_ident = &f.encoding;
            let condition_ident = &f.condition;

            if f.is_event {
                let channel_ident = f
                    .event_channel
                    .clone()
                    .unwrap_or_else(|| Ident::new("ReliableOrdered", f.ident.span()));
                quote! {
                    .event(#field_name, ::pulsar_scenedb::ReplicationCondition::#condition_ident, ::pulsar_scenedb::EventChannel::#channel_ident)
                }
            } else {
                // `get`/`get_mut` are plain, non-capturing field accessors —
                // they coerce to the `fn(&T) -> &F`/`fn(&mut T) -> &mut F`
                // pointers `SchemaBuilder::field` expects, and `F` (the
                // field's own type) is inferred from the accessor's return
                // type — no turbofish needed. `F` must implement
                // `pulsar_scenedb::Replicable`: every `Pod` field type
                // already does (blanket impl), plus `String`/`Vec<T>`/
                // `Option<T>` out of the box; anything else needs a manual
                // `impl Replicable` (see that trait's doc).
                let field_ident = &f.ident;
                quote! {
                    .field(
                        #field_name,
                        |c: &Self| &c.#field_ident,
                        |c: &mut Self| &mut c.#field_ident,
                        ::pulsar_scenedb::ReplicationEncoding::#encoding_ident,
                        ::pulsar_scenedb::ReplicationCondition::#condition_ident,
                    )
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
