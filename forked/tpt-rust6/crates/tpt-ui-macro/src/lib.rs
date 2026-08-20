//! Procedural macros for `tpt-ui`: [`macro@tpt_app`].

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TS;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Error, FnArg, GenericParam, Ident, ItemFn, Pat, Result, Type};

/// Introspects a function's arguments and generates a reactive dashboard app.
///
/// The annotated function is kept as-is. In addition, a struct named after the
/// function (`fn my_app` -> `struct MyApp`) is generated holding one
/// [`Signal`](../tpt_ui/struct.Signal.html) per argument, plus:
///
/// * `fn new() -> Self` / `Default`
/// * `fn view(&self) -> Node` — widgets + the function's rendered result
/// * `fn to_html(&self) -> String`
/// * `fn attach(&self, sink)` — re-renders whenever a widget signal changes
///
/// Recognised argument types: `Slider<f32>`, `Slider<f64>`, `Dropdown`,
/// `FileUpload`, `ColorPicker`. Any other `Default + Clone` type gets a text
/// input.
#[proc_macro_attribute]
pub fn tpt_app(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand(func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

enum Kind {
    Widget,
    Dropdown,
    Plain,
}

fn kind_of(ty: &Type) -> Kind {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return match seg.ident.to_string().as_str() {
                "Slider" | "FileUpload" | "ColorPicker" => Kind::Widget,
                "Dropdown" => Kind::Dropdown,
                _ => Kind::Plain,
            };
        }
    }
    Kind::Plain
}

fn pascal(s: &str) -> String {
    s.split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn expand(func: ItemFn) -> Result<TS> {
    if let Some(a) = &func.sig.asyncness {
        return Err(Error::new_spanned(
            a,
            "#[tpt_app] does not support async fns",
        ));
    }
    for g in &func.sig.generics.params {
        if !matches!(g, GenericParam::Lifetime(_)) {
            return Err(Error::new_spanned(
                g,
                "#[tpt_app] does not support generic fns",
            ));
        }
    }

    let vis = &func.vis;
    let fname = &func.sig.ident;
    let fname_str = fname.to_string();
    let app = Ident::new(&pascal(&fname_str), fname.span());

    let (mut fields, mut defaults) = (Vec::new(), Vec::new());
    let (mut lets, mut widgets, mut args) = (Vec::new(), Vec::new(), Vec::new());

    for input in &func.sig.inputs {
        let pt = match input {
            FnArg::Typed(t) => t,
            FnArg::Receiver(r) => {
                return Err(Error::new_spanned(
                    r,
                    "#[tpt_app] cannot be used on methods",
                ))
            }
        };
        let name = match &*pt.pat {
            Pat::Ident(i) => i.ident.clone(),
            other => {
                return Err(Error::new_spanned(
                    other,
                    "#[tpt_app] arguments must be plain identifiers",
                ))
            }
        };
        let ty = &*pt.ty;
        let label = name.to_string();

        match kind_of(ty) {
            Kind::Dropdown => {
                let choices = format_ident!("{}_choices", name);
                let tmp = format_ident!("__choices_{}", name);
                fields.push(quote! {
                    /// Index of the selected choice.
                    pub #name: ::tpt_ui::Signal<usize>,
                    /// Choices offered by the dropdown.
                    pub #choices: ::std::vec::Vec<::std::string::String>
                });
                defaults.push(quote! {
                    #name: ::tpt_ui::Signal::new(0usize),
                    #choices: ::std::vec::Vec::new()
                });
                lets.push(quote! {
                    let #tmp: ::std::vec::Vec<&str> =
                        self.#choices.iter().map(|s| s.as_str()).collect();
                    let #name = ::tpt_ui::Dropdown { choices: &#tmp, selected: self.#name.get() };
                });
                widgets.push(quote! { ::tpt_ui::Widget::widget_node(&#name, #label) });
            }
            Kind::Widget => {
                fields.push(quote! {
                    /// Reactive widget state.
                    pub #name: ::tpt_ui::Signal<#ty>
                });
                defaults.push(quote! {
                    #name: ::tpt_ui::Signal::new(<#ty as ::core::default::Default>::default())
                });
                lets.push(quote! { let #name: #ty = self.#name.get(); });
                widgets.push(quote! { ::tpt_ui::Widget::widget_node(&#name, #label) });
            }
            Kind::Plain => {
                fields.push(quote! {
                    /// Reactive argument state.
                    pub #name: ::tpt_ui::Signal<#ty>
                });
                defaults.push(quote! {
                    #name: ::tpt_ui::Signal::new(<#ty as ::core::default::Default>::default())
                });
                lets.push(quote! { let #name: #ty = self.#name.get(); });
                widgets.push(quote! { ::tpt_ui::text_widget(#label) });
            }
        }
        args.push(quote! { #name });
    }

    let doc = format!("Reactive dashboard generated from `{fname_str}` by `#[tpt_app]`.");

    Ok(quote! {
        #func

        #[doc = #doc]
        #[derive(Clone)]
        #vis struct #app {
            #(#fields,)*
        }

        impl ::core::default::Default for #app {
            fn default() -> Self {
                Self { #(#defaults,)* }
            }
        }

        impl #app {
            /// Create the app with default widget state.
            #vis fn new() -> Self {
                <Self as ::core::default::Default>::default()
            }

            fn __tpt_render(&self) -> ::tpt_ui::Node {
                #(#lets)*
                let __widgets: ::std::vec::Vec<::tpt_ui::Node> = ::std::vec![#(#widgets),*];
                let __result = #fname(#(#args),*);
                let __out = {
                    use ::tpt_ui::__rt::{ViaDebug as _, ViaView as _};
                    (&&::tpt_ui::__rt::Rendered(__result)).tpt_node()
                };
                ::tpt_ui::Node::element(
                    "div",
                    [("class", "tpt-app"), ("id", #fname_str)],
                    ::std::vec![
                        ::tpt_ui::Node::element("form", [("class", "tpt-controls")], __widgets),
                        ::tpt_ui::Node::element(
                            "div",
                            [("class", "tpt-output")],
                            ::std::vec![__out],
                        ),
                    ],
                )
            }

            /// Build the view tree: one widget per argument plus the result.
            #vis fn view(&self) -> ::tpt_ui::Node {
                self.__tpt_render()
            }

            /// Render the app to an HTML fragment.
            #vis fn to_html(&self) -> ::std::string::String {
                ::tpt_ui::render_to_html(&self.__tpt_render())
            }

            /// Re-render into `sink` whenever any widget signal changes.
            #vis fn attach<F>(&self, mut sink: F) -> ::tpt_ui::Effect
            where
                F: ::core::ops::FnMut(::tpt_ui::Node) + 'static,
            {
                let __this = ::core::clone::Clone::clone(self);
                ::tpt_ui::create_effect(move || sink(__this.__tpt_render()))
            }
        }

        impl ::tpt_ui::View for #app {
            fn view(&self) -> ::tpt_ui::Node {
                self.__tpt_render()
            }
        }
    })
}
