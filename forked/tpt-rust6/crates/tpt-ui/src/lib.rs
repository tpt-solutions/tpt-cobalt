//! # tpt-ui — the serverless Wasm dashboard layer
//!
//! `tpt-ui` turns a plain Rust function into a reactive dashboard. It ships
//! three dependency-light pieces that all work on stable Rust, in tests, and in
//! a browser:
//!
//! 1. [`Signal`] / [`create_effect`] — fine-grained, single-threaded reactivity.
//! 2. [`Node`] — a backend-agnostic view tree with [`render_to_html`] and
//!    [`export_html`] (a single self-contained HTML file).
//! 3. [`tpt_app`] — a proc macro that introspects a function's arguments and
//!    generates the widget state ([`Signal`]s) plus a `view()` method.
//!
//! ```
//! use tpt_ui::prelude::*;
//!
//! #[tpt_app]
//! fn dashboard(threshold: Slider<f32>, method: Dropdown) -> String {
//!     format!("{} @ {:.2}", method.selected_str(), threshold.value)
//! }
//!
//! let mut app = Dashboard::new();
//! app.method_choices = vec!["PCA".into(), "UMAP".into()];
//! app.threshold.set(Slider::new(0.0, 1.0, 0.01, 0.25));
//! assert!(app.view().to_html().contains("<input"));
//! ```
//!
//! The default build pulls in no DOM crates. A browser backend can be added
//! behind the (off-by-default) `wasm` feature.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate self as tpt_ui;

mod reactive;
mod view;
mod widgets;

pub use reactive::{create_effect, untrack, Effect, Signal};
pub use view::{
    escape_attr, escape_text, export_html, export_html_with_title, render_to_html, Node, View,
};
pub use widgets::{text_widget, ColorPicker, Dropdown, FileUpload, Slider, Widget};

/// Generates reactive widget state and a `view()` method from a function's
/// arguments. See the crate docs for an example.
pub use tpt_ui_macro::tpt_app;

/// Internal runtime used by generated code. Not a stable API.
#[doc(hidden)]
pub mod __rt {
    use crate::{Node, View};
    use core::fmt::Debug;

    /// Wrapper enabling autoref specialisation on an app function's result.
    pub struct Rendered<T>(pub T);

    /// Preferred path: the result already knows how to build a [`Node`].
    pub trait ViaView {
        /// Convert to a node via [`View`].
        fn tpt_node(&self) -> Node;
    }

    impl<T: View> ViaView for &Rendered<T> {
        fn tpt_node(&self) -> Node {
            self.0.view()
        }
    }

    /// Fallback path: render any `Debug` result as text.
    pub trait ViaDebug {
        /// Convert to a node via `Debug`.
        fn tpt_node(&self) -> Node;
    }

    impl<T: Debug> ViaDebug for Rendered<T> {
        fn tpt_node(&self) -> Node {
            Node::Text(format!("{:?}", self.0))
        }
    }
}

/// Optional browser backend. Enable with the `wasm` feature.
#[cfg(feature = "wasm")]
pub mod wasm;

/// Everything needed to write a dashboard.
pub mod prelude {
    pub use crate::__rt as _tpt_ui_rt;
    pub use crate::{create_effect, untrack, Effect, Signal};
    pub use crate::{escape_attr, escape_text, export_html, export_html_with_title};
    pub use crate::{render_to_html, Node, View};
    pub use crate::{text_widget, ColorPicker, Dropdown, FileUpload, Slider, Widget};
    pub use tpt_ui_macro::tpt_app;
}
