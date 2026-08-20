//! Browser backend scaffolding (feature `wasm`).
//!
//! Kept free of `wasm-bindgen`/`web-sys` so the crate stays dependency-light.
//! A host supplies a [`Host`] implementation (e.g. one that patches the real
//! DOM) and [`mount`] keeps it in sync with the app's reactive state.

use crate::{create_effect, render_to_html, Effect, Node, View};

/// A rendering target for a mounted app.
pub trait Host: 'static {
    /// Called on the initial render and on every reactive update.
    fn patch(&mut self, html: &str);
}

/// Mount `app` into `host`, re-rendering whenever a widget signal changes.
///
/// The returned [`Effect`] keeps the subscription alive; drop-free by design,
/// call [`Effect::dispose`] to unmount.
pub fn mount<V, H>(app: V, mut host: H) -> Effect
where
    V: View + 'static,
    H: Host,
{
    create_effect(move || {
        let node: Node = app.view();
        host.patch(&render_to_html(&node));
    })
}
