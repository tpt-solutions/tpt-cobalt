//! CI-enforced check that headline-crate public types implement the
//! `tpt_lab::render::Render` rich-display trait (the "Notebook-native"
//! cross-cutting rule from `spec.txt` §6). If a crate drops its `Render` impl,
//! this test fails to compile and the `rich-display` CI job goes red.

use tpt_lab::render::Render;

/// Compile-time assertion that `T` implements [`Render`].
fn assert_render<T: Render>() {}

#[test]
fn headline_public_types_implement_render() {
    // Scalars render via `Display`.
    assert_render::<i32>();
    assert_render::<f64>();
    assert_render::<String>();
    assert_render::<Vec<f64>>();

    // Core data types render (Markdown table / text-grid / SVG heatmap).
    assert_render::<tpt_omni::Tensor<f64>>();
    assert_render::<tpt_omni::Table>();

    // Plots render to SVG when the `viz` feature is enabled.
    #[cfg(feature = "viz")]
    assert_render::<tpt_viz::Plot>();
}
