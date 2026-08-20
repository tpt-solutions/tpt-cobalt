//! End-to-end tests for `#[tpt_app]`, the view tree and reactivity.

use std::cell::RefCell;
use std::rc::Rc;
use tpt_ui::prelude::*;

#[tpt_app]
fn experiment_dashboard(threshold: Slider<f32>, method: Dropdown) -> String {
    format!("{} @ {:.2}", method.selected_str(), threshold.value)
}

#[tpt_app]
fn full_dashboard(data: FileUpload, tint: ColorPicker, gain: Slider<f64>, seed: u32) -> Vec<Node> {
    vec![Node::text(format!(
        "{}|{}|{}|{}",
        data.name,
        tint.hex(),
        gain.value,
        seed
    ))]
}

fn app() -> ExperimentDashboard {
    let mut a = ExperimentDashboard::new();
    a.method_choices = vec!["PCA".into(), "t-SNE".into(), "UMAP".into()];
    a.threshold.set(Slider::new(0.0f32, 1.0, 0.01, 0.25));
    a
}

#[test]
fn original_function_is_preserved() {
    let choices = ["PCA", "UMAP"];
    let out = experiment_dashboard(
        Slider::new(0.0f32, 1.0, 0.1, 0.5),
        Dropdown::new(&choices, 1),
    );
    assert_eq!(out, "UMAP @ 0.50");
}

#[test]
fn generated_view_contains_widgets_and_result() {
    let html = render_to_html(&app().view());
    assert!(html.contains("<input"), "missing input: {html}");
    assert!(html.contains("type=\"range\""));
    assert!(html.contains("<select"));
    assert!(html.contains("<option value=\"0\" selected=\"selected\">PCA</option>"));
    assert!(html.contains("PCA @ 0.25"), "missing result: {html}");
    assert_eq!(html, app().to_html());
}

#[test]
fn view_reacts_to_signal_changes() {
    let a = app();
    let log: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let sink = log.clone();
    let _effect = a.attach(move |node| sink.borrow_mut().push(render_to_html(&node)));

    assert_eq!(log.borrow().len(), 1);
    a.method.set(2);
    a.threshold.set(Slider::new(0.0, 1.0, 0.01, 0.75));

    let renders = log.borrow();
    assert_eq!(
        renders.len(),
        3,
        "one initial render + one per signal write"
    );
    assert!(renders[0].contains("PCA @ 0.25"));
    assert!(renders[1].contains("UMAP @ 0.25"));
    assert!(renders[2].contains("UMAP @ 0.75"));
}

#[test]
fn all_widget_kinds_render() {
    let a = FullDashboard::new();
    a.data.set(FileUpload::new("run.csv"));
    a.tint.set(ColorPicker::new((18, 52, 86)));
    a.gain.set(Slider::new(0.0f64, 10.0, 0.5, 2.5));
    a.seed.set(42);
    let html = a.to_html();
    for needle in [
        "type=\"file\"",
        "type=\"color\"",
        "type=\"range\"",
        "type=\"text\"",
    ] {
        assert!(html.contains(needle), "missing {needle} in {html}");
    }
    assert!(html.contains("run.csv|#123456|2.5|42"), "{html}");
}

#[test]
fn export_html_is_a_full_document() {
    let doc = export_html(app());
    assert!(doc.starts_with("<!doctype html>"));
    assert!(doc.contains("<html"));
    assert!(doc.contains("</html>"));
    assert!(doc.contains("<input"));
    assert!(doc.contains("PCA @ 0.25"));
}

#[test]
fn view_trait_and_effects_compose() {
    let count = Signal::new(0i32);
    let node = Signal::new(Node::text("start"));
    let (c, n) = (count.clone(), node.clone());
    let _e = create_effect(move || n.set(Node::text(format!("n={}", c.get()))));
    count.set(3);
    assert_eq!(render_to_html(&node.get_untracked().view()), "n=3");
    assert!(export_html(node.get_untracked()).contains("n=3"));
}
