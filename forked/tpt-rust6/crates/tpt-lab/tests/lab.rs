//! Integration tests for the reactive notebook.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use tpt_lab::prelude::*;
use tpt_omni::ndarray::{ArrayD, IxDyn};
use tpt_omni::prelude::{col, OmniFrame, Tensor};

fn base() -> Notebook {
    let mut nb = Notebook::new();
    nb.set("A", 1i64).unwrap();
    nb.set_expr("B", "A + 1", &["A"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("A")? + 1)
    })
    .unwrap();
    nb.set_expr("C", "B * 2", &["B"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("B")? * 2)
    })
    .unwrap();
    nb
}

#[test]
fn chain_computes_initial_values() {
    let nb = base();
    assert_eq!(nb.get::<i64>("A"), Some(&1));
    assert_eq!(nb.get::<i64>("B"), Some(&2));
    assert_eq!(nb.get::<i64>("C"), Some(&4));
    assert_eq!(nb.dependencies("C"), vec!["B"]);
    assert_eq!(nb.dependents("A"), vec!["B"]);
    assert!(nb.check().is_ok());
}

#[test]
fn update_recomputes_only_transitive_dependents() {
    let mut nb = Notebook::new();
    let b_runs = Arc::new(AtomicUsize::new(0));
    let d_runs = Arc::new(AtomicUsize::new(0));

    nb.set("A", 1i64).unwrap();
    nb.set("Z", 100i64).unwrap();

    let bc = Arc::clone(&b_runs);
    nb.set_expr("B", "A + 1", &["A"], move |nb: &Notebook| {
        bc.fetch_add(1, Ordering::SeqCst);
        Ok(nb.try_get::<i64>("A")? + 1)
    })
    .unwrap();
    nb.set_expr("C", "B * 2", &["B"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("B")? * 2)
    })
    .unwrap();
    // Independent branch: must NOT re-run when A changes.
    let dc = Arc::clone(&d_runs);
    nb.set_expr("D", "Z + 1", &["Z"], move |nb: &Notebook| {
        dc.fetch_add(1, Ordering::SeqCst);
        Ok(nb.try_get::<i64>("Z")? + 1)
    })
    .unwrap();

    assert_eq!(nb.get::<i64>("B"), Some(&2));
    assert_eq!(nb.get::<i64>("C"), Some(&4));
    assert_eq!(b_runs.load(Ordering::SeqCst), 1);
    assert_eq!(d_runs.load(Ordering::SeqCst), 1);

    nb.set("A", 10i64).unwrap();

    assert_eq!(nb.get::<i64>("B"), Some(&11));
    assert_eq!(nb.get::<i64>("C"), Some(&22));
    assert_eq!(nb.get::<i64>("D"), Some(&101));
    assert_eq!(b_runs.load(Ordering::SeqCst), 2, "B must re-execute");
    assert_eq!(
        d_runs.load(Ordering::SeqCst),
        1,
        "D is not downstream of A and must not re-execute"
    );
}

#[test]
fn cell_macro_captures_dependency_identifiers() {
    let mut nb = Notebook::new();
    cell!(nb, A = 3i64).unwrap();
    cell!(nb, B = [A] "A + 1", |nb: &Notebook| Ok(
        nb.try_get::<i64>("A")? + 1
    ))
    .unwrap();
    assert_eq!(nb.get::<i64>("B"), Some(&4));
    assert_eq!(nb.dependencies("B"), vec!["A"]);

    cell!(nb, A = 41i64).unwrap();
    assert_eq!(nb.get::<i64>("B"), Some(&42));
}

#[test]
fn type_registry_tracks_cell_types() {
    let mut nb = base();
    nb.set("name", "ada".to_string()).unwrap();

    assert_eq!(nb.type_of("A"), Some("i64"));
    assert_eq!(nb.type_of("B"), Some("i64"));
    assert_eq!(nb.type_of("name"), Some("alloc::string::String"));
    assert_eq!(nb.type_of("nope"), None);

    // Reading at the wrong type is an error, not a panic.
    assert!(matches!(
        nb.try_get::<String>("B"),
        Err(LabError::TypeMismatch { .. })
    ));
    assert!(nb.get::<String>("B").is_none());

    // Silently changing a cell's type is refused.
    assert!(matches!(
        nb.set("A", "oops".to_string()),
        Err(LabError::TypeChanged { .. })
    ));
    assert_eq!(nb.get::<i64>("A"), Some(&1));
}

#[test]
fn undefined_references_are_errors() {
    let mut nb = base();

    // Reading an undefined cell.
    assert!(matches!(
        nb.try_get::<i64>("Q"),
        Err(LabError::Undefined(ref n)) if n == "Q"
    ));

    // Declaring a dependency on an undefined cell.
    let err = nb
        .set_expr("X", "Q + 1", &["Q"], |nb: &Notebook| {
            Ok(nb.try_get::<i64>("Q")? + 1)
        })
        .unwrap_err();
    assert!(matches!(err, LabError::MissingDependency { ref dep, .. } if dep == "Q"));
    assert!(!nb.contains("X"));

    // Registering an edge to an undefined cell.
    assert!(matches!(
        nb.depends_on("C", &["Q"]),
        Err(LabError::MissingDependency { .. })
    ));
    assert!(matches!(
        nb.depends_on("Q", &["A"]),
        Err(LabError::Undefined(_))
    ));
}

#[test]
fn cross_cell_type_error_surfaces_on_reexecution() {
    let mut nb = base();
    // `redefine` allows the type change; the downstream closure then fails.
    let err = nb.redefine("A", "not a number".to_string()).unwrap_err();
    assert!(matches!(err, LabError::TypeMismatch { ref cell, .. } if cell == "A"));
    assert!(nb.cell("B").unwrap().error().is_some());
}

#[test]
fn depends_on_registers_edges_and_reruns() {
    let mut nb = Notebook::new();
    nb.set("A", 2i64).unwrap();
    nb.set("K", 5i64).unwrap();
    nb.set_expr("S", "A + K", &["A"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("A")? + nb.try_get::<i64>("K")?)
    })
    .unwrap();
    assert_eq!(nb.get::<i64>("S"), Some(&7));

    // K was read but not declared: register the edge, then updating K reacts.
    nb.depends_on("S", &["K"]).unwrap();
    assert_eq!(nb.dependencies("S"), vec!["A", "K"]);
    nb.set("K", 10i64).unwrap();
    assert_eq!(nb.get::<i64>("S"), Some(&12));
}

#[test]
fn diamond_dependencies_resolve_in_topological_order() {
    let mut nb = Notebook::new();
    nb.set("A", 2i64).unwrap();
    nb.set_expr("L", "A * 3", &["A"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("A")? * 3)
    })
    .unwrap();
    nb.set_expr("R", "A + 4", &["A"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("A")? + 4)
    })
    .unwrap();
    nb.set_expr("T", "L + R", &["L", "R"], |nb: &Notebook| {
        Ok(nb.try_get::<i64>("L")? + nb.try_get::<i64>("R")?)
    })
    .unwrap();
    assert_eq!(nb.get::<i64>("T"), Some(&12));

    nb.set("A", 10i64).unwrap();
    assert_eq!(nb.get::<i64>("T"), Some(&44)); // 30 + 14
}

fn demo_frame() -> OmniFrame {
    let age: ArrayRef = Arc::new(Int64Array::from(vec![17i64, 34, 51]));
    let score: ArrayRef = Arc::new(Float64Array::from(vec![1.5f64, 2.5, 3.5]));
    let city: ArrayRef = Arc::new(StringArray::from(vec!["oslo", "lima", "kyoto"]));
    OmniFrame::from_columns(vec![
        ("age".to_string(), age),
        ("score".to_string(), score),
        ("city".to_string(), city),
    ])
    .unwrap()
}

#[test]
fn display_renders_a_table_as_markdown() {
    let mut nb = Notebook::new();
    nb.set("people", demo_frame().as_table()).unwrap();

    let out = nb.display("people").unwrap();
    assert!(out.contains("age"), "header name missing: {out}");
    assert!(out.contains("score") && out.contains("city"));
    assert!(out.contains("| --- |"));
    assert!(out.contains("oslo") && out.contains("1.5"));
    assert_eq!(nb.type_of("people"), Some("tpt_omni::table::Table"));
}

#[test]
fn derived_table_cell_reacts_to_a_threshold_cell() {
    let mut nb = Notebook::new();
    nb.set("min_age", 18i64).unwrap();
    nb.set("people", demo_frame().as_table()).unwrap();
    nb.set_expr(
        "adults",
        "people.filter(col(\"age\").gt(min_age))",
        &["people", "min_age"],
        |nb: &Notebook| {
            let t = nb.try_get::<tpt_omni::Table>("people")?;
            let min = *nb.try_get::<i64>("min_age")?;
            t.filter(&col("age").gt(min)).map_err(|e| LabError::Eval {
                cell: "adults".to_string(),
                message: e.to_string(),
            })
        },
    )
    .unwrap();
    assert_eq!(nb.get::<tpt_omni::Table>("adults").unwrap().num_rows(), 2);

    nb.set("min_age", 40i64).unwrap();
    assert_eq!(nb.get::<tpt_omni::Table>("adults").unwrap().num_rows(), 1);
    assert!(nb.display("adults").unwrap().contains("kyoto"));
}

#[test]
fn display_renders_a_tensor() {
    let mut nb = Notebook::new();
    let t = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    nb.set("m", t).unwrap();

    let out = nb.display("m").unwrap();
    #[cfg(feature = "viz")]
    {
        // The heatmap title is XML-escaped inside the SVG document.
        assert!(out.starts_with("<svg"), "{out}");
        assert!(
            out.contains("Tensor&lt;f64&gt;") && out.contains("<rect"),
            "{out}"
        );
    }
    #[cfg(not(feature = "viz"))]
    {
        assert!(out.contains("shape=[2, 3]"), "{out}");
        assert!(out.contains("6.0000"), "{out}");
        assert_eq!(out.lines().count(), 3);
    }
}

#[test]
fn display_scalars_and_unknown_types() {
    let mut nb = base();
    assert_eq!(nb.display("C").unwrap(), "4");
    nb.set("v", vec![1.0f64, 2.0]).unwrap();
    assert_eq!(nb.display("v").unwrap(), "[1, 2]");
    nb.set("opt", Some(3u8)).unwrap();
    assert!(nb.display("opt").unwrap().starts_with("<core::option"));
    assert!(matches!(nb.display("ghost"), Err(LabError::Undefined(_))));
}

#[test]
fn custom_render_impl_is_used() {
    struct Point(i32, i32);
    impl Render for Point {
        fn render(&self, _nb: &Notebook) -> String {
            format!("({}, {})", self.0, self.1)
        }
    }
    let mut nb = Notebook::new();
    nb.set_rendered("p", Point(1, 2)).unwrap();
    assert_eq!(nb.display("p").unwrap(), "(1, 2)");
}

#[test]
fn export_rust_emits_cells_in_dependency_order() {
    let mut nb = base();
    nb.set("label", "run-1".to_string()).unwrap();
    nb.set("people", demo_frame().as_table()).unwrap();

    let src = nb.export_rust();
    assert!(!src.is_empty());
    for name in ["A", "B", "C", "label", "people"] {
        assert!(src.contains(name), "missing cell {name} in:\n{src}");
    }
    assert!(src.contains("let A: i64 = 1i64;"));
    assert!(src.contains("let B: i64 = A + 1;"));
    assert!(src.contains("let C: i64 = B * 2;"));
    assert!(src.contains("// cell `B` depends on: A"));
    assert!(src.contains(r#""run-1".to_string()"#));
    // Opaque values are emitted as a comment rather than broken code.
    assert!(src.contains("// let people:"));
    // Dependencies precede dependents.
    assert!(src.find("let A").unwrap() < src.find("let B").unwrap());
    assert!(src.find("let B").unwrap() < src.find("let C").unwrap());
}

#[test]
fn display_all_lists_every_cell() {
    let nb = base();
    let all = nb.display_all();
    assert_eq!(all, "A: 1\nB: 2\nC: 4\n");
    assert_eq!(nb.names(), vec!["A", "B", "C"]);
    assert_eq!(nb.len(), 3);
}
