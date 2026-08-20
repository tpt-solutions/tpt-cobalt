//! Integration tests exercised through the public prelude, as a `.tpt` script
//! would use it.

use tpt_script::prelude::*;

fn sample() -> Table {
    table_of(vec![
        f64s("x", vec![-2.0, -0.5, 1.0, 3.0, 4.0]),
        f64s("val", vec![10.0, 20.0, 30.0, 40.0, 60.0]),
        strs("grp", &["a", "b", "a", "b", "a"]),
    ])
}

#[test]
fn prelude_exposes_the_stack() {
    // Pythonic aliases resolve.
    let _: fn(&str) -> Result<omni::OmniFrame, io::IoError> = read;
    let _: fn(&omni::OmniFrame, &str) -> Result<(), io::IoError> = write;
    let _ = stat::regression::ols(&[vec![1.0, 1.0], vec![1.0, 2.0]], &[1.0, 2.0]);
    let _ = viz::Plot::new().to_svg();
    let _ = sym::Expr::var("x");
}

#[test]
fn filter_drops_rows() {
    let t = sample();
    assert_eq!(t.len(), 5);
    let pos = t.filter("x", ">", 0.0);
    assert_eq!(pos.len(), 3);
    // Integer literals coerce to the column dtype.
    let big = sample().filter("val", ">=", 30);
    assert_eq!(big.len(), 3);
    // Strings and chained calls work too.
    let a_only = sample().filter("grp", "==", "a").filter("x", "<", 2.0);
    assert_eq!(a_only.len(), 2);
}

#[test]
fn filter_reports_bad_column() {
    let e = match sample().try_filter("nope", ">", Value::Float(0.0)) {
        Err(e) => e,
        Ok(_) => panic!("expected a ColumnError"),
    };
    assert_eq!(e.kind, "ColumnError");
    // Script mode: the raise is catchable as a traceback.
    let err = run(|| {
        let _ = sample().filter("nope", ">", 0.0);
        Ok(())
    })
    .unwrap_err();
    assert!(err.traceback().contains("column 'nope' not found"));
}

#[test]
fn groupby_agg_means_and_counts() {
    let g = sample().groupby(&["grp"]);
    assert_eq!(g.n_groups(), 2);
    assert_eq!(g.sizes(), vec![3, 2]); // sorted keys: a, b

    let out = g.agg(&[("val", "mean"), ("*", "count"), ("x", "max")]);
    assert_eq!(out.len(), 2);
    assert_eq!(
        out.column_names(),
        vec!["grp", "val_mean", "count", "x_max"]
    );
    // a: (10 + 30 + 60) / 3 = 33.333..., b: (20 + 40) / 2 = 30
    let means = out.col_f64("val_mean");
    assert!((means[0] - 100.0 / 3.0).abs() < 1e-12);
    assert!((means[1] - 30.0).abs() < 1e-12);
    assert_eq!(out.col_f64("count"), vec![3.0, 2.0]);
    assert_eq!(out.col_f64("x_max"), vec![4.0, 3.0]);

    // Key column keeps its original Utf8 dtype.
    assert_eq!(
        format!("{:?}", out.column("grp").unwrap().data_type()),
        "Utf8"
    );
}

#[test]
fn groupby_multi_key_and_shortcuts() {
    let t = table_of(vec![
        strs("g", &["a", "a", "b", "b"]),
        i64s("k", vec![1, 2, 1, 1]),
        f64s("v", vec![1.0, 2.0, 3.0, 5.0]),
    ]);
    let out = t.groupby(&["g", "k"]).mean("v");
    assert_eq!(out.len(), 3);
    assert_eq!(out.col_f64("v_mean"), vec![1.0, 2.0, 4.0]);
    assert_eq!(t.groupby(&["g"]).count().col_f64("count"), vec![2.0, 2.0]);
}

#[test]
fn plot_builds_a_real_svg() {
    let svg = sample().plot().to_svg();
    assert!(svg.starts_with("<svg") && svg.contains("<circle"));
    let p = sample().plot_xy("x", "val");
    assert!(p.to_svg().contains("val vs x"));
    // Degenerate tables must not panic.
    assert!(table_of(vec![strs("g", &["a"])])
        .plot()
        .to_svg()
        .starts_with("<svg"));
}

#[test]
fn ols_through_the_table() {
    let t = table_of(vec![
        f64s("x", vec![1.0, 2.0, 3.0, 4.0]),
        f64s("y", vec![3.0, 5.0, 7.0, 9.0]),
    ]);
    let r = t.ols("y", &["x"]);
    assert!((r.coefficients[0] - 1.0).abs() < 1e-9);
    assert!((r.coefficients[1] - 2.0).abs() < 1e-9);
}

#[test]
fn script_macro_expands_and_runs() {
    let t = sample();
    script! {
        let pos = t.filter("x", ">", 0.0);
        let summary = pos.groupby(&["grp"]).agg(&[("val", "mean")]);
        let mut n = summary.len();
        n += 0;
        print summary.show();
    }
    assert_eq!(pos.len(), 3);
    assert_eq!(n, 2);
    assert_eq!(summary.col_f64("val_mean"), vec![45.0, 40.0]);
}

mod generated_main {
    tpt_script::script_main! {
        let t = tpt_script::table_of(vec![tpt_script::f64s("v", vec![1.0, 4.0])]);
        let big = t.filter("v", ">", 2.0);
        print big.len();
    }
}

#[test]
fn script_main_macro_runs() {
    generated_main::main();
}

#[test]
fn run_script_executes_the_micro_language() {
    assert!(run_script("# comment\nlet x = 2 + 3 * 4\nassert x == 14\nprint x").is_ok());
    assert!(run_script("let x = (1 + 1) / 4\nassert x < 1\nprint \"ok\"").is_ok());
}

#[test]
fn run_script_catches_panics_with_a_traceback() {
    let err = run_script("let x = 1\npanic \"boom\"").unwrap_err();
    assert!(
        err.starts_with("Traceback (most recent call last):"),
        "{err}"
    );
    assert!(err.contains("boom"), "{err}");

    // A failing assert is a genuine panic, not a returned error.
    let err = run_script("let x = 1\nassert x > 5").unwrap_err();
    assert!(err.contains("assertion failed"), "{err}");

    // Syntax / name errors are reported with the offending line.
    let err = run_script("let x = 1\nlet y = z + 1").unwrap_err();
    assert!(err.contains("NameError") && err.contains("line 2"), "{err}");
    let err = run_script("wat?").unwrap_err();
    assert!(err.contains("SyntaxError"), "{err}");
}
