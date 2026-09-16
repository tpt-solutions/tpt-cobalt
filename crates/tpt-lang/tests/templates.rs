//! Integration test: every file in `examples/templates/` must run cleanly
//! and produce its documented output — so the adoption docs cannot rot.

use std::path::PathBuf;
use tpt_lang::Interpreter;

fn template_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/templates")
        .join(name)
}

fn run_template(name: &str) -> (String, Option<tpt_lang::Value>) {
    let path = template_path(name);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut interp = Interpreter::new();
    let value = interp.run(&src).expect("template must run cleanly");
    (interp.output().to_string(), value)
}

#[test]
fn physics_units_template_runs_with_checked_units() {
    let (out, _) = run_template("physics-units.tpt");
    assert!(out.contains("force   = 19.62"), "{out}");
    assert!(out.contains("impulse = 29.43"), "{out}");
    assert!(out.contains("drop    = 11.0362"), "{out}");
}

#[test]
fn ml_training_template_converges() {
    let (out, _) = run_template("ml-training.tpt");
    assert!(out.contains("final loss = "), "{out}");
    // the parsed loss value must beat the documented target
    let loss: f64 = out
        .split("final loss = ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("loss is a number");
    assert!(loss < 0.05, "loss {loss} misses the documented target");
    assert!(out.contains("predictions shape: Tensor[64, 1]"), "{out}");
}

#[test]
fn modules_template_exercises_defaults_and_kwargs() {
    let (out, _) = run_template("modules.tpt");
    assert!(out.contains("disk  = 12.56636"), "{out}");
    assert!(out.contains("ring  = 9.42476"), "{out}");
    assert!(out.contains("mean  = 3"), "{out}");
}

#[test]
fn data_pipeline_template_slices_and_aggregates() {
    let (out, _) = run_template("data-pipeline.tpt");
    assert!(out.contains("first half sum = 12"), "{out}");
    assert!(
        out.contains("row-weighted first column = [0, 1, 2, 3, 4, 5]"),
        "{out}"
    );
    assert!(out.contains("record total = 6"), "{out}");
}

#[test]
fn every_template_passes_the_static_checker() {
    // units/shape errors are compile errors: enforce that here too
    for name in [
        "physics-units.tpt",
        "ml-training.tpt",
        "modules.tpt",
        "data-pipeline.tpt",
    ] {
        let src = std::fs::read_to_string(template_path(name)).unwrap();
        tpt_lang::check::check_program(&src)
            .unwrap_or_else(|e| panic!("{name} failed the static checker: {e}"));
    }
}
