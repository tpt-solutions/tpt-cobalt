use tpt_sym::prelude::*;

#[test]
fn differentiate_x_squared() {
    let x = sym!(x);
    let f = x.clone() * x.clone(); // x^2
    let df = f.diff("x");
    assert_eq!(df.simplify(), (sym!(x) * 2.0).simplify());
    assert!((df.eval(&[("x", 3.0)]) - 6.0).abs() < 1e-9);
}

#[test]
fn chain_rule_trig() {
    let x = sym!(x);
    let f = x.clone().sin(); // sin(x)
    let df = f.diff("x");
    assert_eq!(df.simplify(), x.clone().cos().simplify());
}

#[test]
fn simplify_identities() {
    let x = sym!(x);
    assert_eq!((x.clone() + 0.0).simplify(), x.clone().simplify());
    assert_eq!((x.clone() * 1.0).simplify(), x.clone().simplify());
    assert_eq!((x.clone() * 0.0).simplify(), Expr::Const(0.0));
    assert_eq!((sym!(2.0) * sym!(3.0)).simplify(), Expr::Const(6.0));
}

#[test]
fn latex_and_eval() {
    let x = sym!(x);
    let f = x.clone() * x.clone() + sym!(2.0);
    assert_eq!(f.to_latex(), "x \\, x + 2");
    assert!((f.eval(&[("x", 3.0)]) - 11.0).abs() < 1e-9);
}

#[test]
fn units_dimension_check() {
    let dist = Quantity::new(10.0, Unit::METER);
    let time = Quantity::new(2.0, Unit::SECOND);
    let speed = dist / time;
    assert_eq!(speed.unit, Unit::METER.div(&Unit::SECOND));
    // adding incompatible units must panic
    let bad = std::panic::catch_unwind(|| {
        let _ = Quantity::new(1.0, Unit::METER) + Quantity::new(1.0, Unit::SECOND);
    });
    assert!(bad.is_err());
}

#[test]
fn mathml_output() {
    let x = sym!(x);
    let f = x.clone().pow(sym!(2.0)) + sym!(2.0); // x^2 + 2
    let ml = f.to_mathml();
    assert!(ml.starts_with("<math xmlns=\"http://www.w3.org/1998/Math/MathML\">"));
    assert!(ml.ends_with("</math>"));
    assert!(ml.contains("<msup><mi>x</mi><mn>2</mn></msup>"));
    assert!(ml.contains("<mo>+</mo>"));
    // LaTeX rendering is untouched.
    assert_eq!(f.to_latex(), "{x}^{2} + 2");
}

#[test]
fn solve_linear_and_quadratic() {
    let x = sym!(x);

    // 2x - 4 == 0  =>  x = 2
    let linear = sym!(2.0) * x.clone() - sym!(4.0);
    assert_eq!(linear.solve("x"), vec![Expr::Const(2.0)]);

    // x^2 - 5x + 6 == 0  =>  x = 2, 3
    let quadratic = x.clone().pow(sym!(2.0)) - sym!(5.0) * x.clone() + sym!(6.0);
    let roots = quadratic.solve("x");
    assert_eq!(roots.len(), 2);
    for r in &roots {
        assert!(quadratic.eval(&[("x", r.eval(&[]))]).abs() < 1e-9);
    }

    // x + y == 0  =>  x = -y
    let transposed = x.clone() + sym!(y);
    let sol = transposed.solve("x");
    assert_eq!(sol.len(), 1);
    assert!((sol[0].eval(&[("y", 4.0)]) + 4.0).abs() < 1e-12);
}

#[test]
fn codegen_rust_tensor() {
    let f = sym!(x) + sym!(y);
    let code = f.to_rust_tensor();
    assert!(code.contains('x'));
    assert!(code.contains('y'));
    assert!(code.contains('+'));
    assert_eq!(code, tpt_sym::codegen::to_rust(&f));
}
