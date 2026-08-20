//! Runtime tape tests: tensor reverse-mode AD and the scalar convenience tape.

use tpt_grad::prelude::*;

fn approx(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-9, "{a:?} != {b:?}");
    }
}

#[test]
fn tape_mul_add() {
    // z = x * y + x  =>  dz/dx = y + 1, dz/dy = x
    let tape = Tape::new();
    let xv = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let yv = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let x = tape.var(tensor(&[2, 2], &xv.to_vec()));
    let y = tape.var(tensor(&[2, 2], &yv.to_vec()));
    let z = (x * y) + x;

    approx(&z.value().to_vec(), &[6.0, 14.0, 24.0, 36.0]);
    assert_eq!(z.value().shape(), &[2, 2]);

    let g = z.backward();
    approx(&g.gradient(&x).to_vec(), &(&yv + 1.0).to_vec()); // y + 1
    approx(&g.gradient(&y).to_vec(), &xv.to_vec()); // x
    assert_eq!(g.gradient(&x).shape(), &[2, 2]);
}

#[test]
fn tape_div_sub_neg() {
    // z = -(x / y) - y  =>  dz/dx = -1/y, dz/dy = x/y^2 - 1
    let tape = Tape::new();
    let x = tape.var(tensor(&[3], &[2.0, 4.0, 6.0]));
    let y = tape.var(tensor(&[3], &[1.0, 2.0, 4.0]));
    let z = -(x / y) - y;
    approx(&z.value().to_vec(), &[-3.0, -4.0, -5.5]);
    let g = z.backward();
    approx(&g.gradient(&x).to_vec(), &[-1.0, -0.5, -0.25]);
    approx(&g.gradient(&y).to_vec(), &[1.0, 0.0, 6.0 / 16.0 - 1.0]);
}

#[test]
fn tape_powf_sum_mean() {
    // l = (x^3).sum()  => dl/dx = 3x^2
    let tape = Tape::new();
    let x = tape.var(tensor(&[3], &[1.0, 2.0, 3.0]));
    let l = x.powf(3.0).sum();
    assert!((l.value_scalar() - 36.0).abs() < 1e-12);
    approx(&l.backward().gradient(&x).to_vec(), &[3.0, 12.0, 27.0]);

    // m = (x * y).mean() => dm/dx = y / n
    let tape = Tape::new();
    let x = tape.var(tensor(&[4], &[1.0, 2.0, 3.0, 4.0]));
    let y = tape.var(tensor(&[4], &[2.0, 2.0, 2.0, 2.0]));
    let m = (x * y).mean();
    assert!((m.value_scalar() - 5.0).abs() < 1e-12);
    let g = m.backward();
    approx(&g.gradient(&x).to_vec(), &[0.5, 0.5, 0.5, 0.5]);
    approx(&g.gradient(&y).to_vec(), &[0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn tape_exp_ln_sqrt_and_scalars() {
    let tape = Tape::new();
    let x = tape.var(tensor(&[2], &[1.0, 2.0]));
    let z = (x.exp() * 2.0).ln() + x.sqrt();
    // z = ln(2) + x + sqrt(x)  =>  dz/dx = 1 + 0.5/sqrt(x)
    let g = z.sum().backward();
    approx(&g.gradient(&x).to_vec(), &[1.5, 1.0 + 0.5 / 2.0_f64.sqrt()]);
}

#[test]
fn tape_broadcast_gradient_is_reduced() {
    // x: [2,3], b: [3] broadcast row vector; grad wrt b sums over rows.
    let tape = Tape::new();
    let x = tape.var(tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = tape.var(tensor(&[3], &[1.0, 1.0, 1.0]));
    let l = (x + b).sum();
    let g = l.backward();
    assert_eq!(g.gradient(&b).shape(), &[3]);
    approx(&g.gradient(&b).to_vec(), &[2.0, 2.0, 2.0]);
    approx(&g.gradient(&x).to_vec(), &[1.0; 6]);
}

#[test]
fn tape_unused_variable_gets_zero_gradient() {
    let tape = Tape::new();
    let x = tape.var(tensor(&[2], &[1.0, 2.0]));
    let y = tape.var(tensor(&[2], &[3.0, 4.0]));
    let g = (x * x).sum().backward();
    approx(&g.gradient(&y).to_vec(), &[0.0, 0.0]);
    approx(&g.gradient(&x).to_vec(), &[2.0, 4.0]);
}

#[test]
fn scalar_tape_basics() {
    let t = ScalarTape::new();
    let x = t.var(3.0);
    let y = t.var(4.0);
    let z = x * y + x.powf(2.0) - y / x;
    assert!((z.value() - (12.0 + 9.0 - 4.0 / 3.0)).abs() < 1e-12);
    let g = z.backward();
    // dz/dx = y + 2x + y/x^2 ; dz/dy = x - 1/x
    assert!((g.gradient(&x) - (4.0 + 6.0 + 4.0 / 9.0)).abs() < 1e-12);
    assert!((g.gradient(&y) - (3.0 - 1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn scalar_tape_transcendental() {
    let t = ScalarTape::new();
    let x = t.var(0.5);
    let z = (x.exp() + x.ln()).sin() * x.sqrt();
    // Finite-difference check.
    let g = z.backward().gradient(&x);
    let f = |v: f64| ((v.exp() + v.ln()).sin()) * v.sqrt();
    let h = 1e-6;
    let fd = (f(0.5 + h) - f(0.5 - h)) / (2.0 * h);
    assert!((g - fd).abs() < 1e-6, "{g} vs {fd}");
}
