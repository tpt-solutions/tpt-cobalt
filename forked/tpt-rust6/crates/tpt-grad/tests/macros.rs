//! Tests for `#[derive_grad]`, `#[derive_vmap]`, `#[derive_jit]` and their
//! composition, plus cross-checks against the runtime tape.

use tpt_grad::prelude::*;

fn approx(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-9, "{a:?} != {b:?}");
    }
}

// All three attributes stacked on one function => f, f_grad, f_vmap, f_jit.
#[derive_grad]
#[derive_vmap]
#[derive_jit]
fn f(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
    (x.clone() * y.clone()) + x.clone()
}

// `let` bindings, unary minus, scalar argument, powf, division.
#[derive_grad]
#[derive_jit]
fn poly(x: Tensor<f64>, y: Tensor<f64>, k: f64) -> Tensor<f64> {
    let a = x.clone().powf(2.0);
    let b = (y.clone() / x.clone()) * -1.0;
    (a + b) * k
}

// A scalar-valued (reduction) function: return type `f64`.
#[derive_grad]
#[derive_jit]
fn loss(x: Tensor<f64>, y: Tensor<f64>) -> f64 {
    ((x.clone() - y.clone()).powf(2.0)).mean()
}

// Single-sample function for vmap: maps a sample to a scalar.
#[derive_vmap]
fn dot_ish(x: Tensor<f64>, y: Tensor<f64>) -> f64 {
    (x.clone() * y.clone()).sum()
}

#[derive_vmap]
fn scale(x: Tensor<f64>, k: f64) -> Tensor<f64> {
    x.clone() * k
}

#[test]
fn grad_matches_value_and_runtime_tape() {
    let (v, g) = f_grad(
        tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
    );

    // Value matches the original function.
    let direct = f(
        tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
    );
    approx(&v.to_vec(), &direct.to_vec());

    // Gradients match the hand-written runtime tape.
    let tape = Tape::new();
    let x = tape.var(tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let y = tape.var(tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]));
    let z = (x * y) + x;
    let rg = z.backward();
    assert_eq!(g.len(), 2);
    approx(&g[0].to_vec(), &rg.gradient(&x).to_vec());
    approx(&g[1].to_vec(), &rg.gradient(&y).to_vec());
    approx(&g[0].to_vec(), &[6.0, 7.0, 8.0, 9.0]); // y + 1
    approx(&g[1].to_vec(), &[1.0, 2.0, 3.0, 4.0]); // x
}

#[test]
fn grad_with_let_bindings_scalar_arg_and_powf() {
    let xs = [1.0, 2.0, 4.0];
    let ys = [2.0, 3.0, 5.0];
    let k = 3.0;
    let (v, g) = poly_grad(tensor(&[3], &xs), tensor(&[3], &ys), k);
    let expect: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x * x - y / x) * k)
        .collect();
    approx(&v.to_vec(), &expect);
    approx(
        &poly(tensor(&[3], &xs), tensor(&[3], &ys), k).to_vec(),
        &expect,
    );

    // d/dx = (2x + y/x^2) * k ; d/dy = -k/x
    let dx: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (2.0 * x + y / (x * x)) * k)
        .collect();
    let dy: Vec<f64> = xs.iter().map(|x| -k / x).collect();
    approx(&g[0].to_vec(), &dx);
    approx(&g[1].to_vec(), &dy);
}

#[test]
fn grad_of_scalar_valued_loss() {
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [0.5, 2.5, 2.0, 5.0];
    let (v, g) = loss_grad(tensor(&[4], &xs), tensor(&[4], &ys));
    assert!((v - loss(tensor(&[4], &xs), tensor(&[4], &ys))).abs() < 1e-12);

    let n = xs.len() as f64;
    let expect_v: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n;
    assert!((v - expect_v).abs() < 1e-12);

    let dx: Vec<f64> = xs.iter().zip(&ys).map(|(x, y)| 2.0 * (x - y) / n).collect();
    let dy: Vec<f64> = dx.iter().map(|d| -d).collect();
    approx(&g[0].to_vec(), &dx);
    approx(&g[1].to_vec(), &dy);
}

#[test]
fn vmap_stacks_per_sample_results() {
    // Tensor-valued samples: [2, 3] batches -> [2, 3] result.
    let xb = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let yb = tensor(&[2, 3], &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let out = f_vmap(xb, yb);
    assert_eq!(out.shape(), &[2, 3]);

    let s0 = f(
        tensor(&[3], &[1.0, 2.0, 3.0]),
        tensor(&[3], &[1.0, 1.0, 1.0]),
    );
    let s1 = f(
        tensor(&[3], &[4.0, 5.0, 6.0]),
        tensor(&[3], &[2.0, 2.0, 2.0]),
    );
    let mut expect = s0.to_vec();
    expect.extend(s1.to_vec());
    approx(&out.to_vec(), &expect);
}

#[test]
fn vmap_scalar_results_stack_into_a_vector() {
    let xb = tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let yb = tensor(&[3, 2], &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let out = dot_ish_vmap(xb, yb);
    assert_eq!(out.shape(), &[3]);
    approx(&out.to_vec(), &[3.0, 14.0, 33.0]);
}

#[test]
fn vmap_shares_scalar_arguments() {
    let xb = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let out = scale_vmap(xb, 10.0);
    assert_eq!(out.shape(), &[2, 2]);
    approx(&out.to_vec(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn jit_matches_the_original_function() {
    let a = || tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = || tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    approx(&f_jit(a(), b()).to_vec(), &f(a(), b()).to_vec());

    let xs = [1.0, 2.0, 4.0];
    let ys = [2.0, 3.0, 5.0];
    approx(
        &poly_jit(tensor(&[3], &xs), tensor(&[3], &ys), 3.0).to_vec(),
        &poly(tensor(&[3], &xs), tensor(&[3], &ys), 3.0).to_vec(),
    );

    // Reduction body: documented fallback to the original function.
    let l = loss_jit(tensor(&[3], &xs), tensor(&[3], &ys));
    assert!((l - loss(tensor(&[3], &xs), tensor(&[3], &ys))).abs() < 1e-12);
}

#[test]
fn jit_broadcasts_like_the_tensor_ops() {
    let x = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let row = tensor(&[3], &[10.0, 20.0, 30.0]);
    let out = f_jit(x, row);
    assert_eq!(out.shape(), &[2, 3]);
    let x2 = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let row2 = tensor(&[3], &[10.0, 20.0, 30.0]);
    approx(&out.to_vec(), &f(x2, row2).to_vec());
}

#[test]
fn generated_functions_compose_with_each_other() {
    // f_grad of one vmap sample equals the gradient of the whole batched row.
    let (_, g) = f_grad(
        tensor(&[3], &[1.0, 2.0, 3.0]),
        tensor(&[3], &[4.0, 5.0, 6.0]),
    );
    approx(&g[0].to_vec(), &[5.0, 6.0, 7.0]);
    approx(&g[1].to_vec(), &[1.0, 2.0, 3.0]);
}
