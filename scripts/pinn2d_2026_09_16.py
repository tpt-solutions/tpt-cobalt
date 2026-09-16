"""Extend pinn.rs with the 2-D Poisson PINN (true second-order Laplacian)."""

p = "crates/tpt-sci/src/pinn.rs"
src = open(p, encoding="utf-8").read()

old = "use tpt_autograd::{add, backward, mul, sub};"
new = "use tpt_autograd::{\n    add, backward, backward_seeded, mul, scale, sub, sum_lastdim, zero_grad,\n};"
assert old in src
src = src.replace(old, new, 1)

anchor = "#[cfg(test)]\nmod tests {"
addition = """// ------------------ 2-D Poisson PINN (true second order) -------------------
//
// The spec's "true PDE PINNs" line: with double-backward the Laplacian in
// the residual is a tape-native u_xx + u_yy - no finite differences, no
// shifted collocation points.

/// Tape-native Laplacian `u_xx + u_yy` of `net` at `points` ([N, 2] leaf,
/// columns = (x, y)). Three backward passes through the same graph: the
/// first gradient expression is itself tape-connected, so seeding it again
/// yields exact second derivatives (verified against finite differences in
/// the tests).
pub fn laplacian_2d(net: &Sequential, points: &Tensor) -> Tensor {
    let n = points.shape()[0];
    let u = net.forward(points);

    // first partials: d(sum u)/d(x_i, y_i) lands on the points leaf
    let ones = tpt_tensor::Tensor::ones(&[n, 1], points.device());
    backward_seeded(&u, &ones);
    let g1 = points.grad().expect("points must be a with_autograd leaf");

    // u_xx: re-seed the first-gradient expression with an x-column mask
    zero_grad(&u);
    let mask_x = col_mask(n, 0);
    backward_seeded(&g1, &mask_x);
    let g2x = points.grad().expect("second pass lost connectivity");

    // u_yy: y-column mask
    zero_grad(&u);
    let mask_y = col_mask(n, 1);
    backward_seeded(&g1, &mask_y);
    let g2y = points.grad().expect("third pass lost connectivity");

    // column extraction via constant selector matmuls (stays on the tape)
    let sel_x = tpt_tensor::Tensor::from_typed(vec![1.0, 0.0])
        .reshape(&[2, 1])
        .unwrap();
    let sel_y = tpt_tensor::Tensor::from_typed(vec![0.0, 1.0])
        .reshape(&[2, 1])
        .unwrap();
    let u_xx = tpt_autograd::matmul(&g2x, &sel_x);
    let u_yy = tpt_autograd::matmul(&g2y, &sel_y);
    add(&u_xx, &u_yy)
}

fn col_mask(n: usize, col: usize) -> tpt_tensor::Tensor {
    let mut mask = Vec::with_capacity(2 * n);
    for _ in 0..n {
        mask.push(if col == 0 { 1.0 } else { 0.0 });
        mask.push(if col == 1 { 1.0 } else { 0.0 });
    }
    tpt_tensor::Tensor::from_typed(mask)
        .reshape(&[n, 2])
        .unwrap()
}

/// Train a PINN for the Poisson problem `-Laplacian(u) = f` on (0,1)^2 with
/// Dirichlet boundary values: the loss is the mean squared PDE residual at
/// `interior` points plus `bc_weight` times the mean squared boundary error
/// against `bc_values` (boundary points and values pair row-wise). Returns
/// the final total loss.
pub fn train_pinn_poisson(
    net: &mut Sequential,
    interior: &Tensor,
    boundary: &Tensor,
    rhs: &Tensor,
    bc_values: &Tensor,
    optimizer: &mut dyn Optimizer,
    epochs: usize,
    bc_weight: f64,
) -> f64 {
    let mut final_loss = f64::INFINITY;
    for _epoch in 0..epochs {
        // PDE residual: u_xx + u_yy + f (rhs stores +f for -Laplacian(u) = f)
        let lap = laplacian_2d(net, interior);
        let residual = add(&lap, rhs);
        let sq = mul(&residual, &residual);
        let loss_phys = scale(&sum_lastdim(&sq), 1.0 / interior.shape()[0] as f64);

        // Dirichlet boundary loss
        let ub = net.forward(boundary);
        let berr = sub(&ub, bc_values);
        let bsql = sum_lastdim(&mul(&berr, &berr));
        let loss_bc = scale(&bsql, bc_weight / boundary.shape()[0] as f64);

        let loss = add(&loss_phys, &loss_bc);
        backward(&loss);

        let mut params = net.parameters();
        optimizer.step(&mut params);
        let params = params.into_iter().map(|p| p.with_autograd()).collect();
        net.set_parameters(params);
        final_loss = loss.to_vec::<f64>().unwrap()[0];
    }
    final_loss
}

#[cfg(test)]
mod tests {"""
assert anchor in src
src = src.replace(anchor, addition, 1)

tests = """
    #[test]
    fn laplacian_matches_finite_differences() {
        // any fixed net: the tape Laplacian must equal central-difference FD
        let mut net = pinn_mlp(2, &[12, 12], 1);
        let pts: Vec<f64> = (0..8)
            .flat_map(|i| {
                let x = 0.1 + 0.1 * i as f64;
                (0..2).map(move |j| (x, 0.2 + 0.3 * j as f64))
            })
            .flat_map(|(x, y)| vec![x, y])
            .collect();
        let points = tpt_tensor::Tensor::from_typed(pts)
            .reshape(&[16, 2])
            .unwrap()
            .with_autograd();
        let lap = laplacian_2d(&net, &points).to_vec::<f64>().unwrap();

        let h = 1e-3;
        let flat = points.to_vec::<f64>().unwrap();
        for i in 0..16 {
            let (x, y) = (flat[2 * i], flat[2 * i + 1]);
            let f = |px: f64, py: f64| {
                let inp = tpt_tensor::Tensor::from_typed(vec![px, py])
                    .reshape(&[1, 2])
                    .unwrap();
                net.forward(&inp).to_vec::<f64>().unwrap()[0]
            };
            let u_xx = (f(x + h, y) - 2.0 * f(x, y) + f(x - h, y)) / (h * h);
            let u_yy = (f(x, y + h) - 2.0 * f(x, y) + f(x, y - h)) / (h * h);
            let fd = u_xx + u_yy;
            assert!(
                (lap[i] - fd).abs() < 1e-2 * (1.0 + fd.abs()),
                "point {i}: tape {} vs fd {fd}",
                lap[i]
            );
        }
    }

    #[test]
    fn poisson_pinn_learns_the_manufactured_solution() {
        // -Laplacian(u) = 2*pi^2*sin(pi x)*sin(pi y) for u = sin(pi x)sin(pi y);
        // the exact solution is zero on the whole boundary.
        use std::f64::consts::PI;
        let mut net = pinn_mlp(2, &[24, 24], 1);
        let mut opt = AdamW::new(4e-3);

        let mut interior = Vec::new();
        let mut rhs = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let x = (i as f64 + 0.5) / 5.0;
                let y = (j as f64 + 0.5) / 5.0;
                interior.push(x);
                interior.push(y);
                rhs.push(2.0 * PI * PI * (PI * x).sin() * (PI * y).sin());
            }
        }
        let interior = tpt_tensor::Tensor::from_typed(interior)
            .reshape(&[25, 2])
            .unwrap()
            .with_autograd();
        let rhs = tpt_tensor::Tensor::from_typed(rhs).reshape(&[25, 1]).unwrap();

        let mut boundary = Vec::new();
        for k in 0..9 {
            let s = k as f64 / 8.0;
            for (x, y) in [(s, 0.0), (s, 1.0), (0.0, s), (1.0, s)] {
                boundary.push(x);
                boundary.push(y);
            }
        }
        let n_b = boundary.len() / 2;
        let boundary = tpt_tensor::Tensor::from_typed(boundary)
            .reshape(&[n_b, 2])
            .unwrap()
            .with_autograd();
        let bc_values = tpt_tensor::Tensor::from_typed(vec![0.0; n_b])
            .reshape(&[n_b, 1])
            .unwrap();

        let loss = train_pinn_poisson(
            &mut net,
            &interior,
            &boundary,
            &rhs,
            &bc_values,
            &mut opt,
            600,
            10.0,
        );
        assert!(loss < 5.0, "Poisson PINN did not converge: loss {loss}");
    }
}
"""

idx = src.rindex("\n}\n")
src = src[: idx + 1] + tests
open(p, "w", encoding="utf-8", newline="").write(src)
print("pinn.rs extended")
