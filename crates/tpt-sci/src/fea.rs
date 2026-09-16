//! # Differentiable Linear Solve (FEA) — Phase 3, spec §5.4
//!
//! A differentiable dense linear solve `K u = f` (the inner kernel of any static
//! FEA/PDE discretization), with an adjoint VJP so gradients flow back into the
//! stiffness matrix `K` and the force vector `f`.
//!
//! The forward pass uses `tpt-math-linalg-dense`'s LU solver. The backward pass
//! solves the adjoint system `K^T λ = grad_output` to obtain gradients w.r.t.
//! `K` and `f`.

// `K` is the conventional FEA stiffness-matrix symbol; keep the capital.
#![allow(non_snake_case)]

use tpt_autograd::custom_vjp;
use tpt_math_linalg_dense::{DMatrix, DVector};
use tpt_tensor::{DType, Tensor};

/// Differentiable dense linear solve: `K u = f`.
///
/// Returns the solution `u` as a differentiable tensor. The autograd tape
/// records the solve so `backward` propagates gradients into `K` and `f`.
///
/// # Arguments
/// * `K` — Stiffness matrix `[n, n]`, must be square and invertible.
/// * `f` — Force vector `[n]` or `[n, 1]`.
///
/// # Panics
/// Panics if `K` is not square, if dimensions don't match, or if the solve fails.
pub fn solve_linear(K: &Tensor, f: &Tensor) -> Tensor {
    // Validate inputs
    assert_eq!(K.dtype(), DType::F64, "solve_linear requires f64");
    assert_eq!(f.dtype(), DType::F64, "solve_linear requires f64");
    assert_eq!(K.ndim(), 2, "K must be 2-D");
    let n = K.shape()[0];
    assert_eq!(K.shape()[1], n, "K must be square");

    // Handle f as vector [n] or [n, 1] (same element order either way)
    let ok_shape = f.ndim() == 1 || (f.ndim() == 2 && f.shape()[1] == 1);
    assert!(ok_shape, "f must be [n] or [n, 1]");
    assert_eq!(f.shape()[0], n, "f dimension mismatch");
    let f_vec = f.to_vec::<f64>().unwrap();

    // Convert to DMatrix/DVector for the solve
    let K_data = K.to_vec::<f64>().unwrap();
    let K_mat = DMatrix::from_row_slice(n, n, &K_data);
    let f_vec = DVector::from_vec(f_vec);

    // Forward solve: K u = f
    let u_vec = K_mat
        .solve(&f_vec)
        .expect("Linear solve failed: K is singular");

    // Result tensor with autograd
    let u_data: Vec<f64> = (0..n).map(|i| u_vec[i]).collect();
    let u_tensor = Tensor::from_typed(u_data).reshape(&[n]).unwrap();

    // Only build autograd if K or f requires grad
    let K_requires = K.requires_grad();
    let f_requires = f.requires_grad();
    if !K_requires && !f_requires {
        return u_tensor;
    }

    // Get autograd nodes
    let K_node = K.node();
    let f_node = f.node();
    let mut parents: Vec<_> = Vec::new();
    if let Some(ref kn) = K_node {
        parents.push(kn.clone());
    }
    if let Some(ref fn_) = f_node {
        parents.push(fn_.clone());
    }

    // Record the operation with custom backward using custom_vjp
    custom_vjp(u_tensor.clone(), parents, move |grad_u: &Tensor| {
        // Adjoint solve: K^T λ = grad_u
        // grad_u shape is [n]
        let grad_u_data = grad_u.to_vec::<f64>().unwrap();
        let grad_u_vec = DVector::from_vec(grad_u_data);

        // K^T = K.transpose() for symmetric K, but we handle general case
        let K_t = K_mat.transpose();
        let lambda_vec = K_t
            .solve(&grad_u_vec)
            .expect("Adjoint solve failed: K^T is singular");

        // Gradient w.r.t. f: dL/df = λ
        if let Some(f_node) = &f_node {
            let lambda_data: Vec<f64> = (0..n).map(|i| lambda_vec[i]).collect();
            let grad_f = Tensor::from_typed(lambda_data).reshape(&[n]).unwrap();
            f_node.accumulate_grad(&grad_f);
        }

        // Gradient w.r.t. K: dL/dK = -λ ⊗ u^T (outer product)
        // For each element K_ij: dL/dK_ij = -λ_i * u_j
        if let Some(K_node) = &K_node {
            let u_data = u_tensor.to_vec::<f64>().unwrap();
            let lambda_data: Vec<f64> = (0..n).map(|i| lambda_vec[i]).collect();

            let mut grad_K = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    grad_K[i * n + j] = -lambda_data[i] * u_data[j];
                }
            }
            let grad_K_tensor = Tensor::from_typed(grad_K).reshape(&[n, n]).unwrap();
            K_node.accumulate_grad(&grad_K_tensor);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn solve_linear_forward() {
        // K = [[4, 1], [1, 3]], f = [1, 2]
        // Solution: u = K^{-1} f = [0.0909, 0.6364] approximately
        let K = Tensor::from_typed(vec![4.0_f64, 1.0, 1.0, 3.0])
            .reshape(&[2, 2])
            .unwrap();
        let f = Tensor::from_typed(vec![1.0_f64, 2.0]);
        let u = solve_linear(&K, &f);
        let u_data = u.to_vec::<f64>().unwrap();

        // Verify K @ u ≈ f
        let K_data = K.to_vec::<f64>().unwrap();
        let Ku = vec![
            K_data[0] * u_data[0] + K_data[1] * u_data[1],
            K_data[2] * u_data[0] + K_data[3] * u_data[1],
        ];
        assert!((Ku[0] - 1.0).abs() < 1e-10);
        assert!((Ku[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_backprop_to_f() {
        // Simple 1D case: K = [2], f = [4] -> u = [2]
        // Loss = u^2 = 4, dL/df = dL/du * du/df = 2u * K^{-1} = 4 * 0.5 = 2
        let K = Tensor::from_typed(vec![2.0_f64]).reshape(&[1, 1]).unwrap();
        let f = Tensor::from_typed(vec![4.0_f64]).with_autograd();
        let u = solve_linear(&K, &f);

        // Loss = sum(u^2) — must use the differentiable autograd mul
        let loss = tpt_autograd::mul(&u, &u);
        backward(&loss);

        // dL/df = dL/du * du/df = 2u * K^{-1} = 4 * 0.5 = 2
        let grad_f = f.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((grad_f - 2.0).abs() < 1e-6, "grad_f = {}", grad_f);
    }

    #[test]
    fn solve_linear_backprop_to_K() {
        // K = [[2, 0], [0, 2]], f = [2, 2] -> u = [1, 1]
        // Loss = sum(u) = 2
        // dL/dK_ij = -λ_i * u_j, where K^T λ = grad_u = [1, 1]
        // λ = [0.5, 0.5], so dL/dK = [[-0.5, -0.5], [-0.5, -0.5]]
        let K = Tensor::from_typed(vec![2.0_f64, 0.0, 0.0, 2.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let f = Tensor::from_typed(vec![2.0_f64, 2.0]);
        let u = solve_linear(&K, &f);

        // Loss = sum(u)
        let ones = Tensor::from_typed(vec![1.0_f64, 1.0]);
        let loss = tpt_autograd::add(&u, &ones); // u + 1, then sum
        let loss = tpt_autograd::sum_lastdim(&loss.reshape(&[1, 2]).unwrap()); // sum over last dim
        backward(&loss);

        let grad_K = K.grad().unwrap().to_vec::<f64>().unwrap();
        // Expected: -λ ⊗ u^T = -[0.5, 0.5] ⊗ [1, 1] = [[-0.5, -0.5], [-0.5, -0.5]]
        for g in &grad_K {
            assert!((g + 0.5).abs() < 1e-6, "grad_K = {:?}", grad_K);
        }
    }

    #[test]
    fn solve_linear_backprop_both() {
        // K = [[2, 1], [1, 2]], f = [3, 3]
        // K^{-1} = 1/3 * [[2, -1], [-1, 2]], u = [1, 1]
        // Loss = 0.5 * ||u||^2 = 1
        let K = Tensor::from_typed(vec![2.0_f64, 1.0, 1.0, 2.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let f = Tensor::from_typed(vec![3.0_f64, 3.0]).with_autograd();
        let u = solve_linear(&K, &f);

        // Loss = 0.5 * sum(u^2)
        let u_sq = tpt_autograd::mul(&u, &u);
        let half = Tensor::from_typed(vec![0.5_f64]);
        let loss = tpt_autograd::mul(&u_sq, &half);
        let loss = tpt_autograd::sum_lastdim(&loss.reshape(&[1, 2]).unwrap());
        backward(&loss);

        let grad_f = f.grad().unwrap().to_vec::<f64>().unwrap();
        let grad_K = K.grad().unwrap().to_vec::<f64>().unwrap();

        // u = [1, 1], grad_u = u = [1, 1]
        // Adjoint: K^T λ = [1, 1] -> λ = [1/3, 1/3]
        // dL/df = λ = [1/3, 1/3]
        assert!((grad_f[0] - 1.0 / 3.0).abs() < 1e-4);
        assert!((grad_f[1] - 1.0 / 3.0).abs() < 1e-4);

        // dL/dK = -λ ⊗ u^T = -[1/3, 1/3] ⊗ [1, 1] = [[-1/3, -1/3], [-1/3, -1/3]]
        for g in &grad_K {
            assert!((g + 1.0 / 3.0).abs() < 1e-4, "grad_K = {:?}", grad_K);
        }
    }
}
