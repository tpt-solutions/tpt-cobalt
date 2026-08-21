//! # tpt-autograd — The Tensor-Graph Tape (Phase 1, spec §5.2)
//!
//! A tensor-aware reverse-mode autodiff engine built over [`tpt_tensor::Tensor`].
//! Starting point: `tpt-rust6::tpt-grad` (proc-macro autograd). This scaffold
//! provides eager execution with a reverse-mode tape; `tpt-tensor` owns the
//! per-tensor [`AutogradNode`] slot and this crate fills it in.
//!
//! Responsibilities covered by this scaffold:
//! - Dynamic (eager) graph execution
//! - Backward pass with gradient accumulation
//! - In-place mutation versioning (via `tpt-tensor`'s `mark_mutated`)
//!
//! Planned (later in Phase 1 / §5.2): traced/compiled graphs, VJP
//! registration for custom ops (FEA/ODE solvers), gradient checkpointing, and
//! cross-device gradient accumulation.

use std::collections::HashSet;
use std::sync::Arc;

use tpt_tensor::{AutogradNode, Tensor};

/// Differentiable element-wise add. Gradient flows equally to both inputs.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.add(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let parents = collect_parents(&a_node, &b_node);
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    record(result, parents, move |grad: &Tensor| {
        if let Some(an) = &a_node {
            an.accumulate_grad(&grad.sum_to(&a_shape));
        }
        if let Some(bn) = &b_node {
            bn.accumulate_grad(&grad.sum_to(&b_shape));
        }
    })
}

/// Differentiable element-wise mul. `grad_a = grad * b`, `grad_b = grad * a`.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.mul(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let parents = collect_parents(&a_node, &b_node);
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let (a_fwd, b_fwd) = (a.clone(), b.clone());
    record(result, parents, move |grad: &Tensor| {
        if let Some(an) = &a_node {
            an.accumulate_grad(&grad.sum_to(&a_shape).mul(&b_fwd));
        }
        if let Some(bn) = &b_node {
            bn.accumulate_grad(&grad.sum_to(&b_shape).mul(&a_fwd));
        }
    })
}

/// Differentiable 2-D matrix multiply. `grad_a = grad @ b^T`, `grad_b = a^T @ grad`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.matmul(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let parents = collect_parents(&a_node, &b_node);
    let b_t = b.transpose();
    let a_t = a.transpose();
    record(result, parents, move |grad: &Tensor| {
        if let Some(an) = &a_node {
            an.accumulate_grad(&grad.matmul(&b_t));
        }
        if let Some(bn) = &b_node {
            an_accumulate(bn, &a_t.matmul(grad));
        }
    })
}

#[inline]
fn an_accumulate(node: &Arc<AutogradNode>, g: &Tensor) {
    node.accumulate_grad(g);
}

/// Gather the require-grad parent nodes into a `Vec` for the tape.
fn collect_parents(
    a: &Option<Arc<AutogradNode>>,
    b: &Option<Arc<AutogradNode>>,
) -> Vec<Arc<AutogradNode>> {
    let mut out = Vec::new();
    if let Some(an) = a {
        out.push(an.clone());
    }
    if let Some(bn) = b {
        out.push(bn.clone());
    }
    out
}

/// Attach an `AutogradNode` (with parents + backward) to `result`.
fn record(
    mut result: Tensor,
    parents: Vec<Arc<AutogradNode>>,
    backward: impl Fn(&Tensor) + Send + Sync + 'static,
) -> Tensor {
    let node = AutogradNode::new(parents, Box::new(backward));
    result.set_node(Arc::new(node));
    result
}

/// Reverse-mode backward from a (possibly non-scalar) output.
///
/// Seeds the output gradient with ones (i.e. differentiates the implicit sum),
/// then walks the tape in reverse topological order, scattering each node's
/// gradient into its parents. Leaf gradients are then readable via
/// [`Tensor::grad`].
pub fn backward(output: &Tensor) {
    let node = output
        .node()
        .expect("backward: output tensor has no autograd node (call .with_autograd() on parameters)");
    node.set_grad(Tensor::ones(output.shape(), output.device()));
    for n in topo(&node) {
        if let Some(g) = n.grad() {
            n.run_backward(&g);
        }
    }
}

/// Reverse topological order: `start` first, then its children, so that every
/// node's gradient is fully accumulated before its own backward runs.
fn topo(start: &Arc<AutogradNode>) -> Vec<Arc<AutogradNode>> {
    let mut seen: HashSet<*const AutogradNode> = HashSet::new();
    let mut order: Vec<Arc<AutogradNode>> = Vec::new();
    fn visit(
        n: &Arc<AutogradNode>,
        seen: &mut HashSet<*const AutogradNode>,
        order: &mut Vec<Arc<AutogradNode>>,
    ) {
        let p = Arc::as_ptr(n);
        if seen.insert(p) {
            for parent in n.parents() {
                visit(parent, seen, order);
            }
            order.push(n.clone());
        }
    }
    visit(start, &mut seen, &mut order);
    order.reverse();
    order
}

/// Differentiable element-wise subtraction.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.sub(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    record(
        result,
        collect_parents(&a_node, &b_node),
        move |grad: &Tensor| {
            if let Some(an) = &a_node {
                an.accumulate_grad(&grad.sum_to(&a_shape));
            }
            if let Some(bn) = &b_node {
                bn.accumulate_grad(&grad.sum_to(&b_shape).scale(-1.0));
            }
        },
    )
}

/// Differentiable element-wise division.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.div(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let (a_fwd, b_fwd) = (a.clone(), b.clone());
    record(
        result,
        collect_parents(&a_node, &b_node),
        move |grad: &Tensor| {
            if let Some(an) = &a_node {
                an.accumulate_grad(&grad.sum_to(&a_shape).div(&b_fwd));
            }
            if let Some(bn) = &b_node {
                let gb = grad.mul(&a_fwd).div(&b_fwd.mul(&b_fwd)).scale(-1.0);
                bn.accumulate_grad(&gb.sum_to(&b_shape));
            }
        },
    )
}

/// Differentiable negation.
pub fn neg(a: &Tensor) -> Tensor {
    let result = a.neg();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        node.accumulate_grad(&grad.sum_to(&shape).scale(-1.0));
    })
}

/// Differentiable exponential.
pub fn exp(a: &Tensor) -> Tensor {
    let result = a.exp();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let out = result.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        node.accumulate_grad(&grad.sum_to(&shape).mul(&out));
    })
}

/// Differentiable natural log.
pub fn log(a: &Tensor) -> Tensor {
    let result = a.log();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let a_fwd = a.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        node.accumulate_grad(&grad.sum_to(&shape).div(&a_fwd));
    })
}

/// Differentiable absolute value (gradient = sign(a) · grad).
pub fn abs(a: &Tensor) -> Tensor {
    let result = a.abs();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let a_fwd = a.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap();
        let av = a_fwd.to_vec::<f64>().unwrap();
        let ga: Vec<f64> = gv.iter().zip(&av).map(|(x, y)| x * y.signum()).collect();
        node.accumulate_grad(&Tensor::from_typed(ga).reshape(&shape).unwrap());
    })
}

/// Differentiable sum of all elements (gradient = ones).
pub fn sum(a: &Tensor) -> Tensor {
    let result = a.sum_all();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    let dev = a.device();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap()[0];
        node.accumulate_grad(&Tensor::ones(&shape, dev).scale(gv));
    })
}

/// Differentiable mean of all elements (gradient = ones / n).
pub fn mean(a: &Tensor) -> Tensor {
    let n = a.numel() as f64;
    let result = a.mean_all();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    let dev = a.device();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap()[0] / n;
        node.accumulate_grad(&Tensor::ones(&shape, dev).scale(gv));
    })
}

/// Differentiable sigmoid.
pub fn sigmoid(a: &Tensor) -> Tensor {
    let result = a.sigmoid();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let out = result.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let one_minus = out.ones_like().sub(&out);
        node.accumulate_grad(&grad.sum_to(&shape).mul(&out).mul(&one_minus));
    })
}

/// Differentiable softmax over the last axis.
pub fn softmax(a: &Tensor) -> Tensor {
    let result = a.softmax();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let out = result.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap();
        let ov = out.to_vec::<f64>().unwrap();
        let mut grad_v = vec![0.0f64; gv.len()];
        if shape.len() == 2 {
            let (r, c) = (shape[0], shape[1]);
            for i in 0..r {
                let dot: f64 = (0..c).map(|j| gv[i * c + j] * ov[i * c + j]).sum();
                for j in 0..c {
                    grad_v[i * c + j] = gv[i * c + j] - ov[i * c + j] * dot;
                }
            }
        } else {
            let dot: f64 = gv.iter().zip(&ov).map(|(x, y)| x * y).sum();
            for j in 0..gv.len() {
                grad_v[j] = gv[j] - ov[j] * dot;
            }
        }
        node.accumulate_grad(&Tensor::from_typed(grad_v).reshape(&shape).unwrap());
    })
}

/// Sum over the last axis of a 2-D tensor (per-row), returning `[batch, 1]`.
pub fn sum_lastdim(a: &Tensor) -> Tensor {
    let v = a.to_vec::<f64>().unwrap();
    let (r, c) = (a.shape()[0], a.shape()[1]);
    let out: Vec<f64> = (0..r).map(|i| v[i * c..(i + 1) * c].iter().sum()).collect();
    let result = Tensor::from_typed(out).reshape(&[r, 1]).unwrap();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap();
        let mut grad_v = vec![0.0f64; r * c];
        for i in 0..r {
            for j in 0..c {
                grad_v[i * c + j] = gv[i];
            }
        }
        node.accumulate_grad(&Tensor::from_typed(grad_v).reshape(&shape).unwrap());
    })
}

/// Differentiable log-softmax over the last axis (2-D input).
pub fn log_softmax(a: &Tensor) -> Tensor {
    let e = exp(a);
    let lse = log(&sum_lastdim(&e)); // [batch, 1]
    sub(a, &lse) // broadcasts [batch, 1] over [batch, c]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_add_mul_chain() {
        // y = a*b + c ; a=2, b=3, c=1 -> y=7
        // dy/da = b = 3, dy/db = a = 2, dy/dc = 1
        let a = Tensor::from_typed(vec![2.0_f64]).with_autograd();
        let b = Tensor::from_typed(vec![3.0_f64]).with_autograd();
        let c = Tensor::from_typed(vec![1.0_f64]).with_autograd();
        let y = add(&mul(&a, &b), &c);
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![7.0]);
        backward(&y);
        assert_eq!(a.grad().unwrap().to_vec::<f64>().unwrap(), vec![3.0]);
        assert_eq!(b.grad().unwrap().to_vec::<f64>().unwrap(), vec![2.0]);
        assert_eq!(c.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0]);
    }

    #[test]
    fn backward_matmul() {
        // Y = A @ B ; seed = ones => dA = ones @ B^T, dB = A^T @ ones
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let b = Tensor::from_typed(vec![5.0_f64, 6.0, 7.0, 8.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let y = matmul(&a, &b);
        backward(&y);
        assert_eq!(
            a.grad().unwrap().to_vec::<f64>().unwrap(),
            vec![11.0, 15.0, 11.0, 15.0]
        );
        assert_eq!(
            b.grad().unwrap().to_vec::<f64>().unwrap(),
            vec![4.0, 4.0, 6.0, 6.0]
        );
    }

    #[test]
    fn no_grad_short_circuits() {
        let a = Tensor::from_typed(vec![2.0_f64]);
        let b = Tensor::from_typed(vec![3.0_f64]);
        let y = add(&a, &b); // neither requires grad -> no node
        assert!(!y.requires_grad());
    }
}
