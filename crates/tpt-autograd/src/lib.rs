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

pub mod accumulate;
pub use accumulate::GradAccumulator;
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
///
/// The VJP is composed from tape-connected ops, so gradients themselves carry a
/// graph and double-backward (second-order derivatives) works.
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
        // Elementwise product FIRST (in broadcast space, via tape ops so the
        // second-order path through the forward operand stays connected),
        // THEN a tracked reduction back to the parent's shape.
        if let Some(an) = &a_node {
            let g = mul(grad, &b_fwd);
            an.accumulate_grad(&sum_to_tracked(&g, &a_shape));
        }
        if let Some(bn) = &b_node {
            let g = mul(grad, &a_fwd);
            bn.accumulate_grad(&sum_to_tracked(&g, &b_shape));
        }
    })
}

/// Tape-connected transpose: raw view plus a recorded linear node whose
/// backward transposes the incoming gradient back. Needed so `matmul`'s VJP
/// keeps its inputs on the tape (the raw `Tensor::transpose` detaches).
fn transpose_tracked(t: &Tensor) -> Tensor {
    let result = t.transpose();
    if !t.requires_grad() {
        return result;
    }
    let node = t.node().expect("autograd node present when requires_grad");
    let parent = node.clone();
    record(result, vec![node], move |grad: &Tensor| {
        parent.accumulate_grad(&grad.transpose());
    })
}

/// Differentiable 2-D matrix multiply. `grad_a = grad @ b^T`, `grad_b = a^T @ grad`.
/// VJP composed from tape ops (double-backward capable).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.matmul(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let parents = collect_parents(&a_node, &b_node);
    let b_t = transpose_tracked(b);
    let a_t = transpose_tracked(a);
    record(result, parents, move |grad: &Tensor| {
        if let Some(an) = &a_node {
            an.accumulate_grad(&matmul(grad, &b_t));
        }
        if let Some(bn) = &b_node {
            bn.accumulate_grad(&matmul(&a_t, grad));
        }
    })
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

/// Like [`record`], but the backward closure receives a cell that will be
/// filled with the output tensor (node attached) *after* recording. Use this
/// when the VJP must reference the op's own forward output as a tape-connected
/// operand (e.g. `exp`, `sigmoid`: the gradient expression contains the output,
/// whose node only exists once recording completes).
fn record_deferred(
    mut result: Tensor,
    parents: Vec<Arc<AutogradNode>>,
    make_backward: impl FnOnce(Arc<std::sync::Mutex<Option<Tensor>>>) -> Box<dyn Fn(&Tensor) + Send + Sync>,
) -> Tensor {
    let cell = Arc::new(std::sync::Mutex::new(None));
    let backward = make_backward(cell.clone());
    result.set_node(Arc::new(AutogradNode::new(parents, backward)));
    *cell.lock().unwrap() = Some(result.clone());
    result
}

/// Tape-connected version of `Tensor::sum_to`.
///
/// Identity when the shapes already match; otherwise a recorded *linear* node
/// whose backward broadcasts the incoming gradient back to the original shape.
/// Keeping this link alive is what preserves second-order connectivity through
/// broadcasting reduction points (a raw `sum_to` would silently detach the
/// gradient expression from the upstream graph and truncate double-backward).
pub(crate) fn sum_to_tracked(t: &Tensor, shape: &[usize]) -> Tensor {
    if t.shape() == shape {
        return t.clone();
    }
    let result = t.sum_to(shape);
    if !t.requires_grad() {
        return result;
    }
    let node = t.node().expect("autograd node present when requires_grad");
    let orig = t.shape().to_vec();
    let reduced = result.shape().to_vec();
    let parent = node.clone();
    record(result, vec![node], move |grad: &Tensor| {
        let gv = grad.to_vec::<f64>().unwrap();
        let bi = broadcast_flat_indices(&orig, &reduced);
        let expanded: Vec<f64> = bi.iter().map(|&s| gv[s]).collect();
        parent.accumulate_grad(&Tensor::from_typed(expanded).reshape(&orig).unwrap());
    })
}

/// Flat target indices into a `source`-shaped row-major buffer for broadcasting
/// `source` up to `target` (dims aligned from the right; a source dim of 1 or a
/// missing dim repeats). Inverse of the gather used by `Tensor::sum_to`.
fn broadcast_flat_indices(target: &[usize], source: &[usize]) -> Vec<usize> {
    let trank = target.len();
    let srank = source.len();
    assert!(srank <= trank, "broadcast: source rank exceeds target rank");
    let mut strides = vec![1usize; srank];
    for i in (0..srank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * source[i + 1];
    }
    let total: usize = target.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; trank];
    for _ in 0..total {
        let mut s = 0usize;
        for d in 0..trank {
            let sd = d + srank - trank;
            let coord = if source[sd] == 1 { 0 } else { idx[d] };
            s += coord * strides[sd];
        }
        out.push(s);
        for d in (0..trank).rev() {
            idx[d] += 1;
            if idx[d] < target[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Register a custom vector-Jacobian product (VJP) for a user op.
///
/// `forward` is the op's output tensor (typically built from raw element values
/// via `Tensor::from_typed`); `parents` are the autograd nodes of the inputs that
/// the VJP scatters into; `backward` receives the output gradient and must call
/// `AccumulateGrad` on each parent (`AutogradNode::accumulate_grad`).
///
/// This is the Phase 3 VJP-registration surface used by the scientific-computing
/// glue crate (`tpt-sci`) to make ODE/FEA solvers differentiable.
pub fn custom_vjp(
    forward: Tensor,
    parents: Vec<Arc<AutogradNode>>,
    backward: impl Fn(&Tensor) + Send + Sync + 'static,
) -> Tensor {
    record(forward, parents, backward)
}

/// Reverse-mode backward from a (possibly non-scalar) output.
///
/// Seeds the output gradient with ones (i.e. differentiates the implicit sum),
/// then walks the tape in reverse topological order, scattering each node's
/// gradient into its parents. Leaf gradients are then readable via
/// [`Tensor::grad`].
pub fn backward(output: &Tensor) {
    backward_seeded(output, &Tensor::ones(output.shape(), output.device()));
}

/// Reverse-mode backward from a (possibly non-scalar) output with an explicit
/// gradient seed `seed` (i.e. differentiates `sum(output * seed)`).
pub fn backward_seeded(output: &Tensor, seed: &Tensor) {
    let node = output
        .node()
        .expect("backward_seeded: output tensor has no autograd node (call .with_autograd() on parameters)");
    node.set_grad(seed.clone());
    for n in topo(&node) {
        if let Some(g) = n.grad() {
            n.run_backward(&g);
        }
    }
}

/// Clear the accumulated gradient on every node reachable from `root`
/// (inclusive). Call between a first-order backward and a second-order
/// (double-backward) pass so the second pass starts from a clean slate.
pub fn zero_grad(root: &Tensor) {
    let node = root.node().expect("zero_grad: tensor has no autograd node");
    for n in topo(&node) {
        n.zero_grad();
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

/// Differentiable element-wise division. VJP composed from tape ops
/// (double-backward capable).
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
                let g = div(grad, &b_fwd);
                an.accumulate_grad(&sum_to_tracked(&g, &a_shape));
            }
            if let Some(bn) = &b_node {
                let num = mul(grad, &a_fwd);
                let den = mul(&b_fwd, &b_fwd); // b² — both operands share b's node
                let g = neg(&div(&num, &den));
                bn.accumulate_grad(&sum_to_tracked(&g, &b_shape));
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

/// Differentiable exponential. VJP composed from tape ops (double-backward
/// capable: d²eˣ/dx² = eˣ flows through the captured forward output's node).
pub fn exp(a: &Tensor) -> Tensor {
    let result = a.exp();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    let parent = node.clone();
    record_deferred(result, vec![node], move |cell| {
        Box::new(move |grad: &Tensor| {
            let out = cell.lock().unwrap().as_ref().unwrap().clone();
            let g = mul(grad, &out);
            parent.accumulate_grad(&sum_to_tracked(&g, &shape));
        })
    })
}

/// Differentiable natural log. VJP composed from tape ops (double-backward
/// capable: d²ln(x)/dx² = −1/x² flows through the captured input's node).
pub fn log(a: &Tensor) -> Tensor {
    let result = a.log();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let a_fwd = a.clone();
    let shape = a.shape().to_vec();
    record(result, vec![node.clone()], move |grad: &Tensor| {
        let g = div(grad, &a_fwd);
        node.accumulate_grad(&sum_to_tracked(&g, &shape));
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

/// Differentiable sigmoid. VJP composed from tape ops (double-backward
/// capable: σ'' = σ(1−σ)(1−2σ) flows through the captured forward output).
pub fn sigmoid(a: &Tensor) -> Tensor {
    let result = a.sigmoid();
    if !a.requires_grad() {
        return result;
    }
    let node = a.node().expect("autograd node present when requires_grad");
    let shape = a.shape().to_vec();
    let dev = a.device();
    let parent = node.clone();
    record_deferred(result, vec![node], move |cell| {
        Box::new(move |grad: &Tensor| {
            let out = cell.lock().unwrap().as_ref().unwrap().clone();
            let one_minus = sub(&Tensor::ones(&shape, dev), &out);
            let g = mul(&mul(grad, &out), &one_minus);
            parent.accumulate_grad(&sum_to_tracked(&g, &shape));
        })
    })
}

/// Differentiable softmax over the last axis (any rank).
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
        if shape.len() >= 2 {
            // rows of length `c` along the last axis
            let c = shape[shape.len() - 1];
            let rows = gv.len() / c;
            for i in 0..rows {
                let base = i * c;
                let dot: f64 = (0..c).map(|j| gv[base + j] * ov[base + j]).sum();
                for j in 0..c {
                    grad_v[base + j] = gv[base + j] - ov[base + j] * dot;
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

/// Differentiable batched 3-D matrix multiply: `[B, M, K] @ [B, K, N]`.
/// Per-batch gradients: `grad_a[b] = grad[b] @ b[b]^T`, `grad_b[b] = a[b]^T @ grad[b]`.
pub fn bmm(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.bmm(b);
    if !a.requires_grad() && !b.requires_grad() {
        return result;
    }
    let a_node = a.node();
    let b_node = b.node();
    let parents = collect_parents(&a_node, &b_node);
    let shape_a = a.shape().to_vec();
    let shape_b = b.shape().to_vec();
    let batch = shape_a[0];
    let (_, m, k) = (shape_a[0], shape_a[1], shape_a[2]);
    let (_, _, n) = (shape_b[0], shape_b[1], shape_b[2]);
    let a_fwd = a.clone();
    let b_fwd = b.clone();
    record(result, parents, move |grad: &Tensor| {
        let g = grad.to_vec::<f64>().unwrap();
        let av = a_fwd.to_vec::<f64>().unwrap();
        let bv = b_fwd.to_vec::<f64>().unwrap();
        // grad_a[b] = grad[b] @ b[b]^T  ([M,N] @ [N,K] -> [M,K])
        if let Some(an) = &a_node {
            let mut ga = vec![0.0f64; batch * m * k];
            for bb in 0..batch {
                let g_off = bb * m * n;
                let b_off = bb * k * n;
                let o_off = bb * m * k;
                for i in 0..m {
                    for j in 0..k {
                        let mut s = 0.0;
                        for kk in 0..n {
                            s += g[g_off + i * n + kk] * bv[b_off + j * n + kk];
                        }
                        ga[o_off + i * k + j] = s;
                    }
                }
            }
            an.accumulate_grad(&Tensor::from_typed(ga).reshape(&shape_a).unwrap());
        }
        // grad_b[b] = a[b]^T @ grad[b]  ([K,M] @ [M,N] -> [K,N])
        if let Some(bn) = &b_node {
            let mut gb = vec![0.0f64; batch * k * n];
            for bb in 0..batch {
                let g_off = bb * m * n;
                let a_off = bb * m * k;
                let o_off = bb * k * n;
                for i in 0..k {
                    for j in 0..n {
                        let mut s = 0.0;
                        for kk in 0..m {
                            s += av[a_off + kk * k + i] * g[g_off + kk * n + j];
                        }
                        gb[o_off + i * n + j] = s;
                    }
                }
            }
            bn.accumulate_grad(&Tensor::from_typed(gb).reshape(&shape_b).unwrap());
        }
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

    #[test]
    fn double_backward_exp() {
        // d/dx e^x = e^x ; d²/dx² e^x = e^x
        let x = Tensor::from_typed(vec![1.0_f64]).with_autograd();
        let y = exp(&x);
        backward(&y);
        let g = x.grad().unwrap();
        assert!((g.to_vec::<f64>().unwrap()[0] - std::f64::consts::E).abs() < 1e-12);
        zero_grad(&y);
        // second pass: differentiate the gradient expression itself
        backward_seeded(&g, &Tensor::ones(g.shape(), g.device()));
        let gg = x.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!(
            (gg - std::f64::consts::E).abs() < 1e-12,
            "expected e, got {gg}"
        );
    }

    #[test]
    fn double_backward_square_mixed_partials() {
        // y = a*a*b ; dy/da = 2ab ; d²y/da² = 2b ; d²y/dadb = 2a
        let a = Tensor::from_typed(vec![2.0_f64]).with_autograd();
        let b = Tensor::from_typed(vec![3.0_f64]).with_autograd();
        let t = mul(&a, &a);
        let y = mul(&t, &b);
        backward(&y);
        let ga = a.grad().unwrap();
        assert!((ga.to_vec::<f64>().unwrap()[0] - 12.0).abs() < 1e-12); // 2ab
        let gb = b.grad().unwrap();
        assert!((gb.to_vec::<f64>().unwrap()[0] - 4.0).abs() < 1e-12); // a²
        zero_grad(&y);
        // second pass over dy/da: read off d²y/da² (in a) and d²y/dadb (in b)
        backward_seeded(&ga, &Tensor::ones(ga.shape(), ga.device()));
        let d2aa = a.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((d2aa - 6.0).abs() < 1e-12, "d²y/da² expected 2b=6, got {d2aa}");
        let d2ab = b.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((d2ab - 4.0).abs() < 1e-12, "d²y/dadb expected 2a=4, got {d2ab}");
    }

    #[test]
    fn double_backward_log_and_sigmoid() {
        // f(x) = ln(x) : f''(x) = -1/x² ; x=2 -> -0.25
        let x = Tensor::from_typed(vec![2.0_f64]).with_autograd();
        let y = log(&x);
        backward(&y);
        let g = x.grad().unwrap();
        assert!((g.to_vec::<f64>().unwrap()[0] - 0.5).abs() < 1e-12);
        zero_grad(&y);
        backward_seeded(&g, &Tensor::ones(g.shape(), g.device()));
        let gg = x.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((gg + 0.25).abs() < 1e-12, "f''(ln) expected -0.25, got {gg}");

        // sigmoid: s''(x) = s(1-s)(1-2s); x=0.7
        let z = Tensor::from_typed(vec![0.7_f64]).with_autograd();
        let s = sigmoid(&z);
        backward(&s);
        let sv = s.to_vec::<f64>().unwrap()[0];
        let gs = z.grad().unwrap();
        let expect1 = sv * (1.0 - sv);
        assert!((gs.to_vec::<f64>().unwrap()[0] - expect1).abs() < 1e-12);
        zero_grad(&s);
        backward_seeded(&gs, &Tensor::ones(gs.shape(), gs.device()));
        let expect2 = sv * (1.0 - sv) * (1.0 - 2.0 * sv);
        let got = z.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((got - expect2).abs() < 1e-12, "σ'' expected {expect2}, got {got}");
    }

    #[test]
    fn double_backward_matmul() {
        // Y = A @ B with A,B [1,1]: y=a*b; d²y/dadb = 1
        let a = Tensor::from_typed(vec![2.0_f64])
            .reshape(&[1, 1])
            .unwrap()
            .with_autograd();
        let b = Tensor::from_typed(vec![3.0_f64])
            .reshape(&[1, 1])
            .unwrap()
            .with_autograd();
        let y = matmul(&a, &b);
        backward(&y);
        let ga = a.grad().unwrap();
        assert_eq!(ga.to_vec::<f64>().unwrap(), vec![3.0]);
        zero_grad(&y);
        backward_seeded(&ga, &Tensor::ones(ga.shape(), ga.device()));
        assert_eq!(b.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0]);
        // d(dy/da)/da = 0 (the seed @ B^T expression does not depend on A):
        // no second-order path reaches `a`, so its slot stays empty.
        let dga_da = a
            .grad()
            .map(|t| t.to_vec::<f64>().unwrap()[0])
            .unwrap_or(0.0);
        assert_eq!(dga_da, 0.0);
    }

    #[test]
    fn backward_bmm() {
        // B=1: A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity) -> Y = A
        // seed = ones -> dA = ones @ B^T = ones, dB = A^T @ ones = col-sums of A^T
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[1, 2, 2])
            .unwrap()
            .with_autograd();
        let b = Tensor::from_typed(vec![1.0_f64, 0.0, 0.0, 1.0])
            .reshape(&[1, 2, 2])
            .unwrap()
            .with_autograd();
        let y = bmm(&a, &b);
        assert_eq!(y.shape(), &[1, 2, 2]);
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        backward(&y);
        assert_eq!(
            a.grad().unwrap().to_vec::<f64>().unwrap(),
            vec![1.0, 1.0, 1.0, 1.0]
        );
        // dB = A^T @ ones = [[1+3, 1+3], [2+4, 2+4]] -> flat [4, 4, 6, 6]
        assert_eq!(
            b.grad().unwrap().to_vec::<f64>().unwrap(),
            vec![4.0, 4.0, 6.0, 6.0]
        );
    }
}
