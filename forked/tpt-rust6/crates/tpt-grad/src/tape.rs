//! Runtime reverse-mode autodiff over [`tpt_omni::Tensor<f64>`].
//!
//! A [`Tape`] records every op applied to a [`Variable`]. Because nodes are only
//! ever appended, and every node's parents have a strictly smaller id, the
//! reverse sweep is a simple backwards scan over the node vector.
//!
//! ```
//! use tpt_grad::prelude::*;
//! let tape = Tape::new();
//! let x = tape.var(tensor(&[3], &[1.0, 2.0, 3.0]));
//! let y = tape.var(tensor(&[3], &[4.0, 5.0, 6.0]));
//! let z = x * y + x;
//! let g = z.backward();
//! assert_eq!(g.gradient(&x).to_vec(), vec![5.0, 6.0, 7.0]); // y + 1
//! assert_eq!(g.gradient(&y).to_vec(), vec![1.0, 2.0, 3.0]); // x
//! ```

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;

use ndarray::{ArrayD, Axis, IxDyn};
use tpt_omni::Tensor;

/// Convenience constructor: `tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])`.
pub fn tensor(shape: &[usize], data: &[f64]) -> Tensor<f64> {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).expect("shape/data length mismatch"),
    )
}

/// Convenience constructor for a rank-0 (scalar) tensor.
pub fn scalar(v: f64) -> Tensor<f64> {
    tensor(&[], &[v])
}

/// A tensor filled with `v`.
pub fn full(shape: &[usize], v: f64) -> Tensor<f64> {
    Tensor::new(ArrayD::from_elem(IxDyn(shape), v))
}

fn clone_t(t: &Tensor<f64>) -> Tensor<f64> {
    Tensor::new(t.inner().clone())
}

/// `tpt_omni::Tensor` does not implement [`Clone`] upstream, so `.clone()` is
/// unavailable on tensors. This extension trait supplies it, which lets a
/// single tensor argument be used several times inside a macro body
/// (`(x.clone() * y.clone()) + x.clone()`). On the tape the call is free:
/// [`Variable`] is `Copy`.
pub trait TensorClone {
    fn clone(&self) -> Tensor<f64>;
}

impl TensorClone for Tensor<f64> {
    #[allow(clippy::should_implement_trait)]
    fn clone(&self) -> Tensor<f64> {
        clone_t(self)
    }
}

/// The element-wise unary ops of the macro subset, for plain tensors.
///
/// `tpt_omni::Tensor<f64>` already provides `+ - * /`, `.sum()` and `.mean()`;
/// this trait adds the remaining whitelisted methods so that a function body
/// type-checks both as written (tensors) and after tape lowering
/// ([`Variable`]) or fusion (`f64`).
///
/// Note: `tpt_omni::Tensor` has no `Neg` impl and the orphan rule forbids
/// adding one here, so write `t * -1.0` instead of `-t` in a macro body.
pub trait TensorOps {
    fn powf(&self, p: f64) -> Tensor<f64>;
    fn exp(&self) -> Tensor<f64>;
    fn ln(&self) -> Tensor<f64>;
    fn sqrt(&self) -> Tensor<f64>;
}

impl TensorOps for Tensor<f64> {
    fn powf(&self, p: f64) -> Tensor<f64> {
        map_t(self, |v| v.powf(p))
    }
    fn exp(&self) -> Tensor<f64> {
        map_t(self, f64::exp)
    }
    fn ln(&self) -> Tensor<f64> {
        map_t(self, f64::ln)
    }
    fn sqrt(&self) -> Tensor<f64> {
        map_t(self, f64::sqrt)
    }
}

fn map_t(t: &Tensor<f64>, f: impl Fn(f64) -> f64) -> Tensor<f64> {
    Tensor::new(t.inner().mapv(f))
}

/// Sum a (possibly broadcast) gradient back down to `shape` (NumPy rules).
fn reduce_to(g: &Tensor<f64>, shape: &[usize]) -> Tensor<f64> {
    let mut a = g.inner().clone();
    while a.ndim() > shape.len() {
        a = a.sum_axis(Axis(0));
    }
    for (i, &s) in shape.iter().enumerate() {
        if s == 1 && a.shape()[i] != 1 {
            a = a.sum_axis(Axis(i)).insert_axis(Axis(i));
        }
    }
    Tensor::new(a)
}

/// The recorded operation that produced a node.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Input or constant; gradient flow stops here.
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    Powf(usize, f64),
    Exp(usize),
    Ln(usize),
    Sqrt(usize),
    Sum(usize),
    Mean(usize),
}

#[derive(Default)]
struct Inner {
    ops: Vec<Op>,
    vals: Vec<Tensor<f64>>,
}

/// Records the computation graph. Create [`Variable`]s with [`Tape::var`].
#[derive(Default)]
pub struct Tape {
    inner: RefCell<Inner>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize {
        self.inner.borrow().ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Register a differentiable input.
    pub fn var(&self, value: Tensor<f64>) -> Variable<'_> {
        self.push(Op::Leaf, value)
    }

    /// Register a constant (a leaf that simply absorbs gradient).
    pub fn constant(&self, value: Tensor<f64>) -> Variable<'_> {
        self.push(Op::Leaf, value)
    }

    /// Register a scalar constant as a rank-0 tensor.
    pub fn scalar(&self, v: f64) -> Variable<'_> {
        self.constant(scalar(v))
    }

    fn push(&self, op: Op, value: Tensor<f64>) -> Variable<'_> {
        let mut inner = self.inner.borrow_mut();
        inner.ops.push(op);
        inner.vals.push(value);
        Variable {
            tape: self,
            id: inner.ops.len() - 1,
        }
    }

    fn val(&self, id: usize) -> Tensor<f64> {
        clone_t(&self.inner.borrow().vals[id])
    }
}

/// A node in the tape: a value plus its position in the graph.
///
/// `Variable` is `Copy`, so expressions such as `x * y + x` need no clones
/// (`x.clone()` also works and is a no-op).
#[derive(Clone, Copy)]
pub struct Variable<'t> {
    tape: &'t Tape,
    id: usize,
}

impl<'t> Variable<'t> {
    /// Node id within the owning tape.
    pub fn id(&self) -> usize {
        self.id
    }
    /// The tape that owns this node.
    pub fn tape(&self) -> &'t Tape {
        self.tape
    }
    /// The forward value of this node.
    pub fn value(&self) -> Tensor<f64> {
        self.tape.val(self.id)
    }
    /// The forward value as a scalar; requires exactly one element.
    pub fn value_scalar(&self) -> f64 {
        let v = self.value().to_vec();
        assert_eq!(v.len(), 1, "value_scalar() on a non-scalar variable");
        v[0]
    }
    /// Shape of the forward value.
    pub fn shape(&self) -> Vec<usize> {
        self.value().shape().to_vec()
    }

    fn same_tape(&self, other: &Variable<'t>) -> bool {
        ptr::eq(self.tape, other.tape)
    }

    fn bin(self, rhs: Variable<'t>, op: fn(usize, usize) -> Op, v: Tensor<f64>) -> Variable<'t> {
        assert!(self.same_tape(&rhs), "variables come from different tapes");
        self.tape.push(op(self.id, rhs.id), v)
    }

    /// Element-wise power, `self ^ p`.
    pub fn powf(self, p: f64) -> Variable<'t> {
        let v = map_t(&self.value(), |a| a.powf(p));
        self.tape.push(Op::Powf(self.id, p), v)
    }
    /// Element-wise `exp`.
    pub fn exp(self) -> Variable<'t> {
        let v = map_t(&self.value(), f64::exp);
        self.tape.push(Op::Exp(self.id), v)
    }
    /// Element-wise natural log.
    pub fn ln(self) -> Variable<'t> {
        let v = map_t(&self.value(), f64::ln);
        self.tape.push(Op::Ln(self.id), v)
    }
    /// Element-wise square root.
    pub fn sqrt(self) -> Variable<'t> {
        let v = map_t(&self.value(), f64::sqrt);
        self.tape.push(Op::Sqrt(self.id), v)
    }
    /// Sum reduction to a rank-0 variable.
    pub fn sum(self) -> Variable<'t> {
        let v = scalar(self.value().sum());
        self.tape.push(Op::Sum(self.id), v)
    }
    /// Mean reduction to a rank-0 variable.
    pub fn mean(self) -> Variable<'t> {
        let v = scalar(self.value().mean());
        self.tape.push(Op::Mean(self.id), v)
    }

    /// Reverse sweep seeded with ones at this node.
    pub fn backward(&self) -> Grads {
        let inner = self.tape.inner.borrow();
        let n = self.id + 1;
        let mut g: Vec<Option<Tensor<f64>>> = (0..n).map(|_| None).collect();
        g[self.id] = Some(full(inner.vals[self.id].shape(), 1.0));

        let acc = |slot: &mut Option<Tensor<f64>>, add: Tensor<f64>| match slot.take() {
            Some(prev) => *slot = Some(&prev + &add),
            None => *slot = Some(add),
        };

        for i in (0..n).rev() {
            let gi = match g[i].as_ref() {
                Some(t) => clone_t(t),
                None => continue,
            };
            let v = |k: usize| clone_t(&inner.vals[k]);
            let shp = |k: usize| inner.vals[k].shape().to_vec();
            match inner.ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    acc(&mut g[a], reduce_to(&gi, &shp(a)));
                    acc(&mut g[b], reduce_to(&gi, &shp(b)));
                }
                Op::Sub(a, b) => {
                    acc(&mut g[a], reduce_to(&gi, &shp(a)));
                    acc(&mut g[b], reduce_to(&map_t(&gi, |x| -x), &shp(b)));
                }
                Op::Mul(a, b) => {
                    acc(&mut g[a], reduce_to(&(&gi * &v(b)), &shp(a)));
                    acc(&mut g[b], reduce_to(&(&gi * &v(a)), &shp(b)));
                }
                Op::Div(a, b) => {
                    acc(&mut g[a], reduce_to(&(&gi / &v(b)), &shp(a)));
                    let num = &(&gi * &v(a)) * &map_t(&v(b), |x| -1.0 / (x * x));
                    acc(&mut g[b], reduce_to(&num, &shp(b)));
                }
                Op::Neg(a) => acc(&mut g[a], reduce_to(&map_t(&gi, |x| -x), &shp(a))),
                Op::Powf(a, p) => {
                    let d = map_t(&v(a), |x| p * x.powf(p - 1.0));
                    acc(&mut g[a], &gi * &d);
                }
                Op::Exp(a) => acc(&mut g[a], &gi * &v(i)),
                Op::Ln(a) => acc(&mut g[a], &gi / &v(a)),
                Op::Sqrt(a) => {
                    let d = map_t(&v(i), |s| 0.5 / s);
                    acc(&mut g[a], &gi * &d);
                }
                Op::Sum(a) => {
                    let s = gi.to_vec()[0];
                    acc(&mut g[a], full(&shp(a), s));
                }
                Op::Mean(a) => {
                    let sh = shp(a);
                    let n: usize = sh.iter().product::<usize>().max(1);
                    let s = gi.to_vec()[0] / n as f64;
                    acc(&mut g[a], full(&sh, s));
                }
            }
        }

        Grads {
            grads: g,
            shapes: inner.vals[..n].iter().map(|t| t.shape().to_vec()).collect(),
        }
    }
}

/// The result of a reverse sweep: gradients of one output w.r.t. every node.
pub struct Grads {
    grads: Vec<Option<Tensor<f64>>>,
    shapes: Vec<Vec<usize>>,
}

impl Grads {
    /// Gradient of the output w.r.t. `var` (zeros if it did not contribute).
    pub fn gradient(&self, var: &Variable<'_>) -> Tensor<f64> {
        match self.grads.get(var.id).and_then(|o| o.as_ref()) {
            Some(t) => clone_t(t),
            None => full(&self.shapes[var.id.min(self.shapes.len() - 1)], 0.0),
        }
    }
    /// Gradient as a scalar; requires a single-element gradient.
    pub fn gradient_scalar(&self, var: &Variable<'_>) -> f64 {
        let v = self.gradient(var).to_vec();
        assert_eq!(v.len(), 1, "gradient_scalar() on a non-scalar variable");
        v[0]
    }
}

macro_rules! impl_var_binop {
    ($tr:ident, $m:ident, $op:tt, $node:ident) => {
        impl<'t> $tr for Variable<'t> {
            type Output = Variable<'t>;
            fn $m(self, rhs: Variable<'t>) -> Variable<'t> {
                let v = &self.value() $op &rhs.value();
                self.bin(rhs, Op::$node, v)
            }
        }
        impl<'t> $tr<f64> for Variable<'t> {
            type Output = Variable<'t>;
            fn $m(self, rhs: f64) -> Variable<'t> {
                let c = self.tape.scalar(rhs);
                self.$m(c)
            }
        }
        impl<'t> $tr<Variable<'t>> for f64 {
            type Output = Variable<'t>;
            fn $m(self, rhs: Variable<'t>) -> Variable<'t> {
                let c = rhs.tape.scalar(self);
                c.$m(rhs)
            }
        }
    };
}
impl_var_binop!(Add, add, +, Add);
impl_var_binop!(Sub, sub, -, Sub);
impl_var_binop!(Mul, mul, *, Mul);
impl_var_binop!(Div, div, /, Div);

impl<'t> Neg for Variable<'t> {
    type Output = Variable<'t>;
    fn neg(self) -> Variable<'t> {
        let v = map_t(&self.value(), |x| -x);
        self.tape.push(Op::Neg(self.id), v)
    }
}

/// Converts a tape output into the return type of a `#[derive_grad]` function.
///
/// Implemented for `Tensor<f64>` and `f64`, which are the two return types the
/// macro supports.
pub trait FromVar {
    fn from_var(v: &Variable<'_>) -> Self;
}
impl FromVar for Tensor<f64> {
    fn from_var(v: &Variable<'_>) -> Tensor<f64> {
        v.value()
    }
}
impl FromVar for f64 {
    fn from_var(v: &Variable<'_>) -> f64 {
        v.value_scalar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dividing by a zero element must not panic: the value is `inf` (or NaN
    /// for `0/0`) and the gradients follow IEEE arithmetic element-wise.
    #[test]
    fn div_by_zero_element_is_infinite_or_nan() {
        let tape = Tape::new();
        let x = tape.var(tensor(&[3], &[1.0, 0.0, -2.0]));
        let y = tape.var(tensor(&[3], &[0.0, 0.0, 2.0]));
        let z = x / y;

        let v = z.value().to_vec();
        assert!(v[0].is_infinite() && v[0].is_sign_positive());
        assert!(v[1].is_nan()); // 0 / 0
        assert_eq!(v[2], -1.0);

        let g = z.backward();
        // dz/dx = 1/y
        let gx = g.gradient(&x).to_vec();
        assert!(gx[0].is_infinite() && gx[0].is_sign_positive());
        assert!(gx[1].is_infinite() && gx[1].is_sign_positive());
        assert!((gx[2] - 0.5).abs() < 1e-12);
        // dz/dy = -x/y^2; the middle element is 0 * -inf = NaN
        let gy = g.gradient(&y).to_vec();
        assert!(gy[0].is_infinite() && gy[0].is_sign_negative());
        assert!(gy[1].is_nan());
        assert!((gy[2] - 0.5).abs() < 1e-12);
    }

    /// `x / x` sends both adjoints into the same slot. Where `x` is zero the
    /// two contributions are `+inf` and NaN, so the sum is NaN; elsewhere they
    /// cancel to zero.
    #[test]
    fn self_division_gradient_is_nan_only_at_zero() {
        let tape = Tape::new();
        let x = tape.var(tensor(&[2], &[0.0, 4.0]));
        let z = x / x;

        let v = z.value().to_vec();
        assert!(v[0].is_nan());
        assert_eq!(v[1], 1.0);

        let g = z.backward().gradient(&x).to_vec();
        assert!(g[0].is_nan(), "got {}", g[0]);
        assert!(g[1].abs() < 1e-12, "got {}", g[1]);
    }

    /// `sqrt` and `ln` have singular derivatives at zero; the reverse sweep
    /// yields `+inf` there and the ordinary value elsewhere.
    #[test]
    fn sqrt_and_ln_at_zero_give_infinite_gradients() {
        let tape = Tape::new();
        let x = tape.var(tensor(&[2], &[0.0, 4.0]));
        let s = x.sqrt();
        assert_eq!(s.value().to_vec(), vec![0.0, 2.0]);
        let g = s.backward().gradient(&x).to_vec();
        assert!(g[0].is_infinite() && g[0].is_sign_positive());
        assert!((g[1] - 0.25).abs() < 1e-12);

        let tape = Tape::new();
        let x = tape.var(tensor(&[2], &[0.0, 4.0]));
        let l = x.ln();
        let lv = l.value().to_vec();
        assert!(lv[0].is_infinite() && lv[0].is_sign_negative());
        let g = l.backward().gradient(&x).to_vec();
        assert!(g[0].is_infinite() && g[0].is_sign_positive());
        assert!((g[1] - 0.25).abs() < 1e-12);
    }

    /// A NaN element in the forward values does not have to poison every
    /// gradient: `d(x*y)/dx = y` stays finite, only `d(x*y)/dy = x` is NaN.
    #[test]
    fn nan_input_poisons_only_the_dependent_gradient() {
        let tape = Tape::new();
        let x = tape.var(tensor(&[2], &[f64::NAN, 2.0]));
        let y = tape.var(tensor(&[2], &[3.0, 4.0]));
        let z = x * y;
        assert!(z.value().to_vec()[0].is_nan());

        let g = z.backward();
        assert_eq!(g.gradient(&x).to_vec(), vec![3.0, 4.0]);
        let gy = g.gradient(&y).to_vec();
        assert!(gy[0].is_nan());
        assert_eq!(gy[1], 2.0);
    }

    /// Broadcast reduction of a non-finite gradient keeps working: summing
    /// `+inf` rows down to the bias shape stays `+inf` and does not panic.
    #[test]
    fn reduce_to_handles_non_finite_gradients() {
        let tape = Tape::new();
        let x = tape.var(tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        let b = tape.var(tensor(&[2], &[0.0, 1.0]));
        let l = (x / b).sum();
        let g = l.backward();
        let gb = g.gradient(&b).to_vec();
        assert_eq!(g.gradient(&b).shape(), &[2]);
        assert!(gb[0].is_infinite() && gb[0].is_sign_negative());
        assert!(gb[1].is_finite());
    }
}
