//! Scalar (f64) reverse-mode tape — a lightweight convenience twin of
//! [`crate::tape`] for plain scalar autodiff.
//!
//! ```
//! use tpt_grad::prelude::*;
//! let t = ScalarTape::new();
//! let x = t.var(3.0);
//! let y = t.var(4.0);
//! let z = x * y + x.powf(2.0); // 12 + 9
//! let g = z.backward();
//! assert_eq!(z.value(), 21.0);
//! assert_eq!(g.gradient(&x), 10.0); // y + 2x
//! assert_eq!(g.gradient(&y), 3.0);  // x
//! ```

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;

#[derive(Clone, Copy)]
enum SOp {
    Leaf,
    /// Linear combination of up to two parents: `d/dp * upstream`.
    Bin(usize, f64, usize, f64),
    Un(usize, f64),
}

/// Tape for scalar reverse-mode AD.
#[derive(Default)]
pub struct ScalarTape {
    inner: RefCell<Vec<(SOp, f64)>>,
}

impl ScalarTape {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Register a differentiable scalar input.
    pub fn var(&self, v: f64) -> Value<'_> {
        self.push(SOp::Leaf, v)
    }
    fn push(&self, op: SOp, v: f64) -> Value<'_> {
        let mut b = self.inner.borrow_mut();
        b.push((op, v));
        Value {
            tape: self,
            id: b.len() - 1,
        }
    }
}

/// A scalar node on a [`ScalarTape`]. `Copy`, so it can be reused freely.
#[derive(Clone, Copy)]
pub struct Value<'t> {
    tape: &'t ScalarTape,
    id: usize,
}

impl<'t> Value<'t> {
    pub fn value(&self) -> f64 {
        self.tape.inner.borrow()[self.id].1
    }
    pub fn id(&self) -> usize {
        self.id
    }

    fn un(self, v: f64, d: f64) -> Value<'t> {
        self.tape.push(SOp::Un(self.id, d), v)
    }
    fn bin(self, r: Value<'t>, v: f64, da: f64, db: f64) -> Value<'t> {
        assert!(ptr::eq(self.tape, r.tape), "values from different tapes");
        self.tape.push(SOp::Bin(self.id, da, r.id, db), v)
    }

    pub fn powf(self, p: f64) -> Value<'t> {
        let a = self.value();
        self.un(a.powf(p), p * a.powf(p - 1.0))
    }
    pub fn exp(self) -> Value<'t> {
        let e = self.value().exp();
        self.un(e, e)
    }
    pub fn ln(self) -> Value<'t> {
        let a = self.value();
        self.un(a.ln(), 1.0 / a)
    }
    pub fn sqrt(self) -> Value<'t> {
        let s = self.value().sqrt();
        self.un(s, 0.5 / s)
    }
    pub fn sin(self) -> Value<'t> {
        let a = self.value();
        self.un(a.sin(), a.cos())
    }
    pub fn cos(self) -> Value<'t> {
        let a = self.value();
        self.un(a.cos(), -a.sin())
    }

    /// Reverse sweep seeded with `1.0` at this node.
    pub fn backward(&self) -> ScalarGrads {
        let b = self.tape.inner.borrow();
        let n = self.id + 1;
        let mut g = vec![0.0f64; n];
        g[self.id] = 1.0;
        for i in (0..n).rev() {
            let gi = g[i];
            if gi == 0.0 {
                continue;
            }
            match b[i].0 {
                SOp::Leaf => {}
                SOp::Un(a, d) => g[a] += gi * d,
                SOp::Bin(a, da, bb, db) => {
                    g[a] += gi * da;
                    g[bb] += gi * db;
                }
            }
        }
        ScalarGrads { g }
    }
}

/// Result of a scalar reverse sweep.
pub struct ScalarGrads {
    g: Vec<f64>,
}

impl ScalarGrads {
    pub fn gradient(&self, v: &Value<'_>) -> f64 {
        self.g.get(v.id).copied().unwrap_or(0.0)
    }
}

macro_rules! impl_val_binop {
    ($tr:ident, $m:ident, $body:expr) => {
        impl<'t> $tr for Value<'t> {
            type Output = Value<'t>;
            fn $m(self, rhs: Value<'t>) -> Value<'t> {
                let f: fn(f64, f64) -> (f64, f64, f64) = $body;
                let (v, da, db) = f(self.value(), rhs.value());
                self.bin(rhs, v, da, db)
            }
        }
        impl<'t> $tr<f64> for Value<'t> {
            type Output = Value<'t>;
            fn $m(self, rhs: f64) -> Value<'t> {
                let c = self.tape.var(rhs);
                self.$m(c)
            }
        }
        impl<'t> $tr<Value<'t>> for f64 {
            type Output = Value<'t>;
            fn $m(self, rhs: Value<'t>) -> Value<'t> {
                let c = rhs.tape.var(self);
                c.$m(rhs)
            }
        }
    };
}
impl_val_binop!(Add, add, |a, b| (a + b, 1.0, 1.0));
impl_val_binop!(Sub, sub, |a, b| (a - b, 1.0, -1.0));
impl_val_binop!(Mul, mul, |a, b| (a * b, b, a));
impl_val_binop!(Div, div, |a, b| (a / b, 1.0 / b, -a / (b * b)));

impl<'t> Neg for Value<'t> {
    type Output = Value<'t>;
    fn neg(self) -> Value<'t> {
        let v = self.value();
        self.un(-v, -1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `x / x` at `x = 0` is `0/0`: the value and the accumulated gradient are
    /// both NaN (`1/b` is `+inf`, `-a/b^2` is NaN, and `inf + NaN` is NaN).
    /// The sweep must produce that quietly rather than panic.
    #[test]
    fn div_by_zero_gradient_is_nan_not_a_panic() {
        let t = ScalarTape::new();
        let x = t.var(0.0);
        let z = x / x;
        assert!(z.value().is_nan());
        let g = z.backward().gradient(&x);
        assert!(g.is_nan(), "expected NaN gradient, got {g}");

        // Away from zero the same expression is exactly self-cancelling.
        let t = ScalarTape::new();
        let x = t.var(3.0);
        let g = (x / x).backward().gradient(&x);
        assert!(g.abs() < 1e-12, "expected ~0 gradient, got {g}");
    }

    /// `1/x` at `x = 0` has an infinite value and an infinite derivative
    /// (`-1/x^2`); nothing about that is a NaN.
    #[test]
    fn reciprocal_at_zero_is_infinite() {
        let t = ScalarTape::new();
        let x = t.var(0.0);
        let z = 1.0 / x;
        assert!(z.value().is_infinite() && z.value().is_sign_positive());
        let g = z.backward().gradient(&x);
        assert!(g.is_infinite() && g.is_sign_negative(), "got {g}");
    }

    /// `sqrt`, `ln` and `powf` all have a singular derivative at zero: the
    /// forward values stay well defined (or `-inf` for `ln`) while the
    /// gradients blow up to `+inf`.
    #[test]
    fn singular_derivatives_at_zero_are_infinite() {
        let t = ScalarTape::new();
        let x = t.var(0.0);
        let s = x.sqrt();
        assert_eq!(s.value(), 0.0);
        assert!(s.backward().gradient(&x).is_infinite());

        let t = ScalarTape::new();
        let x = t.var(0.0);
        let l = x.ln();
        assert!(l.value().is_infinite() && l.value().is_sign_negative());
        assert!(l.backward().gradient(&x).is_infinite());

        let t = ScalarTape::new();
        let x = t.var(0.0);
        let p = x.powf(0.5);
        assert_eq!(p.value(), 0.0);
        assert!(p.backward().gradient(&x).is_infinite());
    }

    /// A NaN *input* poisons the value but not necessarily the gradient:
    /// `d(x*c)/dx = c` is finite even when `x` is NaN. The reverse sweep skips
    /// nodes whose adjoint is exactly `0.0`, and NaN must not be mistaken for
    /// one of those.
    #[test]
    fn nan_input_keeps_a_finite_gradient() {
        let t = ScalarTape::new();
        let x = t.var(f64::NAN);
        let z = x * 2.0;
        assert!(z.value().is_nan());
        let g = z.backward().gradient(&x);
        assert_eq!(g, 2.0);

        // NaN adjoints still propagate through further ops.
        let t = ScalarTape::new();
        let x = t.var(f64::NAN);
        let g = (x * x).backward().gradient(&x);
        assert!(g.is_nan(), "got {g}");
    }
}
