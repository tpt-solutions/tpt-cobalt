//! # Symbolic → Rust tensor code generation
//!
//! Turns an [`Expr`] into a Rust *expression string* that evaluates the same
//! formula element-wise over [`tpt_omni::Tensor<f64>`] values.
//!
//! The emitted code assumes:
//!
//! * every [`Expr::Var`] name is in scope as a binding of type
//!   `tpt_omni::Tensor<f64>` (and is a valid Rust identifier), and
//! * `tpt_omni::Tensor` is imported, because the generated code uses the
//!   tensor operators `+`, `-`, `*`, `/` together with `Tensor::par_map` and
//!   `Tensor::zip_with`.
//!
//! Scalars are kept as plain `f64` literals for as long as possible so that
//! constant sub-expressions fold into the cheap `Tensor op f64` overloads
//! instead of materialising constant tensors. Elementary functions become
//! parallel element-wise maps, e.g. `sin(x)` emits `x.par_map(|v| v.sin())`.
//!
//! ```
//! use tpt_sym::prelude::*;
//!
//! let f = sym!(x) * sym!(y) + sym!(2.0);
//! assert_eq!(f.to_rust_tensor(), "(&(&x * &y) + 2f64)");
//!
//! let g = sym!(x).sin();
//! assert_eq!(g.to_rust_tensor(), "x.par_map(|v| v.sin())");
//! ```
//!
//! [`tpt_omni::Tensor<f64>`]: https://docs.rs/tpt-omni

use crate::Expr;

impl Expr {
    /// Emit a Rust expression string that evaluates this expression over
    /// `tpt_omni::Tensor<f64>` operands. See the [module docs](self) for the
    /// assumptions the generated code makes.
    ///
    /// ```
    /// use tpt_sym::prelude::*;
    ///
    /// let f = sym!(x) + sym!(y);
    /// assert_eq!(f.to_rust_tensor(), "(&x + &y)");
    /// ```
    pub fn to_rust_tensor(&self) -> String {
        to_rust(self)
    }
}

/// Free-function form of [`Expr::to_rust_tensor`].
///
/// An expression with no variables has no tensor operand to take its shape
/// from, so it is emitted as a plain `f64` literal expression (for example
/// `"6f64"`); the caller decides how to broadcast it.
///
/// ```
/// use tpt_sym::{codegen, Expr};
///
/// let e = Expr::var("x").pow(Expr::constant(2.0)) / Expr::constant(3.0);
/// assert_eq!(
///     codegen::to_rust(&e),
///     "(&x.par_map(|v| v.powf(2f64)) / 3f64)"
/// );
/// ```
pub fn to_rust(expr: &Expr) -> String {
    match emit(expr) {
        Code::Tensor(s) | Code::Scalar(s) => s,
    }
}

/// A generated fragment, tracked by the Rust type it evaluates to.
enum Code {
    /// A `f64`-valued fragment (constant sub-expression).
    Scalar(String),
    /// A `Tensor<f64>`-valued fragment.
    Tensor(String),
}

fn emit(expr: &Expr) -> Code {
    match expr {
        Expr::Var(v) => Code::Tensor(v.clone()),
        Expr::Const(c) => Code::Scalar(rust_f64(*c)),
        Expr::Add(a, b) => binary(emit(a), emit(b), "+", true),
        Expr::Sub(a, b) => binary(emit(a), emit(b), "-", false),
        Expr::Mul(a, b) => binary(emit(a), emit(b), "*", true),
        Expr::Div(a, b) => binary(emit(a), emit(b), "/", false),
        Expr::Neg(a) => match emit(a) {
            Code::Scalar(s) => Code::Scalar(format!("(-{})", s)),
            Code::Tensor(t) => Code::Tensor(format!("{}.par_map(|v| -v)", t)),
        },
        Expr::Sin(a) => unary(emit(a), "sin"),
        Expr::Cos(a) => unary(emit(a), "cos"),
        Expr::Exp(a) => unary(emit(a), "exp"),
        Expr::Ln(a) => unary(emit(a), "ln"),
        Expr::Pow(a, b) => power(emit(a), emit(b)),
    }
}

/// `op` is applied as `lhs op rhs`; `commutative` allows a scalar left operand
/// to be folded into the `Tensor op f64` overload.
fn binary(lhs: Code, rhs: Code, op: &str, commutative: bool) -> Code {
    match (lhs, rhs) {
        (Code::Scalar(a), Code::Scalar(b)) => Code::Scalar(format!("({} {} {})", a, op, b)),
        (Code::Tensor(t), Code::Scalar(s)) => Code::Tensor(format!("(&{} {} {})", t, op, s)),
        (Code::Scalar(s), Code::Tensor(t)) if commutative => {
            Code::Tensor(format!("(&{} {} {})", t, op, s))
        }
        // `f64 op Tensor` has no overload, so map element-wise instead.
        (Code::Scalar(s), Code::Tensor(t)) => {
            Code::Tensor(format!("{}.par_map(|v| {} {} v)", t, s, op))
        }
        (Code::Tensor(a), Code::Tensor(b)) => Code::Tensor(format!("(&{} {} &{})", a, op, b)),
    }
}

fn unary(arg: Code, func: &str) -> Code {
    match arg {
        Code::Scalar(s) => Code::Scalar(format!("({}).{}()", s, func)),
        Code::Tensor(t) => Code::Tensor(format!("{}.par_map(|v| v.{}())", t, func)),
    }
}

fn power(base: Code, exp: Code) -> Code {
    match (base, exp) {
        (Code::Scalar(b), Code::Scalar(e)) => Code::Scalar(format!("({}).powf({})", b, e)),
        (Code::Tensor(b), Code::Scalar(e)) => {
            Code::Tensor(format!("{}.par_map(|v| v.powf({}))", b, e))
        }
        (Code::Scalar(b), Code::Tensor(e)) => {
            Code::Tensor(format!("{}.par_map(|v| ({}).powf(v))", e, b))
        }
        (Code::Tensor(b), Code::Tensor(e)) => Code::Tensor(format!(
            "{}.zip_with(&{}, |a, b| a.powf(b)).expect(\"tpt-sym codegen: shape mismatch in powf\")",
            b, e
        )),
    }
}

/// Render an `f64` as a valid Rust literal expression.
fn rust_f64(c: f64) -> String {
    if c.is_nan() {
        return "f64::NAN".to_string();
    }
    if c.is_infinite() {
        return if c > 0.0 {
            "f64::INFINITY".to_string()
        } else {
            "f64::NEG_INFINITY".to_string()
        };
    }
    let body = if c == c.trunc() {
        format!("{}f64", c as i64)
    } else {
        format!("{}f64", c)
    };
    // Parenthesise negatives so they compose safely in any operand position.
    if c.is_sign_negative() {
        format!("({})", body)
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use crate::Expr;

    #[test]
    fn codegen_add_of_two_vars() {
        let e = Expr::var("x") + Expr::var("y");
        let code = e.to_rust_tensor();
        assert!(code.contains('x'), "{}", code);
        assert!(code.contains('y'), "{}", code);
        assert!(code.contains('+'), "{}", code);
        assert_eq!(code, "(&x + &y)");
    }

    #[test]
    fn codegen_scalar_folds_into_tensor_scalar_overload() {
        assert_eq!((Expr::var("x") * 2.0).to_rust_tensor(), "(&x * 2f64)");
        // Commutative: the scalar migrates to the right-hand side.
        assert_eq!(
            (Expr::constant(2.0) + Expr::var("x")).to_rust_tensor(),
            "(&x + 2f64)"
        );
        // Non-commutative with a scalar on the left needs an element-wise map.
        assert_eq!(
            (Expr::constant(1.0) - Expr::var("x")).to_rust_tensor(),
            "x.par_map(|v| 1f64 - v)"
        );
    }

    #[test]
    fn codegen_elementary_functions() {
        assert_eq!(
            Expr::var("x").sin().to_rust_tensor(),
            "x.par_map(|v| v.sin())"
        );
        assert_eq!(
            Expr::var("x").cos().to_rust_tensor(),
            "x.par_map(|v| v.cos())"
        );
        assert_eq!(
            Expr::var("x").ln().to_rust_tensor(),
            "x.par_map(|v| v.ln())"
        );
        assert_eq!(
            Expr::var("x").exp().to_rust_tensor(),
            "x.par_map(|v| v.exp())"
        );
        assert_eq!(
            Expr::Neg(Box::new(Expr::var("x"))).to_rust_tensor(),
            "x.par_map(|v| -v)"
        );
    }

    #[test]
    fn codegen_pow_variants() {
        assert_eq!(
            Expr::var("x").pow(Expr::constant(3.0)).to_rust_tensor(),
            "x.par_map(|v| v.powf(3f64))"
        );
        assert!(Expr::var("x")
            .pow(Expr::var("y"))
            .to_rust_tensor()
            .starts_with("x.zip_with(&y, |a, b| a.powf(b))"));
    }

    #[test]
    fn codegen_nested_expression() {
        // sin(x) * y - 1
        let e = Expr::var("x").sin() * Expr::var("y") - Expr::constant(1.0);
        assert_eq!(
            e.to_rust_tensor(),
            "(&(&x.par_map(|v| v.sin()) * &y) - 1f64)"
        );
    }

    #[test]
    fn codegen_constant_only_expression_is_a_scalar_literal() {
        let e = Expr::constant(2.0) * Expr::constant(3.0);
        assert_eq!(e.to_rust_tensor(), "(2f64 * 3f64)");
        assert_eq!(Expr::constant(-1.5).to_rust_tensor(), "(-1.5f64)");
    }
}
