//! # tpt-sym — Compile-Time Symbolic Mathematics
//!
//! A small, dependency-light computer-algebra core. Expressions are typed
//! `Expr`s you can differentiate, simplify, evaluate, render to LaTeX, and
//! compile into fast numerical closures. A companion unit system performs
//! dimensional analysis on `Quantity` values.
//!
//! ```ignore
//! use tpt_sym::prelude::*;
//! let x = sym!(x);
//! let f = x.clone() * x.clone() + sym!(2.0);   // x^2 + 2
//! let df = f.diff("x");                          // 2*x
//! println!("{}", df.to_latex());                 // 2 x
//! let y = f.eval(&[("x", 3.0)]);                 // 11.0
//! ```

use std::collections::HashMap;

pub mod codegen;

/// A symbolic expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Var(String),
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
}

impl Expr {
    pub fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }
    pub fn constant(v: f64) -> Expr {
        Expr::Const(v)
    }

    pub fn sin(self) -> Expr {
        Expr::Sin(Box::new(self))
    }
    pub fn cos(self) -> Expr {
        Expr::Cos(Box::new(self))
    }
    pub fn exp(self) -> Expr {
        Expr::Exp(Box::new(self))
    }
    pub fn ln(self) -> Expr {
        Expr::Ln(Box::new(self))
    }
    pub fn pow(self, e: Expr) -> Expr {
        Expr::Pow(Box::new(self), Box::new(e))
    }

    /// Constant-fold and apply algebraic identities.
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Add(a, b) => {
                let (a, b) = (a.simplify(), b.simplify());
                match (&a, &b) {
                    (Expr::Const(0.0), _) => b,
                    (_, Expr::Const(0.0)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                    (a, b) if a == b => a.clone() * Expr::Const(2.0),
                    _ => Expr::Add(Box::new(a), Box::new(b)),
                }
            }
            Expr::Sub(a, b) => {
                let (a, b) = (a.simplify(), b.simplify());
                match (&a, &b) {
                    (_, Expr::Const(0.0)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x - y),
                    _ => Expr::Sub(Box::new(a), Box::new(b)),
                }
            }
            Expr::Mul(a, b) => {
                let (a, b) = (a.simplify(), b.simplify());
                match (&a, &b) {
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                    (Expr::Const(1.0), _) => b,
                    (_, Expr::Const(1.0)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                    _ => Expr::Mul(Box::new(a), Box::new(b)),
                }
            }
            Expr::Div(a, b) => {
                let (a, b) = (a.simplify(), b.simplify());
                match (&a, &b) {
                    (_, Expr::Const(1.0)) => a,
                    (Expr::Const(0.0), _) => Expr::Const(0.0),
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x / y),
                    _ => Expr::Div(Box::new(a), Box::new(b)),
                }
            }
            Expr::Neg(a) => {
                let a = a.simplify();
                if let Expr::Const(x) = a {
                    return Expr::Const(-x);
                }
                Expr::Neg(Box::new(a))
            }
            Expr::Pow(a, b) => {
                let (a, b) = (a.simplify(), b.simplify());
                if let (_, Expr::Const(1.0)) = (&a, &b) {
                    return a;
                }
                if let (_, Expr::Const(0.0)) = (&a, &b) {
                    return Expr::Const(1.0);
                }
                Expr::Pow(Box::new(a), Box::new(b))
            }
            Expr::Sin(a) => {
                let a = a.simplify();
                if let Expr::Const(x) = a {
                    return Expr::Const(x.sin());
                }
                Expr::Sin(Box::new(a))
            }
            Expr::Cos(a) => {
                let a = a.simplify();
                if let Expr::Const(x) = a {
                    return Expr::Const(x.cos());
                }
                Expr::Cos(Box::new(a))
            }
            Expr::Exp(a) => {
                let a = a.simplify();
                if let Expr::Const(x) = a {
                    return Expr::Const(x.exp());
                }
                Expr::Exp(Box::new(a))
            }
            Expr::Ln(a) => {
                let a = a.simplify();
                if let Expr::Const(x) = a {
                    return Expr::Const(x.ln());
                }
                Expr::Ln(Box::new(a))
            }
            other => other.clone(),
        }
    }

    /// Symbolic partial derivative with respect to `var`.
    pub fn diff(&self, var: &str) -> Expr {
        match self {
            Expr::Var(v) => Expr::Const(if v == var { 1.0 } else { 0.0 }),
            Expr::Const(_) => Expr::Const(0.0),
            Expr::Add(a, b) => a.diff(var) + b.diff(var),
            Expr::Sub(a, b) => a.diff(var) - b.diff(var),
            Expr::Mul(a, b) => {
                let a = (**a).clone();
                let b = (**b).clone();
                a.diff(var) * b.clone() + a.clone() * b.diff(var)
            }
            Expr::Div(a, b) => {
                let a = (**a).clone();
                let b = (**b).clone();
                (a.diff(var) * b.clone() - a.clone() * b.diff(var)) / (b.clone() * b.clone())
            }
            Expr::Neg(a) => -a.diff(var),
            Expr::Pow(a, b) => {
                // d/dx f^g = f^g * (g' ln f + g f'/f)
                let f = (**a).clone();
                let g = (**b).clone();
                f.clone().pow(g.clone())
                    * (b.diff(var) * f.clone().ln() + g.clone() * a.diff(var) / f.clone())
            }
            Expr::Sin(a) => a.diff(var) * (**a).clone().cos(),
            Expr::Cos(a) => -a.diff(var) * (**a).clone().sin(),
            Expr::Exp(a) => a.diff(var) * self.clone(),
            Expr::Ln(a) => a.diff(var) / (**a).clone(),
        }
        .simplify()
    }

    /// Evaluate to a number given a variable environment.
    pub fn eval(&self, env: &[(&str, f64)]) -> f64 {
        let env: HashMap<&str, f64> = env.iter().copied().collect();
        self.eval_env(&env)
    }

    fn eval_env(&self, env: &HashMap<&str, f64>) -> f64 {
        match self {
            Expr::Var(v) => *env.get(v.as_str()).unwrap_or(&0.0),
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval_env(env) + b.eval_env(env),
            Expr::Sub(a, b) => a.eval_env(env) - b.eval_env(env),
            Expr::Mul(a, b) => a.eval_env(env) * b.eval_env(env),
            Expr::Div(a, b) => a.eval_env(env) / b.eval_env(env),
            Expr::Neg(a) => -a.eval_env(env),
            Expr::Pow(a, b) => a.eval_env(env).powf(b.eval_env(env)),
            Expr::Sin(a) => a.eval_env(env).sin(),
            Expr::Cos(a) => a.eval_env(env).cos(),
            Expr::Exp(a) => a.eval_env(env).exp(),
            Expr::Ln(a) => a.eval_env(env).ln(),
        }
    }

    /// Render to a LaTeX string.
    pub fn to_latex(&self) -> String {
        match self {
            Expr::Var(v) => v.clone(),
            Expr::Const(c) => {
                if *c == c.trunc() {
                    format!("{}", *c as i64)
                } else {
                    format!("{}", c)
                }
            }
            Expr::Add(a, b) => format!("{} + {}", a.to_latex(), b.to_latex()),
            Expr::Sub(a, b) => format!("{} - {}", a.to_latex(), b.to_latex()),
            Expr::Mul(a, b) => format!("{} \\, {}", a.to_latex(), b.to_latex()),
            Expr::Div(a, b) => format!("\\frac{{{}}}{{{}}}", a.to_latex(), b.to_latex()),
            Expr::Neg(a) => format!("-{}", a.to_latex()),
            Expr::Pow(a, b) => format!("{{{}}}^{{{}}}", a.to_latex(), b.to_latex()),
            Expr::Sin(a) => format!("\\sin({})", a.to_latex()),
            Expr::Cos(a) => format!("\\cos({})", a.to_latex()),
            Expr::Exp(a) => format!("\\exp({})", a.to_latex()),
            Expr::Ln(a) => format!("\\ln({})", a.to_latex()),
        }
    }

    /// Render to a Presentation MathML string.
    ///
    /// The output mirrors [`Expr::to_latex`] structurally: `Add` becomes
    /// `<mo>+</mo>`, `Mul` becomes `<mo>&#215;</mo>`, `Div` becomes `<mfrac>`,
    /// `Pow` becomes `<msup>`, variables become `<mi>` and numbers `<mn>`.
    /// Elementary functions render as `<mi>sin</mi>` followed by the invisible
    /// function-application operator `<mo>&#8289;</mo>` and a parenthesised
    /// argument (explicit `<mo>(</mo>` parentheses are used rather than the
    /// deprecated `<mfenced>` element).
    ///
    /// Every node renders as exactly one MathML element, so the two-child
    /// requirements of `<mfrac>` and `<msup>` always hold.
    ///
    /// ```
    /// use tpt_sym::prelude::*;
    /// let f = sym!(x) + sym!(1.0);
    /// assert_eq!(
    ///     f.to_mathml(),
    ///     "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
    ///      <mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow></math>"
    /// );
    /// ```
    pub fn to_mathml(&self) -> String {
        format!(
            "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">{}</math>",
            self.mathml_node()
        )
    }

    /// Render a single MathML element for this node (no `<math>` wrapper).
    fn mathml_node(&self) -> String {
        match self {
            Expr::Var(v) => format!("<mi>{}</mi>", escape_xml(v)),
            Expr::Const(c) => format!("<mn>{}</mn>", escape_xml(&fmt_number(*c))),
            Expr::Add(a, b) => format!(
                "<mrow>{}<mo>+</mo>{}</mrow>",
                a.mathml_node(),
                b.mathml_node()
            ),
            Expr::Sub(a, b) => format!(
                "<mrow>{}<mo>&#8722;</mo>{}</mrow>",
                a.mathml_node(),
                b.mathml_node()
            ),
            Expr::Mul(a, b) => format!(
                "<mrow>{}<mo>&#215;</mo>{}</mrow>",
                a.mathml_node(),
                b.mathml_node()
            ),
            Expr::Div(a, b) => format!("<mfrac>{}{}</mfrac>", a.mathml_node(), b.mathml_node()),
            Expr::Neg(a) => format!("<mrow><mo>&#8722;</mo>{}</mrow>", a.mathml_node()),
            Expr::Pow(a, b) => format!("<msup>{}{}</msup>", a.mathml_node(), b.mathml_node()),
            Expr::Sin(a) => mathml_call("sin", a),
            Expr::Cos(a) => mathml_call("cos", a),
            Expr::Exp(a) => mathml_call("exp", a),
            Expr::Ln(a) => mathml_call("ln", a),
        }
    }

    /// True when `var` occurs anywhere in this expression.
    pub fn contains_var(&self, var: &str) -> bool {
        match self {
            Expr::Var(v) => v == var,
            Expr::Const(_) => false,
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                a.contains_var(var) || b.contains_var(var)
            }
            Expr::Pow(a, b) => a.contains_var(var) || b.contains_var(var),
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) | Expr::Ln(a) => {
                a.contains_var(var)
            }
        }
    }

    /// Solve `self == 0` for `var`, returning every solution that the solver
    /// can establish (an empty `Vec` when it cannot solve the equation).
    ///
    /// The equation is first rewritten as a polynomial in `var` whose
    /// coefficients are themselves expressions free of `var`. Supported cases:
    ///
    /// * **Linear** — `a*var + b == 0` (also `var + b`, `a*var`, `var - c`)
    ///   yields `[-b/a]`. The coefficients may stay symbolic, so simple
    ///   transposition works: `var + y == 0` solves to `-y`.
    /// * **Quadratic** — `a*var^2 + b*var + c == 0` with *numeric* `a`, `b`, `c`
    ///   yields the real roots from the quadratic formula (`f64` arithmetic,
    ///   returned as [`Expr::Const`]), in ascending order. A double root is
    ///   returned once.
    /// * **Monomials** — `a*var^n == 0` for `n >= 1` yields `[0]`.
    ///
    /// # Limitations
    ///
    /// * Only real solutions are reported: a negative discriminant gives `[]`.
    /// * Quadratics with symbolic coefficients are not solved.
    /// * Degree three and above is only handled for pure monomials.
    /// * `var` may not appear inside `sin`, `cos`, `exp`, `ln`, in a divisor,
    ///   or in an exponent; such equations give `[]`.
    /// * Identities such as `0 == 0` (infinitely many solutions) and
    ///   contradictions such as `1 == 0` both give `[]`.
    ///
    /// ```
    /// use tpt_sym::prelude::*;
    /// let eq = sym!(x) * 2.0 - sym!(4.0); // 2x - 4 == 0
    /// assert_eq!(eq.solve("x"), vec![Expr::Const(2.0)]);
    /// ```
    pub fn solve(&self, var: &str) -> Vec<Expr> {
        let mut coeffs = match self.simplify().poly_coeffs(var) {
            Some(c) => c,
            None => return Vec::new(),
        };
        // Drop leading (highest-order) zero coefficients to find the true degree.
        while coeffs.len() > 1 && is_zero(&coeffs[coeffs.len() - 1]) {
            coeffs.pop();
        }

        match coeffs.len() {
            // Constant equation: either an identity or a contradiction.
            0 | 1 => Vec::new(),
            // a*var + b == 0  =>  var = -b/a
            2 => {
                let b = coeffs[0].clone();
                let a = coeffs[1].clone();
                vec![Expr::Div(Box::new(Expr::Neg(Box::new(b))), Box::new(a)).simplify()]
            }
            // a*var^2 + b*var + c == 0
            3 => {
                match (
                    as_number(&coeffs[2]),
                    as_number(&coeffs[1]),
                    as_number(&coeffs[0]),
                ) {
                    (Some(a), Some(b), Some(c)) => {
                        let disc = b * b - 4.0 * a * c;
                        if disc < 0.0 || !disc.is_finite() {
                            return Vec::new();
                        }
                        let sq = disc.sqrt();
                        if sq == 0.0 {
                            return vec![Expr::Const(-b / (2.0 * a))];
                        }
                        let lo = (-b - sq) / (2.0 * a);
                        let hi = (-b + sq) / (2.0 * a);
                        if lo <= hi {
                            vec![Expr::Const(lo), Expr::Const(hi)]
                        } else {
                            vec![Expr::Const(hi), Expr::Const(lo)]
                        }
                    }
                    // Symbolic quadratic: only the pure monomial a*var^2 == 0.
                    _ => monomial_root(&coeffs),
                }
            }
            // Higher degrees: only a*var^n == 0.
            _ => monomial_root(&coeffs),
        }
    }

    /// Coefficients `[c0, c1, c2, ...]` of `self` viewed as a polynomial in
    /// `var`, i.e. `c0 + c1*var + c2*var^2 + ...`, where every `ci` is free of
    /// `var`. Returns `None` when `self` is not polynomial in `var` (or when
    /// the degree exceeds [`MAX_POLY_DEGREE`]).
    fn poly_coeffs(&self, var: &str) -> Option<Vec<Expr>> {
        if !self.contains_var(var) {
            return Some(vec![self.simplify()]);
        }
        match self {
            Expr::Var(_) => Some(vec![Expr::Const(0.0), Expr::Const(1.0)]),
            Expr::Add(a, b) => Some(poly_zip(
                &a.poly_coeffs(var)?,
                &b.poly_coeffs(var)?,
                |x, y| Expr::Add(Box::new(x), Box::new(y)),
            )),
            Expr::Sub(a, b) => Some(poly_zip(
                &a.poly_coeffs(var)?,
                &b.poly_coeffs(var)?,
                |x, y| Expr::Sub(Box::new(x), Box::new(y)),
            )),
            Expr::Neg(a) => Some(
                a.poly_coeffs(var)?
                    .into_iter()
                    .map(|c| Expr::Neg(Box::new(c)).simplify())
                    .collect(),
            ),
            Expr::Mul(a, b) => poly_mul(&a.poly_coeffs(var)?, &b.poly_coeffs(var)?),
            Expr::Div(a, b) => {
                if b.contains_var(var) {
                    return None; // `var` in a divisor: not polynomial
                }
                let den = b.simplify();
                Some(
                    a.poly_coeffs(var)?
                        .into_iter()
                        .map(|c| Expr::Div(Box::new(c), Box::new(den.clone())).simplify())
                        .collect(),
                )
            }
            Expr::Pow(a, b) => {
                if b.contains_var(var) {
                    return None; // `var` in an exponent
                }
                let n = as_number(&b.simplify())?;
                if n < 0.0 || n.fract() != 0.0 || n > MAX_POLY_DEGREE as f64 {
                    return None;
                }
                let base = a.poly_coeffs(var)?;
                let mut acc = vec![Expr::Const(1.0)];
                for _ in 0..(n as u32) {
                    acc = poly_mul(&acc, &base)?;
                }
                Some(acc)
            }
            // sin/cos/exp/ln of something containing `var` is not polynomial.
            _ => None,
        }
    }

    /// Compile into a fast numerical closure.
    pub fn compile(&self) -> impl Fn(f64) -> f64 + '_ {
        move |x: f64| self.eval(&[("x", x)])
    }
}

/// Highest polynomial degree the [`Expr::solve`] coefficient extractor tracks.
const MAX_POLY_DEGREE: usize = 8;

/// `<mrow><mi>f</mi><mo>&#8289;</mo><mo>(</mo> arg <mo>)</mo></mrow>`
fn mathml_call(name: &str, arg: &Expr) -> String {
    format!(
        "<mrow><mi>{}</mi><mo>&#8289;</mo><mo>(</mo>{}<mo>)</mo></mrow>",
        name,
        arg.mathml_node()
    )
}

/// Format a number the way [`Expr::to_latex`] does (integers without `.0`).
fn fmt_number(c: f64) -> String {
    if c == c.trunc() && c.is_finite() {
        format!("{}", c as i64)
    } else {
        format!("{}", c)
    }
}

/// Escape the five XML metacharacters that may appear in a variable name.
fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// The numeric value of an expression, if it simplifies to a constant.
fn as_number(e: &Expr) -> Option<f64> {
    match e.simplify() {
        Expr::Const(c) => Some(c),
        _ => None,
    }
}

/// True when an expression simplifies to the constant zero.
fn is_zero(e: &Expr) -> bool {
    matches!(as_number(e), Some(c) if c == 0.0)
}

/// `a*var^n == 0` has the single root `0`; anything else is unsolved here.
fn monomial_root(coeffs: &[Expr]) -> Vec<Expr> {
    if coeffs.len() >= 2 && coeffs[..coeffs.len() - 1].iter().all(is_zero) {
        vec![Expr::Const(0.0)]
    } else {
        Vec::new()
    }
}

/// Combine two coefficient vectors term by term (padding with zeros).
fn poly_zip(a: &[Expr], b: &[Expr], f: impl Fn(Expr, Expr) -> Expr) -> Vec<Expr> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = a.get(i).cloned().unwrap_or(Expr::Const(0.0));
            let y = b.get(i).cloned().unwrap_or(Expr::Const(0.0));
            f(x, y).simplify()
        })
        .collect()
}

/// Polynomial product of two coefficient vectors.
fn poly_mul(a: &[Expr], b: &[Expr]) -> Option<Vec<Expr>> {
    if a.is_empty() || b.is_empty() {
        return Some(vec![Expr::Const(0.0)]);
    }
    let deg = a.len() + b.len() - 2;
    if deg > MAX_POLY_DEGREE {
        return None;
    }
    let mut out = vec![Expr::Const(0.0); deg + 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let term = Expr::Mul(Box::new(ai.clone()), Box::new(bj.clone())).simplify();
            out[i + j] = Expr::Add(Box::new(out[i + j].clone()), Box::new(term)).simplify();
        }
    }
    Some(out)
}

macro_rules! binop {
    ($trait:ident, $fn:ident, $variant:ident) => {
        impl std::ops::$trait for Expr {
            type Output = Expr;
            fn $fn(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(self), Box::new(rhs))
            }
        }
        impl std::ops::$trait for &Expr {
            type Output = Expr;
            fn $fn(self, rhs: &Expr) -> Expr {
                Expr::$variant(Box::new(self.clone()), Box::new(rhs.clone()))
            }
        }
    };
}
binop!(Add, add, Add);
binop!(Sub, sub, Sub);
binop!(Mul, mul, Mul);
binop!(Div, div, Div);

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

macro_rules! binop_scalar {
    ($trait:ident, $fn:ident, $variant:ident) => {
        impl std::ops::$trait<f64> for Expr {
            type Output = Expr;
            fn $fn(self, rhs: f64) -> Expr {
                self.$fn(Expr::Const(rhs))
            }
        }
    };
}
binop_scalar!(Add, add, Add);
binop_scalar!(Sub, sub, Sub);
binop_scalar!(Mul, mul, Mul);
binop_scalar!(Div, div, Div);

/// Typed symbolic variable.
///
/// `sym!(x)` builds `Expr::Var("x")`. `sym!(x: Real)` is accepted and the type
/// annotation is currently informational (full compile-time dimensional typing
/// is wired through [`Quantity`]).
#[macro_export]
macro_rules! sym {
    ($name:ident) => {
        $crate::Expr::var(stringify!($name))
    };
    ($name:ident : $ty:ident) => {
        $crate::Expr::var(stringify!($name))
    };
    ($lit:literal) => {
        $crate::Expr::constant($lit as f64)
    };
}

/// A physical unit expressed as exponents of the SI base dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Unit {
    pub m: i32,
    pub kg: i32,
    pub s: i32,
    pub a: i32,
    pub k: i32,
    pub mol: i32,
    pub cd: i32,
}

impl Unit {
    pub const SCALAR: Unit = Unit {
        m: 0,
        kg: 0,
        s: 0,
        a: 0,
        k: 0,
        mol: 0,
        cd: 0,
    };
    pub const METER: Unit = Unit {
        m: 1,
        kg: 0,
        s: 0,
        a: 0,
        k: 0,
        mol: 0,
        cd: 0,
    };
    pub const SECOND: Unit = Unit {
        m: 0,
        kg: 0,
        s: 1,
        a: 0,
        k: 0,
        mol: 0,
        cd: 0,
    };
    pub const KILOGRAM: Unit = Unit {
        m: 0,
        kg: 1,
        s: 0,
        a: 0,
        k: 0,
        mol: 0,
        cd: 0,
    };

    fn zip(&self, o: &Unit, f: impl Fn(i32, i32) -> i32) -> Unit {
        Unit {
            m: f(self.m, o.m),
            kg: f(self.kg, o.kg),
            s: f(self.s, o.s),
            a: f(self.a, o.a),
            k: f(self.k, o.k),
            mol: f(self.mol, o.mol),
            cd: f(self.cd, o.cd),
        }
    }
    pub fn mul(&self, o: &Unit) -> Unit {
        self.zip(o, |a, b| a + b)
    }
    pub fn div(&self, o: &Unit) -> Unit {
        self.zip(o, |a, b| a - b)
    }
    pub fn pow(&self, n: i32) -> Unit {
        Unit {
            m: self.m * n,
            kg: self.kg * n,
            s: self.s * n,
            a: self.a * n,
            k: self.k * n,
            mol: self.mol * n,
            cd: self.cd * n,
        }
    }
    pub fn is_scalar(&self) -> bool {
        *self == Unit::SCALAR
    }
}

/// A value carrying a physical unit, with dimensional checking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity<T> {
    pub value: T,
    pub unit: Unit,
}

impl<T: Copy> Quantity<T> {
    pub fn new(value: T, unit: Unit) -> Self {
        Self { value, unit }
    }
}

impl std::ops::Add for Quantity<f64> {
    type Output = Quantity<f64>;
    fn add(self, rhs: Quantity<f64>) -> Quantity<f64> {
        assert_eq!(self.unit, rhs.unit, "dimension mismatch in addition");
        Quantity::new(self.value + rhs.value, self.unit)
    }
}
impl std::ops::Sub for Quantity<f64> {
    type Output = Quantity<f64>;
    fn sub(self, rhs: Quantity<f64>) -> Quantity<f64> {
        assert_eq!(self.unit, rhs.unit, "dimension mismatch in subtraction");
        Quantity::new(self.value - rhs.value, self.unit)
    }
}
impl std::ops::Mul for Quantity<f64> {
    type Output = Quantity<f64>;
    fn mul(self, rhs: Quantity<f64>) -> Quantity<f64> {
        Quantity::new(self.value * rhs.value, self.unit.mul(&rhs.unit))
    }
}
impl std::ops::Div for Quantity<f64> {
    type Output = Quantity<f64>;
    fn div(self, rhs: Quantity<f64>) -> Quantity<f64> {
        Quantity::new(self.value / rhs.value, self.unit.div(&rhs.unit))
    }
}

pub mod prelude {
    pub use crate::Expr;
    pub use crate::Quantity;
    pub use crate::Unit;
    pub use crate::{sym, Expr as Symbol};
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATH_OPEN: &str = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">";

    #[test]
    fn mathml_has_math_root_and_expected_tags() {
        // x^2 + 2
        let f = Expr::var("x").pow(Expr::constant(2.0)) + Expr::constant(2.0);
        let ml = f.to_mathml();

        assert!(ml.starts_with(MATH_OPEN), "{}", ml);
        assert!(ml.ends_with("</math>"), "{}", ml);
        assert!(ml.contains("<mi>x</mi>"), "{}", ml);
        assert!(ml.contains("<mn>2</mn>"), "{}", ml);
        assert!(ml.contains("<msup>"), "{}", ml);
        assert!(ml.contains("<mo>+</mo>"), "{}", ml);
        assert_eq!(
            ml,
            format!(
                "{}<mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn></mrow></math>",
                MATH_OPEN
            )
        );
    }

    #[test]
    fn mathml_fraction_product_and_functions() {
        let frac = Expr::var("a") / Expr::var("b");
        assert!(frac
            .to_mathml()
            .contains("<mfrac><mi>a</mi><mi>b</mi></mfrac>"));

        let prod = Expr::var("a") * Expr::var("b");
        assert!(prod.to_mathml().contains("<mo>&#215;</mo>"));

        let diff = Expr::var("a") - Expr::var("b");
        assert!(diff.to_mathml().contains("<mo>&#8722;</mo>"));

        let s = Expr::var("x").sin().to_mathml();
        assert!(s.contains("<mi>sin</mi>"), "{}", s);
        assert!(s.contains("<mo>&#8289;</mo>"), "{}", s);
        assert!(s.contains("<mo>(</mo><mi>x</mi><mo>)</mo>"), "{}", s);

        // Variable names are XML-escaped.
        assert!(Expr::var("a<b").to_mathml().contains("<mi>a&lt;b</mi>"));
    }

    #[test]
    fn mathml_does_not_change_latex() {
        let f = Expr::var("x") * Expr::var("x") + Expr::constant(2.0);
        assert_eq!(f.to_latex(), "x \\, x + 2");
    }

    #[test]
    fn solve_linear() {
        // 2x - 4 == 0  =>  x = 2
        let eq = Expr::constant(2.0) * Expr::var("x") - Expr::constant(4.0);
        assert_eq!(eq.solve("x"), vec![Expr::Const(2.0)]);

        // x * 2 - 4 == 0 (scalar operator form)
        let eq = Expr::var("x") * 2.0 - Expr::constant(4.0);
        assert_eq!(eq.solve("x"), vec![Expr::Const(2.0)]);

        // x + 3 == 0  =>  x = -3
        let eq = Expr::var("x") + Expr::constant(3.0);
        assert_eq!(eq.solve("x"), vec![Expr::Const(-3.0)]);

        // x - 5 == 0  =>  x = 5
        let eq = Expr::var("x") - Expr::constant(5.0);
        assert_eq!(eq.solve("x"), vec![Expr::Const(5.0)]);

        // 3x == 0  =>  x = 0
        let eq = Expr::constant(3.0) * Expr::var("x");
        assert_eq!(eq.solve("x"), vec![Expr::Const(0.0)]);
    }

    #[test]
    fn solve_linear_symbolic_transposition() {
        // x + y == 0  =>  x = -y
        let eq = Expr::var("x") + Expr::var("y");
        assert_eq!(eq.solve("x"), vec![Expr::Neg(Box::new(Expr::var("y")))]);

        // a*x + b == 0  =>  x = -b/a
        let eq = Expr::var("a") * Expr::var("x") + Expr::var("b");
        let roots = eq.solve("x");
        assert_eq!(roots.len(), 1);
        assert!(
            (roots[0].eval(&[("a", 2.0), ("b", 6.0)]) + 3.0).abs() < 1e-12,
            "{:?}",
            roots[0]
        );
    }

    #[test]
    fn solve_quadratic_two_roots() {
        // x^2 - 5x + 6 == 0  =>  x = 2, 3
        let x = Expr::var("x");
        let eq = x.clone().pow(Expr::constant(2.0)) - Expr::constant(5.0) * x.clone()
            + Expr::constant(6.0);
        let roots = eq.solve("x");
        assert_eq!(roots.len(), 2);
        assert_eq!(roots, vec![Expr::Const(2.0), Expr::Const(3.0)]);

        // x*x - 4 == 0  =>  x = -2, 2 (no explicit Pow node)
        let eq = x.clone() * x.clone() - Expr::constant(4.0);
        let roots = eq.solve("x");
        assert_eq!(roots, vec![Expr::Const(-2.0), Expr::Const(2.0)]);

        // Every returned root really is a root.
        for r in roots {
            let v = r.eval(&[]);
            assert!(eq.eval(&[("x", v)]).abs() < 1e-9);
        }
    }

    #[test]
    fn solve_quadratic_double_root_and_no_real_roots() {
        let x = Expr::var("x");
        // x^2 - 2x + 1 == 0  =>  double root x = 1
        let eq = x.clone().pow(Expr::constant(2.0)) - Expr::constant(2.0) * x.clone()
            + Expr::constant(1.0);
        assert_eq!(eq.solve("x"), vec![Expr::Const(1.0)]);

        // x^2 + 1 == 0 has no real roots
        let eq = x.clone().pow(Expr::constant(2.0)) + Expr::constant(1.0);
        assert!(eq.solve("x").is_empty());
    }

    #[test]
    fn solve_unsupported_cases_return_empty() {
        let x = Expr::var("x");
        assert!(x.clone().sin().solve("x").is_empty()); // transcendental
        assert!((Expr::constant(1.0) / x.clone()).solve("x").is_empty()); // var in divisor
        assert!(Expr::constant(2.0).pow(x.clone()).solve("x").is_empty()); // var in exponent
        assert!(Expr::constant(1.0).solve("x").is_empty()); // contradiction
        assert!(Expr::constant(0.0).solve("x").is_empty()); // identity
        assert!(x.clone().solve("y").is_empty()); // unknown absent
                                                  // cubic that is not a monomial
        let cubic = x.clone().pow(Expr::constant(3.0)) - x.clone();
        assert!(cubic.solve("x").is_empty());
        // pure monomial of high degree still solves
        let mono = Expr::constant(4.0) * x.clone().pow(Expr::constant(3.0));
        assert_eq!(mono.solve("x"), vec![Expr::Const(0.0)]);
    }

    #[test]
    fn contains_var_walks_the_tree() {
        let e = Expr::var("x").sin() * Expr::var("y") + Expr::constant(1.0);
        assert!(e.contains_var("x"));
        assert!(e.contains_var("y"));
        assert!(!e.contains_var("z"));
    }
}
