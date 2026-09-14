//! The static checker (Phase 6 type-system slice).
//!
//! Two analyses over the parsed AST, run *before* execution:
//!
//! - **Compile-time unit checking** — the spec's Four Killer Features line
//!   *"Unit mismatch (meters + seconds) is a compile error"*. Dimension
//!   algebra over base-symbol exponent maps (`m/s^2` → `{m:1, s:-2}`):
//!   `+`/`-` require equal dimensions, `*`/`/` compose them.
//! - **Tensor shape inference** — literal and native shapes flow through
//!   `let`; `matmul(a, b)` checks inner dims statically.
//!
//! Gradual by design: anything unknown is simply not constrained — the
//! checker never blocks a program it cannot understand.

use std::collections::BTreeMap;
use std::fmt;

use crate::interp::{Expr, Stmt};

/// A physical dimension: base symbol → integer exponent. `m/s^2` parses to
/// `{m: +1, s: -2}`. The empty map is the dimensionless unit.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Dim(BTreeMap<String, i32>);

impl Dim {
    /// Parse a dimension string: tokens joined by `*` or `/`, each an
    /// identifier optionally raised with `^` (e.g. `m`, `m/s`, `m*s^2`).
    pub fn parse(s: &str) -> Dim {
        let mut dim = Dim::default();
        let mut sign = 1;
        let mut num = String::new();
        for c in s.chars() {
            match c {
                '*' => {
                    dim.apply_token(&num, sign);
                    num.clear();
                }
                '/' => {
                    dim.apply_token(&num, sign);
                    num.clear();
                    sign = -sign;
                }
                other => num.push(other),
            }
        }
        dim.apply_token(&num, sign);
        dim
    }

    fn add(&mut self, base: &str, exp: i32) {
        if base.is_empty() {
            return;
        }
        let e = self.0.entry(base.to_string()).or_insert(0);
        *e += exp;
        if *e == 0 {
            self.0.remove(base);
        }
    }

    /// Compose two dimensions: `sign = 1` for multiplication, `-1` for division.
    pub fn compose(a: &Dim, b: &Dim, sign: i32) -> Dim {
        let mut out = a.clone();
        for (k, v) in &b.0 {
            out.add(k, v * sign);
        }
        out
    }

    /// Whether this dimension is dimensionless.
    pub fn is_dimensionless(&self) -> bool {
        self.0.is_empty()
    }

    fn apply_token(&mut self, token: &str, sign: i32) {
        let token = token.trim();
        if token.is_empty() {
            return;
        }
        let (base, exp) = match token.split_once('^') {
            Some((b, e)) => (b.trim(), e.trim().parse::<i32>().unwrap_or(1)),
            None => (token, 1),
        };
        if base.chars().all(char::is_alphabetic) {
            self.add(base, exp * sign);
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "(dimensionless)");
        }
        let mut top = Vec::new();
        let mut bottom = Vec::new();
        for (k, v) in &self.0 {
            match v {
                1 => top.push(k.clone()),
                n if *n > 1 => top.push(format!("{k}^{n}")),
                -1 => bottom.push(k.clone()),
                n => bottom.push(format!("{k}^{}", -n)),
            }
        }
        let num = if top.is_empty() {
            "1".to_string()
        } else {
            top.join("*")
        };
        if bottom.is_empty() {
            write!(f, "{num}")
        } else {
            write!(f, "{num}/{}", bottom.join("*"))
        }
    }
}

/// A static-check failure.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckError {
    /// Python-style kind: `UnitError`, `ShapeError`, or `SyntaxError`.
    pub kind: String,
    pub message: String,
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}
impl std::error::Error for CheckError {}

/// Statically check `src` (units + shapes) without executing it.
///
/// This is the compile-time half of the spec's Four Killer Features line
/// *"Unit mismatch (meters + seconds) is a compile error"*.
pub fn check_program(src: &str) -> Result<(), CheckError> {
    let toks = crate::interp::lex(src).map_err(|e| CheckError {
        kind: "SyntaxError".into(),
        message: e.message,
    })?;
    let mut parser = crate::interp::Parser::new(&toks);
    let prog = parser.parse_block().map_err(|e| CheckError {
        kind: "SyntaxError".into(),
        message: e.message,
    })?;
    Checker::default().check_block(&prog)
}

/// Known shapes may contain unknown dims (represented as `None`).
type Shape = Vec<Option<usize>>;

/// Static analysis state: per-variable dimension and shape knowledge.
#[derive(Default)]
struct Checker {
    dims: BTreeMap<String, Dim>,
    shapes: BTreeMap<String, Shape>,
}

impl Checker {
    fn check_block(&mut self, stmts: &[Stmt]) -> Result<(), CheckError> {
        for s in stmts {
            match s {
                Stmt::Let(name, e) => self.bind(name, e)?,
                Stmt::Assign(name, e) => {
                    if let Some(prev) = self.dims.get(name).cloned() {
                        if let Some(d) = self.expr_dim(e)? {
                            if prev != d && !prev.is_dimensionless() && !d.is_dimensionless() {
                                return Err(CheckError {
                                    kind: "UnitError".into(),
                                    message: format!(
                                        "'{name}' has unit '{prev}' but is assigned an expression of unit '{d}'"
                                    ),
                                });
                            }
                        }
                    }
                    self.bind(name, e)?;
                }
                Stmt::Print(e) | Stmt::Assert(e) | Stmt::Expr(e) | Stmt::Return(Some(e)) => {
                    self.expr_dim(e)?;
                    // shape knowledge is best-effort: "Unknown" is not failure
                    if let Err(err) = self.expr_shape(e) {
                        if err.kind != "Unknown" {
                            return Err(err);
                        }
                    }
                }
                Stmt::Return(None) => {}
                Stmt::If(cond, a, b) => {
                    self.expr_dim(cond)?;
                    // branch-local knowledge is not merged (gradual)
                    Checker::default().check_block(a)?;
                    Checker::default().check_block(b)?;
                }
                Stmt::While(cond, body) => {
                    self.expr_dim(cond)?;
                    Checker::default().check_block(body)?;
                }
                Stmt::Module(_, body) => {
                    Checker::default().check_block(body)?;
                }
                Stmt::MemberAssign(base, _, e) => {
                    self.expr_dim(base)?;
                    self.expr_dim(e)?;
                }
                Stmt::Def(_, _, _) => {}
            }
        }
        Ok(())
    }

    /// Evaluate `let`/`assign`: record unit + shape knowledge for `name`.
    fn bind(&mut self, name: &str, e: &Expr) -> Result<(), CheckError> {
        if let Some(d) = self.expr_dim(e)? {
            self.dims.insert(name.to_string(), d);
        }
        match self.expr_shape(e) {
            Ok(sh) => {
                self.shapes.insert(name.to_string(), sh);
            }
            Err(err) if err.kind == "Unknown" => {}
            Err(err) => return Err(err),
        }
        Ok(())
    }

    /// Dimension of an expression (`None` = unknown).
    fn expr_dim(&mut self, e: &Expr) -> Result<Option<Dim>, CheckError> {
        match e {
            Expr::UnitNum(_, dim) => Ok(Some(Dim::parse(dim))),
            Expr::Num(_) | Expr::Bool(_) | Expr::Nil | Expr::Str(_) => Ok(Some(Dim::default())),
            Expr::Ident(name) => Ok(self.dims.get(name).cloned()),
            Expr::List(items) => {
                let mut dim = None;
                for it in items {
                    dim = self.expr_dim(it)?.or(dim);
                }
                Ok(dim)
            }
            Expr::Dict(items) => {
                for (_, ve) in items {
                    self.expr_dim(ve)?;
                }
                Ok(None)
            }
            Expr::Unary(op, inner) => {
                let d = self.expr_dim(inner)?;
                if op == "not" {
                    Ok(Some(Dim::default()))
                } else {
                    Ok(d)
                }
            }
            Expr::Binary(op, l, r) => {
                let ld = self.expr_dim(l)?;
                let rd = self.expr_dim(r)?;
                match op.as_str() {
                    "+" | "-" => match (&ld, &rd) {
                        // the Four Killer Features line item: this is where
                        // `meters + seconds` dies at compile time
                        (Some(a), Some(b)) if !a.is_dimensionless() || !b.is_dimensionless() => {
                            if a != b {
                                return Err(CheckError {
                                    kind: "UnitError".into(),
                                    message: format!("cannot add/subtract '{a}' and '{b}'"),
                                });
                            }
                            Ok(ld)
                        }
                        _ => Ok(ld.or(rd)),
                    },
                    "*" => Ok(match (&ld, &rd) {
                        (Some(a), Some(b)) => Some(Dim::compose(a, b, 1)),
                        _ => ld.or(rd),
                    }),
                    "/" => Ok(match (&ld, &rd) {
                        (Some(a), Some(b)) => Some(Dim::compose(a, b, -1)),
                        _ => ld.or(rd),
                    }),
                    "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                        if let (Some(a), Some(b)) = (&ld, &rd) {
                            if a != b && !a.is_dimensionless() && !b.is_dimensionless() {
                                return Err(CheckError {
                                    kind: "UnitError".into(),
                                    message: format!("comparing '{a}' with '{b}'"),
                                });
                            }
                        }
                        Ok(Some(Dim::default()))
                    }
                    other => Err(CheckError {
                        kind: "SyntaxError".into(),
                        message: format!("unknown operator '{other}'"),
                    }),
                }
            }
            Expr::Call(fexpr, args) => {
                for a in args {
                    let _ = self.expr_shape(a);
                    self.expr_dim(a)?;
                }
                if let (Expr::Ident(name), Some(first)) = (fexpr.as_ref(), args.first()) {
                    if name == "sum" {
                        return self.expr_dim(first);
                    }
                }
                Ok(None)
            }
            Expr::Index(base, idx) => {
                let bd = self.expr_dim(base)?;
                self.expr_dim(idx)?;
                Ok(bd)
            }
            Expr::Member(base, _) => {
                self.expr_dim(base)?;
                // module members have unknown dimensions (gradual)
                Ok(None)
            }
        }
    }

    /// Shape of an expression when statically knowable.
    /// `Err(kind == "Unknown")` means "no opinion", not failure.
    fn expr_shape(&mut self, e: &Expr) -> Result<Shape, CheckError> {
        match e {
            Expr::Ident(name) => match self.shapes.get(name) {
                Some(sh) => Ok(sh.clone()),
                None => Err(CheckError {
                    kind: "Unknown".into(),
                    message: String::new(),
                }),
            },
            Expr::Call(fexpr, args) => {
                for a in args {
                    let _ = self.expr_shape(a);
                }
                if let Expr::Ident(name) = fexpr.as_ref() {
                    if name == "ones" || name == "zeros" {
                        return Ok(args
                            .iter()
                            .flat_map(|a| match a {
                                Expr::Num(n) => vec![Some(*n as usize)],
                                Expr::List(items) => items
                                    .iter()
                                    .map(|i| match i {
                                        Expr::Num(n) => Some(*n as usize),
                                        _ => None,
                                    })
                                    .collect(),
                                _ => vec![None],
                            })
                            .collect());
                    }
                    if name == "matmul" {
                        let sa = args
                            .first()
                            .map(|a| self.expr_shape(a))
                            .transpose()
                            .ok()
                            .flatten();
                        let sb = args
                            .get(1)
                            .map(|a| self.expr_shape(a))
                            .transpose()
                            .ok()
                            .flatten();
                        if let (Some(sa), Some(sb)) = (&sa, &sb) {
                            if sa.len() == 2 && sb.len() == 2 {
                                match (sa[1], sb[0]) {
                                    (Some(k1), Some(k2)) if k1 != k2 => {
                                        return Err(CheckError {
                                            kind: "ShapeError".into(),
                                            message: format!(
                                                "matmul inner dims differ: {k1} vs {k2}"
                                            ),
                                        });
                                    }
                                    _ => return Ok(vec![sa[0], sb[1]]),
                                }
                            }
                        }
                    }
                }
                Err(CheckError {
                    kind: "Unknown".into(),
                    message: String::new(),
                })
            }
            Expr::Binary(op, l, r) => {
                let sa = match self.expr_shape(l) { Ok(s) => s, Err(e) => return Err(e) };
                let sb = match self.expr_shape(r) { Ok(s) => s, Err(e) => return Err(e) };
                if sa == sb && op != "/" {
                    Ok(sa)
                } else if op == "*"
                    && sa.last() == sb.first()
                    && sa.len() >= 2
                    && sb.len() >= 2
                {
                    let mut out = sa;
                    out.pop();
                    out.extend(sb.into_iter().skip(1));
                    Ok(out)
                } else {
                    Err(CheckError {
                        kind: "Unknown".into(),
                        message: String::new(),
                    })
                }
            }
            _ => Err(CheckError {
                kind: "Unknown".into(),
                message: String::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_parse_and_display() {
        assert_eq!(Dim::parse("m/s^2").to_string(), "m/s^2");
        assert_eq!(Dim::parse("m").to_string(), "m");
        assert_eq!(Dim::parse("").to_string(), "(dimensionless)");
        assert_eq!(
            // (m/s) / s = m/s^2
            Dim::compose(&Dim::parse("m/s"), &Dim::parse("s"), -1).to_string(),
            "m/s^2"
        );
        assert_eq!(
            // (m/s) * s = m
            Dim::compose(&Dim::parse("m/s"), &Dim::parse("s"), 1).to_string(),
            "m"
        );
    }

    #[test]
    fn unit_mismatch_is_a_compile_error() {
        // the Four Killer Features line item, verbatim scenario
        let err = check_program(
            "let d = 3.0 m
let t = 5.0 s
let v = d + t",
        )
        .unwrap_err();
        assert_eq!(err.kind, "UnitError");
        assert!(err.message.contains("m") && err.message.contains("s"));
    }

    #[test]
    fn compatible_units_pass_and_compose() {
        check_program(
            "let d = 3.0 m
let t = 5.0 s
let v = d / t
let more = v * t
let total = d + more",
        )
        .unwrap();
    }

    #[test]
    fn matmul_inner_dim_mismatch_is_a_compile_error() {
        let err = check_program(
            "let a = ones([2, 3])
let b = ones([4, 5])
let c = matmul(a, b)",
        )
        .unwrap_err();
        assert_eq!(err.kind, "ShapeError");
        assert!(err.message.contains("3 vs 4"));
        // matching dims pass
        check_program(
            "let a = ones([2, 3])
let b = ones([3, 5])
let c = matmul(a, b)",
        )
        .unwrap();
    }

    #[test]
    fn clean_programs_are_not_blocked() {
        check_program(
            "let x = 10
if x > 5 { print \"big\" } else { print \"small\" }
let l = [1.0, 2.0]
print len(l)",
        )
        .unwrap();
    }
}
