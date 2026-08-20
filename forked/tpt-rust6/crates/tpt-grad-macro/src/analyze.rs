//! Signature + body validation shared by the three attribute macros.
//!
//! The accepted subset is documented in the `tpt-grad` crate docs; this module
//! is the single place where it is enforced.

use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    BinOp, Block, Error, Expr, FnArg, Ident, ItemFn, Lit, Pat, ReturnType, Stmt, Type, UnOp,
};

/// Whitelisted method calls. `sum`/`mean` are the only reductions.
pub const METHODS: &[&str] = &["clone", "powf", "exp", "ln", "sqrt", "sum", "mean"];
const REDUCTIONS: &[&str] = &["sum", "mean"];

const SUBSET: &str = "tpt-grad supports: `+ - * /`, unary `-`, parentheses, identifiers, \
float literals, and the methods .clone() .powf(c) .exp() .ln() .sqrt() .sum() .mean()";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `Tensor<f64>`: differentiated by `_grad`, batched by `_vmap`.
    Tensor,
    /// `f64`: passed through untouched.
    Scalar,
}

pub struct Arg {
    pub ident: Ident,
    pub kind: Kind,
}

pub struct Fun {
    pub item: ItemFn,
    pub args: Vec<Arg>,
    pub ret: Type,
    pub ret_is_tensor: bool,
    /// True when the body contains no reduction, so it can be fused.
    pub elementwise: bool,
}

impl Fun {
    pub fn tensor_args(&self) -> Vec<&Ident> {
        self.args
            .iter()
            .filter(|a| a.kind == Kind::Tensor)
            .map(|a| &a.ident)
            .collect()
    }
}

fn ty_string(t: &Type) -> String {
    t.to_token_stream().to_string().replace(' ', "")
}

/// Reject anything outside the documented subset, and classify the signature.
pub fn analyze(item: ItemFn, mac: &str) -> Result<Fun, Error> {
    let sig = &item.sig;
    let bad = |msg: &str| Error::new_spanned(&item.sig, format!("#[{mac}]: {msg}"));
    if sig.asyncness.is_some() || sig.unsafety.is_some() || sig.abi.is_some() {
        return Err(bad("async / unsafe / extern functions are not supported"));
    }
    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(bad("generic functions are not supported"));
    }

    let mut args = Vec::new();
    for input in &sig.inputs {
        let t = match input {
            FnArg::Receiver(_) => return Err(bad("methods taking `self` are not supported")),
            FnArg::Typed(t) => t,
        };
        let ident = match &*t.pat {
            Pat::Ident(p) if p.by_ref.is_none() && p.subpat.is_none() => p.ident.clone(),
            other => {
                return Err(Error::new_spanned(
                    other,
                    format!("#[{mac}]: arguments must be plain identifiers"),
                ))
            }
        };
        let s = ty_string(&t.ty);
        let kind = if s == "f64" {
            Kind::Scalar
        } else if s.ends_with("Tensor<f64>") {
            Kind::Tensor
        } else {
            return Err(Error::new_spanned(
                &t.ty,
                format!("#[{mac}]: argument types must be `Tensor<f64>` or `f64`, found `{s}`"),
            ));
        };
        args.push(Arg { ident, kind });
    }

    let ret = match &sig.output {
        ReturnType::Default => return Err(bad("the function must return `Tensor<f64>` or `f64`")),
        ReturnType::Type(_, t) => (**t).clone(),
    };
    let rs = ty_string(&ret);
    let ret_is_tensor = rs.ends_with("Tensor<f64>");
    if !ret_is_tensor && rs != "f64" {
        return Err(Error::new_spanned(
            &ret,
            format!("#[{mac}]: return type must be `Tensor<f64>` or `f64`, found `{rs}`"),
        ));
    }

    let mut reduction = false;
    check_block(&item.block, mac, &mut reduction)?;

    Ok(Fun {
        item,
        args,
        ret,
        ret_is_tensor,
        elementwise: !reduction,
    })
}

fn err(tokens: impl ToTokens, mac: &str, what: &str) -> Error {
    Error::new_spanned(
        tokens,
        format!("#[{mac}]: {what} is not supported. {SUBSET}"),
    )
}

fn check_block(b: &Block, mac: &str, red: &mut bool) -> Result<(), Error> {
    if b.stmts.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            format!("#[{mac}]: empty body"),
        ));
    }
    let n = b.stmts.len();
    for (i, st) in b.stmts.iter().enumerate() {
        match st {
            Stmt::Local(l) => {
                if !matches!(&l.pat, Pat::Ident(p) if p.by_ref.is_none() && p.subpat.is_none()) {
                    return Err(err(l, mac, "this `let` pattern (or type annotation)"));
                }
                let init = l
                    .init
                    .as_ref()
                    .ok_or_else(|| err(l, mac, "an uninitialized `let`"))?;
                if init.diverge.is_some() {
                    return Err(err(l, mac, "`let ... else`"));
                }
                check_expr(&init.expr, mac, red)?;
            }
            Stmt::Expr(e, semi) => {
                if semi.is_some() || i + 1 != n {
                    return Err(err(
                        e,
                        mac,
                        "a statement expression (the body must end in a tail expression)",
                    ));
                }
                check_expr(e, mac, red)?;
            }
            other => return Err(err(other, mac, "this statement")),
        }
    }
    Ok(())
}

fn check_expr(e: &Expr, mac: &str, red: &mut bool) -> Result<(), Error> {
    match e {
        Expr::Binary(b) => {
            if !matches!(
                b.op,
                BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) | BinOp::Div(_)
            ) {
                return Err(err(b, mac, "this binary operator"));
            }
            check_expr(&b.left, mac, red)?;
            check_expr(&b.right, mac, red)
        }
        Expr::Unary(u) => {
            if !matches!(u.op, UnOp::Neg(_)) {
                return Err(err(u, mac, "this unary operator"));
            }
            check_expr(&u.expr, mac, red)
        }
        Expr::Paren(p) => check_expr(&p.expr, mac, red),
        Expr::Block(b) if b.label.is_none() => check_block(&b.block, mac, red),
        Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => Ok(()),
        Expr::Lit(l) => match &l.lit {
            Lit::Float(_) => Ok(()),
            Lit::Int(_) => Err(err(l, mac, "an integer literal (write `2.0`, not `2`)")),
            _ => Err(err(l, mac, "this literal")),
        },
        Expr::MethodCall(m) => {
            let name = m.method.to_string();
            if !METHODS.contains(&name.as_str()) {
                return Err(err(&m.method, mac, format!("`.{name}()`").as_str()));
            }
            if name == "powf" {
                if m.args.len() != 1 {
                    return Err(err(m, mac, "`.powf()` with other than one argument"));
                }
                match &m.args[0] {
                    Expr::Lit(l) if matches!(l.lit, Lit::Float(_)) => {}
                    Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {}
                    other => {
                        return Err(err(
                            other,
                            mac,
                            "this `.powf()` exponent (use a float literal or an `f64` argument)",
                        ))
                    }
                }
            } else if !m.args.is_empty() {
                return Err(err(m, mac, "arguments to this method"));
            }
            if REDUCTIONS.contains(&name.as_str()) {
                *red = true;
            }
            check_expr(&m.receiver, mac, red)
        }
        other => Err(err(other, mac, "this expression")),
    }
}
