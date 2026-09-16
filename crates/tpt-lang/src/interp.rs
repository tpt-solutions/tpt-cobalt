//! The eager interpreter (spec §6.2: `ExecutionMode::Eager`).
//!
//! Line-oriented TPT Script: `let` / assignment / `print` / `assert` /
//! `if-else` / `while` / `def` + `return`, with full expressions (arithmetic,
//! comparisons, calls, indexing, list/dict literals). Arithmetic dispatches
//! into [`crate::ops`] so **tensor-first** semantics (broadcasting, element-wise
//! ops) come for free from the shared [`crate::value::Value`] model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tpt_tensor::Tensor;

use crate::env::Environment;
use crate::ops::{LangError, value_add, value_div, value_eq, value_mul, value_sub};
use crate::value::{Function, Truthiness, Value};

/// A runtime error: Python-style kind plus message.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind}: {message}")]
pub struct InterpreterError {
    pub kind: String,
    pub message: String,
}

impl InterpreterError {
    pub fn new(kind: &str, message: impl Into<String>) -> Self {
        InterpreterError {
            kind: kind.to_string(),
            message: message.into(),
        }
    }

    fn from_lang(e: LangError) -> Self {
        match e {
            LangError::DivByZero => InterpreterError::new("ZeroDivisionError", "division by zero"),
            other => InterpreterError::new("TypeError", other.to_string()),
        }
    }
}

pub type Res<T> = Result<T, InterpreterError>;

// ------------------------------- lexer ------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Num(f64),
    /// Numeric literal with a physical-unit suffix (`9.81 m/s^2`).
    Unit(f64, String),
    Str(String),
    Op(String),
    Newline,
}

/// Tokenize a whole program. Newlines are significant (statement ends)
/// except inside brackets.
pub fn lex(src: &str) -> Res<Vec<Tok>> {
    let cs: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut depth = 0usize;
    while i < cs.len() {
        let c = cs[i];
        match c {
            '#' => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '\n' => {
                if depth == 0 && !matches!(out.last(), None | Some(Tok::Newline)) {
                    out.push(Tok::Newline);
                }
                i += 1;
            }
            c if c.is_whitespace() => i += 1,
            c if c.is_ascii_digit() => {
                let start = i;
                let mut dot = false;
                while i < cs.len()
                    && (cs[i].is_ascii_digit()
                        || (cs[i] == '.' && !dot && cs.get(i + 1) != Some(&'.')))
                {
                    if cs[i] == '.' {
                        dot = true;
                    }
                    i += 1;
                }
                let s: String = cs[start..i].iter().collect();
                let n: f64 = s.parse().map_err(|_| {
                    InterpreterError::new("SyntaxError", format!("bad number `{s}`"))
                })?;
                // optional physical-unit suffix: either attached (`3.0m`) or
                // space-separated (`3.0 m/s^2`)
                let mut j = i;
                if j < cs.len() && cs[j] == ' ' {
                    j += 1;
                }
                if j < cs.len() && cs[j].is_alphabetic() {
                    let ustart = j;
                    while j < cs.len()
                        && (cs[j].is_alphabetic()
                            || cs[j] == '/'
                            || cs[j] == '^'
                            || (cs[j].is_ascii_digit() && cs[j - 1] == '^'))
                    {
                        j += 1;
                    }
                    let dim: String = cs[ustart..j].iter().collect();
                    i = j;
                    out.push(Tok::Unit(n, dim));
                } else {
                    out.push(Tok::Num(n));
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Ident(cs[start..i].iter().collect()));
            }
            '"' => {
                i += 1;
                let start = i;
                while i < cs.len() && cs[i] != '"' {
                    i += 1;
                }
                if i >= cs.len() {
                    return Err(InterpreterError::new("SyntaxError", "unterminated string"));
                }
                out.push(Tok::Str(cs[start..i].iter().collect()));
                i += 1;
            }
            '[' | '(' => {
                depth += 1;
                out.push(Tok::Op(c.to_string()));
                i += 1;
            }
            ']' | ')' => {
                depth = depth.saturating_sub(1);
                out.push(Tok::Op(c.to_string()));
                i += 1;
            }
            '.' if cs.get(i + 1) == Some(&'.') => {
                out.push(Tok::Op("..".into()));
                i += 2;
            }
            '{' | '}' | ',' | ':' | '.' => {
                out.push(Tok::Op(c.to_string()));
                i += 1;
            }
            '+' | '-' | '*' | '/' | '%' | '<' | '>' | '=' => {
                let two: String = [c, cs.get(i + 1).copied().unwrap_or(' ')].iter().collect();
                if ["==", "!=", "<=", ">="].contains(&two.as_str()) {
                    out.push(Tok::Op(two));
                    i += 2;
                } else {
                    out.push(Tok::Op(c.to_string()));
                    i += 1;
                }
            }
            other => {
                return Err(InterpreterError::new(
                    "SyntaxError",
                    format!("unexpected character '{other}'"),
                ));
            }
        }
    }
    if !out.is_empty() && !matches!(out.last(), Some(Tok::Newline)) {
        out.push(Tok::Newline);
    }
    Ok(out)
}

// ------------------------------- parser -----------------------------------

/// Expression AST.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Unit-aware numeric literal: value + dimension tag.
    UnitNum(f64, String),
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
    Ident(String),
    List(Vec<Expr>),
    Dict(Vec<(String, Expr)>),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<CallArg>),
    Index(Box<Expr>, Vec<IdxArg>),
    /// Attribute access `base.name`: module members (and dict keys as sugar).
    Member(Box<Expr>, String),
}

/// One argument in a call: positional, or `name = expr` keyword.
#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<String>,
    pub expr: Expr,
}

/// One bracket index: a plain expression, or an `a:b` slice (either side
/// optional).
#[derive(Debug, Clone)]
pub enum IdxArg {
    Expr(Expr),
    Range(Option<Expr>, Option<Expr>),
}

/// A parameter declaration in a `def`: name plus optional default expression
/// (evaluated once at `def` time).
#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: String,
    pub default: Option<Expr>,
}

/// Statement AST.
#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Assign(String, Expr),
    /// `base.name = expr` (module member assignment).
    MemberAssign(Box<Expr>, String, Expr),
    Print(Expr),
    Assert(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    Def(String, Vec<ParamDecl>, Vec<Stmt>),
    /// `module Name { ... }`: execute the body in a child scope, then bind
    /// its bindings as a [`crate::value::Module`] under `Name`.
    Module(String, Vec<Stmt>),
    /// `for name in iterable { body }`. Iterates lists, tensors (flat),
    /// strings (chars), and dicts (sorted keys). The loop variable is bound
    /// in the current scope (Python-like).
    For(String, Expr, Vec<Stmt>),
    Return(Option<Expr>),
    Expr(Expr),
}

pub struct Parser<'t> {
    toks: &'t [Tok],
    pos: usize,
}

impl<'t> Parser<'t> {
    pub fn new(toks: &'t [Tok]) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn at_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if o == op)
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn eat_op(&mut self, op: &str) -> Res<bool> {
        if self.at_op(op) {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect_op(&mut self, op: &str) -> Res<()> {
        if !self.eat_op(op)? {
            return Err(InterpreterError::new(
                "SyntaxError",
                format!("expected '{op}'"),
            ));
        }
        Ok(())
    }

    fn expect_ident(&mut self) -> Res<String> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                Ok(s)
            }
            other => Err(InterpreterError::new(
                "SyntaxError",
                format!("expected identifier, got {other:?}"),
            )),
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Tok::Newline)) {
            self.pos += 1;
        }
    }

    /// Parse statements until EOF or an unmatched `}`.
    pub fn parse_block(&mut self) -> Res<Vec<Stmt>> {
        let mut out = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek() {
                None => break,
                Some(Tok::Op(o)) if o == "}" => break,
                _ => out.push(self.parse_stmt()?),
            }
        }
        Ok(out)
    }

    fn parse_stmt(&mut self) -> Res<Stmt> {
        match self.peek().cloned() {
            Some(Tok::Ident(kw)) => match kw.as_str() {
                "let" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    self.expect_op("=")?;
                    Ok(Stmt::Let(name, self.parse_expr()?))
                }
                "print" => {
                    self.pos += 1;
                    Ok(Stmt::Print(self.parse_expr()?))
                }
                "assert" => {
                    self.pos += 1;
                    Ok(Stmt::Assert(self.parse_expr()?))
                }
                "if" => {
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let then_body = self.parse_block()?;
                    self.expect_op("}")?;
                    let mut else_body = Vec::new();
                    self.skip_newlines();
                    if self.at_kw("else") || self.at_kw("elif") {
                        else_body = self.parse_tail_else()?;
                    }
                    Ok(Stmt::If(cond, then_body, else_body))
                }
                "while" => {
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let body = self.parse_block()?;
                    self.expect_op("}")?;
                    Ok(Stmt::While(cond, body))
                }
                "for" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    if !self.at_kw("in") {
                        return Err(InterpreterError::new(
                            "SyntaxError",
                            "expected 'in' in for loop",
                        ));
                    }
                    self.pos += 1;
                    let iterable = self.parse_expr()?;
                    self.expect_op("{")?;
                    let body = self.parse_block()?;
                    self.expect_op("}")?;
                    Ok(Stmt::For(name, iterable, body))
                }
                "elif" => {
                    // `elif` desugars to a nested if in the else branch
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let then_body = self.parse_block()?;
                    self.expect_op("}")?;
                    let mut else_body = Vec::new();
                    self.skip_newlines();
                    if self.at_kw("else") || self.at_kw("elif") {
                        else_body = self.parse_tail_else()?;
                    }
                    Ok(Stmt::If(cond, then_body, else_body))
                }
                "def" | "fn" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    self.expect_op("(")?;
                    let mut params = Vec::new();
                    let mut seen_default = false;
                    while !self.at_op(")") {
                        let pname = self.expect_ident()?;
                        let mut default = None;
                        if self.eat_op("=")? {
                            default = Some(self.parse_expr()?);
                            seen_default = true;
                        } else if seen_default {
                            return Err(InterpreterError::new(
                                "SyntaxError",
                                format!(
                                    "parameter '{pname}' lacks a default after a defaulted parameter"
                                ),
                            ));
                        }
                        params.push(ParamDecl {
                            name: pname,
                            default,
                        });
                        if !self.eat_op(",")? {
                            break;
                        }
                    }
                    self.expect_op(")")?;
                    self.expect_op("{")?;
                    let body = self.parse_block()?;
                    self.expect_op("}")?;
                    Ok(Stmt::Def(name, params, body))
                }
                "module" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    self.expect_op("{")?;
                    let body = self.parse_block()?;
                    self.expect_op("}")?;
                    Ok(Stmt::Module(name, body))
                }
                "return" => {
                    self.pos += 1;
                    let e = match self.peek() {
                        None | Some(Tok::Newline) => None,
                        Some(Tok::Op(o)) if o == "}" || o == ";" => None,
                        _ => Some(self.parse_expr()?),
                    };
                    Ok(Stmt::Return(e))
                }
                _ => self.parse_assign_or_expr(),
            },
            _ => self.parse_assign_or_expr(),
        }
    }

    /// The `else { ... }` / `else if ...` / `elif ...` tail of an if.
    fn parse_tail_else(&mut self) -> Res<Vec<Stmt>> {
        if self.at_kw("elif") || self.peek_is_ident("if") {
            // recurse: the tail is itself an if-statement
            return Ok(vec![self.parse_stmt()?]);
        }
        self.pos += 1; // 'else'
        self.expect_op("{")?;
        let body = self.parse_block()?;
        self.expect_op("}")?;
        Ok(body)
    }

    fn peek_is_ident(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn parse_assign_or_expr(&mut self) -> Res<Stmt> {
        // lookahead: IDENT '=' expr (but not '=='), and IDENT '.' IDENT '='
        // for module member assignment
        if let Some(Tok::Ident(_)) = self.peek() {
            match self.toks.get(self.pos + 1) {
                Some(Tok::Op(op)) if op == "=" => {
                    let name = self.expect_ident()?;
                    self.pos += 1; // '='
                    let e = self.parse_expr()?;
                    return Ok(Stmt::Assign(name, e));
                }
                Some(Tok::Op(op)) if op == "." => {
                    // IDENT ('.' IDENT)* '.' IDENT '=' → member(-chain)
                    // assignment. Pure lookahead: nothing is consumed unless
                    // the full pattern matches (member *access* falls through).
                    let mut segs: Vec<String> = Vec::new();
                    let mut j = self.pos;
                    while matches!(self.toks.get(j), Some(Tok::Ident(_)))
                        && matches!(self.toks.get(j + 1), Some(Tok::Op(o)) if o == ".")
                    {
                        segs.push(match self.toks.get(j).cloned() {
                            Some(Tok::Ident(s)) => s,
                            _ => unreachable!(),
                        });
                        j += 2;
                    }
                    let is_assign = matches!(self.toks.get(j), Some(Tok::Ident(_)))
                        && matches!(self.toks.get(j + 1), Some(Tok::Op(o)) if o == "=");
                    if is_assign {
                        segs.push(match self.toks.get(j).cloned() {
                            Some(Tok::Ident(s)) => s,
                            _ => unreachable!(),
                        });
                        j += 1; // at '='
                    }
                    if is_assign && segs.len() >= 2 {
                        let mut base = Expr::Ident(segs[0].clone());
                        for seg in &segs[1..segs.len() - 1] {
                            base = Expr::Member(Box::new(base), seg.clone());
                        }
                        self.pos = j + 1; // past '='
                        let e = self.parse_expr()?;
                        return Ok(Stmt::MemberAssign(
                            Box::new(base),
                            segs.last().unwrap().clone(),
                            e,
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(Stmt::Expr(self.parse_expr()?))
    }

    pub fn parse_expr(&mut self) -> Res<Expr> {
        let lhs = self.parse_comparison()?;
        if self.at_op("..") {
            self.pos += 1;
            let rhs = self.parse_comparison()?;
            return Ok(Expr::Binary("..".into(), Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> Res<Expr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if ["==", "!=", "<", "<=", ">", ">="].contains(&o.as_str()) => {
                    o.clone()
                }
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_additive()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Res<Expr> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if o == "+" || o == "-" => o.clone(),
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_term()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Res<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if o == "*" || o == "/" || o == "%" => o.clone(),
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Res<Expr> {
        if self.at_op("-") {
            self.pos += 1;
            return Ok(Expr::Unary("-".into(), Box::new(self.parse_unary()?)));
        }
        if self.at_kw("not") {
            self.pos += 1;
            return Ok(Expr::Unary("not".into(), Box::new(self.parse_unary()?)));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Res<Expr> {
        let mut e = self.parse_atom()?;
        loop {
            if self.at_op(".") {
                self.pos += 1;
                let name = self.expect_ident()?;
                e = Expr::Member(Box::new(e), name);
            } else if self.at_op("[") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op("]") {
                    args.push(self.parse_idx_arg()?);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op("]")?;
                e = Expr::Index(Box::new(e), args);
            } else if self.at_op("(") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op(")") {
                    let arg = if matches!(self.peek(), Some(Tok::Ident(_)))
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Op(o)) if o == "=")
                    {
                        let name = self.expect_ident()?;
                        self.pos += 1; // '='
                        CallArg {
                            name: Some(name),
                            expr: self.parse_expr()?,
                        }
                    } else {
                        CallArg {
                            name: None,
                            expr: self.parse_expr()?,
                        }
                    };
                    args.push(arg);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op(")")?;
                e = Expr::Call(Box::new(e), args);
            } else {
                break;
            }
        }
        Ok(e)
    }

    /// One bracket index: `expr`, `expr? : expr?` (slice), with the slice
    /// form recognized by a leading ':' or a ':' after the first expression.
    fn parse_idx_arg(&mut self) -> Res<IdxArg> {
        let mut start = None;
        if !self.at_op(":") {
            start = Some(self.parse_expr()?);
        }
        if !self.at_op(":") {
            let start = start.ok_or_else(|| InterpreterError::new("SyntaxError", "empty index"))?;
            return Ok(IdxArg::Expr(start));
        }
        self.pos += 1; // ':'
        let mut end = None;
        if !self.at_op("]") && !self.at_op(",") {
            end = Some(self.parse_expr()?);
        }
        Ok(IdxArg::Range(start, end))
    }

    fn parse_atom(&mut self) -> Res<Expr> {
        match self.peek().cloned() {
            Some(Tok::Num(x)) => {
                self.pos += 1;
                Ok(Expr::Num(x))
            }
            Some(Tok::Unit(x, dim)) => {
                self.pos += 1;
                Ok(Expr::UnitNum(x, dim))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Str(s))
            }
            Some(Tok::Ident(s)) => match s.as_str() {
                "true" => {
                    self.pos += 1;
                    Ok(Expr::Bool(true))
                }
                "false" => {
                    self.pos += 1;
                    Ok(Expr::Bool(false))
                }
                "nil" | "None" => {
                    self.pos += 1;
                    Ok(Expr::Nil)
                }
                _ => {
                    self.pos += 1;
                    Ok(Expr::Ident(s))
                }
            },
            Some(Tok::Op(o)) if o == "(" => {
                self.pos += 1;
                let e = self.parse_expr()?;
                self.expect_op(")")?;
                Ok(e)
            }
            Some(Tok::Op(o)) if o == "[" => {
                self.pos += 1;
                let mut items = Vec::new();
                while !self.at_op("]") {
                    items.push(self.parse_expr()?);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op("]")?;
                Ok(Expr::List(items))
            }
            Some(Tok::Op(o)) if o == "{" => {
                self.pos += 1;
                let mut items = Vec::new();
                while !self.at_op("}") {
                    let key = match self.peek().cloned() {
                        Some(Tok::Str(s)) => s,
                        other => {
                            return Err(InterpreterError::new(
                                "SyntaxError",
                                format!("dict key must be a string, got {other:?}"),
                            ));
                        }
                    };
                    self.pos += 1;
                    self.expect_op(":")?;
                    let v = self.parse_expr()?;
                    items.push((key, v));
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op("}")?;
                Ok(Expr::Dict(items))
            }
            other => Err(InterpreterError::new(
                "SyntaxError",
                format!("unexpected token {other:?}"),
            )),
        }
    }
}

// ------------------------------ debugger ----------------------------------

/// What the debug callback decides when the interpreter pauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugAction {
    /// Resume until the next breakpoint hit.
    Continue,
    /// Execute the paused statement, then pause again before the next one.
    Step,
    /// Stop the program: `run_debug` returns a `KeyboardInterrupt` error.
    Abort,
}

/// A watch expression's value, evaluated in the paused scope. Tensor values
/// render element-wise via `Display` (tensor inspection).
#[derive(Debug, Clone)]
pub struct WatchValue {
    pub expr: String,
    pub value: Option<Value>,
    pub error: Option<String>,
}

/// Snapshot handed to the debug callback at each pause.
#[derive(Debug, Clone)]
pub struct DebugFrame {
    /// Profiler-style statement label (`let x`, `call train_step`, `while`…).
    pub label: String,
    /// 1-based ordinal of this statement execution within the run.
    pub statement: usize,
    /// Function-call nesting depth (0 = top level).
    pub depth: usize,
    /// Visible bindings, nearest scope first (locals shadow globals).
    pub locals: Vec<(String, Value)>,
    /// Configured watch expressions evaluated in the paused scope.
    pub watches: Vec<WatchValue>,
    /// `print` output captured so far.
    pub output: String,
}

impl DebugFrame {
    /// Look up a visible binding.
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.locals.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

#[derive(Default)]
struct DebugState {
    single_step: bool,
    /// Breakpoints as label substrings (`"train_step"` hits `call train_step`).
    breakpoints: Vec<String>,
    watches: Vec<String>,
    pending_step: bool,
}

/// Callback signature for [`Interpreter::run_debug`].
type DbgCb<'a> = &'a mut dyn FnMut(&DebugFrame) -> DebugAction;

// ----------------------------- interpreter --------------------------------

enum Flow {
    Normal,
    Return(Value),
}

/// The eager TPT Script interpreter over scoped [`Environment`]s.
pub struct Interpreter {
    globals: Arc<Environment>,
    /// `def` bodies keyed by the function's unique id (names can be shadowed
    /// or live inside modules; ids cannot collide).
    script_bodies: HashMap<u64, (Vec<crate::value::Param>, Vec<Stmt>)>,
    next_fn_id: u64,
    out: String,
    last_value: Option<Value>,
    /// When `Some`, every statement execution is recorded (profiler).
    trace_start: Option<std::time::Instant>,
    trace_events: Vec<TraceEvent>,
    /// Debugger config (breakpoints/watches/stepping) — only consulted when a
    /// callback is passed to [`Interpreter::run_debug`]; `run` is unaffected.
    debug: DebugState,
    /// 1-based statement-execution ordinal, reset per run (debug frames).
    stmt_counter: usize,
    /// Function-call nesting depth (debug frames).
    call_depth: usize,
}

/// One recorded statement execution (profiler).
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Human-readable operation name (statement kind + target).
    pub name: String,
    /// Microseconds since tracing started.
    pub start_us: u64,
    /// Execution duration in microseconds.
    pub dur_us: u64,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    /// New interpreter with the standard native surface registered.
    pub fn new() -> Self {
        let mut it = Interpreter {
            globals: Arc::new(Environment::new()),
            script_bodies: HashMap::new(),
            next_fn_id: 0,
            out: String::new(),
            last_value: None,
            trace_start: None,
            trace_events: Vec::new(),
            debug: DebugState::default(),
            stmt_counter: 0,
            call_depth: 0,
        };
        it.install_natives();
        crate::ml::install(&mut it);
        it
    }

    /// The global environment (parent scope of function calls).
    pub fn global_env(&self) -> &Arc<Environment> {
        &self.globals
    }

    /// Register a Rust function callable from script.
    pub fn register_native<F>(&mut self, name: &'static str, f: F)
    where
        F: Fn(&[Value]) -> Result<Value, String> + Send + Sync + 'static,
    {
        self.globals.define(
            name,
            Value::Function(Arc::new(crate::value::Function::Native {
                name,
                f: Arc::new(f),
            })),
        );
    }

    /// Captured `print` output from the last run.
    pub fn output(&self) -> &str {
        &self.out
    }

    /// Look up a global after a run.
    pub fn get(&self, name: &str) -> Option<Value> {
        self.globals.get(name)
    }

    /// Insert a value into globals before a run.
    pub fn set(&mut self, name: impl Into<String>, v: Value) {
        self.globals.define(name, v);
    }

    /// Sorted list of all visible global names (REPL `:env`).
    pub fn names(&self) -> Vec<String> {
        let mut names = self.globals.names();
        names.sort();
        names
    }

    /// Enable statement-level tracing (profiler). Events accumulate until
    /// [`Self::take_chrome_trace`] or [`Self::disable_tracing`].
    pub fn enable_tracing(&mut self) {
        self.trace_start = Some(std::time::Instant::now());
        self.trace_events.clear();
    }

    /// Stop recording; returns the events recorded so far.
    pub fn disable_tracing(&mut self) -> Vec<TraceEvent> {
        self.trace_start = None;
        std::mem::take(&mut self.trace_events)
    }

    /// Recorded events without stopping tracing.
    pub fn trace_events(&self) -> &[TraceEvent] {
        &self.trace_events
    }

    /// Export the recorded events as a **Chrome Trace Format** JSON document
    /// (loadable in `chrome://tracing`, Perfetto, or `go tool trace` viewers).
    pub fn take_chrome_trace_json(&mut self) -> String {
        let start = self.trace_start.unwrap_or_else(std::time::Instant::now);
        let mut json = String::from(
            "{\"displayTimeUnit\":\"us\",\"traceEvents\":[{\"name\":\"process_name\",\
             \"ph\":\"M\",\"pid\":1,\"tid\":1,\"args\":{\"name\":\"TPT Script\"}},",
        );
        for ev in &self.trace_events {
            let start_us = start.elapsed().as_secs_f64() * 1e6 - ev.dur_us as f64;
            json.push_str(&format!(
                "{{\"name\":{},\"cat\":\"script\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\
                 \"ts\":{:.3},\"dur\":{:.3}}},",
                json_string(&ev.name),
                start_us.max(0.0),
                ev.dur_us as f64
            ));
        }
        if json.ends_with(',') {
            json.pop();
        }
        json.push_str("]}");
        json
    }

    fn record_event(&mut self, name: String, started: std::time::Instant) {
        if self.trace_start.is_some() {
            let dur_us = started.elapsed().as_micros() as u64;
            let start_us = self
                .trace_start
                .map(|t| t.elapsed().as_micros() as u64)
                .unwrap_or(0)
                .saturating_sub(dur_us);
            self.trace_events.push(TraceEvent {
                name,
                start_us,
                dur_us,
            });
        }
    }

    fn install_natives(&mut self) {
        self.register_native("len", |args| match args.first() {
            Some(Value::Str(s)) => Ok(Value::Num(s.len() as f64)),
            Some(Value::List(l)) => Ok(Value::Num(l.lock().unwrap().len() as f64)),
            Some(Value::Dict(d)) => Ok(Value::Num(d.lock().unwrap().len() as f64)),
            Some(Value::Tensor(t)) => Ok(Value::Num(t.numel() as f64)),
            Some(v) => Err(format!("len() not supported for {}", v.type_name())),
            None => Err("len() requires one argument".into()),
        });
        self.register_native("abs", |args| match args.first() {
            Some(v @ Value::Num(_)) => Ok(v.clone()),
            _ => Err("abs() requires a number".into()),
        });
        self.register_native("matmul", |args| {
            let (Some(Value::Tensor(a)), Some(Value::Tensor(b))) = (args.first(), args.get(1))
            else {
                return Err("matmul(a, b) requires two tensors".into());
            };
            Ok(Value::Tensor(a.matmul(b)))
        });
        self.register_native("sum", |args| match args.first() {
            Some(Value::Tensor(t)) => Ok(Value::Num(t.to_vec::<f64>().unwrap().iter().sum())),
            _ => Err("sum() requires a tensor".into()),
        });
        self.register_native("ones", |args| filled_from_shape(args, 1.0));
        self.register_native("zeros", |args| filled_from_shape(args, 0.0));
        self.register_native("upper", |args| match args.first() {
            Some(Value::Str(s)) => Ok(Value::Str(s.to_uppercase())),
            _ => Err("upper() requires a string".into()),
        });
        self.register_native("lower", |args| match args.first() {
            Some(Value::Str(s)) => Ok(Value::Str(s.to_lowercase())),
            _ => Err("lower() requires a string".into()),
        });
        self.register_native("trim", |args| match args.first() {
            Some(Value::Str(s)) => Ok(Value::Str(s.trim().to_string())),
            _ => Err("trim() requires a string".into()),
        });
        self.register_native("contains", |args| match (args.first(), args.get(1)) {
            (Some(Value::Str(h)), Some(Value::Str(n))) => Ok(Value::Bool(h.contains(n.as_str()))),
            _ => Err("contains(haystack, needle) requires two strings".into()),
        });
        self.register_native("split", |args| match (args.first(), args.get(1)) {
            (Some(Value::Str(s)), Some(Value::Str(sep))) => Ok(Value::List(Arc::new(Mutex::new(
                s.split(sep.as_str())
                    .map(|p| Value::Str(p.to_string()))
                    .collect(),
            )))),
            _ => Err("split(s, sep) requires two strings".into()),
        });
        self.register_native("str", |args| {
            args.first()
                .map(|v| Value::Str(format!("{v}")))
                .ok_or_else(|| "str() requires one argument".into())
        });
    }

    /// Lex, parse and execute `src`; returns the value of the last expression
    /// statement or `return`, if any.
    pub fn run(&mut self, src: &str) -> Res<Option<Value>> {
        self.out.clear();
        self.last_value = None;
        let toks = lex(src)?;
        let mut parser = Parser::new(&toks);
        let prog = parser.parse_block()?;
        let flow = {
            let globals = Arc::clone(&self.globals);
            self.exec_block(&prog, &globals, &mut None)?
        };
        let last = match flow {
            Flow::Return(v) => Some(v),
            Flow::Normal => self.last_value.clone(),
        };
        Ok(last)
    }

    // ---------------------------- debugger API ------------------------------

    /// Pause before *every* statement (single-stepping mode).
    pub fn set_single_step(&mut self, on: bool) {
        self.debug.single_step = on;
    }

    /// Break whenever a statement label contains `substr` (e.g. `"train_step"`
    /// hits `call train_step`). Multiple breakpoints are OR-ed.
    pub fn add_breakpoint(&mut self, substr: impl Into<String>) {
        self.debug.breakpoints.push(substr.into());
    }

    /// Remove all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.debug.breakpoints.clear();
    }

    /// Watch `expr` (a script expression) at every pause, evaluated in the
    /// paused scope — this is the tensor-inspection surface.
    pub fn add_watch(&mut self, expr: impl Into<String>) {
        self.debug.watches.push(expr.into());
    }

    /// Remove all watch expressions.
    pub fn clear_watches(&mut self) {
        self.debug.watches.clear();
    }

    /// Run `src` under the debugger: whenever a breakpoint hits (or single-
    /// step mode is on), the interpreter pauses *before* the statement,
    /// evaluates the watches in the current scope, and hands a
    /// [`DebugFrame`] to `on_pause`. The returned [`DebugAction`] decides
    /// whether to resume, step to the next statement, or abort.
    pub fn run_debug(
        &mut self,
        src: &str,
        on_pause: &mut dyn FnMut(&DebugFrame) -> DebugAction,
    ) -> Res<Option<Value>> {
        self.out.clear();
        self.last_value = None;
        self.stmt_counter = 0;
        self.call_depth = 0;
        self.debug.pending_step = false;
        let toks = lex(src)?;
        let mut parser = Parser::new(&toks);
        let prog = parser.parse_block()?;
        let flow = {
            let globals = Arc::clone(&self.globals);
            self.exec_block(&prog, &globals, &mut Some(on_pause))?
        };
        let last = match flow {
            Flow::Return(v) => Some(v),
            Flow::Normal => self.last_value.clone(),
        };
        Ok(last)
    }

    /// Whether this statement should pause, and the frame to hand out.
    fn debug_pause(&mut self, label: &str, env: &Arc<Environment>, dbg: DbgCb) -> Res<()> {
        self.stmt_counter += 1;
        let hit = self.debug.single_step
            || self.debug.pending_step
            || self
                .debug
                .breakpoints
                .iter()
                .any(|b| label.contains(b.as_str()));
        if !hit {
            return Ok(());
        }
        self.debug.pending_step = false;
        let mut watches = Vec::with_capacity(self.debug.watches.len());
        for wexpr in self.debug.watches.clone() {
            match self.eval_snapshot(&wexpr, env) {
                Ok(v) => watches.push(WatchValue {
                    expr: wexpr,
                    value: Some(v),
                    error: None,
                }),
                Err(e) => watches.push(WatchValue {
                    expr: wexpr,
                    value: None,
                    error: Some(e),
                }),
            }
        }
        let locals: Vec<(String, Value)> = env
            .names()
            .into_iter()
            .take(200)
            .filter_map(|n| env.get(&n).map(|v| (n, v)))
            .collect();
        let frame = DebugFrame {
            label: label.to_string(),
            statement: self.stmt_counter,
            depth: self.call_depth,
            locals,
            watches,
            output: self.out.clone(),
        };
        match dbg(&frame) {
            DebugAction::Continue => {}
            DebugAction::Step => self.debug.pending_step = true,
            DebugAction::Abort => {
                return Err(InterpreterError::new(
                    "KeyboardInterrupt",
                    "aborted by debugger",
                ));
            }
        }
        Ok(())
    }

    /// Evaluate a single expression in `env` without disturbing run state
    /// (watch evaluation). Errors are rendered as strings.
    fn eval_snapshot(&mut self, src: &str, env: &Arc<Environment>) -> Result<Value, String> {
        let toks = lex(src).map_err(|e| format!("{}: {}", e.kind, e.message))?;
        let mut parser = Parser::new(&toks);
        let e = parser
            .parse_expr()
            .map_err(|e| format!("{}: {}", e.kind, e.message))?;
        // trailing Newline after a full expression is expected
        self.eval(&e, env, &mut None)
            .map_err(|e| format!("{}: {}", e.kind, e.message))
    }

    fn exec_block(
        &mut self,
        stmts: &[Stmt],
        env: &Arc<Environment>,
        dbg: &mut Option<DbgCb>,
    ) -> Res<Flow> {
        for s in stmts {
            match self.exec_stmt(s, env, dbg)? {
                Flow::Normal => {}
                r @ Flow::Return(_) => return Ok(r),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(
        &mut self,
        s: &Stmt,
        env: &Arc<Environment>,
        dbg: &mut Option<DbgCb>,
    ) -> Res<Flow> {
        let started = std::time::Instant::now();
        let label = stmt_label(s);
        if let Some(cb) = dbg.as_mut() {
            self.debug_pause(&label, env, cb)?;
        }
        let flow = self.exec_stmt_inner(s, env, dbg)?;
        self.record_event(label, started);
        Ok(flow)
    }

    fn exec_stmt_inner(
        &mut self,
        s: &Stmt,
        env: &Arc<Environment>,
        dbg: &mut Option<DbgCb>,
    ) -> Res<Flow> {
        match s {
            Stmt::Let(name, e) => {
                let v = self.eval(e, env, dbg)?;
                env.define(name.clone(), v);
                Ok(Flow::Normal)
            }
            Stmt::Assign(name, e) => {
                let v = self.eval(e, env, dbg)?;
                env.assign(name, v).map_err(|_| {
                    InterpreterError::new("NameError", format!("'{name}' is not defined"))
                })?;
                Ok(Flow::Normal)
            }
            Stmt::Print(e) => {
                let v = self.eval(e, env, dbg)?;
                match &v {
                    Value::Str(s) if s.contains('{') => {
                        let rendered = self.interpolate(s, env, dbg)?;
                        self.out.push_str(&format!("{rendered}\n"));
                    }
                    _ => self.out.push_str(&format!("{v}\n")),
                }
                Ok(Flow::Normal)
            }
            Stmt::Assert(e) => {
                let v = self.eval(e, env, dbg)?;
                if !v.truthy() {
                    return Err(InterpreterError::new("AssertionError", "assert failed"));
                }
                Ok(Flow::Normal)
            }
            Stmt::If(cond, then_body, else_body) => {
                let c = self.eval(cond, env, dbg)?;
                if c.truthy() {
                    self.exec_block(then_body, env, dbg)
                } else {
                    self.exec_block(else_body, env, dbg)
                }
            }
            Stmt::While(cond, body) => loop {
                let c = self.eval(cond, env, dbg)?;
                if !c.truthy() {
                    return Ok(Flow::Normal);
                }
                match self.exec_block(body, env, dbg)? {
                    Flow::Normal => {}
                    r @ Flow::Return(_) => return Ok(r),
                }
            },
            Stmt::Def(name, decls, body) => {
                // defaults evaluate once, at def time (Python semantics)
                let mut params = Vec::with_capacity(decls.len());
                for d in decls {
                    let default = match &d.default {
                        Some(e) => Some(self.eval(e, env, dbg)?),
                        None => None,
                    };
                    params.push(crate::value::Param {
                        name: d.name.clone(),
                        default,
                    });
                }
                let id = self.next_fn_id;
                self.next_fn_id += 1;
                self.script_bodies
                    .insert(id, (params.clone(), body.clone()));
                env.define(
                    name.clone(),
                    Value::Function(Arc::new(Function::Script {
                        name: name.clone(),
                        params,
                        id,
                        env: Arc::clone(env),
                    })),
                );
                Ok(Flow::Normal)
            }
            Stmt::For(name, iterable, body) => {
                let it = self.eval(iterable, env, dbg)?;
                let items: Vec<Value> = match it {
                    Value::List(l) => l.lock().unwrap().clone(),
                    Value::Tensor(t) => t
                        .to_vec::<f64>()
                        .unwrap()
                        .into_iter()
                        .map(Value::Num)
                        .collect(),
                    Value::Str(s) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
                    Value::Dict(d) => {
                        let mut keys: Vec<String> = d.lock().unwrap().keys().cloned().collect();
                        keys.sort(); // deterministic (HashMap order is not)
                        keys.into_iter().map(Value::Str).collect()
                    }
                    other => {
                        return Err(InterpreterError::new(
                            "TypeError",
                            format!("{} is not iterable in for", other.type_name()),
                        ));
                    }
                };
                for item in items {
                    env.define(name.clone(), item);
                    match self.exec_block(body, env, dbg)? {
                        Flow::Normal => {}
                        r @ Flow::Return(_) => return Ok(r),
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Module(name, body) => {
                let child = Arc::new(Environment::child(env));
                self.exec_block(body, &child, dbg)?;
                let module = crate::value::Module::with_members(name.clone(), child.own_bindings());
                env.define(name.clone(), Value::Module(Arc::new(module)));
                Ok(Flow::Normal)
            }
            Stmt::MemberAssign(base, name, e) => {
                let b = self.eval(base, env, dbg)?;
                let v = self.eval(e, env, dbg)?;
                match b {
                    Value::Module(m) => {
                        m.set(name.clone(), v);
                        Ok(Flow::Normal)
                    }
                    Value::Dict(d) => {
                        d.lock().unwrap().insert(name.clone(), v);
                        Ok(Flow::Normal)
                    }
                    other => Err(InterpreterError::new(
                        "TypeError",
                        format!("cannot assign member '{name}' on {}", other.type_name()),
                    )),
                }
            }
            Stmt::Return(e) => {
                let v = e.as_ref().map(|e| self.eval(e, env, dbg)).transpose()?;
                Ok(Flow::Return(v.unwrap_or(Value::Nil)))
            }
            Stmt::Expr(e) => {
                let v = self.eval(e, env, dbg)?;
                if !matches!(v, Value::Nil) {
                    self.last_value = Some(v);
                }
                Ok(Flow::Normal)
            }
        }
    }

    fn eval(&mut self, e: &Expr, env: &Arc<Environment>, dbg: &mut Option<DbgCb>) -> Res<Value> {
        match e {
            Expr::UnitNum(x, dim) => Ok(Value::Unit(Arc::new(crate::value::UnitValue {
                value: *x,
                dim: dim.clone(),
            }))),
            Expr::Nil => Ok(Value::Nil),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Num(x) => Ok(Value::Num(*x)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Ident(name) => env.get(name).ok_or_else(|| {
                InterpreterError::new("NameError", format!("'{name}' is not defined"))
            }),
            Expr::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(self.eval(it, env, dbg)?);
                }
                Ok(Value::List(Arc::new(Mutex::new(out))))
            }
            Expr::Dict(items) => {
                let mut map = HashMap::new();
                for (k, ve) in items {
                    map.insert(k.clone(), self.eval(ve, env, dbg)?);
                }
                Ok(Value::Dict(Arc::new(Mutex::new(map))))
            }
            Expr::Unary(op, inner) => {
                let v = self.eval(inner, env, dbg)?;
                match op.as_str() {
                    "-" => match v {
                        Value::Num(x) => Ok(Value::Num(-x)),
                        Value::Tensor(t) => Ok(Value::Tensor(t.neg())),
                        other => Err(InterpreterError::new(
                            "TypeError",
                            format!("cannot negate {}", other.type_name()),
                        )),
                    },
                    "not" => Ok(Value::Bool(!v.truthy())),
                    other => Err(InterpreterError::new(
                        "SyntaxError",
                        format!("unknown unary operator '{other}'"),
                    )),
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let l = self.eval(lhs, env, dbg)?;
                let r = self.eval(rhs, env, dbg)?;
                if op == ".." {
                    let (Some(a), Some(b)) = (l.as_num(), r.as_num()) else {
                        return Err(InterpreterError::new(
                            "TypeError",
                            "range '..' requires number endpoints",
                        ));
                    };
                    let (a, b) = (a as i64, b as i64);
                    return Ok(Value::List(Arc::new(Mutex::new(
                        (a..b).map(|x| Value::Num(x as f64)).collect(),
                    ))));
                }
                apply_binary(op, &l, &r)
            }
            Expr::Call(fexpr, args) => {
                let f = self.eval(fexpr, env, dbg)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    let v = self.eval(&a.expr, env, dbg)?;
                    vals.push((a.name.clone(), v));
                }
                call_function(self, &f, &vals, dbg)
            }
            Expr::Index(base, idx_args) => {
                let b = self.eval(base, env, dbg)?;
                let mut idx = Vec::with_capacity(idx_args.len());
                for arg in idx_args {
                    match arg {
                        IdxArg::Expr(e) => idx.push(IndexArg::Value(self.eval(e, env, dbg)?)),
                        IdxArg::Range(start, end) => {
                            let s = match start {
                                Some(e) => {
                                    Some(self.eval(e, env, dbg)?.as_num().ok_or_else(|| {
                                        InterpreterError::new(
                                            "TypeError",
                                            "slice bounds must be numbers",
                                        )
                                    })? as i64)
                                }
                                None => None,
                            };
                            let e2 = match end {
                                Some(e) => {
                                    Some(self.eval(e, env, dbg)?.as_num().ok_or_else(|| {
                                        InterpreterError::new(
                                            "TypeError",
                                            "slice bounds must be numbers",
                                        )
                                    })? as i64)
                                }
                                None => None,
                            };
                            idx.push(IndexArg::Slice(s, e2));
                        }
                    }
                }
                index_value(&b, &idx)
            }
            Expr::Member(base, name) => {
                let b = self.eval(base, env, dbg)?;
                match b {
                    Value::Module(m) => m.get(name).ok_or_else(|| {
                        InterpreterError::new(
                            "AttributeError",
                            format!("module '{}' has no member '{name}'", m.name),
                        )
                    }),
                    // dict sugar: d.key == d["key"]
                    Value::Dict(d) => d.lock().unwrap().get(name).cloned().ok_or_else(|| {
                        InterpreterError::new("KeyError", format!("key '{name}' not found"))
                    }),
                    other => Err(InterpreterError::new(
                        "TypeError",
                        format!("cannot access member '{name}' on {}", other.type_name()),
                    )),
                }
            }
        }
    }
}

fn apply_binary(op: &str, l: &Value, r: &Value) -> Res<Value> {
    // unit-aware arithmetic: Unit operands carry dimensions that must agree
    if matches!(l, Value::Unit(_)) || matches!(r, Value::Unit(_)) {
        let Some((a, da)) = unit_parts(l) else {
            return Err(type_err(op, l));
        };
        let (b, db) = unit_parts(r).ok_or_else(|| type_err(op, r))?;
        let (value, dim) = crate::interp::unit_binop(op, a, &da, b, &db)
            .map_err(|m| InterpreterError::new("UnitError", m))?;
        return Ok(Value::Unit(Arc::new(crate::value::UnitValue {
            value,
            dim,
        })));
    }
    match op {
        "+" => value_add(l, r).map_err(InterpreterError::from_lang),
        "-" => value_sub(l, r).map_err(InterpreterError::from_lang),
        "*" => value_mul(l, r).map_err(InterpreterError::from_lang),
        "/" => value_div(l, r).map_err(InterpreterError::from_lang),
        "%" => match (l.as_num(), r.as_num()) {
            (Some(a), Some(b)) => {
                if b == 0.0 {
                    Err(InterpreterError::from_lang(LangError::DivByZero))
                } else {
                    Ok(Value::Num(a % b))
                }
            }
            _ => Err(type_err(op, l)),
        },
        "==" => Ok(value_eq(l, r)),
        "!=" => Ok(Value::Bool(!matches!(value_eq(l, r), Value::Bool(true)))),
        "<" | "<=" | ">" | ">=" => match (l.as_num(), r.as_num()) {
            (Some(a), Some(b)) => Ok(Value::Bool(match op {
                "<" => a < b,
                "<=" => a <= b,
                ">" => a > b,
                _ => a >= b,
            })),
            _ => Err(type_err(op, l)),
        },
        other => Err(InterpreterError::new(
            "SyntaxError",
            format!("unknown operator '{other}'"),
        )),
    }
}

fn type_err(op: &str, l: &Value) -> InterpreterError {
    InterpreterError::new(
        "TypeError",
        format!("unsupported operand for '{op}': {}", l.type_name()),
    )
}

/// `(value, dimension)` view of a numeric-or-unit value.
fn unit_parts(v: &Value) -> Option<(f64, String)> {
    match v {
        Value::Num(n) => Some((*n, String::new())),
        Value::Bool(b) => Some((if *b { 1.0 } else { 0.0 }, String::new())),
        Value::Unit(u) => Some((u.value, u.dim.clone())),
        _ => None,
    }
}

/// `a.b.c` path rendering for profiler labels / diagnostics.
fn expr_path(e: &Expr) -> String {
    match e {
        Expr::Ident(n) => n.clone(),
        Expr::Member(b, n) => format!("{}.{}", expr_path(b), n),
        _ => "<expr>".to_string(),
    }
}

/// Statement label for profiler events.
fn stmt_label(s: &Stmt) -> String {
    match s {
        Stmt::Let(n, _) => format!("let {n}"),
        Stmt::Assign(n, _) => format!("assign {n}"),
        Stmt::MemberAssign(b, n, _) => format!("assign {}.{}", expr_path(b), n),
        Stmt::Print(_) => "print".into(),
        Stmt::Assert(_) => "assert".into(),
        Stmt::If(..) => "if/else".into(),
        Stmt::While(..) => "while".into(),
        Stmt::Def(n, _, _) => format!("def {n}"),
        Stmt::Module(n, _) => format!("module {n}"),
        Stmt::For(n, _, _) => format!("for {n}"),
        Stmt::Return(_) => "return".into(),
        Stmt::Expr(e) => match e {
            Expr::Call(f, _) => match f.as_ref() {
                Expr::Ident(name) => format!("call {name}"),
                Expr::Member(..) => format!("call {}", expr_path(f)),
                _ => "call".into(),
            },
            _ => "expr".into(),
        },
    }
}

/// JSON string literal with quotes and escaping.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// An evaluated bracket index: a value (number for tensors/lists, string
/// for dicts) or an `[a:b]` slice with optional bounds.
pub enum IndexArg {
    Value(Value),
    Slice(Option<i64>, Option<i64>),
}

fn norm(i: f64, len: usize) -> Res<usize> {
    let i = i as i64;
    let i = if i < 0 {
        (len as i64 + i) as usize
    } else {
        i as usize
    };
    if i >= len {
        return Err(InterpreterError::new(
            "IndexError",
            format!("index {i} out of range (len {len})"),
        ));
    }
    Ok(i)
}

/// Resolve an optional slice bound against `len` (None -> 0 or len).
fn slice_bound(v: Option<i64>, len: usize, is_start: bool) -> Res<usize> {
    match v {
        None => Ok(if is_start { 0 } else { len }),
        Some(x) => {
            let x = if x < 0 { (len as i64 + x).max(0) } else { x };
            let x = x.min(len as i64) as usize;
            Ok(x)
        }
    }
}

fn index_value(base: &Value, idx: &[IndexArg]) -> Res<Value> {
    match (base, idx) {
        (Value::List(l), [IndexArg::Value(Value::Num(i))]) => {
            let l = l.lock().unwrap();
            let i = norm(*i, l.len())?;
            Ok(l[i].clone())
        }
        (Value::List(l), [IndexArg::Slice(a, b)]) => {
            let l = l.lock().unwrap();
            let s = slice_bound(*a, l.len(), true)?;
            let e = slice_bound(*b, l.len(), false)?;
            if s > e {
                return Err(InterpreterError::new("IndexError", "slice start > end"));
            }
            Ok(Value::List(Arc::new(Mutex::new(l[s..e].to_vec()))))
        }
        (Value::Dict(d), [IndexArg::Value(Value::Str(k))]) => d
            .lock()
            .unwrap()
            .get(k)
            .cloned()
            .ok_or_else(|| InterpreterError::new("KeyError", format!("key '{k}' not found"))),
        (Value::Tensor(t), [IndexArg::Value(Value::Num(i))]) => {
            let v = t.to_vec::<f64>().unwrap();
            let i = norm(*i, v.len())?;
            Ok(Value::Num(v[i]))
        }
        // multi-index: one number per dimension, row-major offset
        (Value::Tensor(t), args)
            if !args.is_empty()
                && args.len() == t.ndim()
                && args
                    .iter()
                    .all(|a| matches!(a, IndexArg::Value(Value::Num(_)))) =>
        {
            let shape = t.shape();
            let nums: Vec<f64> = args
                .iter()
                .map(|a| match a {
                    IndexArg::Value(Value::Num(n)) => *n,
                    _ => unreachable!(),
                })
                .collect();
            let mut offset = 0usize;
            for (d, &n) in nums.iter().enumerate() {
                let i = norm(n, shape[d])?;
                offset = offset * shape[d] + i;
            }
            let v = t.to_vec::<f64>().unwrap();
            Ok(Value::Num(v[offset]))
        }
        // single slice over the first axis (flat for 1-D tensors)
        (Value::Tensor(t), [IndexArg::Slice(a, b)]) => {
            let shape = t.shape();
            let s = slice_bound(*a, shape[0], true)?;
            let e = slice_bound(*b, shape[0], false)?;
            if s > e {
                return Err(InterpreterError::new("IndexError", "slice start > end"));
            }
            let data = t.to_vec::<f64>().unwrap();
            let row_len: usize = shape[1..].iter().product();
            let mut out = Vec::new();
            for row in s..e {
                out.extend_from_slice(&data[row * row_len..(row + 1) * row_len]);
            }
            let mut new_shape = shape.to_vec();
            new_shape[0] = e - s;
            Ok(Value::Tensor(
                Tensor::from_typed(out).reshape(&new_shape).unwrap(),
            ))
        }
        (Value::Str(s), [IndexArg::Value(Value::Num(i))]) => {
            let chars: Vec<char> = s.chars().collect();
            let i = norm(*i, chars.len())?;
            Ok(Value::Str(chars[i].to_string()))
        }
        (b, _) => Err(InterpreterError::new(
            "TypeError",
            format!("cannot index {} with these arguments", b.type_name()),
        )),
    }
}

/// Render `{expr}` interpolations in a print string against `env`.
/// `{{` / `}}` are literal braces.
impl Interpreter {
    fn interpolate(
        &mut self,
        s: &str,
        env: &Arc<Environment>,
        dbg: &mut Option<DbgCb>,
    ) -> Res<String> {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            match c {
                '{' if matches!(chars.peek(), Some((_, '{'))) => {
                    out.push('{');
                    chars.next();
                }
                '}' if matches!(chars.peek(), Some((_, '}'))) => {
                    out.push('}');
                    chars.next();
                }
                '{' => {
                    let close = s[i..].find('}').ok_or_else(|| {
                        InterpreterError::new("SyntaxError", "unterminated '{' in print string")
                    })?;
                    let expr_src: String = s[i + 1..i + close].trim().to_string();
                    let toks = lex(&expr_src)?;
                    let mut parser = Parser::new(&toks);
                    let e = parser.parse_expr()?;
                    let v = self.eval(&e, env, dbg)?;
                    out.push_str(&v.to_string());
                    for _ in 0..close {
                        chars.next();
                    }
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }
}

/// One evaluated call argument: positional or keyword.
pub type CallVal = (Option<String>, Value);

fn call_function(
    interp: &mut Interpreter,
    f: &Value,
    args: &[CallVal],
    dbg: &mut Option<DbgCb>,
) -> Res<Value> {
    let Value::Function(fun) = f else {
        return Err(InterpreterError::new(
            "TypeError",
            format!("{} is not callable", f.type_name()),
        ));
    };
    match fun.as_ref() {
        Function::Native { f, .. } => {
            if args.iter().any(|(name, _)| name.is_some()) {
                return Err(InterpreterError::new(
                    "TypeError",
                    "native functions take positional arguments only",
                ));
            }
            let positional: Vec<Value> = args.iter().map(|(_, v)| v.clone()).collect();
            f(&positional).map_err(|m| InterpreterError::new("RuntimeError", m))
        }
        Function::Script { name, id, env, .. } => {
            let (params_def, body) = interp.script_bodies.get(id).cloned().ok_or_else(|| {
                InterpreterError::new("NameError", format!("body of '{name}' missing"))
            })?;
            // bind: positionals fill slots left-to-right, keywords target
            // parameters by name, defaults cover whatever is still missing
            let mut slots: Vec<Option<Value>> = vec![None; params_def.len()];
            let mut positional = 0usize;
            for (cname, v) in args {
                match cname {
                    None => {
                        if positional >= params_def.len() {
                            return Err(InterpreterError::new(
                                "TypeError",
                                format!(
                                    "{name}() takes at most {} arguments but more were given",
                                    params_def.len()
                                ),
                            ));
                        }
                        slots[positional] = Some(v.clone());
                        positional += 1;
                    }
                    Some(cname) => {
                        let idx = params_def
                            .iter()
                            .position(|p| &p.name == cname)
                            .ok_or_else(|| {
                                InterpreterError::new(
                                    "TypeError",
                                    format!("{name}() has no parameter '{cname}'"),
                                )
                            })?;
                        if slots[idx].is_some() {
                            return Err(InterpreterError::new(
                                "TypeError",
                                format!("{name}() got multiple values for '{cname}'"),
                            ));
                        }
                        slots[idx] = Some(v.clone());
                    }
                }
            }
            let missing: Vec<String> = params_def
                .iter()
                .zip(&slots)
                .filter(|(p, s)| s.is_none() && p.default.is_none())
                .map(|(p, _)| p.name.clone())
                .collect();
            if !missing.is_empty() {
                return Err(InterpreterError::new(
                    "TypeError",
                    format!(
                        "{name}() missing {} required argument(s): {}",
                        missing.len(),
                        missing.join(", ")
                    ),
                ));
            }
            // calls evaluate in the *defining* scope (closures: a function
            // defined inside a `module` block sees that module's members)
            let parent = Arc::clone(env);
            let local = Arc::new(Environment::child(&parent));
            interp.call_depth += 1;
            for (p, slot) in params_def.iter().zip(&slots) {
                match slot {
                    Some(v) => local.define(p.name.clone(), v.clone()),
                    None => {
                        if let Some(dv) = &p.default {
                            local.define(p.name.clone(), dv.clone());
                        }
                    }
                }
            }
            let result = interp.exec_block(&body, &local, dbg);
            interp.call_depth -= 1;
            match result? {
                Flow::Return(v) => Ok(v),
                Flow::Normal => Ok(Value::Nil),
            }
        }
    }
}

/// `ones(2, 2)` or `ones([2, 2])` both yield the shape `[2, 2]`.
fn filled_from_shape(args: &[Value], fill: f64) -> Result<Value, String> {
    let shape: Vec<usize> = match args.first() {
        Some(Value::List(l)) => l
            .lock()
            .unwrap()
            .iter()
            .map(|v| {
                v.as_num()
                    .map(|x| x as usize)
                    .ok_or_else(|| "bad shape".to_string())
            })
            .collect::<Result<_, _>>()?,
        _ => args
            .iter()
            .map(|v| {
                v.as_num()
                    .map(|x| x as usize)
                    .ok_or_else(|| "bad shape".to_string())
            })
            .collect::<Result<_, _>>()?,
    };
    let n: usize = shape.iter().product();
    Ok(Value::Tensor(
        Tensor::from_typed(vec![fill; n]).reshape(&shape).unwrap(),
    ))
}

/// Runtime unit-aware arithmetic helper shared with the static checker's
/// dimension algebra ([`crate::check::Dim`]).
pub fn unit_binop(
    op: &str,
    a: f64,
    dim_a: &str,
    b: f64,
    dim_b: &str,
) -> Result<(f64, String), String> {
    use crate::check::Dim;
    let (da, db) = (Dim::parse(dim_a), Dim::parse(dim_b));
    match op {
        "+" | "-" => {
            if da != db {
                return Err(format!("cannot add/subtract '{}' and '{}'", da, db));
            }
            Ok((
                match op {
                    "+" => a + b,
                    _ => a - b,
                },
                dim_a.to_string(),
            ))
        }
        "*" => Ok((a * b, Dim::compose(&da, &db, 1).to_string())),
        "/" => {
            if b == 0.0 {
                return Err("division by zero".into());
            }
            Ok((a / b, Dim::compose(&da, &db, -1).to_string()))
        }
        other => Err(format!("unsupported unit op '{other}'")),
    }
}

// --------------------------------- REPL -----------------------------------

/// Outcome of feeding one line to the REPL.
#[derive(Debug)]
pub enum ReplOutcome {
    /// The line was executed; contains the printed output plus the value of
    /// the last expression (if any).
    Done {
        output: String,
        value: Option<Value>,
    },
    /// The line opened an unclosed bracket; keep reading with a continuation.
    NeedMore,
    /// A magic command ran; the string is its output.
    Magic(String),
    /// `:quit` / `:exit` was issued.
    Quit,
    /// The line executed but produced an error.
    Error(InterpreterError),
}

fn bracket_balance(line: &str) -> i64 {
    let mut bal = 0i64;
    let mut in_str = false;
    let mut escaped = false;
    for c in line.chars() {
        match c {
            '"' if !escaped => in_str = !in_str,
            _ if in_str => {}
            '[' | '(' | '{' => bal += 1,
            ']' | ')' | '}' => bal -= 1,
            _ => {}
        }
        escaped = c == '\\' && !escaped;
        if c != '\\' {
            escaped = false;
        }
    }
    if in_str {
        return 1; // unterminated string also needs continuation
    }
    bal
}

/// Stateful read-eval-print loop over [`Interpreter`].
///
/// Feed lines via [`Self::feed`]; multi-line entries accumulate until all
/// brackets balance. Magic commands start with `:`.
pub struct Repl {
    interp: Interpreter,
    pending: String,
    /// Accumulated source of every successfully fed entry — the static
    /// `:type`/`:shape` magics analyze this so session variables are known.
    session: String,
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

impl Repl {
    pub fn new() -> Self {
        Repl {
            interp: Interpreter::new(),
            pending: String::new(),
            session: String::new(),
        }
    }

    /// Access the underlying interpreter (seed globals before feeding lines).
    pub fn interpreter(&mut self) -> &mut Interpreter {
        &mut self.interp
    }

    /// Whether a multi-line entry is still open.
    pub fn needs_continuation(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Prompt for the next input line.
    pub fn prompt(&self) -> &'static str {
        if self.needs_continuation() {
            "... "
        } else {
            "tpt> "
        }
    }

    /// Feed one input line.
    pub fn feed(&mut self, line: &str) -> ReplOutcome {
        let trimmed = line.trim();
        if self.pending.is_empty() && trimmed.starts_with(':') {
            return self.magic(trimmed);
        }
        if !self.pending.is_empty() {
            self.pending.push('\n');
        }
        self.pending.push_str(line);
        if bracket_balance(&self.pending) > 0 {
            return ReplOutcome::NeedMore;
        }
        let src = std::mem::take(&mut self.pending);
        match self.interp.run(&src) {
            Ok(value) => {
                if !self.session.is_empty() {
                    self.session.push('\n');
                }
                self.session.push_str(&src);
                let mut output = self.interp.output().to_string();
                if let Some(v) = &value {
                    output.push_str(&format!("= {v}\n"));
                }
                ReplOutcome::Done { output, value }
            }
            Err(e) => ReplOutcome::Error(e),
        }
    }

    fn magic(&mut self, cmd: &str) -> ReplOutcome {
        let out = match cmd {
            ":help" => ":quit | :exit   leave the REPL\n\
                        :env             list defined variables\n\
                        :clear           forget all variables\n\
                        :output          show captured print output\n\
                        :help            this message"
                .to_string(),
            ":quit" | ":exit" => return ReplOutcome::Quit,
            ":env" => {
                let names = self.interp.names();
                if names.is_empty() {
                    "(no variables defined)".to_string()
                } else {
                    format!("{} variable(s): {}", names.len(), names.join(", "))
                }
            }
            ":clear" => {
                self.interp = Interpreter::new();
                "cleared".to_string()
            }
            ":output" => self.interp.output().to_string(),
            cmd if cmd.starts_with(":type ") || cmd.starts_with(":shape ") => {
                // static analysis of the expression via the checker
                let expr = cmd[cmd.find(' ').unwrap() + 1..].trim();
                let probe = format!(
                    "{}
let __probe = {expr}",
                    self.session
                );
                match tpt_lang_check(&probe) {
                    Ok(info) => {
                        let shape = info.shape.map(|s| {
                            let parts: Vec<String> = s
                                .iter()
                                .map(|d| d.map(|n| n.to_string()).unwrap_or("_".into()))
                                .collect();
                            format!("[{}]", parts.join(", "))
                        });
                        let units = info.dim.map(|d| format!("units {d}"));
                        let want_shape = cmd.starts_with(":shape");
                        match (shape, units) {
                            (Some(s), Some(u)) if want_shape => format!("shape {s} {u}"),
                            (Some(s), None) if want_shape => format!("shape {s}"),
                            (None, _) if want_shape => "(no static shape known)".into(),
                            (Some(s), Some(u)) => format!("tensor shape {s}, {u}"),
                            (Some(s), None) => format!("tensor shape {s}"),
                            (None, Some(u)) => u,
                            (None, None) => "(no static type known)".into(),
                        }
                    }
                    Err(e) => format!("{}: {}", e.kind, e.message),
                }
            }
            cmd if cmd.starts_with(":load ") => {
                let path = cmd[cmd.find(' ').unwrap() + 1..].trim();
                match std::fs::read_to_string(path) {
                    Ok(src) => match self.interp.run(&src) {
                        Ok(_) => format!("loaded {path}"),
                        Err(e) => format!("{}: {}", e.kind, e.message),
                    },
                    Err(e) => format!("cannot read {path}: {e}"),
                }
            }
            other => format!("unknown command '{other}' — try :help"),
        };
        ReplOutcome::Magic(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_str(src: &str) -> Value {
        let mut it = Interpreter::new();
        it.run(src).expect("run failed").expect("no value")
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert!(matches!(eval_str("2 + 3 * 4"), Value::Num(n) if n == 14.0));
        assert!(matches!(eval_str("(2 + 3) * 4"), Value::Num(n) if n == 20.0));
        assert!(matches!(eval_str("10 / 4"), Value::Num(n) if n == 2.5));
        assert!(matches!(eval_str("-5 + 1"), Value::Num(n) if n == -4.0));
        assert!(matches!(eval_str("7 % 3"), Value::Num(n) if n == 1.0));
    }

    #[test]
    fn variables_let_assign_while() {
        let mut it = Interpreter::new();
        it.run(
            "let total = 0
let i = 1
while i < 5 {
    total = total + i
    i = i + 1
}",
        )
        .unwrap();
        assert_eq!(it.get("total"), Some(Value::Num(10.0)));
    }

    #[test]
    fn if_else_branching() {
        let v = eval_str(
            "let x = 10
if x > 5 { \"big\" } else { \"small\" }",
        );
        assert_eq!(v, Value::Str("big".into()));
    }

    #[test]
    fn lists_dicts_indexing() {
        let v = eval_str(
            "let d = {\"alpha\": 1, \"beta\": 2}
let l = [10, 20, 30]
d[\"beta\"] + l[1]",
        );
        assert_eq!(v, Value::Num(22.0));
        let neg = eval_str("let l = [1, 2, 3]\nl[-1]");
        assert_eq!(neg, Value::Num(3.0));
    }

    #[test]
    fn user_functions_with_return() {
        let v = eval_str(
            "def square(x) {
    return x * x
}
def hypot(a, b) {
    return square(a) + square(b)
}
hypot(3, 4)",
        );
        assert_eq!(v, Value::Num(25.0));
    }

    // --------------------------- object model --------------------------------

    #[test]
    fn for_loops_iterate_ranges_lists_tensors_and_dicts() {
        // range 0..5
        assert!(matches!(
            eval_str(
                "let total = 0
for i in 0..5 {
    total = total + i
}
total"
            ),
            Value::Num(n) if n == 10.0
        ));
        // list iteration
        let v = eval_str(
            "let acc = []
let l = [10, 20, 30]
for x in l {
    acc = acc + [x * 2]
}
acc[2]",
        );
        assert_eq!(v, Value::Num(60.0));
        // tensor (flat)
        let v = eval_str(
            "let t = ones([2, 3])
let s = 0
for x in t {
    s = s + x
}
s",
        );
        assert_eq!(v, Value::Num(6.0));
        // dict keys (sorted, deterministic)
        let v = eval_str(
            "let last = \"\"
for k in {\"b\": 1, \"a\": 2} {
    last = k
}
last",
        );
        assert_eq!(v, Value::Str("b".into()));
    }

    #[test]
    fn elif_chains_pick_the_right_branch() {
        let v = eval_str(
            "let x = 15
if x < 10 {
    \"small\"
} elif x < 20 {
    \"medium\"
} else {
    \"large\"
}",
        );
        assert_eq!(v, Value::Str("medium".into()));
        let v = eval_str(
            "let x = 99
if x < 10 { 1 } elif x < 20 { 2 } else { 3 }",
        );
        assert_eq!(v, Value::Num(3.0));
    }

    #[test]
    fn keyword_arguments_bind_by_name() {
        assert!(matches!(
            eval_str("def sub(a, b) { return a - b }
        sub(b = 2, a = 9)"),
            Value::Num(n) if n == 7.0
        ));
        // mixed positional + keyword + defaults
        assert!(matches!(
            eval_str("def f(x, y = 10, z = 100) { return x + y + z }
        f(1, z = 5)"),
            Value::Num(n) if n == 16.0
        ));
        // unknown keyword names the parameter
        let mut it = Interpreter::new();
        let err = it
            .run(
                "def f(a) { return a }
f(nope = 1)",
            )
            .unwrap_err();
        assert_eq!(err.kind, "TypeError");
        assert!(err.message.contains("nope"));
        // duplicate binding is rejected
        let err = it
            .run(
                "def f(a) { return a }
f(1, a = 2)",
            )
            .unwrap_err();
        assert!(err.message.contains("multiple values"), "{}", err.message);
    }

    #[test]
    fn print_interpolates_brace_expressions() {
        let mut it = Interpreter::new();
        it.run(
            "let name = \"tpt\"
let rate = 0.25",
        )
        .unwrap();
        it.run("print \"hello {name}, quarter = {rate * 4}\"")
            .unwrap();
        assert_eq!(
            it.output(),
            "hello tpt, quarter = 1
"
        );
        // escaped braces stay literal
        it.run("print \"{{literal}}\"").unwrap();
        assert!(it.output().ends_with(
            "{literal}
"
        ));
    }

    #[test]
    fn chained_and_dict_member_assignment() {
        // dict members are assignable via dot syntax
        let v = eval_str(
            "let d = {\"a\": 1}
d.b = 7
d.a + d.b",
        );
        assert_eq!(v, Value::Num(8.0));
        // chained member assignment through a module of modules
        let v = eval_str(
            "module inner { let x = 1 }
module outer { let inner = inner }
outer.inner.x = 9
outer.inner.x",
        );
        assert_eq!(v, Value::Num(9.0));
    }

    #[test]
    fn tensor_multi_index_and_slices() {
        let t = eval_str(
            "let t = ones([3, 4])
t[1, 2]",
        );
        assert_eq!(t, Value::Num(1.0));
        // flat index still works
        let v = eval_str(
            "let l = [1, 2, 3, 4]
l[1:3]",
        );
        assert!(matches!(v, Value::List(ref l) if l.lock().unwrap().len() == 2));
        // tensor row slice keeps shape
        let mut it = Interpreter::new();
        let v = it
            .run(
                "let m = ones([4, 2])
let s = m[1:3]
sum(s)",
            )
            .unwrap()
            .unwrap();
        assert_eq!(v, Value::Num(4.0));
        // open-ended slice on a list
        let v = eval_str(
            "let l = [9, 8, 7]
l[1:]",
        );
        assert!(matches!(v, Value::List(ref l) if l.lock().unwrap().len() == 2));
        // string index
        assert_eq!(eval_str("\"cat\"[1]"), Value::Str("a".into()));
    }

    #[test]
    fn string_natives_cover_the_common_set() {
        assert_eq!(eval_str("upper(\"tpt\")"), Value::Str("TPT".into()));
        assert_eq!(eval_str("lower(\"TPT\")"), Value::Str("tpt".into()));
        assert_eq!(eval_str("trim(\"  x  \")"), Value::Str("x".into()));
        assert_eq!(
            eval_str("contains(\"hello world\", \"wor\")"),
            Value::Bool(true)
        );
        let parts = eval_str("split(\"a,b,c\", \",\")");
        assert!(matches!(parts, Value::List(ref l) if l.lock().unwrap().len() == 3));
    }

    #[test]
    fn repl_type_and_shape_magics_use_the_checker() {
        let mut repl = Repl::new();
        match repl.feed(":shape ones([2, 3])") {
            ReplOutcome::Magic(out) => assert!(out.contains("[2, 3]"), "{out}"),
            other => panic!("expected magic, got {other:?}"),
        }
        repl.feed("let v = 3.0 m");
        match repl.feed(":type v") {
            ReplOutcome::Magic(out) => assert!(out.contains("m"), "{out}"),
            other => panic!("expected magic, got {other:?}"),
        }
        match repl.feed(":type 1 + 2") {
            ReplOutcome::Magic(_) => {}
            other => panic!("expected magic, got {other:?}"),
        }
    }

    #[test]
    fn parameters_have_defaults() {
        assert!(
            matches!(eval_str("def f(x, y = 2) { return x * y }\nf(3)"), Value::Num(n) if n == 6.0)
        );
        assert!(
            matches!(eval_str("def f(x, y = 2) { return x * y }\nf(3, 4)"), Value::Num(n) if n == 12.0)
        );
        // signature renders defaults (REPL echo)
        let mut it = Interpreter::new();
        it.run("def f(x, y = 2) { return x * y }").unwrap();
        let f = it.get("f").unwrap();
        assert_eq!(f.to_string(), "<function f(x, y=2)>");
    }

    #[test]
    fn missing_required_argument_names_parameter() {
        let mut it = Interpreter::new();
        let err = it.run("def f(x, y, z = 3) { return x }\nf(1)").unwrap_err();
        assert_eq!(err.kind, "TypeError");
        assert!(
            err.message.contains("y") && err.message.contains("required"),
            "{}",
            err.message
        );
    }

    #[test]
    fn too_many_arguments_is_an_error() {
        let mut it = Interpreter::new();
        let err = it.run("def f(x) { return x }\nf(1, 2)").unwrap_err();
        assert_eq!(err.kind, "TypeError");
        assert!(err.message.contains("at most 1"), "{}", err.message);
    }

    #[test]
    fn defaults_evaluate_once_at_def_time() {
        // Python semantics: the default binds the value at `def`, not the name
        assert!(matches!(
            eval_str(
                "let d = 10
def g(a = d) {
    return a
}
d = 99
g()"
            ),
            Value::Num(n) if n == 10.0
        ));
    }

    #[test]
    fn default_after_required_is_a_syntax_error() {
        let mut it = Interpreter::new();
        let err = it.run("def f(x = 1, y) { return y }").unwrap_err();
        assert_eq!(err.kind, "SyntaxError");
    }

    #[test]
    fn modules_are_namespaces_with_member_access() {
        let v = eval_str(
            "module geom {
    let pi = 3.14159
    def area(r) {
        return pi * r * r
    }
}
geom.area(2.0)",
        );
        assert!(matches!(v, Value::Num(n) if (n - 12.56636).abs() < 1e-4));
        let v = eval_str(
            "module geom {
    let pi = 3.14159
}
geom.pi",
        );
        assert!(matches!(v, Value::Num(n) if (n - 3.14159).abs() < 1e-9));
    }

    #[test]
    fn module_members_can_be_assigned() {
        let v = eval_str(
            "module m {
    let x = 1
}
m.x = 42
m.x",
        );
        assert_eq!(v, Value::Num(42.0));
    }

    #[test]
    fn module_functions_do_not_collide_across_modules() {
        // both modules define `f`; id-keyed bodies keep them distinct
        let v = eval_str(
            "module a {
    def f() {
        return 1
    }
}
module b {
    def f() {
        return 2
    }
}
a.f() + b.f()",
        );
        assert_eq!(v, Value::Num(3.0));
    }

    #[test]
    fn unknown_module_member_is_an_attribute_error() {
        let mut it = Interpreter::new();
        let err = it.run("module m { let x = 1 }\nm.nope").unwrap_err();
        assert_eq!(err.kind, "AttributeError");
    }

    #[test]
    fn dict_member_access_is_sugar_for_keys() {
        assert!(matches!(
            eval_str("let d = {\"alpha\": 7}\nd.alpha"),
            Value::Num(n) if n == 7.0
        ));
    }

    #[test]
    fn nested_member_call_paths_label_profiler_events() {
        let mut it = Interpreter::new();
        it.enable_tracing();
        it.run(
            "module m {
    def f(x) {
        return x + 1
    }
}
m.f(1)",
        )
        .unwrap();
        assert!(it.trace_events().iter().any(|e| e.name == "call m.f"));
    }

    #[test]
    fn tensors_are_first_class() {
        // tensor literal via natives, tensor arithmetic with scalar broadcast,
        // matmul through the shared tpt-tensor ops
        let v = eval_str(
            "let a = ones([2, 2])
let b = a + 1.0
let m = matmul(b, b)
sum(m)",
        );
        assert_eq!(v, Value::Num(32.0));
    }

    #[test]
    fn native_registration_from_rust() {
        let mut it = Interpreter::new();
        it.register_native("double", |args| {
            match args.first().and_then(|v| v.as_num()) {
                Some(x) => Ok(Value::Num(2.0 * x)),
                None => Err("double() requires a number".into()),
            }
        });
        let v = it.run("double(21)").unwrap().unwrap();
        assert_eq!(v, Value::Num(42.0));
    }

    #[test]
    fn print_captures_output() {
        let mut it = Interpreter::new();
        it.run("print \"hello\"\nprint 42").unwrap();
        assert_eq!(it.output(), "hello\n42\n");
    }

    #[test]
    fn errors_have_python_kinds() {
        let mut it = Interpreter::new();
        let err = it.run("undefined_name + 1").unwrap_err();
        assert_eq!(err.kind, "NameError");
        let err = it.run("assert 1 == 2").unwrap_err();
        assert_eq!(err.kind, "AssertionError");
        let err = it.run("1 / 0").unwrap_err();
        assert_eq!(err.kind, "ZeroDivisionError");
    }

    // ------------------------------ REPL tests ------------------------------

    #[test]
    fn profiler_records_statement_events() {
        let mut it = Interpreter::new();
        it.enable_tracing();
        it.run(
            "let acc = 0
let i = 0
while i < 3 {
    acc = acc + i
    i = i + 1
}
print acc",
        )
        .unwrap();
        let events = it.trace_events();
        assert!(!events.is_empty(), "no events recorded");
        // statement kinds are labelled
        assert!(events.iter().any(|e| e.name == "print"));
        assert!(events.iter().any(|e| e.name == "let acc"));
        assert!(events.iter().any(|e| e.name == "while"));
        // durations are non-negative by construction; starts are ordered
        for w in events.windows(2) {
            assert!(w[0].start_us <= w[1].start_us + 1_000);
        }
    }

    #[test]
    fn chrome_trace_export_is_valid_json() {
        let mut it = Interpreter::new();
        it.enable_tracing();
        it.run(
            "let net = mlp(1, 4, 1)
let i = 0
while i < 3 {
    train_step(net, ones([2, 1]), ones([2, 1]), 0.01)
    i = i + 1
}",
        )
        .unwrap();
        let json = it.take_chrome_trace_json();
        // structural sanity without a JSON dependency
        assert!(json.starts_with("{\"displayTimeUnit\""));
        assert!(json.ends_with("]}"));
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("\"cat\":\"script\""));
        assert!(json.contains("train_step"));
        assert!(json.contains("process_name"));
        // balanced quotes: even count
        assert_eq!(json.matches('"').count() % 2, 0);
    }

    #[test]
    fn repl_state_persists_across_entries() {
        let mut repl = Repl::new();
        assert!(matches!(repl.feed("let a = 6"), ReplOutcome::Done { .. }));
        match repl.feed("a * 7") {
            ReplOutcome::Done { value: Some(v), .. } => assert_eq!(v, Value::Num(42.0)),
            other => panic!("expected done with 42, got {other:?}"),
        }
    }

    #[test]
    fn repl_multiline_accumulates_until_balanced() {
        let mut repl = Repl::new();
        assert!(matches!(
            repl.feed("def add3(a, b, c) {"),
            ReplOutcome::NeedMore
        ));
        assert!(matches!(
            repl.feed("return a + b + c"),
            ReplOutcome::NeedMore
        ));
        assert!(repl.needs_continuation());
        match repl.feed("}") {
            ReplOutcome::Done { .. } => {}
            other => panic!("expected done after closing brace, got {other:?}"),
        }
        assert!(!repl.needs_continuation());
        match repl.feed("add3(1, 2, 3)") {
            ReplOutcome::Done { value: Some(v), .. } => assert_eq!(v, Value::Num(6.0)),
            other => panic!("expected 6, got {other:?}"),
        }
    }

    #[test]
    fn repl_magic_commands() {
        let mut repl = Repl::new();
        repl.feed("let w = 1");
        match repl.feed(":env") {
            ReplOutcome::Magic(out) => assert!(out.contains('w'), "env output: {out}"),
            other => panic!("expected magic, got {other:?}"),
        }
        assert!(matches!(repl.feed(":help"), ReplOutcome::Magic(_)));
        assert!(matches!(repl.feed(":clear"), ReplOutcome::Magic(_)));
        assert!(matches!(repl.feed(":quit"), ReplOutcome::Quit));
    }

    #[test]
    fn repl_tensor_pretty_printing() {
        let mut repl = Repl::new();
        match repl.feed("ones([2, 2])") {
            ReplOutcome::Done {
                value: Some(Value::Tensor(_)),
                ..
            } => {}
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn repl_errors_do_not_poison_session() {
        let mut repl = Repl::new();
        assert!(matches!(
            repl.feed("let x = "),
            ReplOutcome::Error(InterpreterError { kind, .. }) if kind == "SyntaxError"
        ));
        assert!(!repl.needs_continuation());
        match repl.feed("x = 5") {
            ReplOutcome::Error(e) => assert_eq!(e.kind, "NameError"),
            other => panic!("expected NameError, got {other:?}"),
        }
        match repl.feed("1 + 1") {
            ReplOutcome::Done { value: Some(v), .. } => assert_eq!(v, Value::Num(2.0)),
            other => panic!("expected 2, got {other:?}"),
        }
    }

    // ------------------------------ debugger --------------------------------

    const DEBUG_PROGRAM: &str = r#"
def double(x) {
    let y = x * 2
    return y
}
let r = double(21)
print r
"#;

    #[test]
    fn breakpoint_pauses_and_frame_shows_locals() {
        let mut it = Interpreter::new();
        it.add_breakpoint("let y");
        let mut pauses = 0usize;
        let result = it.run_debug(DEBUG_PROGRAM, &mut |frame| {
            pauses += 1;
            assert_eq!(frame.label, "let y");
            assert_eq!(frame.depth, 1, "inside the function call");
            // local scope is visible: x is the argument, 21.0
            assert_eq!(frame.lookup("x"), Some(&Value::Num(21.0)));
            DebugAction::Continue
        });
        result.unwrap();
        assert_eq!(pauses, 1);
    }

    #[test]
    fn single_step_pauses_before_every_statement() {
        let mut it = Interpreter::new();
        it.set_single_step(true);
        let mut labels = Vec::new();
        it.run_debug(DEBUG_PROGRAM, &mut |frame| {
            labels.push(frame.label.clone());
            DebugAction::Continue
        })
        .unwrap();
        // def, let r (pauses before the call executes), let y, return, print
        assert_eq!(
            labels,
            vec!["def double", "let r", "let y", "return", "print"]
        );
    }

    #[test]
    fn step_action_pauses_at_the_next_statement() {
        let mut it = Interpreter::new();
        it.add_breakpoint("let y");
        let mut first_hit = true;
        let mut labels = Vec::new();
        it.run_debug(DEBUG_PROGRAM, &mut |frame| {
            labels.push(frame.label.clone());
            if first_hit {
                first_hit = false;
                DebugAction::Step // run `let y`, then pause again
            } else {
                DebugAction::Continue
            }
        })
        .unwrap();
        assert_eq!(labels, vec!["let y", "return"]);
    }

    #[test]
    fn watch_expressions_evaluate_in_the_paused_scope() {
        let mut it = Interpreter::new();
        it.add_breakpoint("return");
        it.add_watch("y * 10");
        it.add_watch("undefined_name + 1");
        let mut seen = None;
        it.run_debug(DEBUG_PROGRAM, &mut |frame| {
            seen = Some(frame.watches.clone());
            DebugAction::Continue
        })
        .unwrap();
        let watches = seen.expect("paused once");
        assert_eq!(watches.len(), 2);
        assert_eq!(watches[0].value, Some(Value::Num(420.0)));
        assert!(watches[0].error.is_none());
        // a broken watch reports the error instead of failing the run
        assert!(watches[1].value.is_none());
        assert!(watches[1].error.as_deref().unwrap().contains("NameError"));
    }

    #[test]
    fn tensors_are_inspectable_through_watches() {
        let mut it = Interpreter::new();
        it.add_breakpoint("print");
        it.add_watch("m");
        let mut watch_val = None;
        it.run_debug("let m = ones([2, 2])\nprint sum(m)", &mut |frame| {
            watch_val = frame.watches[0].value.clone();
            DebugAction::Continue
        })
        .unwrap();
        match watch_val.expect("watch evaluated") {
            Value::Tensor(t) => assert_eq!(t.shape(), &[2, 2]),
            other => panic!("expected tensor watch, got {other:?}"),
        }
    }

    #[test]
    fn abort_stops_execution_like_an_interrupt() {
        let mut it = Interpreter::new();
        it.set_single_step(true);
        let err = it
            .run_debug("let a = 1\nlet b = 2", &mut |_frame| DebugAction::Abort)
            .unwrap_err();
        assert_eq!(err.kind, "KeyboardInterrupt");
        // the aborted statement never ran
        assert!(it.get("a").is_none());
    }

    #[test]
    fn breakpoint_substring_matches_calls_and_loops() {
        let mut it = Interpreter::new();
        it.add_breakpoint("train_step");
        let src = "\
let net = mlp(1, 4, 1)
let i = 0
while i < 3 {
    train_step(net, ones([2, 1]), ones([2, 1]), 0.01)
    i = i + 1
}";
        let mut hits = 0usize;
        it.run_debug(src, &mut |frame| {
            assert_eq!(frame.label, "call train_step");
            hits += 1;
            DebugAction::Continue
        })
        .unwrap();
        assert_eq!(hits, 3, "one pause per loop iteration");
    }

    #[test]
    fn run_is_unaffected_by_debug_configuration() {
        let mut it = Interpreter::new();
        it.set_single_step(true);
        it.add_breakpoint("everything");
        let v = it.run("1 + 1").unwrap().unwrap();
        assert_eq!(v, Value::Num(2.0));
    }
}

/// Static analysis for the REPL `:type`/`:shape` magics: returns the
/// checker's knowledge for the single binding `__probe` defined by `src`.
fn tpt_lang_check(src: &str) -> Result<crate::check::BindingInfo, crate::check::CheckError> {
    use std::collections::BTreeMap;
    let bindings: BTreeMap<String, crate::check::BindingInfo> =
        crate::check::analyze_bindings(src)?;
    Ok(bindings.get("__probe").cloned().unwrap_or_default())
}
