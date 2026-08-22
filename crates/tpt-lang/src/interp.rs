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
use crate::ops::{value_add, value_div, value_eq, value_mul, value_sub, LangError};
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
                while i < cs.len() && (cs[i].is_ascii_digit() || (cs[i] == '.' && !dot)) {
                    if cs[i] == '.' {
                        dot = true;
                    }
                    i += 1;
                }
                let s: String = cs[start..i].iter().collect();
                let n: f64 = s
                    .parse()
                    .map_err(|_| InterpreterError::new("SyntaxError", format!("bad number `{s}`")))?;
                out.push(Tok::Num(n));
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
            '{' | '}' | ',' | ':' => {
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
                ))
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
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
    Ident(String),
    List(Vec<Expr>),
    Dict(Vec<(String, Expr)>),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
}

/// Statement AST.
#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Assign(String, Expr),
    Print(Expr),
    Assert(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    Def(String, Vec<String>, Vec<Stmt>),
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
            return Err(InterpreterError::new("SyntaxError", format!("expected '{op}'")));
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
                    if self.at_kw("else") {
                        self.pos += 1;
                        self.expect_op("{")?;
                        else_body = self.parse_block()?;
                        self.expect_op("}")?;
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
                "def" | "fn" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    self.expect_op("(")?;
                    let mut params = Vec::new();
                    while !self.at_op(")") {
                        params.push(self.expect_ident()?);
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

    fn parse_assign_or_expr(&mut self) -> Res<Stmt> {
        // lookahead: IDENT '=' expr (but not '==')
        if let Some(Tok::Ident(_)) = self.peek() {
            if let Some(Tok::Op(op)) = self.toks.get(self.pos + 1) {
                if op == "=" {
                    let name = self.expect_ident()?;
                    self.pos += 1; // '='
                    let e = self.parse_expr()?;
                    return Ok(Stmt::Assign(name, e));
                }
            }
        }
        Ok(Stmt::Expr(self.parse_expr()?))
    }

    pub fn parse_expr(&mut self) -> Res<Expr> {
        self.parse_comparison()
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
            if self.at_op("[") {
                self.pos += 1;
                let idx = self.parse_expr()?;
                self.expect_op("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            } else if self.at_op("(") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op(")") {
                    args.push(self.parse_expr()?);
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

    fn parse_atom(&mut self) -> Res<Expr> {
        match self.peek().cloned() {
            Some(Tok::Num(x)) => {
                self.pos += 1;
                Ok(Expr::Num(x))
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
                            ))
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

// ----------------------------- interpreter --------------------------------

enum Flow {
    Normal,
    Return(Value),
}

/// The eager TPT Script interpreter over scoped [`Environment`]s.
pub struct Interpreter {
    globals: Arc<Environment>,
    script_bodies: HashMap<String, (Vec<String>, Vec<Stmt>)>,
    out: String,
    last_value: Option<Value>,
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
            out: String::new(),
            last_value: None,
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
            self.exec_block(&prog, &globals)?
        };
        let last = match flow {
            Flow::Return(v) => Some(v),
            Flow::Normal => self.last_value.clone(),
        };
        Ok(last)
    }

    fn exec_block(&mut self, stmts: &[Stmt], env: &Arc<Environment>) -> Res<Flow> {
        for s in stmts {
            match self.exec_stmt(s, env)? {
                Flow::Normal => {}
                r @ Flow::Return(_) => return Ok(r),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(&mut self, s: &Stmt, env: &Arc<Environment>) -> Res<Flow> {
        match s {
            Stmt::Let(name, e) => {
                let v = self.eval(e, env)?;
                env.define(name.clone(), v);
                Ok(Flow::Normal)
            }
            Stmt::Assign(name, e) => {
                let v = self.eval(e, env)?;
                env.assign(name, v).map_err(|_| {
                    InterpreterError::new("NameError", format!("'{name}' is not defined"))
                })?;
                Ok(Flow::Normal)
            }
            Stmt::Print(e) => {
                let v = self.eval(e, env)?;
                self.out.push_str(&format!("{v}\n"));
                Ok(Flow::Normal)
            }
            Stmt::Assert(e) => {
                let v = self.eval(e, env)?;
                if !v.truthy() {
                    return Err(InterpreterError::new("AssertionError", "assert failed"));
                }
                Ok(Flow::Normal)
            }
            Stmt::If(cond, then_body, else_body) => {
                let c = self.eval(cond, env)?;
                if c.truthy() {
                    self.exec_block(then_body, env)
                } else {
                    self.exec_block(else_body, env)
                }
            }
            Stmt::While(cond, body) => loop {
                let c = self.eval(cond, env)?;
                if !c.truthy() {
                    return Ok(Flow::Normal);
                }
                match self.exec_block(body, env)? {
                    Flow::Normal => {}
                    r @ Flow::Return(_) => return Ok(r),
                }
            },
            Stmt::Def(name, params, body) => {
                self.script_bodies
                    .insert(name.clone(), (params.clone(), body.clone()));
                env.define(
                    name.clone(),
                    Value::Function(Arc::new(Function::Script {
                        name: name.clone(),
                        params: params.clone(),
                    })),
                );
                Ok(Flow::Normal)
            }
            Stmt::Return(e) => {
                let v = e.as_ref().map(|e| self.eval(e, env)).transpose()?;
                Ok(Flow::Return(v.unwrap_or(Value::Nil)))
            }
            Stmt::Expr(e) => {
                let v = self.eval(e, env)?;
                if !matches!(v, Value::Nil) {
                    self.last_value = Some(v);
                }
                Ok(Flow::Normal)
            }
        }
    }

    fn eval(&mut self, e: &Expr, env: &Arc<Environment>) -> Res<Value> {
        match e {
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
                    out.push(self.eval(it, env)?);
                }
                Ok(Value::List(Arc::new(Mutex::new(out))))
            }
            Expr::Dict(items) => {
                let mut map = HashMap::new();
                for (k, ve) in items {
                    map.insert(k.clone(), self.eval(ve, env)?);
                }
                Ok(Value::Dict(Arc::new(Mutex::new(map))))
            }
            Expr::Unary(op, inner) => {
                let v = self.eval(inner, env)?;
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
                let l = self.eval(lhs, env)?;
                let r = self.eval(rhs, env)?;
                apply_binary(op, &l, &r)
            }
            Expr::Call(fexpr, args) => {
                let f = self.eval(fexpr, env)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, env)?);
                }
                call_function(self, &f, &vals)
            }
            Expr::Index(base, idx) => {
                let b = self.eval(base, env)?;
                let i = self.eval(idx, env)?;
                index_value(&b, &i)
            }
        }
    }
}

fn apply_binary(op: &str, l: &Value, r: &Value) -> Res<Value> {
    match op {
        "+" => value_add(l, r).map_err(InterpreterError::from_lang),
        "-" => value_sub(l, r).map_err(InterpreterError::from_lang),
        "*" => value_mul(l, r).map_err(InterpreterError::from_lang),
        "/" => value_div(l, r).map_err(InterpreterError::from_lang),
        "%" => match (l.as_num(), r.as_num()) {
            (Some(a), Some(b)) if b != 0.0 => Ok(Value::Num(a % b)),
            (_, Some(b)) if b == 0.0 => Err(InterpreterError::from_lang(LangError::DivByZero)),
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

fn index_value(base: &Value, idx: &Value) -> Res<Value> {
    fn norm(i: f64, len: usize) -> Res<usize> {
        let i = i as i64;
        let i = if i < 0 { (len as i64 + i) as usize } else { i as usize };
        if i >= len {
            return Err(InterpreterError::new(
                "IndexError",
                format!("index {i} out of range (len {len})"),
            ));
        }
        Ok(i)
    }
    match (base, idx) {
        (Value::List(l), Value::Num(i)) => {
            let l = l.lock().unwrap();
            let i = norm(*i, l.len())?;
            Ok(l[i].clone())
        }
        (Value::Dict(d), Value::Str(k)) => d
            .lock()
            .unwrap()
            .get(k)
            .cloned()
            .ok_or_else(|| InterpreterError::new("KeyError", format!("key '{k}' not found"))),
        (Value::Tensor(t), Value::Num(i)) => {
            let v = t.to_vec::<f64>().unwrap();
            let i = norm(*i, v.len())?;
            Ok(Value::Num(v[i]))
        }
        (b, _) => Err(InterpreterError::new(
            "TypeError",
            format!("cannot index {}", b.type_name()),
        )),
    }
}

fn call_function(interp: &mut Interpreter, f: &Value, args: &[Value]) -> Res<Value> {
    let Value::Function(fun) = f else {
        return Err(InterpreterError::new(
            "TypeError",
            format!("{} is not callable", f.type_name()),
        ));
    };
    match fun.as_ref() {
        Function::Native { f, .. } => {
            f(args).map_err(|m| InterpreterError::new("RuntimeError", m))
        }
        Function::Script { name, .. } => {
            let (params_def, body) = interp.script_bodies.get(name).cloned().ok_or_else(|| {
                InterpreterError::new("NameError", format!("body of '{name}' missing"))
            })?;
            if params_def.len() != args.len() {
                return Err(InterpreterError::new(
                    "TypeError",
                    format!(
                        "{name}() takes {} arguments but {} were given",
                        params_def.len(),
                        args.len()
                    ),
                ));
            }
            // function calls evaluate in the global scope
            let parent = Arc::clone(interp.global_env());
            let local = Arc::new(Environment::child(&parent));
            for (p, v) in params_def.iter().zip(args) {
                local.define(p.clone(), v.clone());
            }
            match interp.exec_block(&body, &local)? {
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
            .map(|v| v.as_num().map(|x| x as usize).ok_or_else(|| "bad shape".to_string()))
            .collect::<Result<_, _>>()?,
        _ => args
            .iter()
            .map(|v| v.as_num().map(|x| x as usize).ok_or_else(|| "bad shape".to_string()))
            .collect::<Result<_, _>>()?,
    };
    let n: usize = shape.iter().product();
    Ok(Value::Tensor(
        Tensor::from_typed(vec![fill; n]).reshape(&shape).unwrap(),
    ))
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
        it.register_native("double", |args| match args.first().and_then(|v| v.as_num()) {
            Some(x) => Ok(Value::Num(2.0 * x)),
            None => Err("double() requires a number".into()),
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
}



