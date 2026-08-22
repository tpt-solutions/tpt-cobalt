//! The dynamic [`Value`] model: first-class tensors with zero indirection,
//! shared mutable List/Dict (`Arc<Mutex<…>>`, documented divergence from the
//! spec's tracing-GC sketch), numeric promotion rules, and truthiness.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tpt_tensor::Tensor;

/// A dynamically typed runtime value. `Value::Tensor` is a native variant —
/// tensors are first-class citizens of the language, not boxed user data.
#[derive(Clone)]
pub enum Value {
    /// Unit / absence of a value.
    Nil,
    /// Boolean literal.
    Bool(bool),
    /// Floating-point number (the language has a single numeric type).
    Num(f64),
    /// UTF-8 string.
    Str(String),
    /// Shared mutable list (cycles leak until a GC lands).
    List(Arc<Mutex<Vec<Value>>>),
    /// Shared mutable string-keyed dict.
    Dict(Arc<Mutex<HashMap<String, Value>>>),
    /// A first-class tensor (zero indirection to `tpt_tensor::Tensor`).
    Tensor(Tensor),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "Nil"),
            Value::Bool(b) => write!(f, "Bool({b})"),
            Value::Num(n) => write!(f, "Num({n})"),
            Value::Str(s) => write!(f, "Str({s:?})"),
            Value::List(l) => write!(f, "List(len={})", l.lock().unwrap().len()),
            Value::Dict(d) => write!(f, "Dict(len={})", d.lock().unwrap().len()),
            Value::Tensor(t) => write!(f, "Tensor{:?}", t.shape()),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(true) => write!(f, "true"),
            Value::Bool(false) => write!(f, "false"),
            Value::Num(n) => write!(f, "{n}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::List(l) => {
                let items = l.lock().unwrap();
                let rendered: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", rendered.join(", "))
            }
            Value::Dict(d) => {
                let map = d.lock().unwrap();
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let rendered: Vec<String> =
                    keys.iter().map(|k| format!("{k}: {}", map[*k].to_string())).collect();
                write!(f, "{{{}}}", rendered.join(", "))
            }
            Value::Tensor(_) => write!(f, "<tensor>"),
        }
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Num(n)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}
impl From<Tensor> for Value {
    fn from(t: Tensor) -> Self {
        Value::Tensor(t)
    }
}

/// Truthiness rules for conditionals and boolean coercion.
///
/// - `Nil` and `Bool(false)` are falsy.
/// - `Num(0)` / NaN is falsy; every other number is truthy.
/// - Empty strings / lists / dicts are falsy.
/// - A tensor is truthy iff *every* element is nonzero.
pub trait Truthiness {
    /// Whether this value counts as true in a conditional context.
    fn truthy(&self) -> bool;
}

impl Truthiness for Value {
    fn truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Num(n) => *n != 0.0 && !n.is_nan(),
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.lock().unwrap().is_empty(),
            Value::Dict(d) => !d.lock().unwrap().is_empty(),
            Value::Tensor(t) => t.to_vec::<f64>().unwrap_or_default().iter().all(|x| *x != 0.0),
        }
    }
}