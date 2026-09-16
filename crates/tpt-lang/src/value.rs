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
    /// A callable: native Rust builtin or interpreted user function.
    Function(Arc<Function>),
    /// A named namespace of values.
    Module(Arc<Module>),
    /// A trainable model (opaque to scripts; driven via the `train_*`/`predict`
    /// natives from `tpt_lang::ml`).
    Model(Arc<Mutex<crate::ml::ModelBox>>),
    /// A unit-aware numeric literal/value (e.g. `9.81 m/s^2`).
    Unit(Arc<UnitValue>),
}

/// A unit-aware value: numeric payload plus its dimension tag (e.g. `"m/s"`).
/// Compile-time unit *checking* lives in [`crate::check`].
#[derive(Clone, Debug)]
pub struct UnitValue {
    pub value: f64,
    pub dim: String,
}

/// A script-visible function.
#[derive(Clone)]
pub enum Function {
    /// Rust-implemented builtin.
    Native { name: &'static str, f: NativeFn },
    /// User-defined: parameter list plus the interpreter-internal body id
    /// (bodies live in the interpreter, keyed by id so module-scoped and
    /// shadowed definitions cannot collide by name). The defining
    /// environment is captured so functions close over their scope — a
    /// `def` inside a `module` block sees that module's members.
    Script {
        name: String,
        params: Vec<Param>,
        id: u64,
        env: Arc<crate::env::Environment>,
    },
}

impl Function {
    /// `<function f(x, y=2)>`-style rendering (REPL echo, error messages).
    pub fn signature(&self) -> String {
        match self {
            Function::Native { name, .. } => format!("<function {name}>"),
            Function::Script { name, params, .. } => {
                let ps: Vec<String> = params.iter().map(|p| p.display()).collect();
                format!("<function {}({})>", name, ps.join(", "))
            }
        }
    }
}

/// A function parameter: name plus optional default value. Defaults are
/// evaluated once at `def` time (Python semantics), not per call.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub default: Option<Value>,
}

impl Param {
    /// `x` or `y=2` rendering for signatures.
    pub fn display(&self) -> String {
        match &self.default {
            None => self.name.clone(),
            Some(v) => format!("{}={}", self.name, v),
        }
    }
}

/// A native function callable from TPT Script.
pub type NativeFn = Arc<dyn Fn(&[Value]) -> Result<Value, String> + Send + Sync>;

/// A module: a named namespace of values (the spec's object model has
/// modules and functions but deliberately no classes/inheritance). Members
/// are shared-mutable like List/Dict, so scripts can attach values to an
/// existing module (`m.x = v`).
#[derive(Debug, Default)]
pub struct Module {
    pub name: String,
    pub members: Mutex<HashMap<String, Value>>,
}

impl Module {
    /// An empty module with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Module {
            name: name.into(),
            members: Mutex::new(HashMap::new()),
        }
    }

    /// A module with pre-built members (e.g. a snapshot of a `module`
    /// statement's scope).
    pub fn with_members(name: impl Into<String>, members: HashMap<String, Value>) -> Self {
        Module {
            name: name.into(),
            members: Mutex::new(members),
        }
    }

    /// Member lookup.
    pub fn get(&self, member: &str) -> Option<Value> {
        self.members.lock().unwrap().get(member).cloned()
    }

    /// Define or overwrite a member.
    pub fn set(&self, member: impl Into<String>, value: Value) {
        self.members.lock().unwrap().insert(member.into(), value);
    }

    /// Sorted member names.
    pub fn member_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.members.lock().unwrap().keys().cloned().collect();
        names.sort();
        names
    }
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
            Value::Function(fun) => write!(f, "{}", fun.signature()),
            Value::Module(m) => write!(f, "<module {}>", m.name),
            Value::Model(_) => write!(f, "<model>"),
            Value::Unit(u) => write!(f, "{}{}", u.value, u.dim),
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
                    keys.iter().map(|k| format!("{k}: {}", map[*k])).collect();
                write!(f, "{{{}}}", rendered.join(", "))
            }
            Value::Tensor(t) => {
                write!(f, "Tensor{:?} ", t.shape())?;
                // pretty-print small tensors element-wise
                if t.numel() <= 16
                    && let Ok(v) = t.to_vec::<f64>()
                {
                    let rendered: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                    write!(f, "[{}]", rendered.join(", "))?;
                }
                Ok(())
            }
            Value::Function(fun) => write!(f, "{}", fun.signature()),
            Value::Module(m) => write!(f, "<module {}>", m.name),
            Value::Model(_) => write!(f, "<model>"),
            Value::Unit(u) => write!(f, "{}{}", u.value, u.dim),
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

/// Structural/numeric equality (delegates to `crate::ops::values_equal`).
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        crate::ops::values_equal(self, other)
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
            Value::Tensor(t) => t
                .to_vec::<f64>()
                .unwrap_or_default()
                .iter()
                .all(|x| *x != 0.0),
            Value::Function(_) | Value::Module(_) | Value::Model(_) | Value::Unit(_) => true,
        }
    }
}

impl Value {
    /// Type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Num(_) => "num",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Tensor(_) => "tensor",
            Value::Function(_) => "function",
            Value::Module(_) => "module",
            Value::Model(_) => "model",
            Value::Unit(_) => "unit",
        }
    }

    /// Tensor view.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Numeric view (`Num`, `Bool` promotes); tensors are not numbers.
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}
