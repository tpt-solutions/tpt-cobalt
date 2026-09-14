//! Binary operations over [`Value`] with numeric promotion rules
//! (`Num ⊕ Tensor` promotes the scalar via broadcasting) and structural /
//! numeric equality.

use std::fmt;

use tpt_tensor::Tensor;

use crate::value::Value;

/// Errors raised by the built-in operators.
#[derive(Debug, Clone, PartialEq)]
pub enum LangError {
    /// An operand had a type the operator does not accept.
    TypeMismatch { op: &'static str, found: String },
    /// Division (or modulo) by zero.
    DivByZero,
    /// Unsupported operand combination for an arithmetic op.
    Unsupported { op: &'static str, lhs: String, rhs: String },
}

impl fmt::Display for LangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LangError::TypeMismatch { op, found } => {
                write!(f, "operator `{op}` cannot be applied to {found}")
            }
            LangError::DivByZero => write!(f, "division by zero"),
            LangError::Unsupported { op, lhs, rhs } => {
                write!(f, "unsupported operands for `{op}`: {lhs} and {rhs}")
            }
        }
    }
}

impl std::error::Error for LangError {}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Nil => "nil",
        Value::Bool(_) => "bool",
        Value::Num(_) => "num",
        Value::Str(_) => "string",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
        Value::Tensor(_) => "tensor",
        Value::Unit(_) => "unit",
        Value::Function(_) => "function",
        Value::Module(_) => "module",
        Value::Model(_) => "model",
    }
}

macro_rules! arith {
    ($name:ident, $op:literal, $tensor_op:ident) => {
        /// Arithmetic with promotion:
        ///
        /// - `Num op Num` → scalar float math.
        /// - `Tensor op Tensor` → element-wise tensor op (broadcasting applies).
        /// - `Tensor op Num` / `Num op Tensor` → scalar broadcast onto the tensor.
        pub fn $name(lhs: &Value, rhs: &Value) -> Result<Value, LangError> {
            match (lhs, rhs) {
                (a, b) if as_num(a).is_some() && as_num(b).is_some() => {
                    let (a, b) = (as_num(a).unwrap(), as_num(b).unwrap());
                    Ok(Value::Num(tensor_scalar($op, a, b)?))
                }
                (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(a.$tensor_op(b))),
                (Value::Tensor(a), b) if as_num(b).is_some() => {
                    let sb = Tensor::from_typed(vec![as_num(b).unwrap()]);
                    Ok(Value::Tensor(a.$tensor_op(&sb)))
                }
                (a, Value::Tensor(b)) if as_num(a).is_some() => {
                    let sa = Tensor::from_typed(vec![as_num(a).unwrap()]);
                    Ok(Value::Tensor(sa.$tensor_op(b)))
                }
                (a, b) => Err(LangError::Unsupported {
                    op: $op,
                    lhs: kind(a).to_string(),
                    rhs: kind(b).to_string(),
                }),
            }
        }
    };
}

arith!(value_add, "+", add);
arith!(value_sub, "-", sub);
arith!(value_mul, "*", mul);

/// Division with zero check on both the scalar and tensor paths.
pub fn value_div(lhs: &Value, rhs: &Value) -> Result<Value, LangError> {
    match (lhs, rhs) {
        (a, b) if as_num(a).is_some() && as_num(b).is_some() => {
            let (a, b) = (as_num(a).unwrap(), as_num(b).unwrap());
            if b == 0.0 {
                return Err(LangError::DivByZero);
            }
            Ok(Value::Num(a / b))
        }
        (Value::Tensor(a), b) if as_num(b).is_some() => {
            let s = as_num(b).unwrap();
            if s == 0.0 {
                return Err(LangError::DivByZero);
            }
            let sb = Tensor::from_typed(vec![s]);
            Ok(Value::Tensor(a.div(&sb)))
        }
        (a, b) => Err(LangError::Unsupported {
            op: "/",
            lhs: kind(a).to_string(),
            rhs: kind(b).to_string(),
        }),
    }
}

/// Equality: numbers compare numerically (`Bool` promotes), strings/lists/dicts
/// structurally, tensors element-wise. Cross-kind values are unequal.
#[must_use]
pub fn value_eq(lhs: &Value, rhs: &Value) -> Value {
    Value::Bool(values_equal(lhs, rhs))
}

/// Structural / numeric equality — also exposed as `PartialEq` on `Value`.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Nil, Value::Nil) => true,
        (x, y) if as_num(x).is_some() && as_num(y).is_some() => {
            as_num(x).unwrap() == as_num(y).unwrap()
        }
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            let (x, y) = (x.lock().unwrap(), y.lock().unwrap());
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        (Value::Dict(x), Value::Dict(y)) => {
            let (x, y) = (x.lock().unwrap(), y.lock().unwrap());
            x.len() == y.len()
                && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| values_equal(v, w)))
        }
        (Value::Tensor(x), Value::Tensor(y)) => {
            x.shape() == y.shape() && x.to_vec::<f64>().unwrap() == y.to_vec::<f64>().unwrap()
        }
        _ => false,
    }
}

fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn tensor_scalar(op: &str, a: f64, b: f64) -> Result<f64, LangError> {
    match op {
        "+" => Ok(a + b),
        "-" => Ok(a - b),
        "*" => Ok(a * b),
        "/" => {
            if b == 0.0 {
                Err(LangError::DivByZero)
            } else {
                Ok(a / b)
            }
        }
        other => unreachable!("unknown scalar op {other}"),
    }
}