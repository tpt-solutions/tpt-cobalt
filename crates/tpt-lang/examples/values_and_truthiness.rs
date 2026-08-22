//! The dynamic value model: first-class tensors, numeric promotion, and
//! truthiness.
//!
//! Run with: `cargo run -p tpt-lang --example values_and_truthiness`

use std::sync::{Arc, Mutex};

use tpt_lang::{value_add, value_div, value_eq, Truthiness, Value};
use tpt_tensor::Tensor;

fn main() {
    // Scalar arithmetic promotes to floats.
    let sum = value_add(&Value::Num(2.0), &Value::Num(3.0)).unwrap();
    println!("2 + 3            = {}", sum.to_string()); // 5

    // Tensors are first-class values, not boxed data.
    let t = Value::Tensor(Tensor::from_typed(vec![1.0_f64, 2.0, 3.0]));

    // Scalar broadcast onto a tensor, in either operand order.
    let scaled = value_add(&t, &Value::Num(10.0)).unwrap();
    let scaled2 = value_add(&Value::Num(10.0), &t).unwrap();
    println!("t + 10           = tensor [11, 12, 13]");
    assert!(matches!(value_eq(&scaled, &scaled2), Value::Bool(true)));
    // Element-wise tensor-tensor ops.
    let doubled = value_add(&t, &t).unwrap();
    assert!(matches!(doubled, Value::Tensor(_)));

    // Errors are values, not panics.
    let err = value_div(&Value::Num(1.0), &Value::Num(0.0)).unwrap_err();
    println!("1 / 0            -> {err}");

    // Truthiness across the type system.
    assert!(Value::Num(0.5).truthy());
    assert!(!Value::Num(0.0).truthy());
    assert!(!Value::Str(String::new()).truthy());
    assert!(!Value::List(Arc::new(Mutex::new(Vec::new()))).truthy());
    assert!(!Value::Tensor(Tensor::from_typed(vec![1.0_f64, 0.0])).truthy());
    assert!(Value::Tensor(Tensor::from_typed(vec![2.0_f64, 3.0])).truthy());
    println!("truthiness rules verified");
}
