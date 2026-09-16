//! Lexically scoped environments: define in the current scope, look up and
//! assign through the parent chain.
//!
//! Run with: `cargo run -p tpt-lang --example scoped_env`

use std::sync::Arc;

use tpt_lang::{Environment, Value};
use tpt_tensor::Tensor;

fn eq(a: Option<Value>, b: Value) -> bool {
    matches!(
        a.as_ref().map(|v| tpt_lang::value_eq(v, &b)),
        Some(Value::Bool(true))
    )
}

fn main() {
    // Global scope.
    let global = Arc::new(Environment::new());
    global.define("x", Value::Num(1.0));
    global.define(
        "model",
        Value::Tensor(
            Tensor::from_typed(vec![0.5_f64; 4])
                .reshape(&[2, 2])
                .unwrap(),
        ),
    );

    // Function scope: sees globals, defines locals.
    let local = Environment::child(&global);
    local.define("x", Value::Num(100.0)); // shadows the global
    local.define("y", Value::Num(2.0));

    assert!(eq(local.get("x"), Value::Num(100.0))); // local wins
    assert!(eq(global.get("x"), Value::Num(1.0))); // global untouched
    assert!(eq(local.get("y"), Value::Num(2.0))); // own binding
    assert!(local.contains("model")); // falls through to the parent

    // Assignment walks the chain and never creates bindings.
    local.assign("x", Value::Num(42.0)).unwrap();
    assert!(eq(local.get("x"), Value::Num(42.0)));

    local
        .assign("model", Value::Str(String::from("updated")))
        .unwrap();
    assert!(eq(global.get("model"), Value::Str("updated".into())));

    // Assigning an undefined name is an error.
    assert!(local.assign("nope", Value::Num(0.0)).is_err());
    assert!(!local.contains("nope"));

    println!("scoping rules verified");
}
