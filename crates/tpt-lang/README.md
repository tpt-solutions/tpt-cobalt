# tpt-lang

The TPT script language runtime — the dynamic value model and execution
scaffolding for scripting on top of the tensor stack.

The centerpiece is the [`Value`] enum with **first-class tensors**:
`Value::Tensor` is a native variant with zero indirection, so scripts can pass
tensors around exactly like numbers and strings.

## Features

- **Dynamic value model** — `Nil`, `Bool`, `Num` (single float type), `Str`,
  `List`, `Dict`, and `Tensor` variants.
- **Shared mutable collections** — `List`/`Dict` use `Arc<Mutex<…>>` for
  shared-mutable semantics. This is a deliberate divergence from the spec's
  tracing-GC sketch: cycles leak until a GC lands (documented in todo.md).
- **Numeric promotion** — arithmetic between `Num`, `Bool` (promotes to
  0/1), and `Tensor` follows consistent rules: scalar-scalar float math,
  tensor-tensor element-wise broadcasting, and scalar-broadcast onto tensors,
  in either operand order.
- **Truthiness** — the [`Truthiness`] trait: empty containers/strings are
  falsy; a tensor is truthy iff every element is nonzero.
- **Operators** — `value_add/sub/mul/div/eq` returning `Result<Value,
  LangError>` with division-by-zero detection.
- **Scoped environments** — [`Environment`] chains child scopes to parents via
  `Arc`, with define-in-current-scope and walk-the-chain assignment/lookup.

## Installation

```toml
[dependencies]
tpt-lang = "0.1"
```

## Quick start

```rust
use tpt_lang::{Environment, Truthiness, Value, value_add};
use tpt_tensor::Tensor;

fn main() {
    // Scalars promote as floats.
    let sum = value_add(&Value::Num(2.0), &Value::Num(3.0)).unwrap();

    // Tensors are first-class values.
    let t = Value::Tensor(Tensor::from_typed(vec![1.0_f64, 2.0, 3.0]));
    let scaled = value_add(&t, &Value::Num(10.0)).unwrap(); // broadcast +10

    // Scoped variables.
    let mut global = Environment::new();
    global.define("x", Value::Num(41.0));
    let local = Environment::child(&(std::sync::Arc::new(global)));
    local.assign("x", Value::Num(42.0)).unwrap();
    assert_eq!(local.get("x"), Some(Value::Num(42.0)));

    assert!(sum.truthy());
    let _ = scaled;
}
```

## Examples

```sh
cargo run -p tpt-lang --example values_and_truthiness
cargo run -p tpt-lang --example scoped_env
```

## Status

This crate is the Phase 6 scaffold: value model, promotion rules, operators,
and environments. Parser, evaluator loop, and standard library land later in
the phase.

## License

Dual-licensed under the workspace license (see repository root).
