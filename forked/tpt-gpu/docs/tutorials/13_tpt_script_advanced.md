# Tutorial 13: TPT Script Advanced

**Estimated Time:** 55 minutes  
**Prerequisites:** Tutorial 12

---

## Introduction

This tutorial covers advanced TPT Script features: constraints, introspection, structured errors, and the metadata system.

### Constraint System

```tpts
@doc("Matrix multiplication with constraints")
@requires_gpu(true)
@constraint("M > 0", "M must be positive")
@constraint("N > 0", "N must be positive")
@constraint("K > 0", "K must be positive")
@constraint("M % 32 == 0", "M must be multiple of 32")
fn matmul_constrained(
    a: Tensor[f32, M, K],
    b: Tensor[f32, K, N]
) -> Tensor[f32, M, N] {
    return tpt.matmul(a, b)
}
```

---

## Introspection API

Introspection (`crates/tpt-gpu-script-core/src/introspect/mod.rs`) is a Rust-side API exposed
through the CLI, not something called from within a `.tpts` program — there is no
`tpt.introspect(...)` builtin.

```bash
# Inspect one operation's schema (doc, inputs, complexity, hardware requirements, ...)
tpt inspect matmul

# List all built-in operations
tpt ops
```

Rust API surface backing those commands (`introspect::{list_operations, get_schema}`):

```rust
use tpt_gpu_script_core::introspect::{list_operations, get_schema};

for name in list_operations() {
    if let Some(schema) = get_schema(name) {
        println!("{}: {}", schema.name, schema.description);
    }
}
```

---

## Structured Errors

```json
{
    "code": "SHAPE_MISMATCH",
    "message": "Tensor dimensions incompatible",
    "location": {"file": "model.tpts", "line": 42, "col": 15},
    "suggestions": [
        "Check that inner dimensions match",
        "Consider transposing B: tpt.transpose(b)"
    ],
    "fix_code": "tpt.transpose(b)"
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `SHAPE_MISMATCH` | Tensor dimensions incompatible |
| `DTYPE_MISMATCH` | Operand dtypes incompatible |
| `CONSTRAINT_VIOLATION` | @constraint violated |
| `UNDEFINED_VARIABLE` | Variable not in scope |

---

## Formatter and Linter

`tpt-gpu-script-format` is a library crate — it has no `[[bin]]` target, so
`cargo run -p tpt-gpu-script-format` does not work. Use its functions from Rust:

```rust
use tpt_gpu_script_format::{format, lint};

let source = std::fs::read_to_string("file.tpts")?;
let formatted = format(&source)?;         // -> Result<String, CompileError>
let warnings = lint(&source);             // -> Vec<LintWarning>
for w in &warnings {
    println!("{w:?}");
}
```

---

## Example: Advanced Kernel

```tpts
import tpt

@doc("Fused attention with causal masking")
@requires_gpu(true)
@requires_tensor_cores(true)
@constraint("seq <= 8192", "Sequence length must be <= 8192")
@complexity("O(batch * heads * seq^2 * d_k)")
fn fused_attention(
    q: Tensor[f32, batch, heads, seq, d_k],
    k: Tensor[f32, batch, heads, seq, d_k],
    v: Tensor[f32, batch, heads, seq, d_v],
) -> Tensor[f32, batch, heads, seq, d_v] {
    let scale = tpt.sqrt(tpt.cast(d_k, dtype=f32))
    let logits = tpt.matmul(q, tpt.transpose(k)) * (1.0 / scale)
    let mask = tpt.causal_mask(seq)
    let masked = tpt.where(mask, logits, -1e9)
    let weights = tpt.softmax(masked, dim=-1)
    return tpt.matmul(weights, v)
}
```

---

## Exercises

1. **Constraints**: Add constraints to validate tensor shapes
2. **Introspection**: Query operation metadata at runtime
3. **Error Handling**: Handle structured errors in host code

---

## Summary

- ✅ Constraint system for compile-time validation
- ✅ Introspection API for operation metadata
- ✅ Structured errors with fix suggestions
- ✅ LSP integration for IDE support
- ✅ Formatter and linter

**Next:** [Tutorial 14: End-to-End Workflow](14_end_to_end.md)
