# Tutorial 14: End-to-End Workflow

**Estimated Time:** 60 minutes  
**Prerequisites:** Tutorials 1-13

---

## Introduction

This tutorial walks through the complete compilation pipeline from TPT Script source to hardware execution.

### Pipeline Overview

```
TPT Script Source (.tpts)
         │
         ▼
┌─────────────────┐
│ Lexer           │  Tokenize source
├─────────────────┤
│ Parser          │  Build AST
├─────────────────┤
│ Type Checker    │  Validate types
├─────────────────┤
│ CodeGen         │  Generate Rust + TPTIR (one combined output)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  Rust      TPTIR text
    │         │
    ▼         ▼
  cargo   tpt_gpu_compiler::compile_native()  (runs default pass pipeline)
    │         │
    ▼         ▼
  binary    TPT ISA text / LLVM IR
              │
              ▼
           tptd daemon (BAR0 MMIO)
              │
              ▼
           GPU Hardware
```

There is no `tptc` binary — `tpt-gpu-compiler` is a library crate; its `compile_native(source, target)`
function (target `"tptisa"`/`"text"` or `"llvmir"`) is the entry point that steps 4–5 below
actually use.

---

## Step 1: Write TPT Script

```tpts
// vector_add.tpts
import tpt

@doc("Element-wise vector addition")
@requires_gpu(true)
@complexity("O(n)")
fn vector_add(a: Tensor[f32, *], b: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.add(a, b)
}

@doc("Run vector addition")
@requires_gpu(false)
fn run_add(n: i64) -> f32 {
    let a = tpt.randn([n], dtype=f32)
    let b = tpt.randn([n], dtype=f32)
    let c = vector_add(a, b)
    tpt.sync()
    return tpt.sum(c)
}
```

---

## Step 2: Type Check

```bash
tpt check vector_add.tpts
```

Output:
```
✓ Type-checked vector_add.tpts (0 errors, 0 warnings)
```

---

## Step 3: Compile

```bash
tpt compile vector_add.tpts -o out.rs
```

`-o` names a single output **file**, not a directory — `cmd_compile` writes one combined file
containing the Rust source, a `// === TPTIR Output ===` separator, then the TPTIR source:

- `out.rs` — Rust host code, followed by the TPTIR text in the same file

### Generated Rust (top portion of `out.rs`)

```rust
pub fn run_add(n: i64) -> f32 {
    // Named args like `dtype=f32` have no Rust equivalent and are emitted as a
    // positional value with a comment.
    let a = tptr::randn([n], /*dtype=*/ f32);
    let b = tptr::randn([n], /*dtype=*/ f32);
    let c = vector_add(a, b);
    tptr::sync();
    tptr::sum(c)
}
```

### Generated TPTIR (bottom portion of `out.rs`, after the `// === TPTIR Output ===` separator)

```tptir
module {
  func.func @vector_add(
    %a: memref<*xf32>,
    %b: memref<*xf32>
  ) -> memref<*xf32> attributes {tptir.kernel} {
    ^entry:
      %c0 = tptir.constant 0 : i32
      %tid = tptir.get_thread_id : i32
      %idx = tptir.addi(%tid, %c0)
      %va = tptir.load(%a, %idx)
      %vb = tptir.load(%b, %idx)
      %vc = tptir.addf(%va, %vb)
      tptir.store(%vc, %c, %idx)
      tptir.return
  }
}
```

---

## Step 4: Run the Optimization Pipeline and Lower to TPT ISA

There is no `tptc` CLI. Both optimization and target lowering happen through
`tpt_gpu_compiler::compile_native`, which runs the default pass pipeline
(canonicalize → dce → validate → fusion → quantization — see
[Tutorial 5](05_tptir_passes.md)) internally before emitting the target:

```rust
use tpt_gpu_compiler::compile_native;

let tptisa_text = compile_native(&tptir_source, "tptisa")?; // or "llvmir"
```

---

## Step 6: Build Host Code

`tpt compile` emits Rust source text — it does not scaffold a runnable crate. Copy the Rust
portion of `out.rs` (everything before `// === TPTIR Output ===`) into your own Cargo project
that depends on `tptr` (the layer4 runtime bindings), then:

```bash
cargo build --release
```

---

## Step 7: Execute

```bash
./target/release/your_binary
```

---

## Inspecting Intermediate Representations

There is no `tpt ast`, `tpt typed-ast`, or `tptc ir --debug` subcommand. To see both generated
outputs together without writing a file, use `tpt run`, which type-checks and prints the Rust
and TPTIR output to stdout:

```bash
tpt run vector_add.tpts
# === Rust Output ===
# ...
# === TPTIR Output ===
# ...
```

---

## Debugging

### Enable Debug Output

```bash
RUST_LOG=debug tpt check vector_add.tpts
```

### View Generated Code

```bash
tpt compile vector_add.tpts -o out.rs   # single combined file, not a directory or --emit-ir flag
```

---

## Performance Profiling

There is no `tpt profile` subcommand — see [Tutorial 16](16_performance_tuning.md) for the
real profiling path (hardware perf counters via the driver ABI, plus `tpt-gpu-bench` /
`tpt-gpu-kernel-optimizer`).

---

## Complete Example: Transformer Block

```tpts
import tpt

type Batch = Tensor[f32, batch, seq, d_model]

@doc("Transformer block")
@requires_gpu(true)
@requires_tensor_cores(true)
@complexity("O(batch * seq^2 * d_model)")
fn transformer_block(
    x: Batch,
    attn_w: Tensor[f32, 3, d_model, d_model],
    ffn_w1: Tensor[f32, d_ff, d_model],
    ffn_w2: Tensor[f32, d_model, d_ff],
) -> Batch {
    // Self-attention
    let q = tpt.matmul(x, tpt.transpose(attn_w[0]))
    let k = tpt.matmul(x, tpt.transpose(attn_w[1]))
    let v = tpt.matmul(x, tpt.transpose(attn_w[2]))
    let attn_out = tpt.attention(q, k, v)
    let residual1 = tpt.add(x, attn_out)
    let normed1 = tpt.layer_norm(residual1)
    
    // Feed-forward
    let ff = tpt.matmul(normed1, tpt.transpose(ffn_w1))
    let ff = tpt.gelu(ff)
    let ff = tpt.matmul(ff, tpt.transpose(ffn_w2))
    let residual2 = tpt.add(normed1, ff)
    return tpt.layer_norm(residual2)
}
```

---

## Exercises

1. **Pipeline**: Run the complete pipeline for a matrix multiplication kernel
2. **Optimization**: Compare performance with and without optimization passes
3. **Debugging**: Use debug output to understand code generation

---

## Summary

- ✅ Lexer → Parser → Type Checker → CodeGen pipeline
- ✅ Dual output: Rust (host) + TPTIR (device), combined into one file at the path passed to `-o`
- ✅ `compile_native()` runs the default pass pipeline and lowers to `tptisa`/`llvmir` — no separate `tptc` tool
- ✅ `tpt run` for a quick combined-output preview; `RUST_LOG=debug` for verbose `tpt check` output

**Next:** [Tutorial 15: Building a Model](15_building_a_model.md)
