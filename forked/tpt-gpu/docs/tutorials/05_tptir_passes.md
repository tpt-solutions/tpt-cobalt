# Tutorial 5: TPTIR Passes

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 4

---

## Introduction

TPTIR passes are transformation passes that operate on the IR defined in
`crates/tpt-gpu-compiler/src/ir.rs`. Each pass implements the `Pass` trait
(`crates/tpt-gpu-compiler/src/passes.rs`) and is run over a `Region`.

### Pass Pipeline

`passes::default_pipeline()` builds this fixed sequence:

```
Input TPTIR
    │
    ▼
┌─────────────────┐
│ Canonicalize    │  (currently a no-op stub — see below)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DCE             │  (currently a no-op stub — see below)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate        │  Semantic correctness checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fusion          │  Pattern-match and merge op sequences
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quantization    │  Count/prepare Quantize→Gemm→Dequantize sequences
└────────┬────────┘
         │
         ▼
Output TPTIR
```

There is no separate constant-folding, vectorization, or tensor-lowering pass in the
current implementation — those are aspirational and not yet built.

---

## The `Pass` Trait

```rust
// crates/tpt-gpu-compiler/src/passes.rs
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, region: &Region) -> usize; // returns number of changes/findings
}
```

## Pass Reference

### `CanonicalizePass` / `DeadCodeEliminationPass`

Both are registered in the default pipeline but currently stub implementations —
`run()` always returns `0` and performs no transformation. They are placeholders for
future normalization/dead-code-elimination logic.

### `ValidatePass`

Runs `validate_region()` (`validate.rs`) and returns the number of validation errors found
(`0` if valid). Checks include:
- Use-before-def
- Type mismatches between operands and results
- Missing block terminators
- Wrong operand counts
- Cyclic control flow

### `FusionPass`

Detects fusible operation sequences (`fusion.rs::detect_patterns`) and merges them:

- **`ElementwiseChain`** — runs of `Addf`/`Subf`/`Mulf` collapsed into one fused op
- **`FlashAttention`** — matmul → softmax → matmul pattern
- **`ConvBnRelu`** — conv + batchnorm + relu fusion
- **`QuantGemmFuse`** — `Dequantize → Gemm` collapsed into a single `QuantGemm`

### `QuantizationPass`

Counts `QuantGemm`/`Quantize`/`Dequantize`/`QuantAttention` ops already present in the
region (e.g. emitted by codegen) and reports that count as its change total. It does not
itself rewrite `Gemm` ops — see the doc comment in `passes.rs` for the intended relationship
with `FusionPass`.

---

## Running the Pipeline

```rust
use tpt_gpu_compiler::passes::{PassPipeline, default_pipeline};

let pipeline = default_pipeline(); // canonicalize -> dce -> validate -> fusion -> quantization
let total_changes = pipeline.run(&region);
println!("{total_changes} changes/findings across the pipeline");
```

Or build a custom pipeline:

```rust
use tpt_gpu_compiler::passes::{PassPipeline, ValidatePass};
use tpt_gpu_compiler::fusion::FusionPass;

let mut pipeline = PassPipeline::new();
pipeline.add(Box::new(ValidatePass));
pipeline.add(Box::new(FusionPass));
let changes = pipeline.run(&region);
```

---

## Writing Custom Passes

```rust
use tpt_gpu_compiler::passes::Pass;
use tpt_gpu_compiler::ir::Region;

struct CountOpsPass;

impl Pass for CountOpsPass {
    fn name(&self) -> &str {
        "count-ops"
    }

    fn run(&self, region: &Region) -> usize {
        region.blocks.iter().map(|b| b.operations.len()).sum()
    }
}
```

`Pass::run` takes a shared `&Region` and returns a `usize` count — passes that need to
mutate the IR (like a future canonicalizer) will need the region to expose mutable access,
which `FusionPass`/`QuantizationPass` do not currently do (they only count/report).

---

## Fusion Pattern Example

**Input block** (three chained multiplications, detected as `FlashAttention` by the
current simplified heuristic in `detect_patterns`):

```tptir
%a = tptir.mulf(%q, %k)
%b = tptir.mulf(%a, %scale)
%c = tptir.mulf(%b, %v)
```

`detect_patterns` returns a `FusionResult { pattern: FusedPattern::FlashAttention, start_op: 0, end_op: 2 }`
for this sequence.

---

## Exercises

1. **Validate**: Feed `ValidatePass` a region with a use-before-def and confirm `run()` returns a nonzero error count
2. **Fusion**: Write a block with an elementwise chain (`add`, `sub`, `mul` in sequence) and trace through `detect_patterns` by hand
3. **Custom Pass**: Implement a pass that counts operations by `OpKind`

---

## Summary

- ✅ `Pass` trait: `name()` + `run(&Region) -> usize`
- ✅ `default_pipeline()`: canonicalize → dce → validate → fusion → quantization
- ✅ `Canonicalize`/`DCE` are currently no-op stubs
- ✅ `ValidatePass` checks semantic correctness
- ✅ `FusionPass` detects elementwise/attention/conv/quant-gemm patterns
- ✅ `QuantizationPass` counts quantization-related ops
- ✅ Custom pass development

**Next:** [Tutorial 6: Memory Management](06_memory_management.md)
