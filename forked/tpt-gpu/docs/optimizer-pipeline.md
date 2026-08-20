# TPT Model Optimizer Pipeline

**Version:** 1.0  
**Last Updated:** July 2026

---

## Overview

The TPT Model Optimizer compresses LLMs to smaller sizes while maintaining quality within a user-defined budget (default 5% perplexity increase). The optimizer produces the native `.tptf` format (self-contained: weights + tokenizer + chat template) and can re-export to GGUF/EXL2 for compatibility.

## Pipeline Stages

```
┌─────────────────┐
│ GGUF source     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌───────────────────────┐
│ HardwareProfiler├────►│ ~/.tpt/hardware_profile.json│
└────────┬────────┘     └───────────────────────┘
         │
         ▼
┌─────────────────┐
│ LayerSensitivityMap│ (Wanda-style: |weight| × mean(|activation|))
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DomainMapper    │ (per-neuron domain scores)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌───────────────────────┐
│ SurgicalPruner  ├────►│ PruningMask (sparse)  │
└────────┬────────┘     └───────────────────────┘
         │
         ▼
┌─────────────────┐
│ MixedPrecisionAllocator │
│ - Live perplexity eval │
│ - Per-layer bits search│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌───────────────────────┐
│ KvCacheCalculator├───►│ Context window recommendation│
└────────┬────────┘     └───────────────────────┘
         │
         ▼
┌─────────────────┐
│ TptfWriter    │ (quant + pack)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ model.tptf      │ (self-contained)
└─────────────────┘
```

## Stage Details

### 1. Hardware Profiling

The profiler benchmarks the target GPU and caches results to `~/.tpt/hardware_profile.json`:

- **Memory bandwidth**: Large buffer copy sweep, scaled by PCIe ratio
- **L2 cache size**: Found as knee point in bandwidth-vs-size curve
- **Tensor core generation**: Detected via CUDA/ROCm capability query
- **VRAM**: Queried via NVML/ROCm SMI

Results guide:
- Streaming loader threshold (model > 80% VRAM)
- Group size selection for quantization
- Context window calculation

### 2. Sensitivity Analysis

Implements Wanda-style importance scoring:

```
importance = |weight_norm| × mean(|activation|)
```

**Algorithm:**
1. Run 32 calibration tokens through each layer
2. Capture FFN intermediate activations (post-SwiGLU)
3. Compute mean magnitude per neuron
4. Multiply by weight magnitude for combined score
5. Rank layers by sensitivity (low → high)

### 3. Domain Mapping

Maps neurons to domains using activation patterns:

**Domains:**
- `python` — Python code and idioms
- `typescript` — TypeScript/JavaScript patterns
- `sql` — Database queries and schema
- `math` — Mathematical reasoning
- `reasoning` — Logic puzzles
- `code` — General programming
- `general` — General knowledge
- `science` — Physics, chemistry, biology
- `creative` — Creative writing

**Importance computation:**
```
weight_imp = Σ|W_ij| across input dimension
act_imp = mean(|activation_j|) across samples
neuron_imp = weight_imp × act_imp
```

### 4. Surgical Pruning

Zeros whole neurons (structural pruning) based on domain scores:

1. For each target domain, find neurons where:
   - Domain score ≥ threshold (default 0.05)
   - Is dominant domain for that neuron
2. Zero corresponding rows in `gate_proj` and `up_proj`
3. Zero corresponding columns in `down_proj`
4. Record in `PruningMask` embedded in `.tptf`

### 5. Mixed-Precision Allocation

The "5% loss frontier" algorithm:

```
for layer in sorted(sensitivity):
    for bits in [2, 3, 4, 6, 8]:
        temp_quantize(layer, bits)
        ppl = evaluate_perplexity()
        if (ppl - baseline) / baseline <= 0.05:
            assign_bits(layer, bits)
            break
```

**Floors:**
- Embedding (layer 0): always f16
- LM head (last layer): always f16
- Shallow layers (first 10%): ≥ 4-bit
- Attention Q/K: ≥ 4-bit (not yet enforced)

### 6. KV Cache Calculation

After quantization, compute max context:

```
model_bytes = Σ params_i × bits_i / 8
available = vram_free - model_bytes - 512MB_overhead
kv_per_token = 2 × num_kv_heads × head_dim × 2 bytes (K+V) × num_layers
context_len = available_bytes / kv_per_token
```

### 7. TPTF Writing

**Header (512 bytes):**
```
[0..4]    magic "TPTF"
[4..8]    version u32
[8..12]   flags u32
[12..76]  arch string (len-prefixed, max 63)
[76..80]  context_len u32
[80..84]  vocab_size u32
[84..88]  hidden_dim u32
[88..92]  num_heads u32
[92..96]  num_kv_heads u32
[96..100] ffn_dim u32
[100..104] num_layers u32
[104..232] per_layer_bits[u8; 128]
[460..468] tensor_offset u64
[468..476] tokenizer_offset u64
[476..484] chat_template_offset u64
[484..492] pruning_mask_offset u64
```

**Tensor blocks:**
- 128-byte aligned
- Bit-packed for sub-byte depths
- Scales and zero-points stored per group (default 128)

## Usage

### CLI

```bash
# Profile hardware once
tpt-gpu-model-optimizer profile

# Analyze a model's domain distribution
tpt-gpu-model-optimizer analyze model.gguf --output domain_map.json

# Run full optimization
tpt-gpu-model-optimizer optimize model.gguf --output model-opt.tptf

# Export to GGUF for llama.cpp
tpt-gpu-model-optimizer export model.tptf --format gguf --output model-int4.gguf

# Benchmark quality before/after
tpt-gpu-model-optimizer bench model.gguf model-opt.tptf
```

### Programmatic

```rust
use tpt_gpu_model_optimizer::{
    HardwareProfiler, LayerSensitivityMap, DomainMapper,
    MixedPrecisionAllocator, QualityBenchmark,
};

// Profile hardware
let profile = HardwareProfiler::new().profile()?;

// Build sensitivity map
let sensitivity = LayerSensitivityMap::build(num_layers)?;

// Map domains
let mapper = DomainMapper::with_default_domains();
let domain_map = mapper.build(num_layers, ffn_dim)?;

// Allocate bits
let allocator = MixedPrecisionAllocator::new(0.05);
let bits = allocator.allocate(num_layers, &sensitivity, baseline_ppl)?;
```

## Quality Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perplexity delta | ≤ 5% | Calibration set |
| Task accuracy | - | MCQA benchmark |
| Context window | - | VRAM-aware calc |

## Implementation Status

| Stage | Status |
|-------|--------|
| Hardware profiling | ✅ Heuristic + env vars |
| Sensitivity analysis | ✅ Heuristic (Wanda-style ready) |
| Domain mapping | ✅ Heuristic (activation hooks ready) |
| Surgical pruning | ✅ Implemented |
| Mixed-precision | ✅ Heuristic (live eval API ready) |
| KV cache | ✅ Implemented |
| TPTF format | ✅ Bit-packing implemented |
| GGUF export | ✅ Tensor repacking |
| EXL2 export | ✅ Tensor repacking |