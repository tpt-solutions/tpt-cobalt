# TPT GPU Developer Portal

Welcome to the TPT GPU developer portal. This guide provides comprehensive documentation for building, optimizing, and deploying models on the TPT GPU compute platform.

## Overview

TPT GPU is an open-source, hardware-agnostic, full-stack GPU compute platform designed for high-performance machine learning inference. The system consists of six layers:

- **Layer 1** — TPT ISA (SystemVerilog for custom silicon)
- **Layer 2** — TPT Driver / tptd (C + Rust drivers)
- **Layer 3** — TPTIR Compiler Stack / tptc (C++ + Rust)
- **Layer 4** — TPT Runtime / tptr (Rust)
- **Layer 5** — TPT Primitives / tptp (TPTIR + Rust)
- **Layer 6** — Framework Backends (Python + Rust)
- **Layer 7** — TPT Script (AI-native language)

## Quick Start

For cloning the repo and general prerequisites, see the root [`README.md`](../../../README.md#quick-start) Quick Start.

### Installation

```bash
cd tpt-gpu

# Install the tpt-gpu-model-optimizer CLI
cargo install --path crates/tpt-gpu-model-optimizer
```

### Using the Model Optimizer

```bash
# Profile your GPU hardware
tpt-gpu-model-optimizer profile

# Analyze domain-specific neuron distribution
tpt-gpu-model-optimizer analyze model.gguf --output domain_map.json

# Optimize a model with 5% max quality loss
tpt-gpu-model-optimizer optimize model.gguf --max-loss 0.05 --output optimized.tptf

# Export to GGUF format for compatibility
tpt-gpu-model-optimizer export optimized.tptf --format gguf --output model_fp16.gguf

# Compare quality before/after optimization
tpt-gpu-model-optimizer bench original.gguf optimized.tptf

# Calculate max context window
tpt-gpu-model-optimizer kv-calc optimized.tptf
```

## Model Optimizer Pipeline

### Architecture

The model optimizer transforms any GGUF model into TPTF format (TPT's self-contained format) with mixed-precision quantization and optional surgical pruning.

```text
┌─────────────────────────────────────────────────────────────┐
│                    TPT Model Optimizer                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Hardware   │───▶│ Sensitivity │───▶│   Domain    │    │
│  │  Profiler   │    │    Map      │    │   Mapper    │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ GPU profile │    │ Layer ranks │    │ Neuron map  │    │
│  │ (cached)    │    │ (by sens.)  │    │ (by domain) │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │           │
│         │                  ▼                  │           │
│         │            ┌─────────────┐          │           │
│         │            │   Pruner    │          │           │
│         │            └─────────────┘          │           │
│         │                  │                  │           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Mixed-Precision Allocator                 │  │
│  │  Per-layer bit depths within quality budget (5%)     │  │
│  └─────────────────────────────────────────────────────┘  │
│         │                  │                  │           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   KV Cache  │    │   TPTF      │    │ Calibration │    │
│  │ Calculator  │    │  Writer     │    │ Generator   │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### TPTF File Format

The TPTF format is TPT GPU's native model format, designed for efficient loading and inference:

```
[0..512]     Header (magic, version, flags, arch metadata, per-layer bits, offsets)
[512.....]    Tensor blocks (128-byte aligned, pre-swizzled weights + scales + zero_points)
[after...]     Tokenizer block (verbatim GGUF tokenizer KV section)
[optional]     Chat template block (Jinja2 template string)
[optional]     Pruning mask block (sparse bit array)
```

All multi-byte integers are little-endian. See `docs/optimizer-pipeline.md` for complete specification.

### Quantization Strategy

The optimizer uses a "5% loss frontier" algorithm:

1. Compute baseline perplexity on calibration samples
2. Sort layers by sensitivity (least sensitive first)
3. For each layer, try bit depths `[2, 3, 4, 6, 8]` ascending
4. Assign the minimum bits where perplexity delta ≤ 5%

**Heuristic floors:**
- Layer 0 (embedding) and last layer (lm_head): always ≥ 16-bit (f16)
- Attention Q/K projections: ≥ 4-bit
- FFN layers 0..num_layers/10 (shallow): ≥ 4-bit
- FFN layers in the bulk: can reach 2-bit if quality holds

### Surgical Pruning

Domains to prune are specified via `--domains-drop`:

```bash
tpt-gpu-model-optimizer optimize model.gguf --domains-drop sql,typescript,sql --max-loss 0.05
```

This zeros neurons associated with unwanted domains while preserving model integrity.

### Calibration Generation

Calibration samples are domain-specific prompts generated to expose quantization weaknesses:

- SQL: Window function queries, joins, complex aggregations
- TypeScript: Generic types, mapped tuples, conditional types
- Python: Async patterns, concurrent processing, error handling
- Math: Equation solving, symbolic manipulation, proofs
- Science: Quantum mechanics, thermodynamics, statistical mechanics

If an AI provider is configured (Claude/OpenRouter/Ollama), prompts are generated using `AiProvider::generate()`. Otherwise, heuristic prompts are used.

## API Reference

### Core Types

#### `LayerSensitivityMap`

Ranks layers by quantization sensitivity using perplexity delta.

```rust
use tpt_model_optimizer::{LayerSensitivityMap, SensitivityConfig};

let config = SensitivityConfig {
    model_path: PathBuf::from("model.gguf"),
    samples: calibration_samples,
    eval_tokens: 32,
    group_size: 128,
};

let sensitivity = LayerSensitivityMap::build(32, &config)?;
```

#### `DomainMapper`

Maps neurons to knowledge domains using activation analysis.

```rust
use tpt_model_optimizer::DomainMapper;

let mapper = DomainMapper::with_default_domains();
let domain_map = mapper.build(32, 11008)?; // 32 layers, FFN dim 11008
```

#### `MixedPrecisionAllocator`

Allocates per-layer bit depths within quality budget.

```rust
use tpt_model_optimizer::{MixedPrecisionAllocator, QuantEvalConfig};

let allocator = MixedPrecisionAllocator::new(0.05); // 5% loss budget
let config = QuantEvalConfig::default();

let bits = allocator.allocate(32, &sensitivity, &config, |layer, bits| {
    // Your evaluation function here
    Ok(10.0 * (1.0 + 0.15 * (8 - bits) as f32 / 8.0))
})?;
```

### Export Functions

#### GGUF Export

```rust
use tpt_model_optimizer::{GgufExporter, GgufExportConfig};

GgufExporter::export(&GgufExportConfig {
    source: PathBuf::from("optimized.tptf"),
    dest: PathBuf::from("model_q4.gguf"),
    group_size: 128,
})?;
```

Bit depth mapping: 2→Q2_K, 3→Q3_K_M, 4→Q4_K_M, 6→Q6_K, 8→Q8_0, 16→F16

#### EXL2 Export

```rust
use tpt_model_optimizer::{Exl2Exporter, Exl2ExportConfig};

Exl2Exporter::export(&Exl2ExportConfig {
    source: PathBuf::from("optimized.tptf"),
    dest_dir: PathBuf::from("exl2_output"),
    group_size: 128,
})?;
```

Outputs `config.json`, `quant_config.json`, and `model.safetensors`.

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for calibration prompt generation |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `TPT_GPU_UUID` | GPU UUID for hardware profile caching |
| `TPT_TENSOR_CORE_GEN` | Tensor core generation (volta/turing/ampere/ada) |
| `TPT_VRAM_TOTAL_MB` | Total VRAM in MiB |
| `TPT_VRAM_FREE_MB` | Free VRAM in MiB |

### Provider Configuration

The `tpt-shared` crate provides a unified `AiProvider` trait:

```rust
use tpt_shared::{AiProvider, ProviderFactory};

// Create from environment
let provider = ProviderFactory::from_env()?;

// Or create explicitly
let claude = ProviderFactory::claude("sk-ant-...");
let openrouter = ProviderFactory::openrouter("sk-or-...");
let ollama = ProviderFactory::ollama();
```

## Performance Tuning

### Hardware Profiles

Hardware profiles are cached at `~/.tpt/hardware_profile.json`. The profiler measures:

- Memory bandwidth via large buffer copy benchmarks
- L2 cache size via bandwidth-vs-size knee point
- Tensor core generation via compute capability
- VRAM total/free via NVML/ROCm queries

### Community Tuning

Community-submitted GPU profiles can be added to `tuning/<gpu_model>.json`:

```json
{
  "gpu_name": "RTX 4090",
  "gpu_uuid": "NVIDIA-...",
  "bw_gbps": 1008.0,
  "l2_mb": 72.0,
  "tensor_core_gen": "ada",
  "vram_total_mb": 24564,
  "vram_free_mb": 20000,
  "optimal_tile_m": 128,
  "optimal_tile_n": 128,
  "optimal_tile_k": 64
}
```

### Benchmark Integration

Run benchmarks with the TPT benchmark tool:

```bash
# Quick sanity check (30 seconds)
tpt bench --quick

# Full benchmark suite
tpt bench

# Contribute results
tpt bench --contribute
```

## Troubleshooting

### Common Issues

**"No AI provider available"**
- Set `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` environment variable
- Or start the Ollama server locally

**"Model too large for VRAM"**
- Use `--stream` flag for models > 80% of free VRAM
- The optimizer processes layers one at a time via mmap

**"Quality budget exceeded"**
- Increase `--max-loss` value (default 0.05)
- Try different calibration samples
- Check that the model architecture is supported

## Contributing

Bug reports and feature requests (new GPU tuning profiles, calibration generator extensions, export formats) are welcome via GitHub Issues. We also accept pull requests — see [CONTRIBUTING.md](../../../CONTRIBUTING.md) for how to contribute.

## License

Dual-licensed under MIT or Apache 2.0 with LLVM Exception (Express Patent Grant)