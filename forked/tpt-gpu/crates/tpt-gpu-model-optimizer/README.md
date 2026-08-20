# tpt-gpu-model-optimizer

Hardware-aware LLM model optimizer — quantization, surgical pruning, and TPTF export for TPT GPU targets.

## Overview

`tpt-gpu-model-optimizer` loads GGUF models from the shared `~/.tpt/models/` registry, applies quantization (Q4, Q8, FP16) and optional layer pruning, and exports the result in a format ready for `tpt-gpu-runtime` inference. It uses AI-guided heuristics (via `tpt-gpu-shared`) to suggest optimal quantization strategies per target hardware.

## Installation

```sh
cargo install tpt-gpu-model-optimizer
```

## Usage

```sh
tpt-gpu-model-optimizer --model llama3-8b --quant q4_k_m --target tpt
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
