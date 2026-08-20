# TPT Model Registry

Shared GGUF model registry used across tpt-gpu, tpt-spark, and tpt-crucible. Models are downloaded once to `~/.tpt/models/` and never duplicated — any tool that needs a model opens the registry rather than fetching its own copy. See [`MODELS_REGISTRY.md`](../../MODELS_REGISTRY.md) at the repo root for the manifest format.

## Prerequisites

- Rust toolchain (see repo root [`README.md`](../../README.md#quick-start))

## Installation

```bash
cd crates/tpt-gpu-model-registry
cargo build --release
```

## Usage

```bash
# List all registered models
cargo run --bin tpt-models -- list

# Show the registry directory path (~/.tpt/models/)
cargo run --bin tpt-models -- dir

# Remove a model entry from the manifest (does not delete the underlying file)
cargo run --bin tpt-models -- remove <name>
```

## Using it from code

```rust
use tpt_gpu_model_registry::ModelRegistry;

let mut registry = ModelRegistry::open()?;
```

`ModelRegistry::open()` and HuggingFace download support (`hf.rs`) are consumed directly by other tools in this repo, e.g. [`tpt-gpu-model-optimizer`](../tpt-gpu-model-optimizer/) — see `crates/tpt-gpu-runtime/src/arch.rs` for how the runtime maps a registered GGUF model's `general.architecture` to a forward-pass template.
