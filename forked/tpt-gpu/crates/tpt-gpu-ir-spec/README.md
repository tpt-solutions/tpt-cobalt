# tpt-gpu-ir-spec

Machine-readable specification for the TPTIR compiler IR — operations, types, attributes, and the stable text-format serialization used across the TPT GPU stack.

## Overview

TPTIR (TPT Intermediate Representation) is an SSA-based, MLIR-compatible IR used to express GPU compute kernels at a hardware-agnostic level. This crate provides the canonical Rust types for the TPTIR dialect: operation names, operand kinds, type variants, and the attribute schema that the text serializer and deserializer agree on.

## Usage

```toml
[dependencies]
tpt-gpu-ir-spec = "0.1"
```

Enable Serde support:

```toml
tpt-gpu-ir-spec = { version = "0.1", features = ["serde"] }
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
