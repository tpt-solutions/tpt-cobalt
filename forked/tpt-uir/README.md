# TPT-UIR

**TPT Unified Intermediate Representation (TPT-UIR)** — a compact, `no_std`-friendly
intermediate representation for machine-learning compilers and hardware
accelerators in the TPT stack.

[![CI](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> **Status:** pre-release (`0.1.0`), not yet published to crates.io. APIs may
> still change between minor versions; see each crate's `CHANGELOG.md`.

TPT-UIR models SSA region/block/operation graphs with a small, typed attribute
system, and ships first-class serialization in three formats:

- **postcard** — compact, self-describing binary (`tpt-uir-serde`)
- **FlatBuffers** — zero-copy, memory-mappable binary (`tpt-uir-flatbuffers`)
- **text** — a readable, round-trippable textual format (`tpt-uir-text`)

A CLI (`tpt-uir-cli`) and a C FFI (`tpt-uir-ffi`) are provided for inspection and
integration.

## Contents

- [Architecture](#architecture)
- [Crates](#crates)
- [Installation](#installation)
- [Dialects & versioning](#dialects--versioning)
- [Downstream consumers](#downstream-consumers)
- [Getting started](#getting-started)
- [Quality gates](#quality-gates)
- [License](#license)

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              tpt-uir-core                    │
                 │  types · attr · op_name · ir · builder ·     │
                 │  quant · validate_region (SSA)               │
                 └───────────────┬─────────────────────────────┘
                                 │
        ┌───────────────┬────────┴───────────┬──────────────────┐
        │               │                    │                  │
┌───────▼──────┐ ┌──────▼───────┐   ┌────────▼────────┐  ┌──────▼──────┐
│ tpt-uir-     │ │ tpt-uir-     │   │ tpt-uir-       │  │ tpt-uir-    │
│ serde        │ │ flatbuffers  │   │ text           │  │ ffi         │
│ (postcard)   │ │ (zero-copy)  │   │ (textual)      │  │ (C ABI)     │
└───────┬──────┘ └──────┬───────┘   └────────┬────────┘  └──────┬──────┘
        └───────────────┴─────────┬──────────┴──────────────────┘
                                  │
                       ┌──────────▼──────────┐
                       │   tpt-uir-dialects   │
                       │ GPU · Crucible ·     │
                       │ Memory · ValidateDialect
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │   tpt-uir-cli        │
                       │   tpt-uir-examples   │
                       └──────────────────────┘
```

`dialects` sits on top of `core` and defines the GPU, Crucible, and Memory
dialects plus structural validation (`ValidateDialect`). The serialization,
text, and FFI crates depend only on `core` (and, where relevant, `dialects`).

## Crates

| Crate | Purpose | Docs |
| --- | --- | --- |
| `tpt-uir-core` | Core IR data structures, builder API, quantization helpers, SSA validation (`no_std`). | [README](crates/tpt-uir-core/README.md) · [CHANGELOG](crates/tpt-uir-core/CHANGELOG.md) |
| `tpt-uir-dialects` | GPU / Crucible / Memory dialects and dialect-level validation. | [README](crates/tpt-uir-dialects/README.md) · [CHANGELOG](crates/tpt-uir-dialects/CHANGELOG.md) |
| `tpt-uir-serde` | Compact postcard (de)serialization (`no_std`; `std` for file I/O). | [README](crates/tpt-uir-serde/README.md) · [CHANGELOG](crates/tpt-uir-serde/CHANGELOG.md) |
| `tpt-uir-flatbuffers` | Zero-copy FlatBuffers layout + `mmap` access (`no_std`). | [README](crates/tpt-uir-flatbuffers/README.md) · [CHANGELOG](crates/tpt-uir-flatbuffers/CHANGELOG.md) |
| `tpt-uir-text` | Hand-written textual lexer/parser/printer. | [README](crates/tpt-uir-text/README.md) · [CHANGELOG](crates/tpt-uir-text/CHANGELOG.md) |
| `tpt-uir-ffi` | `cdylib`/`rlib`/`staticlib` C ABI for read-only inspection. | [README](crates/tpt-uir-ffi/README.md) · [CHANGELOG](crates/tpt-uir-ffi/CHANGELOG.md) |
| `tpt-uir-cli` | `validate` / `print` / `convert` command-line tool. | [README](crates/tpt-uir-cli/README.md) · [CHANGELOG](crates/tpt-uir-cli/CHANGELOG.md) |
| `tpt-uir-examples` | Usage examples and end-to-end tests (not published). | [README](crates/tpt-uir-examples/README.md) |

## Installation

These crates are not yet on crates.io. Until the first publish, depend on them
directly from git (pin a commit or tag for reproducible builds):

```toml
[dependencies]
tpt-uir-core = { git = "https://github.com/tpt-solutions/tpt-uir", package = "tpt-uir-core" }
tpt-uir-dialects = { git = "https://github.com/tpt-solutions/tpt-uir", package = "tpt-uir-dialects" }
```

Working inside this workspace (e.g. from a sibling crate or a fork), use path
dependencies instead:

```toml
[dependencies]
tpt-uir-core = { path = "../tpt-uir/crates/tpt-uir-core" }
```

Building the CLI from source:

```sh
git clone https://github.com/tpt-solutions/tpt-uir
cd tpt-uir
cargo build --release -p tpt-uir-cli
./target/release/tpt-uir-cli --help
```

## Dialects & versioning

Each dialect exposes a `current_version()` via the `ValidateDialect` trait.
Producers pin the targeted version with a `dialect_version` (or per-op
`op_dialect_version`) attribute — both stored as `i64`, so they add **no new
wire surface**. The migration story is additive: new capabilities bump the
version, older readers reject or degrade instead of misinterpreting the IR, and
the postcard wire format never renumbers existing enum variants.

## Downstream consumers

- **`tpt-gpu`** — ingests TPTIR (incl. GGUF-loaded Llama-3 blocks) into TPT-UIR
  via `tpt-gpu-uir-adapter`; wired into `tpt-gpu-kernelgen` and `tpt-gpu-runtime`.
- **`tpt-crucible`** — ingests `ComputationalGraph`s into TPT-UIR via
  `tpt-crucible-uir-adapter` (static `Fixed` shapes).
- **`tpt-telos`** — formal memory-bound proofs over TPT-UIR regions via
  `tpt-telos-uir-bridge` (Z3 / Fourier-Motzkin).

## Getting started

**1. Build a region and validate it.** Regions contain blocks, blocks contain
operations — the same shape whether you're modeling an SSA function or a
dataflow graph (see [spec.txt](spec.txt) for the full data model).

```rust
use tpt_uir_core::builder::{BlockBuilder, OpBuilder, RegionBuilder};
use tpt_uir_core::OpName;
use tpt_uir_dialects::{GpuDialect, ValidateDialect};

let region = RegionBuilder::new()
    .block(BlockBuilder::new().op(
        OpBuilder::new(1)
            .name(OpName::parse("tpt_gpu.launch").unwrap())
            .result(2)
            .build(),
    ).build())
    .build();

tpt_uir_core::validate_region(&region).unwrap(); // SSA well-formedness
GpuDialect::validate_all(&region).unwrap();      // GPU dialect invariants
```

**2. Serialize it.** Pick postcard for compact storage, FlatBuffers for
zero-copy reads, or text for something you can diff and hand-edit:

```rust
let bytes = tpt_uir_serde::serialize_region(&region).unwrap();      // postcard
let fb    = tpt_uir_flatbuffers::to_flatbuffer(&region);            // flatbuffers
let text  = tpt_uir_text::to_text(&region);                         // text
```

**3. Inspect it from the command line.** Write a region to a `.tptuir`
(postcard) file, then:

```sh
tpt-uir-cli validate region.tptuir --dialect gpu
tpt-uir-cli print region.tptuir
tpt-uir-cli convert --in region.tptuir --out region.uir --to-format text
```

For runnable, end-to-end versions of the above (including round-trips through
all three formats and the C FFI), see
[`crates/tpt-uir-examples/`](crates/tpt-uir-examples/README.md) and
[`crates/tpt-uir-cli/README.md`](crates/tpt-uir-cli/README.md) for the full CLI
reference.

## Quality gates

`cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`, `cargo test --doc --workspace`, and
`cargo build --workspace --no-default-features` are all run on every push by the
[CI workflow](.github/workflows/ci.yml).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
