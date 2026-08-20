# tpt-uir-flatbuffers

[![CI](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Zero-copy (memory-mappable) **FlatBuffers** format for the **TPT Unified
Intermediate Representation (TPT-UIR)**.

This crate is the format to reach for when you want true zero-copy access: a
serialized buffer can be read directly via [`root_as_region`] without decoding
into owned `Operation` / `Region` values. For a compact, self-describing
copy-based format, see [`tpt-uir-serde`](../tpt-uir-serde).

This crate is `no_std` by default; the `mmap` feature adds memory-mapped file
access (and implies `std`).

## Installation

Not yet on crates.io — depend on it via git or a path within this workspace:

```toml
[dependencies]
tpt-uir-flatbuffers = { git = "https://github.com/tpt-solutions/tpt-uir", package = "tpt-uir-flatbuffers", features = ["mmap"] }
```

## Example

```rust
use tpt_uir_core::{OpName, Region, Block, Operation, validate_region};
use tpt_uir_flatbuffers::{to_flatbuffer, from_flatbuffer, root_as_region};

let region = Region {
    blocks: vec![Block {
        arguments: vec![(0u32, tpt_uir_core::Type::Index)],
        operations: vec![Operation {
            id: 1,
            op_name: OpName::parse("core.add").unwrap(),
            operands: vec![0],
            results: vec![1],
            regions: vec![],
            attributes: vec![],
        }],
    }],
};

let bytes = to_flatbuffer(&region);

// Zero-copy view without decoding:
let view = root_as_region(&bytes).unwrap();
assert_eq!(view.blocks().unwrap().iter().count(), 1);

// Full decode round-trip:
let back = from_flatbuffer(&bytes).unwrap();
assert_eq!(region, back);
assert!(validate_region(&back).is_ok());
```

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
