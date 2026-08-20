# tpt-uir-text

[![CI](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-uir/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Human-readable **textual format** for the **TPT Unified Intermediate
Representation (TPT-UIR)**.

This crate provides a hand-written lexer and recursive-descent parser (no
`nom`/`pest`) plus a pretty-printer. It complements the binary formats in
[`tpt-uir-serde`](../tpt-uir-serde) and [`tpt-uir-flatbuffers`](../tpt-uir-flatbuffers).

## Installation

Not yet on crates.io — depend on it via git or a path within this workspace:

```toml
[dependencies]
tpt-uir-text = { git = "https://github.com/tpt-solutions/tpt-uir", package = "tpt-uir-text" }
```

## Format

- Blocks start with `^bb<n>` and may declare arguments: `^bb0(%0: tensor<f32, shape<1, n>>, %1: index):`
- Operations: `#<id>: %<result> = "dialect.op"(%operands) { attrs } [nested regions]`
- Types: `index`, `tensor<scalar, shape<dims>>`
- Shapes: `shape<d0, d1, ...>` where a dim is `128` (fixed), `n` (symbolic), or
  `m:1024` (bounded)
- Quantization: `quant<block_size, bytes_per_block, num_blocks, scales?>`
- Attributes: `key = value` with values `i64`, `f64`, `"string"`,
  `tensor<...>`, `shape<...>`, `opname("dialect.op")`, or `quant<...>`

## Example

```rust
use tpt_uir_core::{OpName, Region, Block, Operation, validate_region, Type};
use tpt_uir_text::{to_text, parse_text, Pretty};

let region = Region {
    blocks: vec![Block {
        arguments: vec![(0u32, Type::Index)],
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

let text = to_text(&region);
let back = parse_text(&text).unwrap();
assert_eq!(region, back);
assert!(validate_region(&back).is_ok());
```

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
