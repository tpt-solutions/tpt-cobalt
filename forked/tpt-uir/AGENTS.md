# AGENTS.md — TPT-UIR

Compact guidance for working in this repo. Verified against `Cargo.toml`,
`ci.yml`, and the crate `Cargo.toml`/`schema` files.

## Repo shape

- Virtual Cargo workspace (`resolver = "2"`), `members = ["crates/*"]`. Every
  crate lives under `crates/`.
- Crates (dependency order, lowest first):
  - `tpt-uir-core` — IR types, builder API, quant helpers, SSA validation.
    **Zero external dependencies**, `#![no_std]`.
  - `tpt-uir-dialects` — GPU / Crucible / Memory dialects + `ValidateDialect`.
    Depends only on `core`.
  - `tpt-uir-serde` (postcard), `tpt-uir-flatbuffers` (zero-copy), `tpt-uir-text`
    (hand-written lexer/parser/printer) — depend on `core` (and `dialects` where
    relevant).
  - `tpt-uir-ffi` — C ABI (`cdylib`/`rlib`/`staticlib`), read-only inspection only.
  - `tpt-uir-cli` — `validate`/`print`/`convert` tool.
  - `tpt-uir-examples` — `publish = false`; usage examples + e2e tests.

## Hard constraints (would break consumers if missed)

- **Postcard wire format is a contract.** Never renumber existing enum variants.
  Add new variants only at the *end* of each enum (see `AttributeValue::Quantization`
  appended last in `tpt-uir-core/src/attr.rs`). The sibling repos
  `tpt-gpu`/`tpt-crucible`/`tpt-telos` already consume this format; keep it stable.
- **`no_std` is required for `core`/`serde`/`flatbuffers`.** Don't add heap/std
  assumptions (except behind the optional `std`/`mmap` features). CI proves this
  with `cargo build --workspace --no-default-features`.
- **FlatBuffers codegen is manual**, not `build.rs` and not run in CI. Edit
  `crates/tpt-uir-flatbuffers/schema/tpt_uir.fbs`, then regenerate with
  **`flatc 24.3.25`** exactly:
  `flatc --rust -o ../src/generated tpt_uir.fbs`
  Do not bump the flatc version without re-verifying `src/convert.rs`.
- **New dialect capabilities use additive versioning**, not new wire surface:
  reuse `AttributeValue::I64` via `Attribute::dialect_version()` /
  `op_dialect_version()` and `ValidateDialect::current_version()`.

## Quality gates (run before pushing; these are exactly what CI runs)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --no-default-features -- -D warnings
cargo test --workspace
cargo build --workspace --no-default-features
```

- Run a single crate: `cargo test -p tpt-uir-core`, etc.
- Doc tests are real (README samples are `#![doc = include_str!]`-included):
  also run `cargo test --doc --workspace`.

## CLI

```bash
cargo run -p tpt-uir-cli -- validate <file> [--dialect gpu|crucible]
cargo run -p tpt-uir-cli -- print <file>
cargo run -p tpt-uir-cli -- convert --in <file> --out <file> [--from-format ...] [--to-format postcard|text|flatbuffers]
```
Format defaults from extension: `*.tptuir`→postcard, `*.txt`/`*.uir`→text,
`*.fbs`/`*.flat`→flatbuffers.

## Out of scope in this repo

Ingestion adapters (`tpt-gpu`/`tpt-crucible`) and the prover bridge (`tpt-telos`)
live in their own repos — do not add them here. Spec/design context: `spec.txt`;
implementation status: `todo.md`.
