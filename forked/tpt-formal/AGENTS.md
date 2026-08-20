# AGENTS.md — tpt-formal

Rust Cargo workspace of small, mostly `no_std` formal-verification / design-by-contract crates.

## Commands

- Format gate: `cargo fmt --check` (`rustfmt.toml`: edition 2021, `max_width = 100`)
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace`
- `no_std` build (explicit crates only, see below):
  `cargo build --target thumbv6m-none-eabi --no-default-features -p tpt-for-typestate -p ...`
- CI sets `RUSTFLAGS=-D warnings`, so **any** compiler or clippy warning fails the pipeline.
  Run `cargo fmt --check` and the clippy command locally before pushing.

## Layout

- 21 member crates under `crates/`, each with its own `Cargo.toml`, `README.md`,
  `src/lib.rs`, and a descriptive example under `examples/` (registered via a
  uniquely-named `[[example]]` entry, e.g. `examples/sensor_fusion.rs` →
  `name = "redundancy_basic"`).
- `[workspace.package]` centralizes `license`, `edition`, `rust-version`; members
  inherit with `*.workspace = true`. Every crate is `license = "MIT OR Apache-2.0"`,
  `edition = "2021"`, `rust-version = "1.75"`.
- `rust-toolchain.toml` pins stable + `rustfmt`/`clippy`, targets
  `thumbv6m-none-eabi` and `wasm32-unknown-unknown`.

## Conventions an agent would miss

- **`no_std` crates opt out via features** (`default = ["std"]`, `std`, `alloc`);
  they are NOT built with `--workspace` (ADR 0001). The CI `no_std` job enumerates
  them with explicit `-p` flags. Adding a `no_std` crate requires: (a) the
  `std`/`alloc` feature set, (b) a new `-p` line in the CI `no_std` job, and
  (c) a row in the README crate table. Do not "simplify" to
  `cargo build --workspace --target thumbv6m-none-eabi`.
- **Internal dependency references are inconsistent**: some crates use
  `tpt-for-foo = { workspace = true }`, others a direct `{ path = "../tpt-for-foo" }`
  (and dev-deps do the same). Prefer `workspace = true`; every crate is already
  declared in `[workspace.dependencies]`.
- **Release profile keeps `overflow-checks = true`** (safety-critical; also `lto = "fat"`,
  `codegen-units = 1`, `strip = "symbols"`). Don't disable overflow checks.
- **`Cargo.toml` is the source of truth, not the README graph.** The README
  dependency graph can lag real edges (e.g. `tpt-for-smt-lite` actually depends on
  `tpt-for-sat`). When in doubt, read `Cargo.toml`.
- Authoritative design notes live in `spec.txt`; the per-task tracker is `todo.md`.

## Adding a crate

1. Add the path to `[workspace] members`.
2. Add it to `[workspace.dependencies]` (so dependents can use `workspace = true`).
3. Inherit workspace `license`/`edition`/`rust-version`; add a descriptive
   `examples/*.rs` with a uniquely-named `[[example]]` entry (repo norm: target
   names are unique, e.g. `redundancy_basic`, to avoid cargo filename collisions).
4. If `no_std`: add the `std`/`alloc` features, a `-p` line in the CI `no_std` job,
   and a README crate-table row.

## Hygiene

- `deny.toml` (cargo-deny) allow-lists permissive licenses; new external deps must
  fit the allow-list. `cargo-deny` and `cargo audit` both run in CI.
- Apache-2.0-ONLY wrap targets are acceptable as dependencies but should be noted
  in `deny.toml` (see its `licenses` note about `registry.toml` externally).
