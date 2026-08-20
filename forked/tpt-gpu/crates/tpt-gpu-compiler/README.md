# tpt-gpu-compiler — Rust Port of TPTIR Compiler Stack

Rust port of the TPTIR compiler stack: IR types, optimization passes, and
native codegen, as the migration target from the C++ `tptc` compiler.

## Build

```bash
# From the repo root (single root Cargo workspace)
cargo build -p tpt-gpu-compiler
cargo test -p tpt-gpu-compiler
```

## Strategy

The port proceeds incrementally, keeping the C++ `tptc` authoritative until each
piece is replaced:

1. FFI bindings to C++ `tptc`
2. Native Rust IR + parser
3. Native Rust passes
4. Native Rust codegen
5. Complete Rust migration

## Layout

- `ir.rs` — Core IR types (`Type`, `Operation`, `Block`, `Region`)
- `passes.rs` — Optimization passes (DCE, constant fold, vectorize)
- `lib.rs` — `compile_native(source, target)` where `target` is `"tptisa"` or `"llvmir"`
- `ffi.rs` — C API boundary to the C++ `tptc` compiler (active during early phases)
- `fusion.rs`, `validate.rs`, `provenance.rs`, `tuning.rs`, `dispatch.rs`, `bench.rs`

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
