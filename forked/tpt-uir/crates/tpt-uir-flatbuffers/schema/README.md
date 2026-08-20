# TPT-UIR FlatBuffers schema

The `tpt_uir.fbs` schema mirrors the `tpt-uir-core` IR
(`Operation` / `Block` / `Region` / `Type` / `Attribute` / `OpName` /
`Dimension` / `QuantizationParams`) as a zero-copy, memory-mappable layout.

## Regenerating the bindings

The Rust bindings checked in under `../src/generated/tpt_uir_generated.rs` are
produced with **flatc 24.3.25** and are *not* regenerated in CI (codegen is
manual, not via `build.rs`). To regenerate after editing the schema:

```bash
flatc --rust -o ../src/generated tpt_uir.fbs
```

Pin the exact tool version above: the generated code uses flatc-specific
idioms (e.g. `UnionWIPOffset`, `as_union_value`) and must be produced by a
matching compiler. Do not bump the flatc version without re-verifying the
conversion in `src/convert.rs`.

## Design notes

- FlatBuffers unions can only hold tables, so scalar attribute payloads
  (`i64`, `f64`, `string`) and the `Type` / `Shape` / `OpName` /
  `Quantization` payloads are wrapped in small wrapper tables (the
  "wrapper-table + union" pattern).
- `Dimension` is a single table with a `kind` discriminant plus the union of
  fields used by its three variants.
- `Type` is a union over `ScalarTypeWrapper`, `TensorType`, and `IndexMarker`.
- The `QuantizationParams` table stores `scales` as `uint` with a `has_scales`
  flag (mirroring `Option<u32>`).
