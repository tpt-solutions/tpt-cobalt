# tpt-lsp

The TPT Script language server (LSP 3.17 over stdio), superseding the forked
`tpt-gpu-script-lsp` for Cobalt's language — built directly on `tpt-lang`.

- **Diagnostics** from the same static checker the runtime uses: unit
  mismatches (`m + s`) and impossible matmul shapes are compile errors.
- **Hover**: variable values with tensor shapes, function signatures
  (`<function circle(r)>`), module member listings.
- **Completions**: globals, natives, keywords, and — for dotted prefixes —
  module members, all reflecting the real interpreted state of the document
  *above the cursor*.

Analysis logic is pure and unit-tested (`tpt_lsp::{analyze, hover_at,
completions_at}`); `server.rs` is the tower-lsp adapter. Run with
`cargo run -p tpt-lsp --bin tpt-lsp` and point an editor at it.
