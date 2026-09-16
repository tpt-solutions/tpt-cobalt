# tpt-lsp

The TPT Script language server (LSP 3.17 over stdio), superseding the
forked `tpt-gpu-script-lsp` for Cobalt's language — built directly on
`tpt-lang` (same supersession pattern as the REPL).

## Features

- **Diagnostics** from the same static checker the runtime uses: unit
  mismatches (`m + s`) and impossible matmul shapes are reported as compile
  errors on save/change, before anything runs.
- **Hover** — variable values with tensor shapes, function signatures
  (`<function circle(r)>`), module member listings, unit-tagged numbers.
  Dotted paths resolve into the module (`geom.circle` hovers as the
  function).
- **Completions** — globals, natives, language keywords, and (for dotted
  prefixes) module members, all reflecting the interpreted state of the
  document *above the cursor*: each request runs the prefix in a fresh,
  side-effect-free interpreter, so results are state-accurate rather than
  name-guessed.

## Architecture

- `lib.rs` — pure analysis (`analyze`, `hover_at`, `completions_at`),
  fully unit-tested, no I/O. Editor-independent; usable from tests, the
  notebook, or other front-ends.
- `server.rs` — thin tower-lsp adapter (full-document sync, hover,
  completions) and the stdio entry point.

```sh
cargo run -p tpt-lsp --bin tpt-lsp   # then point your editor at it
```

## Testing

```sh
cargo test -p tpt-lsp
```

Covers diagnostics (unit, shape, syntax), prefix-aware completions
(globals, keywords, natives, module members, unknown bases), hover value /
signature / module / tensor rendering, and state recovery from
error-containing prefixes.

## Limitations (documented, deliberate)

- Diagnostics are position-coarse (the AST does not carry source offsets
  yet) — messages are exact, spans default to the document start.
- The prefix interpreter re-runs on each request; notebook-scale files are
  the target, not million-line modules.
- No refactorings/formatting (see `tpt-gpu-script-format` upstream for the
  old language; a TPT-Script formatter is future work).
