# tpt-gpu-script-lsp

Language Server Protocol implementation for TPT Script — powers IDE features (hover, completion, diagnostics, formatting) in any LSP-compatible editor.

## Overview

`tpt-gpu-script-lsp` wraps `tpt-gpu-script-core` (parsing, type checking) and `tpt-gpu-script-format` (formatting) behind a `tower-lsp` server. It runs as a standalone binary and communicates over stdio.

## Usage

```toml
[dependencies]
tpt-gpu-script-lsp = "0.1"
```

Or install the binary:

```sh
cargo install tpt-gpu-script-lsp
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
