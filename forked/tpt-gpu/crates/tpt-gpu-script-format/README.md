# tpt-gpu-script-format

TPT Script formatter and linter — canonical pretty-printer and style enforcer for `.tpts` source files.

## Overview

`tpt-gpu-script-format` consumes a `tpt-gpu-script-core` AST and re-emits normalized TPT Script source. It is the backend for the `tpt fmt` subcommand and the LSP formatting provider.

## Usage

```toml
[dependencies]
tpt-gpu-script-format = "0.1"
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
