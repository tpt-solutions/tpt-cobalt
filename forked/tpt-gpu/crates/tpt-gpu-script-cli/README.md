# tpt-gpu-script-cli

Command-line interface for the TPT Script compiler — the `tpt` binary (also installed as `tpt-gpu-script`).

## Overview

`tpt-gpu-script-cli` exposes `tpt-gpu-script-core`'s full pipeline (lex → parse → type-check → codegen) as a command-line tool. It compiles `.tpts` source files to TPTIR or Rust output.

`cargo install tpt-gpu-script-cli` installs **both** the `tpt` and `tpt-gpu-script` binaries, so either name works.

## Installation

```sh
cargo install tpt-gpu-script-cli
```

## Usage

```sh
tpt build my_kernel.tpts        # compile to TPTIR
tpt check my_kernel.tpts        # type-check only
tpt fmt my_kernel.tpts          # format in-place
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
