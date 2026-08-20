# TPT Playground

A browser-based, in-memory playground for TPT Script. Paste or write a `.tpts` program, compile it in the browser (no server round-trip — the layer7 compiler runs as WASM), and inspect the generated TPTIR, generated Rust, a rough perf estimate, and any type/constraint errors.

## What it does

- Live editor with line numbers and Ctrl+Enter / auto-compile (600ms debounce)
- Example picker preloaded with a ReLU kernel, GEMM, scaled dot-product attention, a mixed host+GPU function, Conv2D, and a deliberate shape-mismatch error demo
- Four output tabs: **TPTIR** (syntax-highlighted), **Rust** (syntax-highlighted), **Perf Estimate** (simulated GFLOPs/time/memory via TPT SimGPU), **Errors** (with fix suggestions where available)

It runs entirely client-side — the `tpt-gpu-script-core` compiler (lexer → parser → type checker → codegen) is compiled to WASM via `wasm-bindgen` and loaded by `index.html`.

## Hosted playground

A hosted copy is published to GitHub Pages on every push to `master` (see `.github/workflows/docs.yml`):

**https://tpt-solutions.github.io/tpt-gpu/playground/**

The steps below are only needed when you want to build and serve the playground locally.

## Prerequisites

- Rust toolchain (see repo root `README.md`)
- `wasm-pack` — installed automatically by the build scripts below if missing (`cargo install wasm-pack`)
- Any local HTTP server to serve static files (Python's `http.server` is used in the examples below)

## Build & run

```bash
cd crates/out-gpu-playground

# Linux/macOS
./build.sh

# Windows
./build.ps1
```

This runs `wasm-pack build --target web --out-dir pkg --release`, producing `pkg/tpt_playground.js` and the accompanying `.wasm` binary that `index.html` imports.

Then serve the directory and open it in a browser:

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

`index.html` must be served over HTTP (not opened via `file://`) — browsers block ES module + WASM loading from the filesystem.

## Usage

1. Pick an example from the dropdown, or write your own TPT Script in the left pane.
2. It compiles automatically as you type (or press Ctrl+Enter / click **Run**).
3. Switch between the **TPTIR**, **Rust**, **Perf Estimate**, and **Errors** tabs to inspect the output.

See [`docs/user-guide.md`](../../docs/user-guide.md) for the full TPT Script language reference and [`layer7_tptb/examples/`](../../layer7_tptb/examples/) for larger worked examples than fit comfortably in the playground.
