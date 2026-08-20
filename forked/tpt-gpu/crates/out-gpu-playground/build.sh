#!/usr/bin/env bash
# Build the TPT Playground WASM package.
# Requires: wasm-pack (https://rustwasm.github.io/wasm-pack/installer/)
# Install:  cargo install wasm-pack
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v wasm-pack &>/dev/null; then
    echo "wasm-pack not found — installing via cargo..."
    cargo install wasm-pack
fi

echo "Building WASM package..."
wasm-pack build --target web --out-dir pkg --release

echo ""
echo "Build complete. Serve the playground with:"
echo "  python3 -m http.server 8080"
echo "  then open http://localhost:8080 in your browser"
