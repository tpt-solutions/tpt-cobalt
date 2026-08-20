#!/usr/bin/env bash
# TPT GPU — Rust quickstart bootstrap
# Covers: check cargo, build tpt-gpu-script-cli, print next steps.
# Does NOT bootstrap C++, SystemVerilog, or driver toolchains.

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

info()  { echo -e "${GREEN}[setup]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[setup]${RESET} $*"; }
fatal() { echo -e "${RED}[setup]${RESET} ERROR: $*" >&2; exit 1; }

# ── 1. Check Rust toolchain ───────────────────────────────────────────────────
info "Checking Rust toolchain..."
if ! command -v cargo >/dev/null 2>&1; then
    fatal "cargo not found. Install Rust from https://rustup.rs/ then re-run this script."
fi

RUST_VERSION=$(rustc --version | awk '{print $2}')
REQUIRED="1.75.0"
if [ "$(printf '%s\n' "$REQUIRED" "$RUST_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
    fatal "Rust $REQUIRED+ required, found $RUST_VERSION. Run: rustup update"
fi
info "Rust $RUST_VERSION found."

# ── 2. Build tpt-gpu-script-cli ──────────────────────────────────────────────
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_ROOT"

info "Building tpt-gpu-script-cli (this may take a minute on first run)..."
cargo build --release -p tpt-gpu-script-cli

# Locate the binary
BINARY="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['target_directory'])" \
    2>/dev/null || echo "target")/release/tpt-gpu-script"

if [ ! -f "$BINARY" ]; then
    # Fallback: search the workspace target dir
    BINARY="$(find "$SCRIPT_ROOT/target/release" -maxdepth 1 -name "tpt-gpu-script" 2>/dev/null | head -1)"
fi

if [ -f "$BINARY" ]; then
    info "Build successful: $BINARY"
else
    warn "Build completed but binary location is unknown. Check target/release/."
fi

# ── 3. Quick sanity check ─────────────────────────────────────────────────────
if [ -f "$BINARY" ]; then
    info "Running: $BINARY --help"
    "$BINARY" --help 2>&1 | head -5 || true
fi

# ── 4. Next steps ─────────────────────────────────────────────────────────────
echo ""
info "Setup complete. Next steps:"
echo "  1. Run your first TPT Script program:"
echo "       echo '@requires_gpu(true) fn hello() {}' > hello.tpts"
echo "       ./target/release/tpt-gpu-script check hello.tpts"
echo "  2. Build the runtime:"
echo "       cargo build -p tpt-gpu-runtime"
echo "  3. Install the Python framework (Layer 6):"
echo "       cd layer6_framework && pip install -e '.[dev]'"
echo "  4. Read the tutorials: docs/tutorials/README.md"
echo ""
