# TPT GPU — Rust quickstart bootstrap (PowerShell)
# Covers: check cargo, build tpt-gpu-script-cli, print next steps.
# Does NOT bootstrap C++, SystemVerilog, or driver toolchains.

$ErrorActionPreference = "Stop"

function Info  { param($msg) Write-Host "[setup] $msg" -ForegroundColor Green  }
function Warn  { param($msg) Write-Host "[setup] $msg" -ForegroundColor Yellow }
function Fatal { param($msg) Write-Host "[setup] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ── 1. Check Rust toolchain ───────────────────────────────────────────────────
Info "Checking Rust toolchain..."
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Fatal "cargo not found. Install Rust from https://rustup.rs/ then re-run this script."
}

$rustVersion = (rustc --version) -replace "rustc (\S+).*",'$1'
$required    = [System.Version]"1.75.0"
if ([System.Version]$rustVersion -lt $required) {
    Fatal "Rust $required+ required, found $rustVersion. Run: rustup update"
}
Info "Rust $rustVersion found."

# ── 2. Build tpt-gpu-script-cli ──────────────────────────────────────────────
$ScriptRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ScriptRoot

Info "Building tpt-gpu-script-cli (this may take a minute on first run)..."
cargo build --release -p tpt-gpu-script-cli

$binary = Join-Path $ScriptRoot "target\release\tpt-gpu-script.exe"
if (Test-Path $binary) {
    Info "Build successful: $binary"
} else {
    Warn "Build completed but binary not found at $binary. Check target\release\."
}

# ── 3. Quick sanity check ─────────────────────────────────────────────────────
if (Test-Path $binary) {
    Info "Running: $binary --help"
    & $binary --help 2>&1 | Select-Object -First 5
}

# ── 4. Next steps ─────────────────────────────────────────────────────────────
Write-Host ""
Info "Setup complete. Next steps:"
Write-Host "  1. Run your first TPT Script program:"
Write-Host "       echo '@requires_gpu(true) fn hello() {}' > hello.tpts"
Write-Host "       .\target\release\tpt-gpu-script.exe check hello.tpts"
Write-Host "  2. Build the runtime:"
Write-Host "       cargo build -p tpt-gpu-runtime"
Write-Host "  3. Install the Python framework (Layer 6):"
Write-Host "       cd layer6_framework; pip install -e '.[dev]'"
Write-Host "  4. Read the tutorials: docs\tutorials\README.md"
Write-Host ""
