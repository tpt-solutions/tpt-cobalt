# Build the TPT Playground WASM package.
# Requires: wasm-pack (https://rustwasm.github.io/wasm-pack/installer/)
# Install:  cargo install wasm-pack

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $script:dir

try {
    if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
        Write-Host "wasm-pack not found. Installing via cargo..." -ForegroundColor Yellow
        cargo install wasm-pack
    }

    Write-Host "Building WASM package..." -ForegroundColor Cyan
    wasm-pack build --target web --out-dir pkg --release

    Write-Host ""
    Write-Host "Build complete. Serve the playground with:" -ForegroundColor Green
    Write-Host "  python -m http.server 8080" -ForegroundColor White
    Write-Host "  then open http://localhost:8080 in your browser" -ForegroundColor White
} finally {
    Pop-Location
}
