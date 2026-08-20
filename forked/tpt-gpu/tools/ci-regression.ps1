# CI Regression Detector — blocks merge if any kernel's efficiency drops > 5%.
#
# Standalone script (no GitHub Actions / workflows required).
#
# Usage:
#   .\tools\ci-regression.ps1                  # Run against stored baseline
#   .\tools\ci-regression.ps1 -Threshold 3.0   # Custom threshold
#   .\tools\ci-regression.ps1 -Quick           # Quick mode (faster)
#   .\tools\ci-regression.ps1 -UpdateBaseline  # Regenerate the baseline first
#
# Exit codes:
#   0 = PASS (no regressions)
#   1 = BLOCK (regression detected)
#   2 = Error

param(
    [double]$Threshold = 5.0,
    [switch]$Quick,
    [switch]$UpdateBaseline,
    [string]$BaselineFile = "baseline_report.json",
    [string]$CurrentFile = "current_report.json"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CI Regression Detector" -ForegroundColor Cyan
Write-Host "  Threshold: > $Threshold% efficiency drop = BLOCK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build the project
Write-Host "[1/4] Building out-gpu-primitives-benches..." -ForegroundColor Yellow
cargo build -p out-gpu-primitives-benches --example ci_regression --release 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 2
}
Write-Host "      Build OK." -ForegroundColor Green

$ciArgs = if ($Quick) { "--quick" } else { "" }

# Step 2: (Optional) Update baseline
if ($UpdateBaseline) {
    Write-Host "[2/4] Generating baseline report..." -ForegroundColor Yellow
    $outputArg = "--output `"$BaselineFile`""
    $cmd = "cargo run -p out-gpu-primitives-benches --example ci_regression --release -- --baseline $ciArgs $outputArg"
    Write-Host "      Running: $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Baseline generation failed!" -ForegroundColor Red
        exit 2
    }
    Write-Host "      Baseline saved to: $BaselineFile" -ForegroundColor Green
} else {
    Write-Host "[2/4] Skipping baseline generation (use -UpdateBaseline to regenerate)" -ForegroundColor DarkGray
}

# Step 3: Check baseline file exists
if (-not (Test-Path $BaselineFile)) {
    Write-Host "ERROR: Baseline file '$BaselineFile' not found!" -ForegroundColor Red
    Write-Host "       Run with -UpdateBaseline first, or provide a baseline file." -ForegroundColor Yellow
    exit 2
}

# Step 4: Run current benchmarks and compare
Write-Host "[3/4] Running current benchmarks..." -ForegroundColor Yellow
$outputArg = "--output `"$CurrentFile`""
$thresholdArg = "--threshold $Threshold"
$cmd = "cargo run -p out-gpu-primitives-benches --example ci_regression --release -- --baseline-file `"$BaselineFile`" $ciArgs $outputArg $thresholdArg"
Write-Host "      Running: $cmd"
Invoke-Expression $cmd
$exitCode = $LASTEXITCODE

# Step 5: Report result
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[4/4] RESULT: PASS - No regressions detected." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    exit 0
} elseif ($exitCode -eq 1) {
    Write-Host "[4/4] RESULT: BLOCK - Regression detected! Merge blocked." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
} else {
    Write-Host "[4/4] RESULT: ERROR - Benchmark or comparison failed." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 2
}
