$ErrorActionPreference = "Stop"
$srcRoot = "D:\Programming\1PRODUCTION\Open Source"
$dstRoot = "D:\Programming\1PRODUCTION\Open Source\tpt-cobalt\forked"
$excludesDirs = @("target", ".git", "node_modules")
$roboArgs = @("/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/XD") + $excludesDirs

# Per-repo include filter. If a repo is absent, include ALL its crates.
# Return $true to EXCLUDE a crate, $false to include.
$filters = @{
    "tpt-crucible" = { param($c) $c -notin @("tpt-catalyst","tpt-alloy","tpt-crucible-uir-adapter") }
    "tpt-fem"      = { param($c) $c -in @("tpt-fem-py","tpt-fem-cli") }
    "tpt-physics"  = { param($c) $c -eq "tpt-phys-gallery" }
    "tpt-telos"    = { param($c) $c -in @("vscode-telos","playground") }
    "tpt-uir"      = { param($c) $c -in @("tpt-uir-examples","tpt-uir-cli") }
    # tpt-math, tpt-gpu, tpt-engineering, tpt-formal: include everything
}

# Extra top-level dirs to copy for repos that keep docs/tooling (no target/)
$extraDirs = @{
    "tpt-gpu" = @("docs","scripts","tools","tuning","v","layer1_isa","layer2_tptd","layer3_tptc",
                  "layer4_tptr","layer5_tptp","layer6_framework","layer7_tptb")
}

function Copy-Crate($repo, $crate, $dstBase) {
    $src = Join-Path $srcRoot "$repo\crates\$crate"
    $dst = Join-Path $dstBase "crates\$crate"
    if (-not (Test-Path $src)) { Write-Warning "MISSING crate $repo/$crate"; return }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    & robocopy $src $dst @roboArgs | Out-Null
}
function Copy-RootFiles($repo, $dstBase) {
    New-Item -ItemType Directory -Force -Path $dstBase | Out-Null
    foreach ($f in @("Cargo.toml","Cargo.lock","LICENSE-APACHE","LICENSE-MIT","rust-toolchain.toml",
                     "rustfmt.toml","deny.toml","README.md","AGENTS.md","CLAUDE.md","SECURITY.md",
                     "CONTRIBUTING.md","spec.txt")) {
        $sf = Join-Path $srcRoot "$repo\$f"
        if (Test-Path $sf) { Copy-Item -Force $sf $dstBase }
    }
}
function Copy-Dir($repo, $dir, $dstBase) {
    $src = Join-Path $srcRoot "$repo\$dir"
    if (-not (Test-Path $src)) { return }
    $dst = Join-Path $dstBase $dir
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    & robocopy $src $dst @roboArgs | Out-Null
}
function Set-Members($dstBase, $crates) {
    $toml = Join-Path $dstBase "Cargo.toml"
    $content = Get-Content $toml -Raw
    $lines = ($crates | ForEach-Object { '    "crates/' + $_ + '",' }) -join "`n"
    $new = "members = [`n$lines`n]"
    $content = [regex]::Replace($content, '(?s)members\s*=\s*\[.*?\]', $new,
        [System.Text.RegularExpressions.RegexOptions]::Singleline)
    Set-Content -NoNewline -Path $toml -Value $content
}

$repos = @("tpt-math","tpt-gpu","tpt-crucible","tpt-fem","tpt-physics","tpt-engineering",
           "tpt-science","tpt-formal","tpt-telos","tpt-rust6","tpt-uir")

foreach ($repo in $repos) {
    $srcCrates = Join-Path $srcRoot "$repo\crates"
    if (-not (Test-Path $srcCrates)) { Write-Warning "no crates dir for $repo"; continue }
    $all = @(Get-ChildItem -Directory -Force $srcCrates | ForEach-Object { $_.Name })
    $filter = $filters[$repo]
    if ($filter) {
        $inc = $all | Where-Object { -not (& $filter $_) }
    } else {
        $inc = $all
    }
    $dst = Join-Path $dstRoot $repo
    foreach ($c in $inc) { Copy-Crate $repo $c $dst }
    Copy-RootFiles $repo $dst
    if ($extraDirs[$repo]) { foreach ($d in $extraDirs[$repo]) { Copy-Dir $repo $d $dst } }
    Set-Members $dst $inc
    Write-Output "$repo : $($inc.Count) crates"
}

Write-Output "FORK COMPLETE"
