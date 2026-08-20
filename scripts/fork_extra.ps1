$ErrorActionPreference = "Stop"
$srcRoot = "D:\Programming\1PRODUCTION\Open Source"
$dstRoot = "D:\Programming\1PRODUCTION\Open Source\tpt-cobalt\forked"
$excludes = @("target", ".git", "node_modules")
$robo = @("/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/XD") + $excludes

function Copy-Crate($repo, $crate, $dstBase) {
    $src = Join-Path $srcRoot "$repo\crates\$crate"
    $dst = Join-Path $dstBase "crates\$crate"
    if (-not (Test-Path $src)) { Write-Warning "MISSING $repo/$crate"; return }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    & robocopy $src $dst @robo | Out-Null
}
function Copy-Root($repo, $dstBase) {
    New-Item -ItemType Directory -Force -Path $dstBase | Out-Null
    foreach ($f in @("Cargo.toml","Cargo.lock","LICENSE-APACHE","LICENSE-MIT","rust-toolchain.toml","rustfmt.toml")) {
        if (Test-Path (Join-Path $srcRoot "$repo\$f")) { Copy-Item -Force (Join-Path $srcRoot "$repo\$f") $dstBase }
    }
}
function Set-Members($dstBase, $crates) {
    $toml = Join-Path $dstBase "Cargo.toml"
    $content = Get-Content $toml -Raw
    $lines = ($crates | ForEach-Object { '    "crates/' + $_ + '",' }) -join "`n"
    $new = "members = [`n$lines`n]"
    $content = [regex]::Replace($content, '(?s)members\s*=\s*\[.*?\]', $new, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    Set-Content -NoNewline -Path $toml -Value $content
}

# --- fork tpt-uir (core crates only; exclude examples/cli) ---
$uir = @("tpt-uir-core","tpt-uir-dialects","tpt-uir-ffi","tpt-uir-flatbuffers","tpt-uir-serde","tpt-uir-text")
$dst = Join-Path $dstRoot "tpt-uir"
foreach ($c in $uir) { Copy-Crate "tpt-uir" $c $dst }
Copy-Root "tpt-uir" $dst
Set-Members $dst $uir

# --- expand tpt-rust6 subset with tpt-stat, tpt-viz ---
$r6 = @("tpt-omni","tpt-grad","tpt-grad-macro","tpt-learn","tpt-io","tpt-script","tpt-stat","tpt-viz")
$dst = Join-Path $dstRoot "tpt-rust6"
foreach ($c in @("tpt-stat","tpt-viz")) { Copy-Crate "tpt-rust6" $c $dst }
Set-Members $dst $r6

Write-Output "EXTRA FORK DONE"
