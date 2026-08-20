# UPSTREAM.md — Fork provenance for TPT Cobalt

This file records, for each forked pillar, the source repository and the exact commit
Cobalt forked from. Cobalt's copies are allowed to diverge on purpose (one-way fork);
this table is **provenance, not a to-do list** — it answers "where did this fork start"
if anyone needs to understand a difference later. The original `tpt-solutions/*` repos
keep shipping independently; these are not kept in sync.

| Forked tree | Source repo | Forked at commit | Diverged how |
|---|---|---|---|
| `forked/tpt-math` | `tpt-solutions/tpt-math` | `b22b2b373d337c0e719ffc80a08a7a48cb09c525` | Boundary-only so far; concrete manifests |
| `forked/tpt-gpu` | `tpt-solutions/tpt-gpu` | `e346589a5ee4591486693ee520d7cdf3db7440aa` | Concrete manifests; all crate families kept |
| `forked/tpt-crucible` | `tpt-solutions/tpt-crucible` | `c3fcadf2e3a69ccd74d8e544e1b89f01789b52ca` | Only `tpt-catalyst` + `tpt-alloy` + `tpt-crucible-uir-adapter` |
| `forked/tpt-fem` | `tpt-solutions/tpt-fem` | `bcdf67b870c530f3b6a53710de0bc9d42e229950` | `tpt-fem-py`/`tpt-fem-cli`/`fuzz` excluded |
| `forked/tpt-physics` | `tpt-solutions/tpt-physics` | `2822dd9cfadbd632b824d04ed5be499d9df963d4` | `tpt-phys-gallery` excluded; paths repointed |
| `forked/tpt-engineering` | `tpt-solutions/tpt-engineering` | `69d6d62d294d6924342079d2eea8391c183d9a65` | `tpt-eng-crystallography`/`cli`/`examples` excluded |
| `forked/tpt-science` | `tpt-solutions/tpt-science` | `c3213aab6de1a8f03fe5a62fca3bc78793ebc3c0` | All 18 crates kept (incl. the 9 spec-named) |
| `forked/tpt-formal` | `tpt-solutions/tpt-formal` | `7aa176539cbe3fd6c9e9b30fd56ad35d64b32a07` | All 19 crates kept |
| `forked/tpt-telos` | `tpt-solutions/tpt-telos` | `426a4cad778a8c779b888701e6acc88efb3bf348` | 11 core crates; `vscode-telos`/`playground` excluded |
| `forked/tpt-rust6` | `tpt-solutions/tpt-rust6` | `3510ba455c4c13803a8f34b5036792d377eb7ea1` | Glue-crate starting points (`tpt-omni`,`tpt-grad`(+macro),`tpt-learn`,`tpt-io`,`tpt-script`) + their deps |
| `forked/tpt-uir` | `tpt-solutions/tpt-uir` | `e7a1756f5f23fbf09d50348881f1574b13eeddcd` | Core crates; `examples`/`cli` excluded (needed by uir-bridge) |

## How the fork was produced

1. Crate source was copied from each source repo's `crates/` (and, for `tpt-gpu`, its
   `docs/`, `layer1_isa`–`layer7_tptb`, `scripts/`, `tools/`, `tuning/`); `target/`,
   `.git/`, and repo-specific excluded dirs were never copied.
2. Each forked crate manifest was **concretized**: `*.workspace = true` inheritance was
   replaced with concrete values / relative `path` deps, so the whole tree is a single
   flat Cargo workspace (`members = ["crates/*", "forked/*"]` is *not* used — instead every
   crate is listed explicitly in the root `Cargo.toml`). This avoids nested-workspace
   conflicts and Cargo 1.97's "cannot override `default-features` on a `workspace = true`
   dependency" rule.
3. Cross-repo `../tpt-other/...` path deps were repointed to `../../tpt-other/...`.

Re-fork by re-running `scripts/fork.ps1` then `scripts/concretize.py` (idempotent).
