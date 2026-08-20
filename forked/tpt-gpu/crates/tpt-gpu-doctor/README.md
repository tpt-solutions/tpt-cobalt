# TPT-GPU Doctor

Environment health check for TPT GPU. Verifies a machine is ready to build and
contribute to the project by mirroring the exact gates CI enforces, plus a few
optional informational checks.

## What it checks

- **Rust toolchain** — present and >= 1.75
- **`cargo fmt --all -- --check`** — same formatting gate CI runs
- **`cargo clippy --all-targets -- -D warnings`** — same lint gate CI runs
- **Vendor SDKs** (CUDA, ROCm, Metal) — informational only, not required
- **Python + `tptr` importability** — informational only, not required

Exit code is non-zero if any *required* check fails, so it doubles as a
pre-commit hook.

## Usage

```bash
cargo run -p tpt-gpu-doctor                  # full check (Rust + fmt/clippy + SDKs)
cargo run -p tpt-gpu-doctor -- --pre-commit  # just Rust + fmt + clippy
cargo run -p tpt-gpu-doctor -- --fast        # just the Rust toolchain
```

Each check reports `PASS`, `FAIL`, `WARN`, or `SKIP`.

## License

Dual-licensed under MIT or Apache-2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
