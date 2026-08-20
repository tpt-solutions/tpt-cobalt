# TPT GPU Tutorial Series

**Version:** 1.0 | **Last Updated:** June 2026

---

New to TPT GPU? Start with the root [`README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/README.md#quick-start) Quick Start (5 minutes: clone, build, compile your first `.tpts` file) before working through this series. The tutorials below are the deep-dive path once you're past hello-world.

## Tutorial Overview

| # | Title | Layer | Time | Topics |
|---|-------|-------|------|--------|
| 1 | Introduction & Setup | All | 30m | Architecture, installation |
| 2 | ISA Fundamentals | 1 | 45m | Instructions, pipeline |
| 3 | Kernel Drivers | 2 | 40m | Drivers, ioctls |
| 4 | TPTIR Overview | 3 | 35m | SSA form, operations |
| 5 | TPTIR Passes | 3 | 40m | DCE, vectorization |
| 6 | Memory Management | 4 | 50m | Allocators, hierarchy |
| 7 | Kernel Scheduling | 4 | 45m | Queues, events |
| 8 | GPU Primitives | 5 | 60m | GEMM, Attention |
| 9 | Python API | 6 | 40m | tptr from Python |
| 10 | PyTorch Integration | 6 | 50m | Custom ops, autograd |
| 11 | TPT Script Basics | 7 | 45m | Types, functions |
| 12 | TPT Script Kernels | 7 | 50m | GPU codegen |
| 13 | TPT Script Advanced | 7 | 55m | Introspection |
| 14 | End-to-End Workflow | All | 60m | Full pipeline |
| 15 | Building a Model | All | 75m | Transformer example |
| 16 | Performance Tuning | All | 60m | Profiling |
| 17 | Distributed Computing | All | 70m | Multi-GPU |

**Total:** ~17 hours

---

## Learning Paths

### Application Developer (Python)
1 → 9 → 10 → 11 → 14 (4 hours)

### Systems Programmer (Rust)
1 → 6 → 7 → 8 → 14 (4.5 hours)

### Compiler Engineer
1 → 2 → 4 → 5 → 11 → 13 → 14 (6 hours)

### Complete
All tutorials in order (17 hours)

---

## Build Commands

```bash
cargo build --release -p tpt-gpu-script-cli
cargo build -p tpt-gpu-runtime
cd layer6_framework && pip install -e ".[dev]"
```

## CLI Binary Name

The compiled binary is **`tpt-gpu-script`** (built with `cargo build -p tpt-gpu-script-cli`).
All tutorial command examples use `tpt` as a short alias — set it up once:

```bash
alias tpt=tpt-gpu-script          # bash / zsh
Set-Alias tpt tpt-gpu-script       # PowerShell
```

Or invoke the binary directly, e.g. `./target/release/tpt-gpu-script check file.tpts`.

## Common Operations

```bash
tpt check file.tpts
tpt compile file.tpts -o out/
tpt inspect matmul
tpt ops
python layer6_framework/examples/basic_usage.py
```

## Additional Resources

- **User Guide**: `docs/user-guide.md`
- **Examples**: `layer7_tptb/examples/`
- **Use Cases**: `docs/use-cases.md`
- **Browser Playground**: `crates/out-gpu-playground/` — try TPT Script live, no install required
- **Specifications**: `layer*/spec/`
- **Architecture**: `CLAUDE.md`

---

*Happy learning with TPT GPU!*
