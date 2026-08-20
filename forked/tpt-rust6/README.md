# TPT — The Rust-Native Scientific Stack (`tpt-rust6`)

> **Leapfrog, don't replicate.** Every crate does something Python fundamentally cannot.

TPT is a purely native, Rust-first scientific computing, mathematics, and AI stack.
It refuses to wrap Python and refuses to replicate Python: it exploits Rust's unique
superpowers — Wasm, fearless concurrency, zero-cost abstractions, and compile-time
codegen — to make Python's architectural flaws obsolete.

The full design lives in [`spec.txt`](spec.txt). This repository tracks the roadmap in
[`TODO.md`](TODO.md).

## The 12 Crates

| # | Crate | What it does that Python cannot | Phase | Status |
|---|-------|--------------------------------|-------|--------|
| 1 | **tpt-omni** | One Arrow-backed memory layout for tables, tensors, N-D arrays, and sparse matrices with zero-copy views | 1 | **implemented** |
| 2 | **tpt-io** | A single `read()` for every format, auto-detected by extension + magic bytes | 1 | **implemented** |
| 3 | tpt-grad | Autograd via procedural macros — gradients emitted at compile time, zero runtime graph | 2 | implemented |
| 4 | tpt-stat | Unified distributions, hypothesis tests, regression, and Bayesian inference (incl. NUTS), SIMD + Rayon-parallel | 2 | implemented |
| 5 | tpt-sym | Compile-time computer-algebra: typed symbols, simplification, differentiation, LaTeX, unit-aware arithmetic | 2 | partial |
| 6 | tpt-dag | In-process concurrent pipeline executor: `#[task]`, `pipeline![]`, content-addressed caching, retries | 3 | implemented |
| 7 | tpt-viz | Grammar-of-graphics plotting with level-of-detail; GPU/Wasm/3D rendering pending | 3 | partial |
| 8 | tpt-learn | High-level ML training with compile-time tensor shape verification; sweeps/export pending | 3 | implemented |
| 9 | tpt-ui | Serverless Wasm dashboards via `#[tpt_app]`, deployable as a single HTML file; `cargo tpt-serve` pending | 4 | partial |
| 10 | tpt-lab | Reactive, strictly-typed notebook with an automatic dependency DAG | 4 | implemented |
| 11 | tpt-script | Pythonic scripting layer: `tpt script file.tpt` with zero ceremony | 5 | implemented |
| 12 | tpt-doc | Type-checked documents with compile-time reference/citation verification; EPUB/Wasm/HarfBuzz pending | 5 | partial |

> **Status legend:** **implemented** = core API shipped and tested; **partial** = core shipped but some roadmap features still pending (see `TODO.md`); **scaffolded** = skeleton only. `tpt-io` is the most format-limited of the "implemented" crates: it reads CSV/JSON/Parquet/Arrow-IPC (HDF5 behind `--features hdf5`) and writes CSV/JSON only — FITS/NetCDF/Zarr/Excel/SQLite readers and PNG/PDF writers are not yet present.

## Quickstart

```rust
use tpt_io::prelude::*;
use tpt_omni::prelude::*;

let frame = read("data.csv")?;                 // auto-detect, infer schema
let adults = frame.as_table().filter(&col("age").ge(18))?;
let tensor = frame.as_tensor::<f64>("score", &[frame.num_rows()])?;
let normalized = (&tensor - tensor.mean()) / tensor.std();   // auto-broadcasting
let head = Tensor::from_view(slice![normalized, 0..5]);     // NumPy-style slice
```

Run the example:

```bash
cargo run -p example-01-quickstart
```

## Workspace Layout

```
crates/      # the 12 TPT crates
examples/   # runnable examples (one per milestone)
docs/       # mdBook documentation
benches/    # Criterion benchmarks vs. Python equivalents
```

## Benchmarks vs. Python

The "leapfrog Python" claim is backed by evidence, not assertion. `benches/`
runs the same numerical kernels on the TPT side (Criterion) and, with
`TPT_COMPARE_PYTHON=1`, against a numpy/pandas baseline
(`benches/python_baseline/ops.py`). See [`benches/README.md`](benches/README.md)
for the methodology and a results table.

```bash
cargo bench -p tpt-benches                              # TPT side only
TPT_COMPARE_PYTHON=1 cargo bench -p tpt-benches --bench python_cmp  # + numpy/pandas
```

## Building

```bash
cargo build --workspace
cargo test  --workspace
```

## License

Dual-licensed under MIT or Apache-2.0 (see [`LICENSE`](LICENSE)).
