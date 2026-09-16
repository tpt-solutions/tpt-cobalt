import os

ROOT = r"crates"

def w(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(content.lstrip("\n"))
    print("wrote", full)

# =============================== tpt-approx ================================
w("tpt-approx/README.md", """
# tpt-approx

Clean-room floating-point approximate-comparison macros for the
`tpt-cobalt` workspace: `relative_eq!` / `abs_diff_eq!` predicates plus
`assert_*` forms with `epsilon = ...` / `max_relative = ...` named
arguments.

Floating-point results should almost never be compared with `==`: rounding
makes exact equality wrong for transitive math (`0.1 + 0.2 != 0.3`) and for
any reordered but mathematically identical expression. This crate gives the
workspace one small, dependency-free vocabulary for "equal up to rounding".

## Features

- **Two predicates** — `relative_eq!(a, b)` scales the tolerance with the
  magnitude of the operands; `abs_diff_eq!(a, b)` uses an absolute
  tolerance. Both return `bool` and compose in conditions, not just
  assertions.
- **Assert forms** — `assert_relative_eq!` / `assert_abs_diff_eq!` panic
  with a message showing both operands on failure, so test failures are
  readable.
- **Named tolerances** — `epsilon = 1e-12`, `max_relative = 1e-9` style
  arguments, with sensible defaults (machine epsilon and a small multiple).
- **Macro-only, zero dependencies** — nothing to pull into a dependency
  tree; the crate compiles nothing but macro definitions.
- **Clean-room** — implemented from the mathematics of floating-point
  comparison, not copied from any existing crate (no code, doc text, or
  API surface was transcribed from third-party sources).

## Usage

```rust
use tpt_approx::{assert_relative_eq, assert_abs_diff_eq, relative_eq};

let a: f64 = 0.1 + 0.2;
assert!(relative_eq!(a, 0.3));                       // default tolerances
assert_relative_eq!(a, 0.3, epsilon = 1e-12);
assert_abs_diff_eq!(3.0_f64.sqrt(), 1.7320508075688772, epsilon = 1e-12);
```

## Testing

```sh
cargo test -p tpt-approx
```

## Status

Stable at 0.1.x. Part of the first-party `tpt-cobalt` workspace; consumed
by the numerical test suites of the scientific crates.
""")

w("tpt-approx/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- `relative_eq!` and `abs_diff_eq!` predicates with `epsilon` /
  `max_relative` named-argument tolerances and machine-epsilon defaults.
- `assert_relative_eq!` / `assert_abs_diff_eq!` macro forms with
  operand-rendering failure messages.
- README with usage, feature list, and the clean-room provenance note.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-approx-v0.1.0
""")

# ============================== tpt-columnar ===============================
w("tpt-columnar/README.md", """
# tpt-columnar

The clean-room columnar data engine of the `tpt-cobalt` workspace: typed
arrays, schemas, record batches, compute kernels, display helpers, and a
native self-describing container format ("TPTC") — with **zero
dependencies**.

## Features

- **Typed arrays** — `PrimitiveArray<T>` (`Int32Array`, `Int64Array`,
  `UInt32Array`, `Float32Array`, `Float64Array`), `BooleanArray`,
  `StringArray`, and `BinaryArray` behind a type-erased `Array` trait
  (`ArrayRef = Arc<dyn Array>`) so batches can hold mixed column types.
- **Schemas as data** — `Schema`, `Field`, `DataType` describe every batch;
  `RecordBatch::try_new` validates column count, types, and lengths.
- **Compute kernels** — `filter`, `take`, `and`, `or`, `not`, and
  `concat_batches` operate on `ArrayRef`s (boolean-mask and index-list
  selection, logical combination, concatenation).
- **Native TPTC container** — a self-describing file format
  (`ipc::FileWriter` / `ipc::FileReader`) for persisting batches; the
  format `tpt-hub`'s Arrow-IPC bridge aligns with.
- **Display helpers** — `array_value_to_string` and friends for
  human-readable table rendering (notebook/REPL output).
- **Clean-room, dependency-free** — implemented from the columnar data
  model itself; no code, formats, or specifications were copied from any
  third-party project.

## Usage

```rust
use std::sync::Arc;
use tpt_columnar::prelude::*;

let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::Int32, false),
    Field::new("score", DataType::Float64, false),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])),
    ],
)?;

// select rows where score > 1.0
let mask = ...; // BooleanArray from a compute/comparison kernel
let taken = take(batch.column(0), &mask)?;
```

See the crate docs (`cargo doc -p tpt-columnar --open`) for the full
kernel and IPC surface.

## Testing

```sh
cargo test -p tpt-columnar
```

## Status

Stable at 0.1.x. Serves as the in-workspace columnar foundation for
`tpt-hub`'s serialization/IPC story; a native Rust foundation for the
tabular side of the ML stack.
""")

w("tpt-columnar/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Typed arrays (`PrimitiveArray` family, `BooleanArray`, `StringArray`,
  `BinaryArray`) behind the type-erased `Array` trait / `ArrayRef`.
- Schemas and validation: `Schema`, `Field`, `DataType`,
  `RecordBatch::try_new`.
- Compute kernels: `filter`, `take`, `and`, `or`, `not`, `concat_batches`.
- Native TPTC container format (`ipc::FileWriter` / `ipc::FileReader`).
- Display helpers for human-readable rendering.
- README with feature list, usage, and clean-room provenance note.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-columnar-v0.1.0
""")

# ============================ tpt-fusion README ============================
w("tpt-fusion/README.md", """
# tpt-fusion

The Fusion (FPGA) backend of TPT Cobalt, written natively in Rust — no port
of the upstream Python module, and no raw-bitstream path (declined in
`docs/phase5-exotic-backends.md`). The deliverable is **artifact emission**:
Catalyst IR in, HLS-C++ kernel sources plus a vendor toolchain manifest out,
with a **memory-fit proof computed before anything is emitted**.

## Features

- **HLS kernel emission** — deterministic, byte-stable tiled GEMM kernels
  (m_axi interfaces, `ram_2p` local buffers, `PIPELINE II=1` inner compute,
  shapes fixed from IR node attributes).
- **On-chip memory-fit proof** — per-kernel buffer accounting (A/B tiles
  double-buffered, C accumulating in f32) checked against the device budget
  *before* emission; over-budget kernels are errors with a per-buffer
  breakdown, never silently emitted.
- **Global-memory proofs (UIR bridge)** — `build_manifest_proved` lowers
  the model's real allocations (weights, activations, symbolic batch dims)
  to TPT-UIR `tpt_memory.alloc` ops and refuses to emit unless the
  Fourier-Motzkin proof shows every admissible assignment fits the device's
  global memory. Counterexamples name the overflowing assignment.
- **Vendor toolchain manifests** — Xilinx `v++` compile + link command
  lines and a `manifest.json` written out with the kernel sources; other
  vendors error honestly instead of emitting pretend commands.
- **Honest scope** — unsupported ops are reported together and skipped
  never; missing shape attributes name the node; raw bitstream synthesis is
  explicitly out of scope (vendor place-and-route is not reimplementable
  honestly).

## Usage

```rust,ignore
use tpt_fusion::{build_manifest, build_manifest_proved, DeviceConfig, Vendor, TileConfig, DType};

let device = DeviceConfig::new("xilinx.platform", Vendor::Xilinx,
                               /* on-chip */ 1 << 20, /* global */ 16 << 20, 300.0);
let tile = TileConfig { m: 32, n: 32, k: 32, double_buffer: true };

// tile-fit only
let manifest = build_manifest(&ir, &device, tile, DType::F32)?;

// tile-fit + global memory-bound proof over the real model
let (manifest, proof) = build_manifest_proved(&ir, &device, tile, DType::F32, &activations)?;
manifest.write_out(&out_dir)?; // manifest.json + <kernel>.cpp
```

## Testing

```sh
cargo test -p tpt-fusion
```

Covers buffer accounting, fit rejection with breakdowns, emission
determinism, IR lowering with unsupported-op/missing-attribute errors,
manifest JSON round-trips, file output, and the proof gate (valid, symbolic
quantification, counterexample witnesses, real-model allocations).

## Status

0.1.x, first slice per `docs/phase5-exotic-backends.md`. Remaining:
conv/attention kernels, an Intel toolchain template, and sourcing the
device budget from live toolchain reports.
""")

w("tpt-fusion/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-14

### Added
- Deterministic HLS-C++ tiled GEMM kernel emission from Catalyst IR
  (`emit_hls_gemm`), with m_axi interfaces, local `ram_2p` buffers, and a
  pipelined inner loop.
- On-chip memory-fit accounting and checking (`check_memory_fit`,
  `FitReport` with per-buffer breakdown); over-budget kernels are hard
  errors.
- Vendor toolchain manifest (`build_manifest`): Xilinx `v++` per-kernel
  compile + link commands, `manifest.json` + kernel sources via
  `ToolchainManifest::write_out`, JSON round-trip.
- Honest error surface: `UnsupportedOps` (reported together),
  `MissingShape` (names the node), `UnsupportedVendor`.

## [0.1.1] - 2026-09-15

### Added
- `proof` module: TPT-UIR lowering of model allocations
  (`AllocTensor`, `Dim::Bounded`, `alloc_region`) and Fourier-Motzkin
  memory-bound proofs via `tpt-telos-uir-bridge`.
- `build_manifest_proved`: emission gated on a `Valid` global-memory proof
  over kernel operands plus caller activations; counterexample witnesses
  surface as `ProofFailed` with the overflow arithmetic.
- `allocs_from_module`: real-model allocations from any `tpt-ml::Module`.
- `DeviceConfig::global_mem_bytes` (the proof's budget).

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-fusion-v0.1.0
[0.1.1]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-fusion-v0.1.1
""")

# ========================= tpt-alloy-deploy README =========================
w("tpt-alloy-deploy/README.md", """
# tpt-alloy-deploy

The Alloy (MCU swarm) deploy layer of TPT Cobalt, written natively in Rust.
The forked `tpt-alloy` crate partitions models and generates per-node
firmware sources; upstream had *"no deployment mechanism"* — this crate is
the missing deploy half. Protocol knowledge only; no port of
tpt-basestation's Python (see `docs/phase5-exotic-backends.md`).

## Features

- **RP2040 UF2 generation** (`uf2`) — raw flash images to UF2 blocks
  (USB mass-storage drop format): spec byte layout
  (magics `0x0A324655`/`0x9E5D5157`/`0x0AB16F30`, family `0xE48BFF56`,
  flags `0x2000`, 256-byte chunks zero-padded, seq/total counters),
  known-vector tested.
- **ESP32 ROM-UART protocol** (`esp`) — SLIP encode/decode with escape
  handling, 9-byte-header command packets (Sync / FlashBegin / FlashData /
  FlashEnd, XOR-0xEF data checksums), and `EspFlasher`, a flash driver
  over the `Transport` trait: a real serial backend is one impl away, and
  tests drive an in-memory mock device (success, error-status, and
  multi-packet sequences).
- **Fleet OTA** (`ota`) — sha256-digested `NodeArtifact`s keyed by node id
  + `tpt_alloy::FirmwareTarget`; manifest validation rejects duplicate
  node ids, foreign release ids, and digest tampering; two-phase staged
  rollout (stage every node in id order, then a commit gate) with
  abort-on-failure and skip modes; JSON export strips image bytes but
  keeps digests.

## Usage

```rust,ignore
use tpt_alloy_deploy::{to_uf2, UF2_FAMILY_RP2040, Rollout};

let uf2 = to_uf2(&image, 0x1000_0000, UF2_FAMILY_RP2040);

let mut rollout = Rollout::begin(manifest)?;
while let Some(node) = rollout.next_node() {
    /* stage via your transport */
    rollout.mark_staged();
}
rollout.commit()?;
```

## Testing

```sh
cargo test -p tpt-alloy-deploy
```

17 tests cover the UF2 block layout against the spec, SLIP round-trips and
escape edge cases, protocol packet layout field-by-field, the flash
sequence over a mock transport (chunking, sequence numbers, header
checksums, reboot flag), device-error propagation, and the rollout state
machine.

## Status

0.1.x, first slice. Remaining (optional): a real `serialport` transport
implementation and signed release bundles.
""")

w("tpt-alloy-deploy/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-14

### Added
- RP2040 UF2 block generation from raw flash images (`to_uf2`), spec-verified
  byte layout with known-vector tests.
- ESP32 ROM-UART SLIP framing, command packets (Sync/FlashBegin/FlashData/
  FlashEnd) with XOR-0xEF checksums, and `EspFlasher` over a `Transport`
  trait; in-memory mock transport drives the tests.
- Fleet OTA: sha256-digested `NodeArtifact`s, `UpdateManifest` validation
  (duplicates, foreign releases, digest tampering), staged two-phase
  `Rollout` state machine with abort-on-failure and skip modes, and
  image-stripping JSON export.

### Fixed
- Response parsing reads the data section after the full 9-byte header
  (direction 1 + command 2 + size 2 + checksum 4); an earlier off-by-one
  happened to pass the first status checks.
- `Rollout::begin` now actually stages nodes in ascending node-id order.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-alloy-deploy-v0.1.0
""")

# ============================= tpt-lsp README ==============================
w("tpt-lsp/README.md", """
# tpt-lsp

The TPT Script language server (LSP 3.17 over stdio), superseding the
forked `tpt-gpu-script-lsp` for Cobalt's language — built directly on
`tpt-lang` (same supersession pattern as the REPL).

## Features

- **Diagnostics** from the same static checker the runtime uses: unit
  mismatches (`m + s`) and impossible matmul shapes are reported as compile
  errors on save/change, before anything runs.
- **Hover** — variable values with tensor shapes, function signatures
  (`<function circle(r)>`), module member listings, unit-tagged numbers.
  Dotted paths resolve into the module (`geom.circle` hovers as the
  function).
- **Completions** — globals, natives, language keywords, and (for dotted
  prefixes) module members, all reflecting the interpreted state of the
  document *above the cursor*: each request runs the prefix in a fresh,
  side-effect-free interpreter, so results are state-accurate rather than
  name-guessed.

## Architecture

- `lib.rs` — pure analysis (`analyze`, `hover_at`, `completions_at`),
  fully unit-tested, no I/O. Editor-independent; usable from tests, the
  notebook, or other front-ends.
- `server.rs` — thin tower-lsp adapter (full-document sync, hover,
  completions) and the stdio entry point.

```sh
cargo run -p tpt-lsp --bin tpt-lsp   # then point your editor at it
```

## Testing

```sh
cargo test -p tpt-lsp
```

Covers diagnostics (unit, shape, syntax), prefix-aware completions
(globals, keywords, natives, module members, unknown bases), hover value /
signature / module / tensor rendering, and state recovery from
error-containing prefixes.

## Limitations (documented, deliberate)

- Diagnostics are position-coarse (the AST does not carry source offsets
  yet) — messages are exact, spans default to the document start.
- The prefix interpreter re-runs on each request; notebook-scale files are
  the target, not million-line modules.
- No refactorings/formatting (see `tpt-gpu-script-format` upstream for the
  old language; a TPT-Script formatter is future work).
""")

w("tpt-lsp/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
- Pure analysis layer: `analyze` (checker + syntax diagnostics),
  `hover_at` (values with tensor shapes, function signatures, module
  listings, dotted-path resolution), `completions_at` (state-aware
  globals/keywords/natives/module members).
- tower-lsp stdio server (`tpt-lsp` binary) with full-document sync,
  hover, and completion handlers.
- README with architecture, testing, and documented limitations.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-lsp-v0.1.0
""")

# ============================ tpt-bench README =============================
w("tpt-bench/README.md", """
# tpt-bench

The Cobalt benchmark suite (Phase 7): criterion micro-benchmarks over the
core kernels, a deterministic wall-clock report generator, and the fixed
protocol that makes PyTorch comparisons meaningful.

## Benchmarks

- `benches/matmul.rs` — f64 matmul at 64x64, 128x128, 256x256 with
  element-throughput tracking.
- `benches/ml.rs` — Linear forward+backward through the autograd tape
  (128x256x128), transformer-block forward ([B, T, D] = [2, 8, 64],
  4 heads), and the full TPT-Script `train_step` interpreter path
  (forward -> MSE -> backward -> AdamW, batch 16).

## Report generator

```sh
cargo run --release -p tpt-bench --bin tpt-bench-report -- 200
cargo bench -p tpt-bench
```

`tpt-bench-report` measures the same kernels with a fixed rule (10 untimed
warmup runs, then the timed loop) and prints a Markdown table plus JSON.
Kernel inputs are seeded with a deterministic LCG, so reports are
reproducible for a given binary.

## PyTorch comparison

[benches/PYTORCH_PROTOCOL.md](benches/PYTORCH_PROTOCOL.md) fixes the
matching recipe — same machine, same shapes, dtypes, warmup and iteration
counts, thread budget recorded — and lists the by-design differences that
must be quoted when publishing (notably: the Cobalt `train_step` includes
interpreter dispatch overhead; the suite is the CPU path). The PyTorch
half is a ~40-line reference script kept out of this repo (no Python in
the workspace).

## Testing

```sh
cargo test -p tpt-bench
```

Runs the full fixed suite in fast mode and validates both renderers.

## Status

0.1.x, first slice. Candidates for the next cut: conv benchmarks, the
differentiable-science kernels (`tpt-sci`), and WGPU/CUDA cross-device
numbers as a separate comparison.
""")

w("tpt-bench/CHANGELOG.md", """
# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
- Criterion benchmarks: f64 matmul (64/128/256), Linear forward+backward
  through the tape, transformer-block forward, and the TPT-Script
  `train_step` interpreter path.
- `tpt-bench-report` wall-clock report generator (Markdown + JSON,
  deterministic LCG-seeded inputs, 10-run warmup rule).
- `benches/PYTORCH_PROTOCOL.md`: the fixed PyTorch comparison recipe and
  the by-design differences to quote.
- README with benchmark list, usage, and status.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-bench-v0.1.0
""")

# ============================ manifest patches =============================
def patch_manifest(path, additions):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for key, value in additions.items():
        if key + " =" in src:
            continue
        # insert after the description line
        marker = "description = "
        idx = src.index(marker)
        end = src.index("\n", idx)
        src = src[: end + 1] + f'{key} = {value}\n' + src[end + 1 :]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src)
    print("patched", path)

patch_manifest(
    "tpt-approx/Cargo.toml",
    {
        "readme": '"README.md"',
        "keywords": '["floating-point", "comparison", "assert", "testing", "numerical"]',
        "categories": '["development-tools::testing", "science", "mathematics"]',
    },
)
patch_manifest(
    "tpt-columnar/Cargo.toml",
    {
        "readme": '"README.md"',
        "keywords": '["columnar", "arrays", "ipc", "dataframe", "batch"]',
        "categories": '["database-implementations", "data-structures", "encoding"]',
    },
)

print("done")
