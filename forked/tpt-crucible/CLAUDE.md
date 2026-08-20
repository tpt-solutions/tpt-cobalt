# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TPT Crucible is a hardware-agnostic AI compiler suite that compiles standard AI models (GGUF, ONNX, PyTorch, TensorFlow, SafeTensors, EXL2, AWQ/GPTQ, JAX, TFLite, and more) onto non-traditional hardware: FPGAs, analog compute circuits, microcontroller swarms, photonic MZI meshes, neuromorphic chips, and compute-in-memory arrays. It is **not** a GPU-targeting tool — it explicitly bypasses the GPU path.

## Model Formats

| Role | Format | Notes |
|------|--------|-------|
| **Input** | GGUF (primary) | Quantization-preserving ingestion; reads from Spark's model library |
| **Input** | ONNX, PyTorch `.pt`, TensorFlow SavedModel | Secondary ingestion paths |
| **Internal IR** | TPT-IR (`.tptir`) | Custom MLIR dialect; hardware-agnostic compiled graph |
| **Output** | TPT Package (`.tptpkg`) | ZIP container — the canonical deliverable |

**TPT Package (`.tptpkg`)** is a ZIP-compatible container bundling everything needed to deploy a model to custom hardware: TPT-IR, compiled artifacts per target (firmware, RTL bitstream, SPICE netlist), hardware profiles, pre-flight compatibility report, quantization profile, and Mosaic partition plan. All module outputs write into a `.tptpkg` — there are no loose output files. A package with only `ir/` and `compat/` is a valid "analyzed, not compiled" artifact.

```
model.tptpkg/
├── manifest.json         # version, model name, SHA-256 hashes
├── ir/model.tptir
├── targets/alloy/        # firmware binaries + topology + flash script
├── targets/fusion/       # bitstream + mem_init + board profile
├── targets/element/      # SPICE netlist + KiCad PCB + confidence score
├── compat/preflight.json
├── quant/quant_profile.json
└── mosaic/partition.json
```

## Directory Structure

All Python packages live under `python/`. The Go Observer service lives under `services/tpt-observer/`. Rust crates live under `crates/`. Cloud workers live under `cloud/`.

```
tpt-crucible/
├── python/
│   ├── tpt_catalyst/     # ingestion + IR compiler
│   ├── tpt_alloy/        # MCU swarm backend
│   ├── tpt_fusion/       # FPGA backend
│   ├── tpt_element/      # analog SPICE backend
│   ├── tpt_photon/       # photonic MZI mesh backend (experimental)
│   ├── tpt_pulse/        # neuromorphic ANN→SNN compiler
│   ├── tpt_silicon/      # compute-in-memory backend
│   ├── tpt_emulator/     # software-in-the-loop emulator
│   ├── tpt_mosaic/       # hybrid orchestrator
│   ├── tpt_drivers/      # driver SDK client
│   ├── tpt_fl/           # federated learning orchestration
│   ├── tpt_shell/        # interactive hardware REPL
│   └── tpt_train/        # training hooks → .tptprofile
├── services/tpt-observer/ # Go + Next.js dashboard
├── crates/               # Rust workspace (compiler backends, firmware gen)
├── cloud/                # self-hostable workers
│   ├── synthesis-worker/ # Go worker for Yosys+Nextpnr
│   ├── synthesis-broker/ # Redis-backed job broker
│   └── crucible-cloud/   # full pipeline web service
├── drivers/              # driver SDK + community registry
│   └── certification/    # driver signing/verification tools
├── validator/            # accuracy validation package
├── frontend/             # Next.js app (served by Observer)
└── tests/                # integration tests
```

## Architecture

The system follows a **Core + Modules** pattern:

1. **TPT Catalyst** (`python/tpt_catalyst/`) — Python ingestion layer + Rust compiler backend. Parses AI models and emits **TPT-IR**, a hardware-agnostic intermediate representation built as a custom [MLIR](https://mlir.llvm.org/) dialect. Uses Apache TVM for operator fusion. All other modules consume TPT-IR. Extended capabilities include: carbon-aware compilation, model provenance/lineage graph, AI-powered diagnostics (`tpt-catalyst doctor`), adaptive quantization, streaming preflight, model comparison and tournament ranking, community cache, marketplace, and natural-language config.

2. **TPT Alloy** (`python/tpt_alloy/`) — Swarm/microcontroller module. Partitions TPT-IR into sub-graphs using METIS/KaFFPa, then generates unique per-node C++/Rust firmware for each microcontroller (ESP32, RP2040, RISC-V). Integrates PlatformIO or Zephyr RTOS for flashing.

3. **TPT Fusion** (`python/tpt_fusion/`) — FPGA module. Takes TPT-IR and generates synthesizable RTL via Amaranth HDL (Python → Verilog). Wraps Yosys (synthesis) and Nextpnr (place-and-route). Uses LiteX/LiteDRAM for HBM controller auto-routing.

4. **TPT Element** (`python/tpt_element/`) — Analog compute module. Maps AI weights to physical components (resistors, memristors, op-amps). Runs SPICE simulation via Xyce/PySpice and uses a trained PyTorch model to predict thermal/noise drift fast. Outputs SPICE netlists and KiCad PCB files.

5. **TPT Photon** (`python/tpt_photon/`) *(experimental)* — Photonic compute backend. Maps TPT-IR weight matrices onto a Mach-Zehnder Interferometer (MZI) mesh. Generates MZI phase-shift configurations for photonic matrix-vector processors.

6. **TPT Pulse** (`python/tpt_pulse/`) — Neuromorphic compiler. Converts a standard ANN (from TPT-IR) to a Spiking Neural Network (SNN) using Leaky-Integrate-and-Fire (LIF) neurons. Exports spike schedules and weight maps for neuromorphic hardware targets.

7. **TPT Silicon** (`python/tpt_silicon/`) — Compute-in-Memory backend. Packs TPT-IR weight tensors into CIM array layouts, generates bitline operation sequences for in-memory matrix-vector multiply on CIM accelerators.

8. **TPT Observer** (`services/tpt-observer/`) — Unified dashboard. Go backend streams real-time telemetry via WebSockets. React/Next.js frontend with Three.js for 3D swarm topology/PCB visualization, React Flow for the Visual IR Graph Editor, telemetry replay engine (`.tptlog`), model compare, provenance graph viewer, and tournament leaderboard.

9. **TPT Emulator** (`python/tpt_emulator/`) — Software-in-the-Loop emulator for all hardware types. Alloy SiL: virtual N-node swarm. Fusion SiL: Verilator-backed cycle-accurate RTL simulation. Element SiL: Xyce/ngspice analog simulation. All emit the same telemetry schema as real hardware so Observer treats them identically.

10. **TPT Mosaic** (`python/tpt_mosaic/`) — Hybrid cross-hardware deployment orchestrator. Reads per-layer hardware annotations from TPT-IR, calls the appropriate module per partition, and generates inter-hardware communication glue (USB/UART/Ethernet bridges). Enables a single model to run across FPGA + Swarm + Analog simultaneously.

11. **TPT Drivers** (`drivers/`) — Hardware driver SDK and community registry. Defines a standardized driver interface (Rust trait + Python protocol) covering board identity, pin/resource map, synthesis constraints, telemetry adapter, and flash protocol. Includes a `probe/` submodule for USB auto-detection. The community registry is a public index of versioned, signed driver packages and verified compilation recipes.

12. **TPT FL** (`python/tpt_fl/`) — Federated learning orchestration. Splits a model across a live Crucible hardware deployment, runs local training rounds on each partition, and aggregates updates privately without exposing raw gradients.

13. **TPT Shell** (`python/tpt_shell/`) — Interactive hardware REPL. Provides a `tpt-shell` command for live introspection and ad-hoc tensor operations against a connected hardware target or SiL instance.

14. **tpt-train** (`python/tpt_train/`) — Standalone pip package providing PyTorch/JAX training hooks (`TPTProbeCallback`) that record per-layer activation ranges and weight distributions into a `model.tptprofile` file. Catalyst consumes `.tptprofile` to make better quantization decisions than static weight analysis alone.

15. **Cloud Workers** (`cloud/`) — Self-hostable infrastructure only; TPT provides Docker images and docs, does not operate these services.

16. **TPT Validator** (`validator/`) — Model accuracy validator. Connects to a live hardware deployment (or SiL) and a reference backend (Spark IPC or local CPU), sends a standardized prompt suite, and compares outputs via token-level similarity + perplexity delta. For analog: additionally checks per-layer output voltage vs. SPICE-expected values.

17. **AI Generation Subsystems** — Each hardware module has an AI-assisted design layer:
    - `drivers/ai-gen/` — LLM-based driver generator from datasheets; uses pluggable LLM backend
    - `python/tpt_alloy/ai-topology/` — Swarm topology advisor; starts LLM-based, accumulates SiL training data, graduates to a trained ML model
    - `python/tpt_fusion/ai-rtl/` — LLM-assisted Verilog MAC array generation with static timing pre-check; falls back to Amaranth templates if no LLM configured
    - `python/tpt_element/ai-circuit/` — Generative analog circuit designer; retrieval-augmented (Phase 1), generative model (Phase 2); validated by Reality Check after each candidate
    - `cloud/synthesis-worker/` — Go worker + Redis queue for offloading slow FPGA synthesis (Yosys + Nextpnr) to remote machines
    - `cloud/crucible-cloud/` *(optional)* — Full pipeline web service: upload GGUF, select target, download `.tptpkg`. Docker Compose + Helm chart for self-hosted deployment.

## Data Flow

```
AI Model (.gguf / .pt / .onnx / .safetensors / .tflite / ...)
        ↓
   TPT Catalyst  →  TPT-IR (.tptir)
        ↓
  ┌──┬──┼──┬──┬──┐
Alloy Fusion Element Photon Pulse Silicon
  ↓     ↓     ↓      ↓      ↓      ↓
 MCU   RTL  SPICE  MZI   SNN    CIM
              ↓
        TPT Observer (live telemetry)
```

## Technology Stack

| Component | Languages | Key Dependencies |
|-----------|-----------|-----------------|
| Catalyst | Python + Rust | MLIR, Apache TVM, PyTorch, ONNX Runtime, gguf-py |
| Alloy | Python + Rust | METIS/KaFFPa, PlatformIO, Zephyr RTOS |
| Fusion | Python | Amaranth HDL, Yosys, Nextpnr, LiteX/LiteDRAM |
| Element | Python | Xyce/ngspice, PySpice, PyTorch |
| Photon | Python | (MZI mesh generation; no external sim dep in Phase 1) |
| Pulse | Python | (LIF neuron model; optional Brian2/Norse for SiL) |
| Silicon | Python | (bitline op gen; no external dep) |
| Observer | Go + TypeScript | React, Next.js, Three.js/R3F, React Flow, Tailwind CSS |
| Emulator | Python + Rust | Verilator (Fusion SiL), Xyce/ngspice (Element SiL) |
| Mosaic | Python + Rust | (orchestrates all hardware modules) |
| FL | Python | (federated aggregation; no external ML framework required) |
| Shell | Python | (REPL over Observer WebSocket API) |
| Drivers | Rust + Python | (SDK + registry client; probe uses udev/WMI/IOKit) |
| tpt-train | Python | PyTorch, JAX/Flax |
| Cloud Workers | Go | Redis, Docker, Yosys + Nextpnr (synthesis worker) |

## Development Phases

- **Phase 1 (Months 1–6):** Catalyst + Alloy. Milestone: TinyLlama on 16x ESP32. ✓ (complete)
- **Phase 2 (Months 6–12):** Fusion. Milestone: Xilinx Alveo bitstream from UI.
- **Phase 3 (Year 2):** Element + Photon + Pulse + Silicon. Milestone: 3-layer analog NN → KiCad PCB; photonic MZI export; SNN spike schedule output.
- **Phase 4 (Year 2+):** Observer unifying all hardware types + FL + cloud workers.

## Key Design Decisions

- **Do not build a custom IR from scratch.** Use MLIR and define a TPT-IR dialect on top of it.
- **Wrap CLI tools (Yosys, Nextpnr, Xyce), never expose them raw** to the user — the UI abstracts all synthesis/simulation toolchains.
- **Reality Check (TPT Element):** Brute-force SPICE is too slow for interactive use. A PyTorch model trained on SPICE runs provides fast drift prediction; full SPICE is reserved for final validation.
- **Pre-flight check before compiling.** Always run `tpt-catalyst check` before `tpt-catalyst ingest` when targeting a new hardware type. The operator support matrix lives in `catalyst/compat/`.
- **SiL emulator uses the same telemetry schema as real hardware.** Observer cannot tell the difference — this is intentional. Don't add emulator-specific telemetry fields.
- **Mosaic layer annotations live in TPT-IR, not in module code.** Each module reads its own tagged layers; the Mosaic orchestrator handles cross-module coordination.
- **GGUF ingestion is quantization-preserving.** Unlike PyTorch/ONNX/TF models (ingested as float32 graphs), GGUF models arrive pre-quantized. TPT-IR must carry quantization type metadata (Q4_K, Q8_0, etc.) so Fusion and Alloy can generate hardware-appropriate compute (e.g., INT4 MAC arrays for Q4 models).
- **Reality Check and Circuit Designer are separate models with different tasks.** Reality Check is *discriminative* (given a circuit, predict failure probability). The AI Circuit Designer is *generative* (given a target operation, produce a circuit). Both are trained on the same SPICE dataset but solve different problems. Don't conflate them.
- **Driver manifests include `[bom]` and `[power]` sections.** `[bom]` carries part numbers + supplier SKUs (DigiKey/Mouser/LCSC). `[power]` carries idle/active/peak mW. Both are required for BOM generation and cost/power estimation to work.
- **AI generation features degrade gracefully.** Driver Generator, Topology Advisor, and RTL Assistant are hidden when no LLM provider is configured. Fusion falls back to Amaranth template generation. The Circuit Designer's retrieval-augmented Phase 1 works without an LLM.
- **Hardware drivers are a standardized SDK, not hardcoded board configs.** All board support lives in `drivers/` as versioned, community-contributable packages. The Xilinx Alveo profile is the reference implementation. New boards require only a driver package — no changes to core modules.
- **Natural language features are LLM-agnostic and optional.** The provider interface supports OpenRouter, Anthropic API, any Ollama-compatible endpoint, and TPT Spark IPC. If no provider is configured, NL features are hidden — no degraded experience and no hard dependency on any LLM service.
- **Cloud features are self-hostable infrastructure.** TPT provides Docker images + deployment docs; the project does not operate any cloud services. All cloud features are opt-in via user-configured URLs in settings.
- **`.tptprofile` improves but never blocks compilation.** If `tpt-train` hooks were used during training, Catalyst uses the profile for better quantization. If not present, Catalyst falls back to static weight analysis — the pipeline always works.
- **Open-core:** Underlying compilers (MLIR, Yosys, Xyce) are open-source. TPT-proprietary layers are the glue, UI, and AI-driven optimization passes.
- **UI theme:** Tailwind CSS, dark mode, "industrial blueprint" aesthetic — dark grays, neon cyan/amber accents, monospaced fonts for data readouts.
- **Photon, Pulse, Silicon are parallel expansion targets, not replacements.** They follow the same TPT-IR → `.tptpkg` contract as Alloy/Fusion/Element. Mosaic can partition a model across all six hardware types simultaneously.
- **Catalyst carbon accounting is non-blocking.** Carbon estimates are advisory; they appear in preflight reports and the Observer dashboard but never prevent compilation.
- **Model provenance is always recorded.** Every Catalyst ingestion writes a `provenance/lineage.json` into the `.tptpkg` tracking every transformation decision. This is not optional — it is required for the community cache to accept a package.
- **Spark auto-detection priority.** On startup, Catalyst and Observer check for a running Spark IPC socket. If found, Spark becomes the default LLM backend automatically. Priority order: Spark IPC → Ollama-compatible → OpenRouter → Anthropic API.
- **TPT Shell uses the Observer WebSocket API.** `tpt-shell` connects to a running Observer instance; it does not open its own hardware connections. This means SiL and real hardware are both reachable via the same REPL without special-casing.
- **Federated learning does not expose raw gradients.** TPT FL aggregates model updates using secure aggregation. The orchestrator (`python/tpt_fl/`) never receives plaintext gradients from individual hardware nodes.

## TPT Spark Integration

[TPT Spark](https://github.com/PhillipC05/tpt-spark) is a sibling app — a local GGUF runtime (Tauri v2, Rust, wgpu GPU / HuggingFace Candle CPU fallback). They share the GGUF model format and form a complementary stack: Spark runs models on standard hardware, Crucible compiles them for custom hardware.

**Integration points:**
- **Spark auto-detection** (`python/tpt_catalyst/tpt_catalyst/spark_autodetect.py`): On startup, checks for `$XDG_RUNTIME_DIR/tpt-spark.sock` (Linux), `\\.\pipe\tpt-spark` (Windows), `/tmp/tpt-spark.sock` (macOS). If found, Spark becomes the default LLM backend.
- **Shared model library** (`~/.tpt/models/`): Catalyst reads from Spark's local model directory; no re-downloading. `--spark-model <id>` flag in the Catalyst CLI.
- **Baseline benchmarks** (`python/tpt_catalyst/tpt_catalyst/spark_benchmark.py`): Reads `~/.tpt/benchmarks/spark-*.json`; displays GPU tok/s alongside Crucible hardware metrics in Observer.
- **Prompt replay** (`python/tpt_catalyst/tpt_catalyst/spark_replay.py`): The SiL emulator can consume Spark's conversation JSON as regression benchmark input.
- **Spark integration IPC** (`python/tpt_catalyst/tpt_catalyst/spark_integration.py`): Full IPC client for Spark's headless API.
- **Spark UI hook**: A "Compile for Custom Hardware" button in Spark's sidebar hands off the loaded model to Crucible (file path + model ID via IPC or temp file).

**Boundary**: Do not add Crucible as a Spark dependency. Integration is filesystem + optional IPC only. Both apps must remain independently runnable.
