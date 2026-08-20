# TPT Crucible

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-cyan)](CHANGELOG.md)

**Hardware-agnostic AI compiler suite.** Compile standard AI models (GGUF, ONNX, PyTorch, TensorFlow, SafeTensors, EXL2, AWQ/GPTQ, JAX, TFLite) onto non-traditional hardware: FPGAs, analog compute circuits, microcontroller swarms, photonic processors, neuromorphic chips, and compute-in-memory arrays.

> TPT Crucible is **not** a GPU compiler. It explicitly targets edge and custom silicon: Alloy (ESP32/RP2040 swarms), Fusion (Xilinx FPGA), Element (analog/SPICE), Photon (photonic MZI mesh), Pulse (neuromorphic SNN), and Silicon (CIM accelerators).

---

## Quick Start (5 minutes)

```bash
# 1. Install Rust, Python 3.10+, Go 1.22+, Node 18+

# 2. Build the Rust backend
cargo build --release

# 3. Install Python packages
pip install -e python/tpt_catalyst -e python/tpt_alloy

# 4. Ingest a GGUF model and compile for ESP32 swarm
tpt-catalyst ingest models/tinyllama.gguf --target alloy --output dist/tinyllama.tptpkg

# 5. Start the Observer dashboard
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

No hardware required — use the Software-in-the-Loop emulator for all targets.

---

## Modules

| Module | Target | Language |
|--------|--------|----------|
| **TPT Catalyst** | Ingestion → TPT-IR (all formats) | Python + Rust |
| **TPT Alloy** | MCU swarm (ESP32, RP2040, RISC-V) | Python + Rust |
| **TPT Fusion** | FPGA (Amaranth HDL → Yosys → Nextpnr) | Python |
| **TPT Element** | Analog (SPICE/KiCad) | Python |
| **TPT Photon** | Photonic MZI mesh *(experimental)* | Python |
| **TPT Pulse** | Neuromorphic ANN→SNN compiler | Python |
| **TPT Silicon** | Compute-in-Memory accelerators | Python |
| **TPT Observer** | Real-time dashboard | Go + Next.js |
| **TPT Emulator** | Software-in-the-Loop | Python + Rust |
| **TPT Mosaic** | Hybrid cross-hardware orchestration | Python + Rust |
| **TPT Drivers** | Board SDK + community registry | Rust + Python |
| **TPT FL** | Federated learning orchestration | Python |
| **TPT Shell** | Interactive hardware REPL | Python |
| **TPT Validator** | Accuracy validation vs. reference | Python |
| **tpt-train** | Training hooks → `.tptprofile` | Python |

## Output: `.tptpkg`

Every compilation produces a `.tptpkg` (ZIP) containing:

```
model.tptpkg/
├── manifest.json         # version, model name, SHA-256 hashes
├── ir/model.tptir        # hardware-agnostic IR
├── targets/alloy/        # firmware binaries + flash script
├── targets/fusion/       # bitstream + board profile
├── targets/element/      # SPICE netlist + KiCad PCB
├── targets/photon/       # MZI mesh configuration
├── targets/pulse/        # SNN weight + spike schedule export
├── targets/silicon/      # CIM weight arrays + bitline ops
├── compat/preflight.json
├── quant/quant_profile.json
├── mosaic/partition.json
└── provenance/lineage.json  # full audit trail of compilation decisions
```

## Architecture

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
   TPT Observer (live telemetry + 3D topology)
```

## Key Features

- **12+ ingestion formats** — GGUF, ONNX, PyTorch, TensorFlow, SafeTensors, HuggingFace Hub, TFLite, AWQ/GPTQ, EXL2, JAX/Flax, Llamafile, Keras
- **Carbon-aware compilation** — estimates and minimizes grid carbon footprint per target
- **Model provenance graph** — full lineage audit trail of every compilation decision
- **AI-powered diagnostics** — LLM-backed `tpt-catalyst doctor` for debugging failed compilations
- **Federated learning** — split a model across a Crucible hardware deployment, train locally, aggregate privately
- **Model tournament** — benchmark multiple models head-to-head on the same hardware target
- **Community cache & marketplace** — share and discover pre-compiled `.tptpkg` artifacts
- **Interactive REPL** — `tpt-shell` for live hardware introspection and ad-hoc tensor ops
- **Spark auto-detection** — detects a running TPT Spark instance and uses it as the local LLM backend

## Development Phases

- **Phase 1 (Months 1–6):** Catalyst + Alloy. Milestone: TinyLlama on 16x ESP32. ✓
- **Phase 2 (Months 6–12):** Fusion. Milestone: Xilinx Alveo bitstream from UI.
- **Phase 3 (Year 2):** Element + Photon + Pulse + Silicon. Milestone: analog/photonic/neuromorphic/CIM targets.
- **Phase 4 (Year 2+):** Observer unifying all hardware types + FL + cloud workers.

## TPT Spark Integration

[TPT Spark](https://github.com/PhillipC05/tpt-spark) is the companion local GGUF runtime (Tauri v2). Spark runs models on standard hardware; Crucible compiles them for custom hardware. Crucible auto-detects a running Spark instance, uses it as the default offline LLM backend, reads its benchmark baselines for emulator validation, and can replay Spark conversation JSON as regression input. Integration is filesystem + optional IPC only — both apps are independently runnable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs are welcome.

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache 2.0 — Copyright 2026 TPT Solutions. See [LICENSE](LICENSE).
