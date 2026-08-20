# TPT ISA — Layer 1: Instruction Set Architecture

**Tensor Processing Technology — Hardware Description Layer**

## Overview

Layer 1 defines the TPT Instruction Set Architecture (ISA) implemented in SystemVerilog. This layer provides the hardware foundation for the entire TPT GPU compute stack.

### Directory Structure

```
layer1_isa/
├── spec/
│   └── tpt_isa_spec.md      — ISA specification document
├── rtl/
│   ├── tpt_pkg.sv           — Package: constants, types, opcodes
│   ├── tpt_core.sv          — Top-level core integration
│   ├── tpt_decode.sv        — Instruction decoder
│   ├── tpt_regfile.sv       — Scalar register file (32×32-bit)
│   ├── tpt_alu.sv           — Integer/FP ALU unit
│   ├── tpt_lsu.sv           — Load/store unit
│   ├── tpt_ctrl.sv          — Control unit (hazards, forwarding, branch)
│   ├── tpt_pipeline.sv      — Pipeline stage registers
│   └── tpt_tensor_unit.sv   — Tensor/MMA compute unit
├── sim/
│   ├── tpt_tb.sv            — Self-checking testbench
│   ├── sim.do               — ModelSim/Questa simulation script
│   ├── tpt_assemble.py      — Python assembler (asm → hex)
│   ├── Makefile             — Simulation build
│   └── programs/
│       ├── simple_add.asm   — Assembly test program
│       └── simple_add.hex   — Pre-assembled hex
└── README.md                — This file
```

### Key Features

- **32-bit fixed-length instructions** — 5 opcode groups (R/I/M/B/J/V)
- **9-stage pipeline** — F1/F2, D1/D2, E1/E2, E3/E4, W1
- **Large register file** — 32 scalar + 64 vector (512-bit) + 8 predicate
- **SIMT execution** — 32-lane warp execution with predication
- **Tensor acceleration** — Native MMA operations (FP16/INT8)
- **Forwarding + hazard detection** — Minimizes pipeline stalls

### Running Simulation

#### Prerequisites
- Icarus Verilog (iverilog) or ModelSim/Questa
- Python 3.x

#### With Makefile
```bash
cd layer1_isa/sim
make sim
```

#### With ModelSim/Questa
```bash
cd layer1_isa/sim
vsim -do sim.do
```

#### Manually assemble + run
```bash
cd layer1_isa/sim
python tpt_assemble.py programs/simple_add.asm
iverilog -g2012 -o sim.vvp ../rtl/*.sv tpt_tb.sv
vvp sim.vvp
```

### Instruction Formats

| Format | Description | Fields |
|---|---|---|
| R-Type | Register-register | opcode, rd, rs1, rs2, func |
| I-Type | Immediate | opcode, rd, rs1, imm12, func |
| M-Type | Memory | opcode, rd, rs1, offset12, func |
| B-Type | Branch | opcode, rs1, rs2, func, offset12 |
| J-Type | Jump | opcode, rd, target22, func |
| V-Type | Vector/Tensor | opcode, vd, vs1, vs2, sz, dm, func, subop |

### Status

- [x] ISA specification document
- [x] RTL implementation (core units)
- [x] Testbench / simulation
- [ ] FPGA validation
- [ ] Formal verification

### License

Apache License 2.0 (with Express Patent Grant)
