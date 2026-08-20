# TPT ISA Specification v1.0

**Tensor Processing Technology — Instruction Set Architecture**

**Version:** 1.0  
**Status:** Draft  
**License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Architecture Overview

The TPT ISA defines a 32-bit, load-store, register-based architecture designed for general-purpose GPU compute. It supports SIMT (Single Instruction, Multiple Threads) execution with explicit tensor acceleration units.

### 1.1 Key Design Goals

- **32-bit fixed-length instructions** — simple decode, no alignment issues
- **Large register file** (256 × 32-bit scalar + 64 × 512-bit vector registers) — enables warp-level computation without spilling
- **Unified memory addressing** — 48-bit virtual address space (256 TiB)
- **Explicit memory hierarchy** — global, shared, local, and constant address spaces
- **SIMT execution model** — warps of 32 lanes (SIMD width 32)
- **Tensor acceleration** — native matrix multiply-accumulate (MMA) operations

### 1.2 Execution Model

The TPT core executes in a SIMT fashion:
- **Threads** are grouped into **warps** of 32 lanes
- **Warps** are grouped into **thread blocks (CTAs)**
- All threads in a warp execute the same instruction (with predication for divergence)
- Hardware tracks warp state and schedules warps onto compute units

### 1.3 Memory Model

| Address Space | Access Scope | Width | Description |
|---|---|---|---|
| Global | All threads | 48-bit | Main device memory |
| Shared | Single CTA | 32-bit | Low-latency shared memory |
| Local | Single thread | 32-bit | Thread-private stack/memory |
| Constant | All threads (read-only) | 48-bit | Read-only constant cache |

---

## 2. Instruction Formats

All instructions are 32 bits. There are 6 formats:

### 2.1 R-Type (Register — Register)

```
 31    27  26   22  21   17  16   12  11    7  6    0
┌────────┬──────┬──────┬──────┬──────┬────────────┐
│  opcode │  rd   │  rs1  │  rs2  │  func │  0000000  │
└────────┴──────┴──────┴──────┴──────┴────────────┘
```

- `opcode` (5 bits): Major opcode
- `rd` (5 bits): Destination register (0–31 scalar, or vector via mode)
- `rs1` (5 bits): Source register 1
- `rs2` (5 bits): Source register 2
- `func` (5 bits): Function selector within opcode group
- Reserved: 7 bits (zero)

### 2.2 I-Type (Immediate)

```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │    immediate      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

- `immediate` (12 bits): Signed immediate value (sign-extended to 32 bits)

### 2.3 M-Type (Memory)

```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │      offset      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

- `offset` (12 bits): Signed byte offset (sign-extended for address calculation)
- `rs1`: Base address register
- `rd`: Data register (load) or zero (store)

### 2.4 B-Type (Branch)

```
 31    27  26   22  21   17  16   12  11         0
┌────────┬──────┬──────┬──────┬──────────────────┐
│  opcode │  rs1  │  rs2  │ func │    branch_offset │
└────────┴──────┴──────┴──────┴──────────────────┘
```

- `branch_offset` (12 bits): Signed PC-relative offset in instructions (×4 byte addressing)

### 2.5 J-Type (Jump)

```
 31    27  26                    5  4   0
┌────────┬────────────────────────┬──────┐
│  opcode │     jump_target        │ func │
└────────┴────────────────────────┴──────┘
```

- `jump_target` (22 bits): Absolute or PC-relative target

### 2.6 V-Type (Vector / Tensor)

```
 31    27  26   22  21   17  16   12  11  10  9   5  4   0
┌────────┬──────┬──────┬──────┬───┬───┬───────┬──────┐
│  opcode │  vd   │  vs1  │  vs2  │ sz│ dm│  func  │ subop│
└────────┴──────┴──────┴──────┴───┴───┴───────┴──────┘
```

- `vd` (5 bits): Vector destination register (0–63)
- `vs1` (5 bits): Vector source register 1
- `vs2` (5 bits): Vector source register 2
- `sz` (2 bits): Data size (00=8b, 01=16b, 10=32b, 11=64b)
- `dm` (2 bits): Destination modifier (packed, scatter, mask)
- `func` (5 bits): Vector function selector
- `subop` (5 bits): Sub-operation



---

## 3. Registers

### 3.1 Scalar Register File

- **32 × 32-bit general-purpose scalar registers** (R0–R31)
- R0 is hardwired to zero (writes ignored)
- R1–R31 are general purpose

### 3.2 Vector Register File

- **64 × 512-bit vector registers** (V0–V63)
- Each vector holds 32 × 16-bit elements, 16 × 32-bit elements, or 8 × 64-bit elements
- Accessible as 4 × 128-bit sub-registers for mixed-precision

### 3.3 Predicate Register File

- **8 × 32-bit predicate registers** (P0–P7)
- Each bit corresponds to one lane in a warp
- Used for predicated execution and vector compare results

### 3.4 Special Registers

| Number | Name | Width | Description |
|---|---|---|---|
| SR0 | `LANE_ID` | 5 | Current lane index within warp (0–31) |
| SR1 | `WARP_ID` | 10 | Current warp index within CTA |
| SR2 | `CTA_ID_X` | 16 | CTA index in X dimension |
| SR3 | `CTA_ID_Y` | 16 | CTA index in Y dimension |
| SR4 | `CTA_ID_Z` | 16 | CTA index in Z dimension |
| SR5 | `NTID_X` | 16 | Number of threads in X dimension |
| SR6 | `NTID_Y` | 16 | Number of threads in Y dimension |
| SR7 | `NTID_Z` | 16 | Number of threads in Z dimension |
| SR8 | `CLOCK` | 64 | Cycle counter (read-only) |
| SR9 | `STATUS` | 32 | Status/exception flags |
| SR10 | `MASK` | 32 | Active lane mask for current warp |
| SR11–SR15 | Reserved | — | Reserved for future use |
| SR16–SR31 | User | 32 | User-defined special registers |

### 3.5 Special Register Access

Special registers are accessed via the `RDSR` (read special register) and `WRSR` (write special register) instructions.

---

## 4. Opcode Map

### 4.1 Major Opcode Groups

| Opcode[4:0] | Group | Description |
|---|---|---|
| `00000` | ALU_INT | Integer ALU operations |
| `00001` | ALU_FP | Floating-point ALU operations |
| `00010` | ALU_COMP | Compare operations |
| `00011` | ALU_LOG | Logical operations |
| `00100` | MEM_LD | Load operations |
| `00101` | MEM_ST | Store operations |
| `00110` | MEM_ATOM | Atomic memory operations |
| `00111` | CTRL_BR | Branch operations |
| `01000` | CTRL_J | Jump operations |
| `01001` | CTRL_SYNC | Synchronization operations |
| `01010` | VEC | Vector/SIMD operations |
| `01011` | TENSOR | Tensor/MMA operations |
| `01100` | TEX | Texture/sampler operations |
| `01101` | CONV | Convolution operations |
| `01110` | SYSTEM | System/coprocessor operations |
| `01111` | PRED | Predicate operations |
| `10000` | MISC | Miscellaneous |
| Others | Reserved | Future expansion |

### 4.2 Integer ALU (Opcode = 00000)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `ADD` | rd = rs1 + rs2 |
| `00001` | `ADDI` | rd = rs1 + imm (I-Type) |
| `00010` | `SUB` | rd = rs1 - rs2 |
| `00011` | `SUBI` | rd = rs1 - imm (I-Type) |
| `00100` | `MUL` | rd = rs1 × rs2 (low 32 bits) |
| `00101` | `MULHI` | rd = rs1 × rs2 (high 32 bits) |
| `00110` | `DIV` | rd = rs1 ÷ rs2 |
| `00111` | `MOD` | rd = rs1 % rs2 |
| `01000` | `AND` | rd = rs1 & rs2 |
| `01001` | `OR` | rd = rs1 \| rs2 |
| `01010` | `XOR` | rd = rs1 ^ rs2 |
| `01011` | `SLL` | rd = rs1 << rs2 |
| `01100` | `SRL` | rd = rs1 >> rs2 (logical) |
| `01101` | `SRA` | rd = rs1 >> rs2 (arithmetic) |
| `01110` | `CLZ` | rd = count_leading_zeros(rs1) |
| `01111` | `POPC` | rd = popcount(rs1) |
| `10000` | `MIN` | rd = min(rs1, rs2) |
| `10001` | `MAX` | rd = max(rs1, rs2) |
| `10010` | `ABS` | rd = abs(rs1) |
| `10011` | `NEG` | rd = -rs1 |
| Others | Reserved | — |

### 4.3 Floating-Point ALU (Opcode = 00001)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `FADD` | rd = rs1 + rs2 (FP32) |
| `00001` | `FSUB` | rd = rs1 - rs2 (FP32) |
| `00010` | `FMUL` | rd = rs1 × rs2 (FP32) |
| `00011` | `FDIV` | rd = rs1 ÷ rs2 (FP32) |
| `00100` | `FADD16` | rd = rs1 + rs2 (FP16) |
| `00101` | `FMUL16` | rd = rs1 × rs2 (FP16) |
| `00110` | `FADD64` | rd = rs1 + rs2 (FP64) |
| `00111` | `FMUL64` | rd = rs1 × rs2 (FP64) |
| `01000` | `F2I` | Convert FP32 to int32 |
| `01001` | `I2F` | Convert int32 to FP32 |
| `01010` | `F2F16` | Convert FP32 to FP16 |
| `01011` | `F16Tof` | Convert FP16 to FP32 |
| `01100` | `FMA` | rd = rs1 × rs2 + rd (FP32 fused multiply-add) |
| `01101` | `FMA16` | rd = rs1 × rs2 + rd (FP16 FMA) |
| `01110` | `FMA64` | rd = rs1 × rs2 + rd (FP64 FMA) |
| `01111` | `FSQRT` | rd = sqrt(rs1) |
| Others | Reserved | — |



### 4.4 Memory Operations (Opcode = 00100, 00101)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `LB` | Load byte (sign-extended) |
| `00001` | `LBU` | Load byte (zero-extended) |
| `00010` | `LH` | Load halfword (sign-extended) |
| `00011` | `LHU` | Load halfword (zero-extended) |
| `00100` | `LW` | Load word (32-bit) |
| `00101` | `LD` | Load doubleword (64-bit, two registers) |
| `00110` | `LV` | Load vector (512-bit, to V-reg) |
| `00111` | `SB` | Store byte |
| `01000` | `SH` | Store halfword |
| `01001` | `SW` | Store word |
| `01010` | `SD` | Store doubleword |
| `01011` | `SV` | Store vector |

### 4.5 Control Flow (Opcode = 00111, 01000)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `BEQ` | Branch if rs1 == rs2 |
| `00001` | `BNE` | Branch if rs1 != rs2 |
| `00010` | `BLT` | Branch if rs1 < rs2 (signed) |
| `00011` | `BGE` | Branch if rs1 >= rs2 (signed) |
| `00100` | `BLTU` | Branch if rs1 < rs2 (unsigned) |
| `00101` | `BGEU` | Branch if rs1 >= rs2 (unsigned) |
| `00110` | `JAL` | Jump and link (rd = PC+4, jump to target) |
| `00111` | `JALR` | Jump and link register (rd = PC+4, jump to rs1 + imm) |
| `01000` | `RET` | Return (jump to rd, where rd holds return address) |
| `01001` | `BAR` | Barrier synchronization (within CTA) |

### 4.6 Vector Operations (Opcode = 01010)

| func[4:0] | subop[4:0] | Mnemonic | Description |
|---|---|---|---|
| `00000` | `00000` | `VADD` | vd = vs1 + vs2 (element-wise) |
| `00001` | `00000` | `VSUB` | vd = vs1 - vs2 (element-wise) |
| `00010` | `00000` | `VMUL` | vd = vs1 × vs2 (element-wise) |
| `00011` | `00000` | `VFMADD` | vd = vs1 × vs2 + vd (fused multiply-add) |
| `00100` | `00000` | `VSHFL` | Vector shuffle (cross-lane permute) |
| `00101` | `00000` | `VRED` | Vector reduction (sum, min, max) |
| `00110` | `00000` | `VBCAST` | Broadcast scalar to all lanes |
| `00111` | `00000` | `VCMP` | Vector compare (result in predicate register) |
| Others | — | Reserved | — |

### 4.7 Tensor / MMA Operations (Opcode = 01011)

| func[4:0] | subop[4:0] | Mnemonic | Description |
|---|---|---|---|
| `00000` | `00000` | `MMA_16x16x16` | 16×16×16 FP16 matrix multiply-accumulate |
| `00001` | `00000` | `MMA_32x32x8` | 32×32×8 FP16 MMA |
| `00010` | `00000` | `MMA_8x8x32` | 8×8×32 INT8 MMA |
| Others | — | Reserved | — |

### 4.8 Predicate Operations (Opcode = 01111)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `PAND` | pd = ps1 & ps2 (bitwise, per-lane) |
| `00001` | `POR` | pd = ps1 \| ps2 |
| `00010` | `PXOR` | pd = ps1 ^ ps2 |
| `00011` | `PNOT` | pd = ~ps1 |
| `00100` | `PSETP` | Set predicate based on condition |
| `00101` | `SELP` | Select rd = rs1 if predicate true else rs2 |

### 4.9 Synchronization (Opcode = 01001)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `BARRIER` | CTA barrier synchronization |
| `00001` | `MEMFENCE` | Memory fence (ensure visibility) |
| `00010` | `WARPSYNC` | Synchronize within warp |
| `00011` | `ATOMIC_ADD` | Atomic add to memory location |
| `00100` | `ATOMIC_EXCH` | Atomic exchange |
| `00101` | `ATOMIC_CAS` | Atomic compare-and-swap |

### 4.10 System Operations (Opcode = 01110)

| func[4:0] | Mnemonic | Description |
|---|---|---|
| `00000` | `RDSR` | Read special register |
| `00001` | `WRSR` | Write special register |
| `00010` | `TRAP` | Software trap/interrupt |
| `00011` | `RETI` | Return from interrupt |
| `00100` | `WFI` | Wait for interrupt |
| Others | Reserved | — |

---

## 5. Pipeline Architecture

The TPT core implements a 9-stage pipeline:

| Stage | Name | Description |
|---|---|---|
| F1 | Fetch | Instruction fetch from I-cache |
| F2 | Fetch2 | I-cache hit/miss resolution |
| D1 | Decode | Instruction decode, register read |
| D2 | Decode2 | Operand forwarding, hazard detection |
| E1 | Execute ALU | Integer/FP ALU operations |
| E2 | Execute CMP | Compare operations, predicate generation |
| E3 | Execute MEM | Address generation, D-cache access |
| E4 | Execute MEM2 | D-cache hit/miss, load alignment |
| W1 | Writeback | Register writeback |

### 5.1 Pipeline Hazards

- **RAW hazards**: Detected in D2 stage; resolved via forwarding (E1→D2, W1→D2)
- **WAW hazards**: Detected in D2 stage; stall if write-after-write to same register
- **Memory hazards**: RAW on memory is resolved by scoreboard; load-use penalty = 2 cycles
- **Structural hazards**: Handled by arbitration between ALU and LSU units

### 5.2 Branch Prediction

- Simple 2-bit saturating counter predictor (4096-entry BTB)
- Branch misprediction penalty = 4 cycles (F1→E2 flush)
- Return address stack (RAS): 16 entries

---

## 6. Exception and Interrupt Handling

### 6.1 Exception Types

| Code | Exception | Cause |
|---|---|---|
| 0 | Reserved | — |

---

## 7. Initialization and Reset

### 7.1 Reset State

On reset:
- PC = reset vector (0x0000_0000)
- All scalar registers = 0 (R0 hardwired to 0)
- All vector registers = 0
- All predicate registers = 0
- All special registers = implementation-defined reset values
- All caches invalidated
- Pipeline flushed

### 7.2 Boot Sequence

1. CPU loads GPU firmware to device memory address 0x0000_0000
2. CPU writes to MMIO register `TPT_CTRL.BOOT` to release reset
3. GPU fetches first instruction from address 0x0000_0000
4. Firmware initializes thread scheduler and memory manager
5. Firmware signals ready via MMIO register `TPT_STATUS.READY`

---

## 8. Instruction Encoding Reference

| Instruction | Format | Opcode | func | subop | Description |
|---|---|---|---|---|---|
| ADD | R | 00000 | 00000 | — | Integer add |
| ADDI | I | 00000 | 00001 | — | Integer add immediate |
| SUB | R | 00000 | 00010 | — | Integer subtract |
| MUL | R | 00000 | 00100 | — | Integer multiply |
| DIV | R | 00000 | 00110 | — | Integer divide |
| AND | R | 00000 | 01000 | — | Bitwise AND |
| OR | R | 00000 | 01001 | — | Bitwise OR |
| XOR | R | 00000 | 01010 | — | Bitwise XOR |
| SLL | R | 00000 | 01011 | — | Shift left logical |
| SRL | R | 00000 | 01100 | — | Shift right logical |
| SRA | R | 00000 | 01101 | — | Shift right arithmetic |
| FADD | R | 00001 | 00000 | — | FP32 add |
| FSUB | R | 00001 | 00001 | — | FP32 subtract |
| FMUL | R | 00001 | 00010 | — | FP32 multiply |
| FDIV | R | 00001 | 00011 | — | FP32 divide |
| FMA | R | 00001 | 01100 | — | FP32 fused multiply-add |
| LB | M | 00100 | 00000 | — | Load byte |
| LH | M | 00100 | 00010 | — | Load halfword |
| LW | M | 00100 | 00100 | — | Load word |
| LD | M | 00100 | 00101 | — | Load doubleword |
| LV | M | 00100 | 00110 | — | Load vector |
| SB | M | 00101 | 00111 | — | Store byte |
| SH | M | 00101 | 01000 | — | Store halfword |
| SW | M | 00101 | 01001 | — | Store word |
| SD | M | 00101 | 01010 | — | Store doubleword |
| SV | M | 00101 | 01011 | — | Store vector |
| BEQ | B | 00111 | 00000 | — | Branch if equal |
| BNE | B | 00111 | 00001 | — | Branch if not equal |
| BLT | B | 00111 | 00010 | — | Branch if less than |
| JAL | J | 01000 | 00110 | — | Jump and link |
| JALR | J | 01000 | 00111 | — | Jump and link register |
| BARRIER | R | 01001 | 00000 | — | CTA barrier |
| MEMFENCE | R | 01001 | 00001 | — | Memory fence |
| ATOMIC_ADD | M | 01001 | 00011 | — | Atomic add |
| VADD | V | 01010 | 00000 | 00000 | Vector add |
| VSUB | V | 01010 | 00001 | 00000 | Vector subtract |
| VMUL | V | 01010 | 00010 | 00000 | Vector multiply |
| VFMADD | V | 01010 | 00011 | 00000 | Vector FMA |
| VSHFL | V | 01010 | 00100 | 00000 | Vector shuffle |
| VRED | V | 01010 | 00101 | 00000 | Vector reduction |
| MMA_16x16x16 | V | 01011 | 00000 | 00000 | Tensor MMA 16×16×16 |
| MMA_32x32x8 | V | 01011 | 00001 | 00000 | Tensor MMA 32×32×8 |
| MMA_8x8x32 | V | 01011 | 00010 | 00000 | Tensor MMA 8×8×32 |
| RDSR | R | 01110 | 00000 | — | Read special register |
| WRSR | R | 01110 | 00001 | — | Write special register |
| TRAP | R | 01110 | 00010 | — | Software trap |
| PAND | R | 01111 | 00000 | — | Predicate AND |
| POR | R | 01111 | 00001 | — | Predicate OR |
| PNOT | R | 01111 | 00011 | — | Predicate NOT |
| SELP | R | 01111 | 00101 | — | Select with predicate |

---

## 9. Warp Scheduling

The TPT core manages a warp pool of up to 64 warps. The scheduler uses a round-robin policy with priority boost for memory-bound warps. Scheduling decisions are made at every cycle:

1. Check ready warps (not waiting for memory, sync, or dependencies)
2. Select next warp using round-robin pointer
3. Fetch instruction from warp's PC
4. Execute in SIMT fashion across 32 lanes

---

## 10. Performance Counters

The following performance counters are accessible via special registers:

| SR | Name | Description |
|---|---|---|
| SR20 | `INST_RETIRED` | Total instructions retired |
| SR21 | `CORE_CYCLES` | Total core cycles |
| SR22 | `L1D_MISSES` | L1 data cache misses |
| SR23 | `L1I_MISSES` | L1 instruction cache misses |
| SR24 | `BRANCH_MISPRED` | Branch mispredictions |
| SR25 | `WARP_STALLS` | Cycles any warp is stalled |

---

*End of TPT ISA Specification v1.0*

| 1 | `INST_PAGE_FAULT` | Instruction fetch page fault |
| 2 | `DATA_PAGE_FAULT` | Load/store page fault |
| 3 | `INST_ILLEGAL` | Illegal instruction |
| 4 | `PRIVILEGE` | Privilege violation |
| 5 | `MISALIGNED` | Misaligned memory access |
| 6 | `DIV_BY_ZERO` | Division by zero |
| 7 | `OVERFLOW` | Arithmetic overflow |
| 8–15 | Reserved | — |
| 16–31 | User-defined | Software-defined exceptions |

### 6.2 Trap Vector

The trap handler base address is held in special register `SR_TVEC` (SR12). Each exception type has a 32-byte handler slot at `SR_TVEC + (exception_code × 32)`.
