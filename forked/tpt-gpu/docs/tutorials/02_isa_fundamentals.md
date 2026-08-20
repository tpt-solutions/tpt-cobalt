# Tutorial 2: TPT ISA Fundamentals

**Estimated Time:** 45 minutes  
**Prerequisites:** Tutorial 1, basic computer architecture

---

## Introduction

The TPT ISA (Instruction Set Architecture) is the lowest software abstraction in the TPT GPU platform. It defines the hardware operations that execute directly on the GPU.

### Design Principles

1. **Fixed-Length 32-bit Instructions**: Simplifies decode and scheduling
2. **SIMT Execution**: Single Instruction, Multiple Thread
3. **Load/Store Architecture**: Explicit memory operations
4. **9-Stage Pipeline**: Fetch, Decode, Issue, Execute, Writeback
5. **Large Register File**: 256 scalar + 64 vector registers

---

## Instruction Format

All TPT ISA instructions are exactly 32 bits wide with 6 formats:

### R-Type (Register)
```
 31    27  26   22  21   17  16   12  11    7  6    0
┌────────┬──────┬──────┬──────┬──────┬────────────┐
│  opcode │  rd   │  rs1  │  rs2  │  func │  0000000  │
└────────┴──────┴──────┴──────┴──────┴────────────┘
```

### I-Type (Immediate)
```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │    immediate      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

### M-Type (Memory)
```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │      offset      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

---

## Register File

### General Purpose Registers
- **R0-R31**: 32 general-purpose registers per thread
- **R0**: Hardwired to zero
- **R1**: Link register (return address)
- **R2**: Stack pointer

### Special Registers

| Register | Description |
|----------|-------------|
| `PC` | Program counter |
| `SR` | Status register |
| `TID` | Thread ID (read-only) |
| `BID` | Block ID (read-only) |
| `BSZ` | Block size (read-only) |

---

## Pipeline Stages

```
┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│ FETCH │→│DECODE │→│ISSUE  │→│READ   │→│EXEC   │→│MEMORY │→│WRITE  │→│COMMIT │→│TRAP   │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘
```

### SIMT Execution Model

```
┌─────────────────────────────────────────────────────────────┐
│                         Warp (32 threads)                    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────────────────────┬─────┤
│ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ ...                 │ T31 │
├─────┴─────┴─────┴─────┴─────┴─────┴─────────────────────┴─────┤
│                    One instruction in flight                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Instruction Reference

### Arithmetic Instructions

| Instruction | Opcode | Format | Description |
|-------------|--------|--------|-------------|
| `ADD` | 0x01 | R-Type | rd = rs1 + rs2 |
| `SUB` | 0x02 | R-Type | rd = rs1 - rs2 |
| `MUL` | 0x03 | R-Type | rd = rs1 * rs2 |
| `FMA` | 0x05 | R-Type | rd = (rs1 * rs2) + rs3 |

```assembly
add r3, r1, r2      // r3 = r1 + r2
fma r5, r1, r2, r5  // r5 += r1 * r2
```

### Memory Instructions

| Instruction | Opcode | Format | Description |
|-------------|--------|--------|-------------|
| `LOAD` | 0x40 | M-Type | rd = memory[rs1 + offset] |
| `STORE` | 0x41 | M-Type | memory[rs1 + offset] = rs2 |

```assembly
load r6, [r1 + 0]   // r6 = memory[r1]
store [r1 + 0], r6   // memory[r1] = r6
```

### Control Flow

| Instruction | Opcode | Format | Description |
|-------------|--------|--------|-------------|
| `JMP` | 0x60 | J-Type | Unconditional jump |
| `BEQ` | 0x70 | B-Type | Branch if equal |
| `BNE` | 0x71 | B-Type | Branch if not equal |
| `BLT` | 0x72 | B-Type | Branch if less than |

```assembly
    mov r10, 10
loop:
    sub r10, r10, 1
    bne r10, loop    // branch if r10 != 0
```

### Vector/SIMT Instructions

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| `VADD` | 0xA0 | Vector add (32 lanes) |
| `VMUL` | 0xA2 | Vector multiply |
| `VFMADD` | 0xA3 | Vector FMA |
| `BARRIER` | 0x90 | CTA barrier sync |

---

## Memory Model

| Address Space | Access Scope | Width | Description |
|---------------|--------------|-------|-------------|
| Global | All threads | 48-bit | Main device memory |
| Shared | Single CTA | 32-bit | Low-latency shared memory |
| Local | Single thread | 32-bit | Thread-private stack |
| Constant | All threads (RO) | 48-bit | Read-only constant cache |

---

## Example: Vector Addition

```assembly
// Vector Add Kernel
// Inputs: r1 = A base, r2 = B base, r3 = C base, r4 = length

    mov r5, 4
    mul r4, r4, r5           // r4 = length in bytes
    tid_read r6               // r6 = thread ID
    mul r6, r6, r5           // r6 = thread offset

    add r10, r1, r6          // r10 = A + offset
    add r11, r2, r6          // r11 = B + offset
    add r12, r3, r6          // r12 = C + offset
    mov r20, 0               // r20 = index

loop:
    mul r21, r20, 4          // r21 = byte offset
    cmp r21, r4
    bge r21, r4, done

    load r30, [r10 + r21]    // r30 = A[i]
    load r31, [r11 + r21]    // r31 = B[i]
    add r32, r30, r31        // r32 = A[i] + B[i]
    store [r12 + r21], r32   // C[i] = r32

    add r20, r20, 1
    jmp loop

done:
    ret
```

### Assembling and Running

```bash
cd layer1_isa/sim
# tpt_assemble.py takes positional args (input, then optional output) — no -o flag.
# With no second argument it defaults to replacing the .asm extension with .hex.
python tpt_assemble.py add.asm add.hex
iverilog -g2012 -o sim.vvp ../rtl/*.sv tpt_tb.sv
vvp sim.vvp
```

---

## Performance Considerations

### Memory Coalescing
```assembly
// Bad: Strided access
load r10, [r1 + r6 * 32]

// Good: Coalesced access
load r10, [r1 + r6 * 4]
```

### Optimization Tips
1. **Minimize Divergence**: Avoid branching within a warp
2. **Use FMA**: Combine multiply-add for better throughput
3. **Occupancy**: Keep enough warps to hide latency

---

## Exercises

1. **Dot Product**: Write a kernel that computes the dot product of two vectors
2. **Reduction**: Implement a sum reduction using shared memory
3. **Matrix Multiply**: Write a naive matmul and optimize memory access

---

## Summary

- ✅ TPT ISA uses 32-bit fixed-length instructions
- ✅ 9-stage SIMT pipeline with warp scheduling
- ✅ 6 instruction formats: R, I, M, B, J, V
- ✅ Memory hierarchy: global, shared, local, constant
- ✅ Vector operations for warp-level parallelism

**Next:** [Tutorial 3: Kernel Drivers](03_kernel_drivers.md)
