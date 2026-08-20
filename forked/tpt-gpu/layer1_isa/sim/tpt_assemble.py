# ==============================================================================
# tpt_assemble.py — TPT ISA Assembler
# ==============================================================================
# Converts TPT assembly (.asm) to hex files for simulation.
# Supports a subset of TPT ISA instructions for testing.
# ==============================================================================

import struct
import re
import sys
import os


class TPTAssembler:
    """TPT ISA assembler — converts assembly to hex machine code."""

    OPCODES = {
        'OP_ALU_INT': 0b00000, 'OP_ALU_FP': 0b00001,
        'OP_ALU_COMP': 0b00010, 'OP_ALU_LOG': 0b00011,
        'OP_MEM_LD': 0b00100, 'OP_MEM_ST': 0b00101,
        'OP_CTRL_BR': 0b00111, 'OP_CTRL_J': 0b01000,
        'OP_CTRL_SYNC': 0b01001, 'OP_VEC': 0b01010,
        'OP_TENSOR': 0b01011, 'OP_SYSTEM': 0b01110,
        'OP_PRED': 0b01111,
    }

    ALU_FUNCS = {
        'ADD': (0b00000, 'OP_ALU_INT', 'R'), 'ADDI': (0b00001, 'OP_ALU_INT', 'I'),
        'SUB': (0b00010, 'OP_ALU_INT', 'R'), 'SUBI': (0b00011, 'OP_ALU_INT', 'I'),
        'MUL': (0b00100, 'OP_ALU_INT', 'R'), 'MULHI': (0b00101, 'OP_ALU_INT', 'R'),
        'DIV': (0b00110, 'OP_ALU_INT', 'R'), 'MOD': (0b00111, 'OP_ALU_INT', 'R'),
        'AND': (0b01000, 'OP_ALU_INT', 'R'), 'OR': (0b01001, 'OP_ALU_INT', 'R'),
        'XOR': (0b01010, 'OP_ALU_INT', 'R'),
        'SLL': (0b01011, 'OP_ALU_INT', 'R'), 'SRL': (0b01100, 'OP_ALU_INT', 'R'),
        'SRA': (0b01101, 'OP_ALU_INT', 'R'),
        'CLZ': (0b01110, 'OP_ALU_INT', 'R'), 'POPC': (0b01111, 'OP_ALU_INT', 'R'),
        'MIN': (0b10000, 'OP_ALU_INT', 'R'), 'MAX': (0b10001, 'OP_ALU_INT', 'R'),
        'ABS': (0b10010, 'OP_ALU_INT', 'R'), 'NEG': (0b10011, 'OP_ALU_INT', 'R'),
    }

    FP_FUNCS = {
        'FADD': (0b00000, 'OP_ALU_FP', 'R'), 'FSUB': (0b00001, 'OP_ALU_FP', 'R'),
        'FMUL': (0b00010, 'OP_ALU_FP', 'R'), 'FDIV': (0b00011, 'OP_ALU_FP', 'R'),
        'FADD16': (0b00100, 'OP_ALU_FP', 'R'), 'FMUL16': (0b00101, 'OP_ALU_FP', 'R'),
        'F2I': (0b01000, 'OP_ALU_FP', 'R'), 'I2F': (0b01001, 'OP_ALU_FP', 'R'),
        'FMA': (0b01100, 'OP_ALU_FP', 'R'), 'FSQRT': (0b01111, 'OP_ALU_FP', 'R'),
    }

    MEM_FUNCS = {
        'LB': (0b00000, 'OP_MEM_LD', 'M'), 'LBU': (0b00001, 'OP_MEM_LD', 'M'),
        'LH': (0b00010, 'OP_MEM_LD', 'M'), 'LHU': (0b00011, 'OP_MEM_LD', 'M'),
        'LW': (0b00100, 'OP_MEM_LD', 'M'), 'LD': (0b00101, 'OP_MEM_LD', 'M'),
        'LV': (0b00110, 'OP_MEM_LD', 'M'),
        'SB': (0b00111, 'OP_MEM_ST', 'M'), 'SH': (0b01000, 'OP_MEM_ST', 'M'),
        'SW': (0b01001, 'OP_MEM_ST', 'M'), 'SD': (0b01010, 'OP_MEM_ST', 'M'),
        'SV': (0b01011, 'OP_MEM_ST', 'M'),
    }

    CTRL_FUNCS = {
        'BEQ': (0b00000, 'OP_CTRL_BR', 'B'), 'BNE': (0b00001, 'OP_CTRL_BR', 'B'),
        'BLT': (0b00010, 'OP_CTRL_BR', 'B'), 'BGE': (0b00011, 'OP_CTRL_BR', 'B'),
        'BLTU': (0b00100, 'OP_CTRL_BR', 'B'), 'BGEU': (0b00101, 'OP_CTRL_BR', 'B'),
        'JAL': (0b00110, 'OP_CTRL_J', 'J'), 'JALR': (0b00111, 'OP_CTRL_J', 'J'),
        'RET': (0b01000, 'OP_CTRL_J', 'J'),
    }

    def __init__(self):
        self.labels = {}
        self.current_addr = 0
        self.org_addr = 0

    def parse_operand(self, op_str):
        op_str = op_str.strip()
        if op_str.startswith('R'):
            return ('reg', int(op_str[1:]))
        elif op_str.startswith('0x'):
            return ('imm', int(op_str, 16))
        else:
            return ('imm', int(op_str))

    def encode_r_type(self, func, opcode, rd, rs1, rs2):
        return (opcode << 27) | (rd << 22) | (rs1 << 17) | (rs2 << 12) | (func << 7)

    def encode_i_type(self, func, opcode, rd, rs1, imm):
        return (opcode << 27) | (rd << 22) | (rs1 << 17) | ((imm & 0xFFF) << 5) | func

    def encode_m_type(self, func, opcode, rd, rs1, offset):
        return (opcode << 27) | (rd << 22) | (rs1 << 17) | ((offset & 0xFFF) << 5) | func

    def encode_b_type(self, func, opcode, rs1, rs2, offset):
        offset_shifted = (offset >> 2) & 0xFFF
        return (opcode << 27) | (rs1 << 22) | (rs2 << 17) | (func << 12) | offset_shifted

    def encode_j_type(self, func, opcode, rd, target):
        return (opcode << 27) | ((target & 0x3FFFFF) << 5) | func

    def parse_line(self, line):
        """Parse a single assembly line."""
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            return None
        if line.startswith('.org'):
            self.org_addr = int(line.split()[1], 0)
            self.current_addr = self.org_addr
            return None

        line = re.sub(r'[;#].*$', '', line).strip()
        if not line:
            return None

        # Label
        if ':' in line:
            label_part, rest = line.split(':', 1)
            label_part = label_part.strip()
            if label_part:
                self.labels[label_part] = self.current_addr
            line = rest.strip()
            if not line:
                return None

        parts = line.replace(',', ' ').split()
        if not parts:
            return None

        mnemonic = parts[0].upper()
        operands = parts[1:]

        for table in [self.ALU_FUNCS, self.FP_FUNCS, self.MEM_FUNCS, self.CTRL_FUNCS]:
            if mnemonic in table:
                func, opcode_name, fmt = table[mnemonic]
                opcode = self.OPCODES[opcode_name]
                return self._encode(opcode, func, fmt, operands)

        print(f"ERROR: Unknown instruction '{mnemonic}'", file=sys.stderr)
        return None

    def _encode(self, opcode, func, fmt, operands):
        """Encode based on format type."""
        if fmt == 'R':
            rd = self.parse_operand(operands[0])[1]
            rs1 = self.parse_operand(operands[1])[1] if len(operands) > 1 else 0
            rs2 = self.parse_operand(operands[2])[1] if len(operands) > 2 else 0
            return self.encode_r_type(func, opcode, rd, rs1, rs2)
        elif fmt == 'I':
            rd = self.parse_operand(operands[0])[1]
            rs1 = self.parse_operand(operands[1])[1] if len(operands) > 1 else 0
            imm = self.parse_operand(operands[2])[1] if len(operands) > 2 else 0
            return self.encode_i_type(func, opcode, rd, rs1, imm)
        elif fmt == 'M':
            rd = self.parse_operand(operands[0])[1]
            rs1 = self.parse_operand(operands[1])[1] if len(operands) > 1 else 0
            offset = self.parse_operand(operands[2])[1] if len(operands) > 2 else 0
            return self.encode_m_type(func, opcode, rd, rs1, offset)
        elif fmt == 'B':
            rs1 = self.parse_operand(operands[0])[1]
            rs2 = self.parse_operand(operands[1])[1]
            target_str = operands[2] if len(operands) > 2 else '0'
            if target_str in self.labels:
                offset = self.labels[target_str] - self.current_addr
            else:
                offset = int(target_str)
            return self.encode_b_type(func, opcode, rs1, rs2, offset)
        elif fmt == 'J':
            rd = self.parse_operand(operands[0])[1] if len(operands) > 0 else 0
            target_str = operands[1] if len(operands) > 1 else '0'
            if target_str in self.labels:
                target = self.labels[target_str]
            else:
                target = int(target_str)
            return self.encode_j_type(func, opcode, rd, target)
        return 0


def assemble(input_path, output_path):
    """Assemble .asm to .hex file."""
    asm = TPTAssembler()

    # First pass: scan labels
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.org'):
                asm.current_addr = int(line.split()[1], 0)
            elif ':' in line and not line.lstrip().startswith((';', '#')):
                label = line.split(':', 1)[0].strip()
                if label and not any(c in label for c in ' \t'):
                    asm.labels[label] = asm.current_addr
                # Also count instructions after label on same line
                rest = line.split(':', 1)[1].strip()
                if rest and not rest.startswith((';', '#')):
                    asm.current_addr += 4
            elif line and not line.startswith((';', '#', '.', ' ')):
                asm.current_addr += 4

    # Second pass: encode
    asm.current_addr = asm.org_addr
    instructions = []
    with open(input_path, 'r') as f:
        for line in f:
            instr = asm.parse_line(line)
            if instr is not None:
                instructions.append(instr)
                asm.current_addr += 4

    with open(output_path, 'w') as f:
        for i, instr in enumerate(instructions):
            f.write(f"{instr:08x}\n")

    print(f"Assembled {len(instructions)} instructions -> {output_path}")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tpt_assemble.py <input.asm> [output.hex]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.asm', '.hex')
    assemble(inp, out)
