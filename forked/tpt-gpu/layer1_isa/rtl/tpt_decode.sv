//==============================================================================
// tpt_decode.sv — TPT Instruction Decoder
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Decodes a 32-bit instruction word into the decoded_instr_t struct.
// Determines instruction format type, opcode, register addresses,
// immediates, and control flags.
//==============================================================================

module tpt_decode (
    input  logic [31:0]           instr_i,
    output tpt_pkg::decoded_instr_t decoded_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Field extraction
  //--------------------------------------------------------------------------
  logic [4:0]  opcode_field;
  logic [4:0]  rd_field;
  logic [4:0]  rs1_field;
  logic [4:0]  rs2_field;
  logic [4:0]  func_field;
  logic [4:0]  subop_field;
  logic [11:0] imm_field;
  logic [21:0] jump_field;
  logic [1:0]  vsz_field;
  logic [1:0]  vdm_field;

  assign opcode_field = instr_i[OPCODE_MSB:OPCODE_LSB];
  assign rd_field     = instr_i[RD_MSB:RD_LSB];
  assign rs1_field    = instr_i[RS1_MSB:RS1_LSB];
  assign rs2_field    = instr_i[RS2_MSB:RS2_LSB];
  assign func_field   = instr_i[FUNC_MSB:FUNC_LSB];
  assign subop_field  = instr_i[VSUBOP_MSB:VSUBOP_LSB];
  assign imm_field    = instr_i[IMM_I_MSB:IMM_I_LSB];
  assign jump_field   = instr_i[JMP_TGT_MSB:JMP_TGT_LSB];
  assign vsz_field    = instr_i[VSZ_MSB:VSZ_LSB];
  assign vdm_field    = instr_i[VDM_MSB:VDM_LSB];

  //--------------------------------------------------------------------------
  // Main decode logic
  //--------------------------------------------------------------------------
  always_comb begin
    decoded_o                   = '0;
    decoded_o.valid             = 1'b0;
    decoded_o.opcode            = opcode_t'(opcode_field);
    decoded_o.is_r_type         = 1'b0;
    decoded_o.is_i_type         = 1'b0;
    decoded_o.is_m_type         = 1'b0;
    decoded_o.is_b_type         = 1'b0;
    decoded_o.is_j_type         = 1'b0;
    decoded_o.is_v_type         = 1'b0;

    unique case (opcode_t'(opcode_field))

      OP_ALU_INT: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_r_type = 1'b1;
        decoded_o.is_alu    = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.rs2       = rs2_field;
        decoded_o.func      = func_field;
        if (func_field inside {FUNC_ADDI, FUNC_SUBI}) begin
          decoded_o.is_r_type = 1'b0;
          decoded_o.is_i_type = 1'b1;
          decoded_o.imm       = imm_field;
        end
      end

      OP_ALU_FP: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_r_type = 1'b1;
        decoded_o.is_alu    = 1'b1;
        decoded_o.is_fp     = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.rs2       = rs2_field;
        decoded_o.func      = func_field;
      end

      OP_ALU_COMP, OP_ALU_LOG: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_r_type = 1'b1;
        decoded_o.is_alu    = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.rs2       = rs2_field;
        decoded_o.func      = func_field;
      end

      OP_MEM_LD: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_m_type = 1'b1;
        decoded_o.is_load   = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.func      = func_field;
        decoded_o.imm       = imm_field;
      end

      OP_MEM_ST: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_m_type = 1'b1;
        decoded_o.is_store  = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.func      = func_field;
        decoded_o.imm       = imm_field;
      end

      OP_MEM_ATOM: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_m_type = 1'b1;
        decoded_o.is_store  = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.rs2       = rs2_field;
        decoded_o.func      = func_field;
        decoded_o.imm       = imm_field;
      end

      OP_CTRL_BR: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_b_type = 1'b1;
        decoded_o.is_branch = 1'b1;
        decoded_o.rs1       = rs1_field;
        decoded_o.rs2       = rs2_field;
        decoded_o.func      = func_field;
        decoded_o.imm       = imm_field;
      end

      OP_CTRL_J: begin
        decoded_o.valid      = 1'b1;
        decoded_o.is_j_type  = 1'b1;
        decoded_o.is_jump    = 1'b1;
        decoded_o.rd         = rd_field;
        decoded_o.rs1        = rs1_field;
        decoded_o.jump_target = jump_field;
        decoded_o.func       = func_field;
      end

      OP_CTRL_SYNC: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_r_type = 1'b1;
        decoded_o.func      = func_field;
      end

      OP_VEC: begin
        decoded_o.valid      = 1'b1;
        decoded_o.is_v_type  = 1'b1;
        decoded_o.is_vector  = 1'b1;
        decoded_o.rd         = rd_field;
        decoded_o.rs1        = rs1_field;
        decoded_o.rs2        = rs2_field;
        decoded_o.func       = func_field;
        decoded_o.subop      = subop_field;
        decoded_o.vec_size   = vsz_field;
        decoded_o.vec_dest_mod = vdm_field;
      end

      OP_TENSOR: begin
        decoded_o.valid      = 1'b1;
        decoded_o.is_v_type  = 1'b1;
        decoded_o.is_tensor  = 1'b1;
        decoded_o.rd         = rd_field;
        decoded_o.rs1        = rs1_field;
        decoded_o.rs2        = rs2_field;
        decoded_o.func       = func_field;
        decoded_o.subop      = subop_field;
        decoded_o.vec_size   = vsz_field;
      end

      OP_SYSTEM: begin
        decoded_o.valid     = 1'b1;
        decoded_o.is_r_type = 1'b1;
        decoded_o.rd        = rd_field;
        decoded_o.rs1       = rs1_field;
        decoded_o.func      = func_field;
        decoded_o.imm       = imm_field;
      end

      OP_PRED: begin
        decoded_o.valid         = 1'b1;
        decoded_o.is_r_type     = 1'b1;
        decoded_o.uses_predicate = 1'b1;
        decoded_o.rd             = rd_field;
        decoded_o.rs1            = rs1_field;
        decoded_o.rs2            = rs2_field;
        decoded_o.func           = func_field;
      end

      default: begin
        decoded_o.valid = 1'b0;
      end
    endcase
  end

endmodule : tpt_decode
