//==============================================================================
// tpt_ctrl.sv — TPT Control Unit
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Generates all control signals for the pipeline, including:
// - Hazard detection and stall generation
// - Forwarding control
// - Branch / jump target calculation
// - Exception handling
//==============================================================================

module tpt_ctrl (
    input  logic              clk_i,
    input  logic              rst_n_i,
    input  tpt_pkg::decoded_instr_t decoded_i,
    input  logic [4:0]        ex_rd_i,
    input  logic              ex_reg_wren_i,
    input  logic [4:0]        mem_rd_i,
    input  logic              mem_reg_wren_i,
    input  logic [4:0]        wb_rd_i,
    input  logic              wb_reg_wren_i,
    input  logic              alu_zero_i,
    input  logic              alu_negative_i,
    input  logic [31:0]       alu_result_i,
    input  logic [31:0]       pc_i,
    output logic [31:0]       next_pc_o,
    output logic              pc_sel_o,
    output logic              stall_fetch_o,
    output logic              stall_decode_o,
    output logic              flush_decode_o,
    output logic              flush_execute_o,
    output logic [1:0]        forward_a_o,
    output logic [1:0]        forward_b_o,
    output logic              exception_o,
    output tpt_pkg::exception_t exception_code_o,
    output logic [31:0]       exception_pc_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Hazard Detection
  //--------------------------------------------------------------------------
  logic        load_use_hazard;
  logic [4:0]  data_hazard_rs1, data_hazard_rs2;

  assign data_hazard_rs1 = decoded_i.rs1;
  assign data_hazard_rs2 = decoded_i.rs2;

  // Load-use hazard: next instruction reads the register being loaded
  assign load_use_hazard = decoded_i.valid
        && (ex_reg_wren_i && (ex_rd_i != 5'd0)
            && ((data_hazard_rs1 == ex_rd_i) || (data_hazard_rs2 == ex_rd_i)));

  //--------------------------------------------------------------------------
  // Stall / Flush generation
  //--------------------------------------------------------------------------
  always_comb begin
    stall_fetch_o   = load_use_hazard;
    stall_decode_o  = load_use_hazard;
    flush_decode_o  = decoded_i.is_branch || decoded_i.is_jump;
    flush_execute_o = decoded_i.is_branch || decoded_i.is_jump;
  end

  //--------------------------------------------------------------------------
  // Forwarding logic
  //--------------------------------------------------------------------------
  always_comb begin
    forward_a_o = 2'b00;
    forward_b_o = 2'b00;

    if (decoded_i.valid && (data_hazard_rs1 != 5'd0)) begin
      if      (ex_reg_wren_i && (ex_rd_i == data_hazard_rs1)) forward_a_o = 2'b01;
      else if (mem_reg_wren_i && (mem_rd_i == data_hazard_rs1)) forward_a_o = 2'b10;
      else if (wb_reg_wren_i && (wb_rd_i == data_hazard_rs1)) forward_a_o = 2'b11;
    end

    if (decoded_i.valid && (data_hazard_rs2 != 5'd0)) begin
      if      (ex_reg_wren_i && (ex_rd_i == data_hazard_rs2)) forward_b_o = 2'b01;
      else if (mem_reg_wren_i && (mem_rd_i == data_hazard_rs2)) forward_b_o = 2'b10;
      else if (wb_reg_wren_i && (wb_rd_i == data_hazard_rs2)) forward_b_o = 2'b11;
    end
  end


  //--------------------------------------------------------------------------
  // Branch / Jump target calculation
  //--------------------------------------------------------------------------
  logic [31:0] branch_target;
  logic [31:0] jump_target;
  logic        branch_taken;

  assign branch_target = pc_i + {{18{decoded_i.imm[11]}}, decoded_i.imm, 2'b00};
  assign jump_target = {decoded_i.jump_target, 2'b00};

  always_comb begin
    branch_taken = 1'b0;
    if (decoded_i.valid && decoded_i.is_branch) begin
      unique case (ctrl_func_t'(decoded_i.func))
        CTRL_BEQ:  branch_taken = alu_zero_i;
        CTRL_BNE:  branch_taken = ~alu_zero_i;
        CTRL_BLT:  branch_taken = alu_negative_i;
        CTRL_BGE:  branch_taken = ~alu_negative_i || alu_zero_i;
        CTRL_BLTU: branch_taken = alu_negative_i;
        CTRL_BGEU: branch_taken = ~alu_negative_i || alu_zero_i;
        default:   branch_taken = 1'b0;
      endcase
    end
  end

  always_comb begin
    if (decoded_i.is_jump) begin
      next_pc_o = jump_target;
      pc_sel_o  = 1'b1;
    end else if (branch_taken) begin
      next_pc_o = branch_target;
      pc_sel_o  = 1'b1;
    end else begin
      next_pc_o = pc_i + 32'd4;
      pc_sel_o  = 1'b0;
    end
  end

  //--------------------------------------------------------------------------
  // Exception handling
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      exception_o      <= 1'b0;
      exception_code_o <= EXC_NONE;
      exception_pc_o   <= '0;
    end else begin
      exception_o      <= 1'b0;
      exception_code_o <= EXC_NONE;
      exception_pc_o   <= pc_i;

      if (decoded_i.valid && !decoded_i.is_r_type && !decoded_i.is_i_type
          && !decoded_i.is_m_type && !decoded_i.is_b_type
          && !decoded_i.is_j_type && !decoded_i.is_v_type) begin
        exception_o      <= 1'b1;
        exception_code_o <= EXC_ILLEGAL_INST;
      end
    end
  end

endmodule : tpt_ctrl
