//==============================================================================
// tpt_pipeline.sv — TPT Pipeline Control
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Manages pipeline stages and pipeline register interfaces.
// Handles stall, flush, and inter-stage handshaking.
//==============================================================================

module tpt_pipeline (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Stall / flush from control
    input  logic              stall_fetch_i,
    input  logic              stall_decode_i,
    input  logic              flush_decode_i,
    input  logic              flush_execute_i,

    // PC tracking
    output logic [31:0]       pc_fetch_o,
    input  logic [31:0]       next_pc_i,
    input  logic              pc_sel_i,

    // Pipeline register dumps for forwarding
    output logic [4:0]        ex_rd_o,
    output logic              ex_reg_wren_o,
    output logic [4:0]        mem_rd_o,
    output logic              mem_reg_wren_o,
    output logic [4:0]        wb_rd_o,
    output logic              wb_reg_wren_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // PC (F1 stage)
  //--------------------------------------------------------------------------
  logic [31:0] pc;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      pc <= '0;
    end else if (~stall_fetch_i) begin
      pc <= next_pc_i;
    end
  end

  assign pc_fetch_o = pc;

  //--------------------------------------------------------------------------
  // Pipeline registers (simplified — actual forwarding addresses)
  //--------------------------------------------------------------------------
  logic [4:0]  ex_rd, mem_rd, wb_rd;
  logic        ex_wren, mem_wren, wb_wren;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      ex_rd    <= '0;
      ex_wren  <= 1'b0;
      mem_rd   <= '0;
      mem_wren <= 1'b0;
      wb_rd    <= '0;
      wb_wren  <= 1'b0;
    end else begin
      // D1 → E1 (Decode → Execute)
      if (flush_decode_i) begin
        ex_rd   <= '0;
        ex_wren <= 1'b0;
      end else if (~stall_decode_i) begin
        // In a full implementation, these would come from decode stage
      end

      // E1 → E3 (Execute → Memory)
      if (flush_execute_i) begin
        mem_rd   <= '0;
        mem_wren <= 1'b0;
      end else begin
        mem_rd   <= ex_rd;
        mem_wren <= ex_wren;
      end

      // E3 → W1 (Memory → Writeback)
      wb_rd   <= mem_rd;
      wb_wren <= mem_wren;
    end
  end

  assign ex_rd_o  = ex_rd;
  assign ex_reg_wren_o = ex_wren;
  assign mem_rd_o = mem_rd;
  assign mem_reg_wren_o = mem_wren;
  assign wb_rd_o  = wb_rd;
  assign wb_reg_wren_o = wb_wren;

endmodule : tpt_pipeline
