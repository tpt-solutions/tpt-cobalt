//==============================================================================
// tpt_core.sv — TPT Core Top-Level Module
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Top-level integration of all TPT pipeline stages:
// - F1/F2: Fetch (instruction memory)
// - D1/D2: Decode + Register read
// - E1/E2: ALU / Branch
// - E3/E4: LSU (memory access)
// - W1:    Writeback
//==============================================================================

module tpt_core (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Instruction memory interface
    output logic [31:0]       imem_addr_o,
    input  logic [31:0]       imem_rdata_i,
    output logic              imem_req_o,
    input  logic              imem_ack_i,

    // Data memory interface
    output logic [31:0]       dmem_addr_o,
    output logic              dmem_req_o,
    output logic              dmem_we_o,
    output logic [3:0]        dmem_be_o,
    output logic [31:0]       dmem_wdata_o,
    input  logic              dmem_ack_i,
    input  logic [31:0]       dmem_rdata_i
);

  import tpt_pkg::*;

  //==========================================================================
  // Signal declarations
  //==========================================================================
  logic [31:0] pc_fetch, next_pc;
  logic        pc_sel;
  logic [31:0] instr_word;
  decoded_instr_t decoded;
  logic [31:0] rf_rdata0, rf_rdata1, rf_rdata2;
  logic [31:0] forward_src_a, forward_src_b;
  logic [1:0]  forward_a, forward_b;
  logic [31:0] alu_result;
  logic        alu_valid, alu_zero, alu_negative, alu_overflow;
  logic [31:0] lsu_data;
  logic        lsu_valid, lsu_misaligned;
  logic [4:0]  ex_rd, mem_rd, wb_rd;
  logic        ex_wren, mem_wren, wb_wren;
  logic        stall_fetch, stall_decode, flush_decode, flush_execute;
  logic        exception;
  exception_t  exc_code;
  logic [31:0] exc_pc;

  //==========================================================================
  // Stage F1: Fetch — Instruction memory access
  //==========================================================================
  assign imem_addr_o = pc_fetch;
  assign imem_req_o  = ~stall_fetch;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      instr_word <= '0;
    end else if (imem_ack_i) begin
      instr_word <= imem_rdata_i;
    end
  end

  //==========================================================================
  // Stage D1: Decode
  //==========================================================================
  tpt_decode decode_inst (
      .instr_i    (instr_word),
      .decoded_o  (decoded)
  );

  //==========================================================================
  // Stage D1: Register file read
  //==========================================================================
  tpt_regfile regfile_inst (
      .clk_i      (clk_i),
      .rst_n_i    (rst_n_i),
      .raddr0_i   (decoded.rs1),
      .rdata0_o   (rf_rdata0),
      .raddr1_i   (decoded.rs2),
      .rdata1_o   (rf_rdata1),
      .raddr2_i   (decoded.rs2),
      .rdata2_o   (rf_rdata2),
      .wren0_i    (wb_wren),
      .waddr0_i   (wb_rd),
      .wdata0_i   (alu_result),
      .wren1_i    (lsu_valid),
      .waddr1_i   (mem_rd),
      .wdata1_i   (lsu_data)
  );

  //==========================================================================
  // Stage D2: Forwarding mux
  //==========================================================================
  always_comb begin
    unique case (forward_a)
      2'b00:    forward_src_a = rf_rdata0;
      2'b01:    forward_src_a = alu_result;
      2'b10:    forward_src_a = lsu_data;
      default:  forward_src_a = rf_rdata0;
    endcase

    unique case (forward_b)
      2'b00:    forward_src_b = rf_rdata1;
      2'b01:    forward_src_b = alu_result;
      2'b10:    forward_src_b = lsu_data;
      default:  forward_src_b = rf_rdata1;
    endcase
  end

  //==========================================================================
  // Stage E1/E2: ALU / Compute
  //==========================================================================
  tpt_alu alu_inst (
      .clk_i      (clk_i),
      .rst_n_i    (rst_n_i),
      .valid_i    (decoded.valid && (decoded.is_alu || decoded.is_fp)),
      .is_fp_i    (decoded.is_fp),
      .func_i     (decoded.func),
      .rs1_i      (forward_src_a),
      .rs2_i      (forward_src_b),
      .imm_i      ({{20{decoded.imm[11]}}, decoded.imm}),
      .use_imm_i  (decoded.is_i_type),
      .result_o   (alu_result),
      .valid_o    (alu_valid),
      .zero_o     (alu_zero),
      .negative_o (alu_negative),
      .overflow_o (alu_overflow)
  );

  //==========================================================================
  // Stage E3/E4: Load/Store Unit
  //==========================================================================
  tpt_lsu lsu_inst (
      .clk_i        (clk_i),
      .rst_n_i      (rst_n_i),
      .valid_i      (decoded.valid && (decoded.is_load || decoded.is_store)),
      .is_load_i    (decoded.is_load),
      .is_store_i   (decoded.is_store),
      .func_i       (decoded.func),
      .base_addr_i  (forward_src_a + {{20{decoded.imm[11]}}, decoded.imm}),
      .store_data_i (forward_src_b),
      .mem_addr_o   (dmem_addr_o),
      .mem_req_o    (dmem_req_o),
      .mem_we_o     (dmem_we_o),
      .mem_be_o     (dmem_be_o),
      .mem_wdata_o  (dmem_wdata_o),
      .mem_ack_i    (dmem_ack_i),
      .mem_rdata_i  (dmem_rdata_i),
      .load_data_o  (lsu_data),
      .valid_o      (lsu_valid),
      .misaligned_o (lsu_misaligned)
  );

  //==========================================================================
  // W1: Writeback tracking
  //==========================================================================
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      ex_rd   <= '0;
      ex_wren <= 1'b0;
      mem_rd   <= '0;
      mem_wren <= 1'b0;
      wb_rd    <= '0;
      wb_wren  <= 1'b0;
    end else begin
      if (flush_decode) begin
        ex_rd   <= '0;
        ex_wren <= 1'b0;
      end else begin
        ex_rd   <= decoded.rd;
        ex_wren <= decoded.valid && (decoded.is_alu || decoded.is_fp || decoded.is_load);
      end
      mem_rd   <= ex_rd;
      mem_wren <= ex_wren;
      wb_rd    <= mem_rd;
      wb_wren  <= mem_wren && decoded.is_alu;
    end
  end

  //==========================================================================
  // Control Unit
  //==========================================================================
  tpt_ctrl ctrl_inst (
      .clk_i           (clk_i),
      .rst_n_i         (rst_n_i),
      .decoded_i       (decoded),
      .ex_rd_i         (ex_rd),
      .ex_reg_wren_i   (ex_wren),
      .mem_rd_i        (mem_rd),
      .mem_reg_wren_i  (mem_wren),
      .wb_rd_i         (wb_rd),
      .wb_reg_wren_i   (wb_wren),
      .alu_zero_i      (alu_zero),
      .alu_negative_i  (alu_negative),
      .alu_result_i    (alu_result),
      .pc_i            (pc_fetch),
      .next_pc_o       (next_pc),
      .pc_sel_o        (pc_sel),
      .stall_fetch_o   (stall_fetch),
      .stall_decode_o  (stall_decode),
      .flush_decode_o  (flush_decode),
      .flush_execute_o (flush_execute),
      .forward_a_o     (forward_a),
      .forward_b_o     (forward_b),
      .exception_o     (exception),
      .exception_code_o (exc_code),
      .exception_pc_o  (exc_pc)
  );

  //==========================================================================
  // PC update
  //==========================================================================
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      pc_fetch <= '0;
    end else if (~stall_fetch) begin
      pc_fetch <= next_pc;
    end
  end

endmodule : tpt_core
