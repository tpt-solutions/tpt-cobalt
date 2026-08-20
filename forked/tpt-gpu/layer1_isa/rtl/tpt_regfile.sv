//==============================================================================
// tpt_regfile.sv — TPT Scalar Register File (32 x 32-bit)
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements 32 general-purpose scalar registers (R0-R31).
// R0 is hardwired to zero.
// Three read ports, two write ports (for pipeline forwarding).
//==============================================================================

module tpt_regfile (
    input  logic        clk_i,
    input  logic        rst_n_i,

    // Read port 0 (D1 stage)
    input  logic [4:0]  raddr0_i,
    output logic [31:0] rdata0_o,

    // Read port 1 (D1 stage)
    input  logic [4:0]  raddr1_i,
    output logic [31:0] rdata1_o,

    // Read port 2 (D2 stage, for branches)
    input  logic [4:0]  raddr2_i,
    output logic [31:0] rdata2_o,

    // Write port 0 (W1 stage, from ALU)
    input  logic        wren0_i,
    input  logic [4:0]  waddr0_i,
    input  logic [31:0] wdata0_i,

    // Write port 1 (W1 stage, from LSU)
    input  logic        wren1_i,
    input  logic [4:0]  waddr1_i,
    input  logic [31:0] wdata1_i
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Register array — 32 registers, 32 bits each
  //--------------------------------------------------------------------------
  logic [31:0] regs [0:NUM_SCALAR_REGS-1];

  //--------------------------------------------------------------------------
  // Write ports (synchronous write)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int i = 0; i < NUM_SCALAR_REGS; i++) begin
        regs[i] <= '0;
      end
    end else begin
      if (wren0_i && (waddr0_i != 5'd0)) begin
        regs[waddr0_i] <= wdata0_i;
      end
      if (wren1_i && (waddr1_i != 5'd0)) begin
        regs[waddr1_i] <= wdata1_i;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Read ports (asynchronous read)
  //--------------------------------------------------------------------------
  assign rdata0_o = (raddr0_i == 5'd0) ? '0 : regs[raddr0_i];
  assign rdata1_o = (raddr1_i == 5'd0) ? '0 : regs[raddr1_i];
  assign rdata2_o = (raddr2_i == 5'd0) ? '0 : regs[raddr2_i];

endmodule : tpt_regfile
