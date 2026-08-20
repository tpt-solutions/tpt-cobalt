//==============================================================================
// tpt_pregfile.sv — TPT Predicate Register File (8 x 32-bit)
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements 8 predicate registers (P0-P7), each 32 bits wide.
// Each bit corresponds to one lane in a warp (32 lanes = 32 bits).
// Used for:
//   - Predicated execution (conditional per-lane execution)
//   - Vector compare results
//   - Warp divergence tracking
//
// P0 is conventionally the "all-ones" mask (active lanes), but is not
// hardwired — it is set by the warp scheduler on kernel launch.
//==============================================================================

module tpt_pregfile (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Read port 0 (D1 stage)
    input  logic [2:0]        raddr0_i,
    output logic [31:0]       rdata0_o,

    // Read port 1 (D1 stage)
    input  logic [2:0]        raddr1_i,
    output logic [31:0]       rdata1_o,

    // Write port 0 (W1 stage — from predicate ALU)
    input  logic              wren0_i,
    input  logic [2:0]        waddr0_i,
    input  logic [31:0]       wdata0_i,

    // Write port 1 (W1 stage — from vector compare / branch)
    input  logic              wren1_i,
    input  logic [2:0]        waddr1_i,
    input  logic [31:0]       wdata1_i
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Register array — 8 registers, 32 bits each (one bit per warp lane)
  //--------------------------------------------------------------------------
  logic [31:0] pregs [0:NUM_PRED_REGS-1];

  //--------------------------------------------------------------------------
  // Write ports (synchronous write)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int i = 0; i < NUM_PRED_REGS; i++) begin
        pregs[i] <= '0;
      end
    end else begin
      if (wren0_i) begin
        pregs[waddr0_i] <= wdata0_i;
      end
      if (wren1_i) begin
        pregs[waddr1_i] <= wdata1_i;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Read ports (asynchronous read)
  //--------------------------------------------------------------------------
  assign rdata0_o = pregs[raddr0_i];
  assign rdata1_o = pregs[raddr1_i];

endmodule : tpt_pregfile
