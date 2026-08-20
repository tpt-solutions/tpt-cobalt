//==============================================================================
// tpt_vregfile.sv — TPT Vector Register File (64 x 512-bit)
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements 64 vector registers (V0-V63), each 512 bits wide.
// Each vector holds:
//   - 32 x 16-bit elements (FP16/INT16)
//   - 16 x 32-bit elements (FP32/INT32)
//   -  8 x 64-bit elements (FP64/INT64)
//
// Organized as 4 x 128-bit sub-registers for mixed-precision access.
// V0 is NOT hardwired to zero (unlike scalar R0); all 64 are general-purpose.
//==============================================================================

module tpt_vregfile (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Read port 0 (D1 stage — operand vs1)
    input  logic [5:0]        raddr0_i,
    output logic [511:0]      rdata0_o,

    // Read port 1 (D1 stage — operand vs2)
    input  logic [5:0]        raddr1_i,
    output logic [511:0]      rdata1_o,

    // Read port 2 (D1 stage — accumulator vd)
    input  logic [5:0]        raddr2_i,
    output logic [511:0]      rdata2_o,

    // Write port 0 (W1 stage — from vector/tensor unit)
    input  logic              wren0_i,
    input  logic [5:0]        waddr0_i,
    input  logic [511:0]      wdata0_i,

    // Write port 1 (W1 stage — from LSU vector load)
    input  logic              wren1_i,
    input  logic [5:0]        waddr1_i,
    input  logic [511:0]      wdata1_i
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Register array — 64 registers, 512 bits each
  //--------------------------------------------------------------------------
  logic [511:0] vregs [0:NUM_VECTOR_REGS-1];

  //--------------------------------------------------------------------------
  // Write ports (synchronous write)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int i = 0; i < NUM_VECTOR_REGS; i++) begin
        vregs[i] <= '0;
      end
    end else begin
      if (wren0_i) begin
        vregs[waddr0_i] <= wdata0_i;
      end
      if (wren1_i) begin
        vregs[waddr1_i] <= wdata1_i;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Read ports (asynchronous read)
  //--------------------------------------------------------------------------
  assign rdata0_o = vregs[raddr0_i];
  assign rdata1_o = vregs[raddr1_i];
  assign rdata2_o = vregs[raddr2_i];

  //--------------------------------------------------------------------------
  // Simulation debug: display register contents on request
  //--------------------------------------------------------------------------
  `ifdef SIMULATION
  task automatic dump_vregs;
    for (int i = 0; i < NUM_VECTOR_REGS; i++) begin
      $display("  V%02d = 0x%0128h", i, vregs[i]);
    end
  endtask
  `endif

endmodule : tpt_vregfile
