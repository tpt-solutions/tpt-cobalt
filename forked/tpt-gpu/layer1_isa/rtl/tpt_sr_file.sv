//==============================================================================
// tpt_sr_file.sv — TPT Special Register File (32 x 64-bit)
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements the special-purpose register file for system control,
// thread identification, performance counters, and exception handling.
//
// Register Map (as per ISA Spec §3.4):
//   SR0  LANE_ID    — Current lane index within warp (0–31), RO
//   SR1  WARP_ID    — Current warp index within CTA, RO
//   SR2–SR4  CTA_ID_X/Y/Z — CTA index dimensions, RO
//   SR5–SR7  NTID_X/Y/Z   — Thread count dimensions, RO
//   SR8  CLOCK      — 64-bit cycle counter, RO
//   SR9  STATUS     — Status/exception flags, RW
//   SR10 MASK       — Active lane mask, RW
//   SR12 TVEC       — Trap vector base address, RW
//   SR20–SR25       — Performance counters, RO
//   SR26–SR31       — User-defined, RW
//==============================================================================

module tpt_sr_file (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Current warp/thread context (set by warp scheduler)
    input  logic [4:0]        lane_id_i,
    input  logic [5:0]        warp_id_i,
    input  logic [15:0]       cta_id_x_i,
    input  logic [15:0]       cta_id_y_i,
    input  logic [15:0]       cta_id_z_i,
    input  logic [15:0]       ntid_x_i,
    input  logic [15:0]       ntid_y_i,
    input  logic [15:0]       ntid_z_i,

    // Read port (RDSR instruction)
    input  logic [4:0]        raddr_i,
    output logic [63:0]       rdata_o,

    // Write port (WRSR instruction)
    input  logic              wren_i,
    input  logic [4:0]        waddr_i,
    input  logic [63:0]       wdata_i,

    // Performance counter events (increment inputs)
    input  logic              event_inst_retired_i,
    input  logic              event_l1d_miss_i,
    input  logic              event_l1i_miss_i,
    input  logic              event_branch_mispred_i,
    input  logic              event_warp_stall_i
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Register storage — 32 x 64-bit
  //--------------------------------------------------------------------------
  logic [63:0] sregs [0:NUM_SR_REGS-1];

  //--------------------------------------------------------------------------
  // Performance counter accumulation (free-running)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      sregs[SR_CLOCK]     <= '0;
      sregs[SR_INST_RET]  <= '0;
      sregs[SR_CORE_CYCL] <= '0;
      sregs[SR_L1D_MISS]  <= '0;
      sregs[SR_L1I_MISS]  <= '0;
      sregs[SR_BR_MISPRED] <= '0;
      sregs[SR_WAR_STALL] <= '0;
    end else begin
      sregs[SR_CLOCK]     <= sregs[SR_CLOCK] + 64'd1;
      sregs[SR_CORE_CYCL] <= sregs[SR_CORE_CYCL] + 64'd1;
      if (event_inst_retired_i)
        sregs[SR_INST_RET] <= sregs[SR_INST_RET] + 64'd1;
      if (event_l1d_miss_i)
        sregs[SR_L1D_MISS] <= sregs[SR_L1D_MISS] + 64'd1;
      if (event_l1i_miss_i)
        sregs[SR_L1I_MISS] <= sregs[SR_L1I_MISS] + 64'd1;
      if (event_branch_mispred_i)
        sregs[SR_BR_MISPRED] <= sregs[SR_BR_MISPRED] + 64'd1;
      if (event_warp_stall_i)
        sregs[SR_WAR_STALL] <= sregs[SR_WAR_STALL] + 64'd1;
    end
  end

  //--------------------------------------------------------------------------
  // Writable registers (STATUS, MASK, TVEC, user-defined)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      sregs[SR_STATUS] <= '0;
      sregs[SR_MASK]   <= {32{1'b1}};  // All lanes active by default
      sregs[SR_TVEC]   <= '0;
      for (int i = 16; i < 32; i++) begin
        sregs[i] <= '0;
      end
    end else if (wren_i) begin
      unique case (waddr_i)
        5'd9:  sregs[SR_STATUS] <= wdata_i;
        5'd10: sregs[SR_MASK]   <= wdata_i;
        5'd12: sregs[SR_TVEC]   <= wdata_i;
        default: begin
          if (waddr_i >= 5'd16) begin
            sregs[waddr_i] <= wdata_i;
          end
        end
      endcase
    end
  end

  //--------------------------------------------------------------------------
  // Read port — mux based on address
  //--------------------------------------------------------------------------
  always_comb begin
    rdata_o = '0;
    unique case (raddr_i)
      5'd0:  rdata_o = {59'd0, lane_id_i};
      5'd1:  rdata_o = {58'd0, warp_id_i};
      5'd2:  rdata_o = {48'd0, cta_id_x_i};
      5'd3:  rdata_o = {48'd0, cta_id_y_i};
      5'd4:  rdata_o = {48'd0, cta_id_z_i};
      5'd5:  rdata_o = {48'd0, ntid_x_i};
      5'd6:  rdata_o = {48'd0, ntid_y_i};
      5'd7:  rdata_o = {48'd0, ntid_z_i};
      5'd8:  rdata_o = sregs[SR_CLOCK];
      5'd9:  rdata_o = sregs[SR_STATUS];
      5'd10: rdata_o = sregs[SR_MASK];
      5'd12: rdata_o = sregs[SR_TVEC];
      5'd20: rdata_o = sregs[SR_INST_RET];
      5'd21: rdata_o = sregs[SR_CORE_CYCL];
      5'd22: rdata_o = sregs[SR_L1D_MISS];
      5'd23: rdata_o = sregs[SR_L1I_MISS];
      5'd24: rdata_o = sregs[SR_BR_MISPRED];
      5'd25: rdata_o = sregs[SR_WAR_STALL];
      default: begin
        if (raddr_i >= 5'd16) begin
          rdata_o = sregs[raddr_i];
        end
      end
    endcase
  end

endmodule : tpt_sr_file