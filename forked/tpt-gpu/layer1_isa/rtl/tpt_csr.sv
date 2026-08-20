//==============================================================================
// tpt_csr.sv — TPT Control/Status Registers (MMIO Interface)
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Host-accessible MMIO control/status registers.
// Register Map:
//   0x000  TPT_CTRL       — Control (boot, reset, enable)
//   0x004  TPT_STATUS     — Status (ready, idle, error)
//   0x008  TPT_INTR       — Interrupt status
//   0x00C  TPT_INTR_MASK  — Interrupt mask
//   0x010  TPT_FENCE_SEQ  — Fence sequence number
//   0x014  TPT_DOORBELL   — Doorbell (submit work)
//   0x020  TPT_SCHED_CTRL — Scheduler control
//   0x024  TPT_NUM_WARPS  — Active warp count (RO)
//   0x030  TPT_QUERY_VRAM — Total VRAM
//   0x038  TPT_QUERY_CTAS — Max concurrent CTAs
//   0x03C  TPT_QUERY_VER  — Driver version
//==============================================================================

module tpt_csr (
    input  logic              clk_i,
    input  logic              rst_n_i,

    input  logic [11:0]       mmio_addr_i,
    input  logic              mmio_req_i,
    input  logic              mmio_we_i,
    input  logic [31:0]       mmio_wdata_i,
    output logic [31:0]       mmio_rdata_o,
    output logic              mmio_ack_o,

    input  logic              gpu_ready_i,
    input  logic              gpu_idle_i,
    input  logic              gpu_error_i,
    input  logic [5:0]        num_active_warps_i,
    input  logic [63:0]       vram_total_i,
    input  logic [63:0]       vram_free_i,

    output logic              sched_enable_o,
    output logic              gpu_reset_o,
    output logic              gpu_boot_o,
    output logic [5:0]        doorbell_warp_o,
    output logic              doorbell_valid_o,

    output logic              intr_o
);

  import tpt_pkg::*;

  logic [31:0] reg_ctrl;
  logic [31:0] reg_intr;
  logic [31:0] reg_intr_mask;
  logic [31:0] reg_fence_seq;
  logic [31:0] reg_sched_ctrl;

  // TPT_CTRL: [0]=BOOT, [1]=RESET, [2]=ENABLE, [3]=FLUSH
  assign gpu_boot_o     = reg_ctrl[0];
  assign gpu_reset_o    = reg_ctrl[1];
  assign sched_enable_o = reg_sched_ctrl[0];

  // Doorbell (write-only, one-cycle pulse)
  logic [5:0]  doorbell_warp;
  logic        doorbell_pulse;

  assign doorbell_warp_o  = doorbell_warp;
  assign doorbell_valid_o = doorbell_pulse;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      doorbell_pulse <= 1'b0;
      doorbell_warp  <= '0;
    end else begin
      doorbell_pulse <= 1'b0;
      if (mmio_req_i && mmio_we_i && (mmio_addr_i == 12'h014)) begin
        doorbell_warp  <= mmio_wdata_i[5:0];
        doorbell_pulse <= 1'b1;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Register write decode
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      reg_ctrl      <= '0;
      reg_intr      <= '0;
      reg_intr_mask <= '0;
      reg_fence_seq <= '0;
      reg_sched_ctrl <= '0;
    end else if (mmio_req_i && mmio_we_i) begin
      unique case (mmio_addr_i)
        12'h000: reg_ctrl       <= mmio_wdata_i;
        12'h008: reg_intr       <= reg_intr & ~mmio_wdata_i; // W1C (write-1-clear)
        12'h00C: reg_intr_mask  <= mmio_wdata_i;
        12'h010: reg_fence_seq  <= mmio_wdata_i;
        12'h020: reg_sched_ctrl <= mmio_wdata_i;
        default: ;  // Read-only or reserved registers — ignore writes
      endcase
    end
  end

  //--------------------------------------------------------------------------
  // Read decode
  //--------------------------------------------------------------------------
  always_comb begin
    mmio_rdata_o = '0;
    unique case (mmio_addr_i)
      12'h000: mmio_rdata_o = reg_ctrl;
      12'h004: mmio_rdata_o = {28'd0, gpu_error_i, gpu_idle_i, gpu_ready_i, reg_ctrl[0]};
      12'h008: mmio_rdata_o = reg_intr;
      12'h00C: mmio_rdata_o = reg_intr_mask;
      12'h010: mmio_rdata_o = reg_fence_seq;
      12'h024: mmio_rdata_o = {26'd0, num_active_warps_i};
      12'h030: mmio_rdata_o = vram_total_i[31:0];
      12'h038: mmio_rdata_o = {26'd0, 5'd16};  // NUM_CTAS = 16
      12'h03C: mmio_rdata_o = {(16'd1), (16'd0)};  // version 1.0
      default: mmio_rdata_o = '0;
    endcase
  end

  // Ack always returns 1 cycle after request (simplified)
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i)
      mmio_ack_o <= 1'b0;
    else
      mmio_ack_o <= mmio_req_i;
  end

  //--------------------------------------------------------------------------
  // Interrupt generation
  //--------------------------------------------------------------------------
  assign intr_o = |(reg_intr & reg_intr_mask);

endmodule : tpt_csr