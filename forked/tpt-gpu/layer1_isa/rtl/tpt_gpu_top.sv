//==============================================================================
// tpt_gpu_top.sv — TPT GPU Top-Level Integration
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Top-level module integrating the full TPT GPU subsystem for silicon.
// Host Interface: PCIe-like bus with MMIO + DMA.
//
// Silicon hierarchy:
//   tpt_gpu_top
//     ├─ tpt_csr          (MMIO / BAR0 register block)
//     ├─ tpt_warp_sched   (global warp dispatcher)
//     ├─ tpt_core × NUM_SM (scalar+vector+tensor pipelines)
//     ├─ tpt_icache × NUM_SM
//     ├─ tpt_dcache × NUM_SM
//     ├─ tpt_l2cache      (unified L2, 2 MiB, 4-way SA)
//     └─ tpt_mem_ctrl     (GDDR6 / HBM2 memory controller)
//==============================================================================

module tpt_gpu_top #(
    parameter int NUM_SM = 1
) (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Host MMIO interface (PCIe BAR0)
    input  logic [11:0]       host_mmio_addr_i,
    input  logic              host_mmio_req_i,
    input  logic              host_mmio_we_i,
    input  logic [31:0]       host_mmio_wdata_i,
    output logic [31:0]       host_mmio_rdata_o,
    output logic              host_mmio_ack_o,
    output logic              host_intr_o,

    // GDDR6 / HBM2 PHY interface (to off-chip memory)
    output logic [2:0]        phy_cmd_o,
    output logic [3:0]        phy_bank_o,
    output logic [14:0]       phy_row_o,
    output logic [9:0]        phy_col_o,
    output logic [511:0]      phy_wdata_o,
    input  logic [511:0]      phy_rdata_i,
    input  logic              phy_rdata_valid_i,
    output logic              phy_cs_n_o,
    output logic              phy_cke_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Internal control signals
  //--------------------------------------------------------------------------
  logic        sched_enable, gpu_reset, gpu_boot;
  logic [5:0]  doorbell_warp;
  logic        doorbell_valid;
  logic [5:0]  active_warp_id;
  logic [31:0] active_warp_pc, active_warp_mask;
  logic [5:0]  num_active_warps;
  logic [NUM_WARPS-1:0] warp_active;
  logic        gpu_ready, gpu_idle, gpu_error;
  logic [63:0] vram_total, vram_free;

  assign vram_total = 64'd8589934592;  // 8 GiB
  assign vram_free  = 64'd8589934592;

  //--------------------------------------------------------------------------
  // CSR (MMIO register block — PCIe BAR0)
  //--------------------------------------------------------------------------
  tpt_csr csr_inst (
      .clk_i              (clk_i),
      .rst_n_i            (rst_n_i),
      .mmio_addr_i        (host_mmio_addr_i),
      .mmio_req_i         (host_mmio_req_i),
      .mmio_we_i          (host_mmio_we_i),
      .mmio_wdata_i       (host_mmio_wdata_i),
      .mmio_rdata_o       (host_mmio_rdata_o),
      .mmio_ack_o         (host_mmio_ack_o),
      .gpu_ready_i        (gpu_ready),
      .gpu_idle_i         (gpu_idle),
      .gpu_error_i        (gpu_error),
      .num_active_warps_i (num_active_warps),
      .vram_total_i       (vram_total),
      .vram_free_i        (vram_free),
      .sched_enable_o     (sched_enable),
      .gpu_reset_o        (gpu_reset),
      .gpu_boot_o         (gpu_boot),
      .doorbell_warp_o    (doorbell_warp),
      .doorbell_valid_o   (doorbell_valid),
      .intr_o             (host_intr_o)
  );

  //--------------------------------------------------------------------------
  // Warp Scheduler
  //--------------------------------------------------------------------------
  logic        warp_stall, warp_done, warp_branch_taken;
  logic [31:0] warp_branch_target;
  logic [15:0] cta_id_x, cta_id_y, cta_id_z;

  tpt_warp_sched sched_inst (
      .clk_i                (clk_i),
      .rst_n_i              (rst_n_i),
      .sched_enable_i       (sched_enable),
      .active_warp_id_o     (active_warp_id),
      .active_warp_pc_o     (active_warp_pc),
      .active_warp_mask_o   (active_warp_mask),
      .active_warp_next_pc_i(active_warp_pc + 32'd4),
      .warp_stall_i         (warp_stall),
      .warp_done_i          (warp_done),
      .warp_branch_taken_i  (warp_branch_taken),
      .warp_branch_target_i (warp_branch_target),
      .dispatch_valid_i     (doorbell_valid),
      .dispatch_warp_id_i   (doorbell_warp),
      .dispatch_pc_i        (32'd0),
      .dispatch_mask_i      (32'hFFFFFFFF),
      .dispatch_cta_id_x_i  (16'd0),
      .dispatch_cta_id_y_i  (16'd0),
      .dispatch_cta_id_z_i  (16'd0),
      .cta_id_x_o           (cta_id_x),
      .cta_id_y_o           (cta_id_y),
      .cta_id_z_o           (cta_id_z),
      .warp_active_o        (warp_active),
      .num_active_warps_o   (num_active_warps)
  );

  //--------------------------------------------------------------------------
  // Per-SM: core + I-cache + D-cache
  // L1 fill ports to L2: [sm*2+0] = I-cache fill, [sm*2+1] = D-cache fill
  //--------------------------------------------------------------------------
  localparam int L2_PORTS = NUM_SM * 2;

  logic [L2_PORTS-1:0]        l2_req;
  logic [L2_PORTS-1:0]        l2_we;
  logic [L2_PORTS-1:0][39:0]  l2_addr;
  logic [L2_PORTS-1:0][511:0] l2_wdata;
  logic [L2_PORTS-1:0]        l2_ack;
  logic [L2_PORTS-1:0][511:0] l2_rdata;

  generate
    for (genvar sm = 0; sm < NUM_SM; sm++) begin : gen_sm

      logic [31:0] core_imem_addr, core_imem_rdata;
      logic        core_imem_req,  core_imem_ack;
      logic [31:0] core_dmem_addr, core_dmem_wdata, core_dmem_rdata;
      logic        core_dmem_req,  core_dmem_we,    core_dmem_ack;
      logic [3:0]  core_dmem_be;

      logic        ic_miss, dc_miss;
      logic [39:0] ic_fill_addr, dc_fill_addr;
      logic [511:0] ic_fill_data, dc_fill_data;

      tpt_core core_inst (
          .clk_i         (clk_i),
          .rst_n_i       (rst_n_i & gpu_boot & ~gpu_reset),
          .imem_addr_o   (core_imem_addr),
          .imem_rdata_i  (core_imem_rdata),
          .imem_req_o    (core_imem_req),
          .imem_ack_i    (core_imem_ack),
          .dmem_addr_o   (core_dmem_addr),
          .dmem_req_o    (core_dmem_req),
          .dmem_we_o     (core_dmem_we),
          .dmem_be_o     (core_dmem_be),
          .dmem_wdata_o  (core_dmem_wdata),
          .dmem_ack_i    (core_dmem_ack),
          .dmem_rdata_i  (core_dmem_rdata)
      );

      tpt_icache icache_inst (
          .clk_i        (clk_i),
          .rst_n_i      (rst_n_i),
          .addr_i       (core_imem_addr),
          .req_i        (core_imem_req),
          .ack_o        (core_imem_ack),
          .rdata_o      (core_imem_rdata),
          .miss_o       (ic_miss),
          .fill_valid_i (l2_ack[sm*2]),
          .fill_addr_i  (l2_addr[sm*2][31:0]),
          .fill_data_i  (l2_rdata[sm*2]),
          .event_miss_o ()
      );

      tpt_dcache dcache_inst (
          .clk_i        (clk_i),
          .rst_n_i      (rst_n_i),
          .addr_i       (core_dmem_addr),
          .req_i        (core_dmem_req),
          .we_i         (core_dmem_we),
          .be_i         (core_dmem_be),
          .wdata_i      (core_dmem_wdata),
          .ack_o        (core_dmem_ack),
          .rdata_o      (core_dmem_rdata),
          .miss_o       (dc_miss),
          .fill_valid_i (l2_ack[sm*2+1]),
          .fill_addr_i  (l2_addr[sm*2+1][31:0]),
          .fill_data_i  (l2_rdata[sm*2+1]),
          .wb_valid_o   (l2_we[sm*2+1]),   // dirty writeback → L2
          .wb_addr_o    (l2_addr[sm*2+1]),
          .wb_data_o    (l2_wdata[sm*2+1]),
          .wb_ack_i     (l2_ack[sm*2+1]),
          .event_miss_o ()
      );

      // I-cache fill port (read-only misses → L2)
      assign l2_req  [sm*2]   = ic_miss;
      assign l2_we   [sm*2]   = 1'b0;
      assign l2_addr [sm*2]   = {8'd0, core_imem_addr};
      assign l2_wdata[sm*2]   = '0;

      // D-cache miss port (reads — writeback driven above by dc_wb_*)
      assign l2_req  [sm*2+1] = dc_miss & ~l2_we[sm*2+1];
      assign l2_addr [sm*2+1] = dc_miss ? {8'd0, core_dmem_addr} : l2_addr[sm*2+1];
      assign l2_wdata[sm*2+1] = dc_miss ? '0 : l2_wdata[sm*2+1];

    end : gen_sm
  endgenerate

  //--------------------------------------------------------------------------
  // L2 Unified Cache
  //--------------------------------------------------------------------------
  logic [39:0]  mc_addr;
  logic         mc_req, mc_we;
  logic [511:0] mc_wdata, mc_rdata;
  logic         mc_ack;

  tpt_l2cache #(
      .SIZE_BYTES (2097152),
      .LINE_BYTES (64),
      .WAYS       (4),
      .NUM_PORTS  (L2_PORTS)
  ) l2_inst (
      .clk_i       (clk_i),
      .rst_n_i     (rst_n_i),
      .l1_req_i    (l2_req),
      .l1_we_i     (l2_we),
      .l1_addr_i   (l2_addr),
      .l1_wdata_i  (l2_wdata),
      .l1_ack_o    (l2_ack),
      .l1_rdata_o  (l2_rdata),
      .mc_addr_o   (mc_addr),
      .mc_req_o    (mc_req),
      .mc_we_o     (mc_we),
      .mc_wdata_o  (mc_wdata),
      .mc_ack_i    (mc_ack),
      .mc_rdata_i  (mc_rdata),
      .event_hit_o (),
      .event_miss_o(),
      .event_wb_o  ()
  );

  //--------------------------------------------------------------------------
  // Memory Controller (GDDR6 / HBM2)
  //--------------------------------------------------------------------------
  tpt_mem_ctrl #(
      .CLK_MHZ  (1000),
      .CHANNELS (4),
      .BANKS    (16)
  ) memctrl_inst (
      .clk_i            (clk_i),
      .rst_n_i          (rst_n_i),
      .l2_addr_i        (mc_addr),
      .l2_req_i         (mc_req),
      .l2_we_i          (mc_we),
      .l2_wdata_i       (mc_wdata),
      .l2_ack_o         (mc_ack),
      .l2_rdata_o       (mc_rdata),
      .phy_cmd_o        (phy_cmd_o),
      .phy_bank_o       (phy_bank_o),
      .phy_row_o        (phy_row_o),
      .phy_col_o        (phy_col_o),
      .phy_wdata_o      (phy_wdata_o),
      .phy_rdata_i      (phy_rdata_i),
      .phy_rdata_valid_i(phy_rdata_valid_i),
      .phy_cs_n_o       (phy_cs_n_o),
      .phy_cke_o        (phy_cke_o)
  );

  //--------------------------------------------------------------------------
  // Status generation
  //--------------------------------------------------------------------------
  assign gpu_ready = gpu_boot & ~gpu_reset;
  assign gpu_idle  = (num_active_warps == '0);
  assign gpu_error = 1'b0;

  assign warp_stall        = 1'b0;
  assign warp_done         = 1'b0;
  assign warp_branch_taken = 1'b0;
  assign warp_branch_target = '0;

endmodule : tpt_gpu_top
