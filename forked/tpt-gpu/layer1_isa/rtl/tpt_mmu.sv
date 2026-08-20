//==============================================================================
// tpt_mmu.sv — TPT Memory Management Unit
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Translates 48-bit virtual addresses to physical addresses.
// Uses a 64-entry ITLB and 64-entry DTLB (fully associative).
// Shared/Local memory spaces are identity-mapped (bypass TLB).
//==============================================================================

module tpt_mmu (
    input  logic              clk_i,
    input  logic              rst_n_i,

    input  logic [47:0]       va_i,
    input  logic              req_i,
    input  logic              is_fetch_i,
    input  logic              is_write_i,
    input  aspace_t           aspace_i,

    output logic [39:0]       pa_o,
    output logic              valid_o,
    output logic              fault_o,
    output logic              permission_fault_o,

    output logic [47:0]       ptw_va_o,
    output logic              ptw_req_o,
    input  logic              ptw_ack_i,
    input  logic [39:0]       ptw_pa_i,

    output logic              event_tlb_miss_o
);

  import tpt_pkg::*;

  typedef struct packed {
    logic [25:0]  vpn;
    logic [17:0]  pfn;
    logic         valid;
    logic         dirty;
    logic         readable;
    logic         writable;
    logic         executable;
    logic         global;
  } tlb_entry_t;

  localparam TLB_ENTRIES = 64;
  tlb_entry_t itlb [0:TLB_ENTRIES-1];
  tlb_entry_t dtlb [0:TLB_ENTRIES-1];

  //--------------------------------------------------------------------------
  // Address space bypass for shared/local memory
  //--------------------------------------------------------------------------
  logic [39:0] bypass_pa;
  logic        bypass_en;

  always_comb begin
    bypass_en = 1'b0;
    bypass_pa = '0;
    unique case (aspace_i)
      ASPACE_SHARED: begin bypass_en = 1'b1; bypass_pa = {8'd0, va_i[31:0]}; end
      ASPACE_LOCAL:  begin bypass_en = 1'b1; bypass_pa = {8'd0, va_i[31:0]}; end
      ASPACE_GLOBAL, ASPACE_CONSTANT: bypass_en = 1'b0;
      default: bypass_en = 1'b0;
    endcase
  end

  //--------------------------------------------------------------------------
  // TLB lookup
  //--------------------------------------------------------------------------
  logic [25:0] req_vpn;
  logic        itlb_hit, dtlb_hit;
  logic [17:0] itlb_pfn, dtlb_pfn;
  logic        itlb_readable, itlb_executable;
  logic        dtlb_readable, dtlb_writable;

  assign req_vpn = va_i[47:22];

  always_comb begin
    itlb_hit        = 1'b0; itlb_pfn = '0;
    itlb_readable   = 1'b0; itlb_executable = 1'b0;
    for (int i = 0; i < TLB_ENTRIES; i++) begin
      if (itlb[i].valid && (itlb[i].vpn == req_vpn)) begin
        itlb_hit        = 1'b1;
        itlb_pfn        = itlb[i].pfn;
        itlb_readable   = itlb[i].readable;
        itlb_executable = itlb[i].executable;
      end
    end
  end

  always_comb begin
    dtlb_hit      = 1'b0; dtlb_pfn = '0;
    dtlb_readable = 1'b0; dtlb_writable = 1'b0;
    for (int i = 0; i < TLB_ENTRIES; i++) begin
      if (dtlb[i].valid && (dtlb[i].vpn == req_vpn)) begin
        dtlb_hit      = 1'b1;
        dtlb_pfn      = dtlb[i].pfn;
        dtlb_readable = dtlb[i].readable;
        dtlb_writable = dtlb[i].writable;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Translation output
  //--------------------------------------------------------------------------
  always_comb begin
    pa_o               = '0;
    valid_o            = 1'b0;
    fault_o            = 1'b0;
    permission_fault_o = 1'b0;

    if (bypass_en) begin
      pa_o    = bypass_pa;
      valid_o = 1'b1;
    end else if (is_fetch_i) begin
      if (itlb_hit) begin
        pa_o    = {itlb_pfn, va_i[21:0]};
        valid_o = 1'b1;
        if (!itlb_executable) permission_fault_o = 1'b1;
      end else begin
        fault_o = 1'b1;
      end
    end else begin
      if (dtlb_hit) begin
        pa_o    = {dtlb_pfn, va_i[21:0]};
        valid_o = 1'b1;
        if (is_write_i && !dtlb_writable) permission_fault_o = 1'b1;
        if (!is_write_i && !dtlb_readable) permission_fault_o = 1'b1;
      end else begin
        fault_o = 1'b1;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Page table walk request
  //--------------------------------------------------------------------------
  assign ptw_va_o  = va_i;
  assign ptw_req_o = req_i && fault_o;

  //--------------------------------------------------------------------------
  // TLB fill (from page table walk completion)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int i = 0; i < TLB_ENTRIES; i++) begin
        itlb[i] <= '0;
        dtlb[i] <= '0;
      end
    end else if (ptw_ack_i) begin
      if (is_fetch_i) begin
        itlb[0] <= '{vpn: req_vpn, pfn: ptw_pa_i[39:22],
                     valid: 1'b1, dirty: 1'b0,
                     readable: 1'b1, writable: 1'b0,
                     executable: 1'b1, global: 1'b0};
      end else begin
        dtlb[0] <= '{vpn: req_vpn, pfn: ptw_pa_i[39:22],
                     valid: 1'b1, dirty: is_write_i,
                     readable: 1'b1, writable: 1'b1,
                     executable: 1'b0, global: 1'b0};
      end
    end
  end

  assign event_tlb_miss_o = req_i && fault_o;

endmodule : tpt_mmu