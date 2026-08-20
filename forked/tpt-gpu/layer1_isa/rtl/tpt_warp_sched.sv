//==============================================================================
// tpt_warp_sched.sv — TPT Warp Scheduler
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Manages a pool of up to 64 warps using round-robin scheduling with
// priority boost for memory-bound warps. Each warp has its own PC,
// active mask, and state. The scheduler selects the next ready warp
// every cycle for instruction fetch.
//
// Warp States: READY, RUNNING, STALLED, DONE
//==============================================================================

module tpt_warp_sched (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Scheduler enable (from CSR)
    input  logic              sched_enable_i,

    // Current warp selection
    output logic [5:0]        active_warp_id_o,
    output logic [31:0]       active_warp_pc_o,
    output logic [31:0]       active_warp_mask_o,
    input  logic [31:0]       active_warp_next_pc_i,

    // Warp state updates from pipeline
    input  logic              warp_stall_i,
    input  logic              warp_done_i,
    input  logic              warp_branch_taken_i,
    input  logic [31:0]       warp_branch_target_i,

    // Warp dispatch (from host/driver)
    input  logic              dispatch_valid_i,
    input  logic [5:0]        dispatch_warp_id_i,
    input  logic [31:0]       dispatch_pc_i,
    input  logic [31:0]       dispatch_mask_i,
    input  logic [15:0]       dispatch_cta_id_x_i,
    input  logic [15:0]       dispatch_cta_id_y_i,
    input  logic [15:0]       dispatch_cta_id_z_i,

    // CTA ID outputs for selected warp
    output logic [15:0]       cta_id_x_o,
    output logic [15:0]       cta_id_y_o,
    output logic [15:0]       cta_id_z_o,

    // Status
    output logic [NUM_WARPS-1:0] warp_active_o,
    output logic [5:0]        num_active_warps_o
);

  import tpt_pkg::*;

  typedef enum logic [1:0] {
    WARP_READY   = 2'b00,
    WARP_RUNNING = 2'b01,
    WARP_STALLED = 2'b10,
    WARP_DONE    = 2'b11
  } warp_state_t;

  //--------------------------------------------------------------------------
  // Per-warp state storage
  //--------------------------------------------------------------------------
  warp_state_t warp_state [0:NUM_WARPS-1];
  logic [31:0] warp_pc    [0:NUM_WARPS-1];
  logic [31:0] warp_mask  [0:NUM_WARPS-1];
  logic [15:0] warp_cta_x [0:NUM_WARPS-1];
  logic [15:0] warp_cta_y [0:NUM_WARPS-1];
  logic [15:0] warp_cta_z [0:NUM_WARPS-1];
  logic        warp_mem_bound [0:NUM_WARPS-1];

  logic [5:0]  rr_ptr;
  logic [5:0]  selected_warp;
  logic [5:0]  next_rr_ptr;
  logic        found_ready;

  //--------------------------------------------------------------------------
  // Warp selection — round-robin with priority boost for memory-bound warps
  //--------------------------------------------------------------------------
  always_comb begin
    selected_warp = '0;
    found_ready   = 1'b0;
    next_rr_ptr   = rr_ptr;

    if (sched_enable_i) begin
      // Priority 1: Memory-bound ready warps get priority boost
      for (int i = 0; i < NUM_WARPS && !found_ready; i++) begin
        logic [5:0] idx;
        idx = (rr_ptr + 5'(i)) % NUM_WARPS;
        if (warp_state[idx] == WARP_READY && warp_mem_bound[idx]) begin
          selected_warp = idx;
          found_ready   = 1'b1;
          next_rr_ptr   = idx + 5'd1;
        end
      end

      // Priority 2: Any ready warp (round-robin)
      for (int i = 0; i < NUM_WARPS && !found_ready; i++) begin
        logic [5:0] idx;
        idx = (rr_ptr + 5'(i)) % NUM_WARPS;
        if (warp_state[idx] == WARP_READY) begin
          selected_warp = idx;
          found_ready   = 1'b1;
          next_rr_ptr   = idx + 5'd1;
        end
      end
    end
  end

  //--------------------------------------------------------------------------
  // State update
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      rr_ptr <= '0;
      for (int i = 0; i < NUM_WARPS; i++) begin
        warp_state[i]     <= WARP_DONE;
        warp_pc[i]        <= '0;
        warp_mask[i]      <= '0;
        warp_cta_x[i]     <= '0;
        warp_cta_y[i]     <= '0;
        warp_cta_z[i]     <= '0;
        warp_mem_bound[i] <= 1'b0;
      end
    end else begin
      rr_ptr <= next_rr_ptr;

      // Handle dispatch of new warps
      if (dispatch_valid_i) begin
        warp_state[dispatch_warp_id_i] <= WARP_READY;
        warp_pc[dispatch_warp_id_i]    <= dispatch_pc_i;
        warp_mask[dispatch_warp_id_i]  <= dispatch_mask_i;
        warp_cta_x[dispatch_warp_id_i] <= dispatch_cta_id_x_i;
        warp_cta_y[dispatch_warp_id_i] <= dispatch_cta_id_y_i;
        warp_cta_z[dispatch_warp_id_i] <= dispatch_cta_id_z_i;
        warp_mem_bound[dispatch_warp_id_i] <= 1'b0;
      end

      // Update selected warp state
      if (found_ready) begin
        warp_state[selected_warp] <= WARP_RUNNING;
      end

      if (warp_stall_i) begin
        warp_state[selected_warp] <= WARP_STALLED;
        warp_mem_bound[selected_warp] <= 1'b1;
      end

      if (warp_done_i) begin
        warp_state[selected_warp] <= WARP_DONE;
        warp_mask[selected_warp]  <= '0;
        warp_mem_bound[selected_warp] <= 1'b0;
      end

      // Branch taken: update PC for the current warp
      if (warp_branch_taken_i) begin
        warp_pc[selected_warp] <= warp_branch_target_i;
      end else if (found_ready) begin
        // Normal advance: PC+4 for running warp
        warp_pc[selected_warp] <= warp_pc[selected_warp] + 32'd4;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Outputs
  //--------------------------------------------------------------------------
  assign active_warp_id_o  = selected_warp;
  assign active_warp_pc_o  = warp_pc[selected_warp];
  assign active_warp_mask_o = warp_mask[selected_warp];
  assign cta_id_x_o        = warp_cta_x[selected_warp];
  assign cta_id_y_o        = warp_cta_y[selected_warp];
  assign cta_id_z_o        = warp_cta_z[selected_warp];

  // Active warp bit vector and count
  always_comb begin
    warp_active_o    = '0;
    num_active_warps_o = '0;
    for (int i = 0; i < NUM_WARPS; i++) begin
      if (warp_state[i] != WARP_DONE) begin
        warp_active_o[i] = 1'b1;
        num_active_warps_o = num_active_warps_o + 5'd1;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Stalled warps become ready when they are not selected (simulates
  // memory latency completion — a real design would have a completion signal)
  //--------------------------------------------------------------------------
  integer i;
  always_ff @(posedge clk_i) begin
    if (rst_n_i) begin
      for (i = 0; i < NUM_WARPS; i++) begin
        // Auto-recover stalled warps after 1 cycle (simplified)
        // In real silicon, this would be driven by LSU completion signals
        if (warp_state[i] == WARP_STALLED && i != selected_warp) begin
          warp_state[i] <= WARP_READY;
          warp_mem_bound[i] <= 1'b0;
        end
      end
    end
  end

endmodule : tpt_warp_sched