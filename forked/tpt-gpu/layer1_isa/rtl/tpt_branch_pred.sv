//==============================================================================
// tpt_branch_pred.sv — TPT Branch Predictor
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements a 2-bit saturating counter branch predictor with a 4096-entry
// Branch Target Buffer (BTB) and 16-entry Return Address Stack (RAS).
//
// Spec §5.2:
//   - 2-bit saturating counter predictor (4096-entry BTB)
//   - Branch misprediction penalty = 4 cycles (F1→E2 flush)
//   - Return Address Stack (RAS): 16 entries
//==============================================================================

module tpt_branch_pred (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Predict interface (F1 stage)
    input  logic [31:0]       pc_i,
    output logic              pred_taken_o,
    output logic [31:0]       pred_target_o,

    // Update interface (E2 stage — after branch resolution)
    input  logic              update_valid_i,
    input  logic [31:0]       update_pc_i,
    input  logic              update_actual_i,     // actual branch outcome
    input  logic [31:0]       update_target_i,     // actual branch target
    input  logic              update_is_call_i,    // JAL with rd=RA
    input  logic              update_is_ret_i,     // RET instruction

    // RAS interface
    output logic [31:0]       ras_top_o,
    output logic              ras_valid_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Branch Target Buffer (BTB) — 4096 entries, direct-mapped
  //--------------------------------------------------------------------------
  localparam BTB_ENTRIES = 4096;
  localparam BTB_IDX_W   = $clog2(BTB_ENTRIES);

  logic [31:0] btb_tag    [0:BTB_ENTRIES-1];
  logic [31:0] btb_target [0:BTB_ENTRIES-1];
  logic        btb_valid  [0:BTB_ENTRIES-1];
  logic [1:0]  btb_counter[0:BTB_ENTRIES-1];  // 2-bit saturating counter

  logic [BTB_IDX_W-1:0] btb_idx;
  logic                  btb_hit;

  assign btb_idx  = pc_i[BTB_IDX_W+1:2];  // word-aligned indexing
  assign btb_hit  = btb_valid[btb_idx] && (btb_tag[btb_idx] == pc_i);

  // Prediction output
  assign pred_taken_o  = btb_hit && (btb_counter[btb_idx] >= 2'b10);
  assign pred_target_o = btb_target[btb_idx];

  //--------------------------------------------------------------------------
  // BTB update (on branch resolution)
  //--------------------------------------------------------------------------
  logic [BTB_IDX_W-1:0] update_idx;
  assign update_idx = update_pc_i[BTB_IDX_W+1:2];

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int i = 0; i < BTB_ENTRIES; i++) begin
        btb_tag[i]     <= '0;
        btb_target[i]  <= '0;
        btb_valid[i]   <= 1'b0;
        btb_counter[i] <= 2'b01;  // weakly not-taken
      end
    end else if (update_valid_i) begin
      btb_tag[update_idx]     <= update_pc_i;
      btb_target[update_idx]  <= update_target_i;
      btb_valid[update_idx]   <= 1'b1;

      // 2-bit saturating counter update
      if (update_actual_i) begin
        if (btb_counter[update_idx] < 2'b11)
          btb_counter[update_idx] <= btb_counter[update_idx] + 2'b01;
      end else begin
        if (btb_counter[update_idx] > 2'b00)
          btb_counter[update_idx] <= btb_counter[update_idx] - 2'b01;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Return Address Stack (RAS) — 16 entries
  //--------------------------------------------------------------------------
  localparam RAS_DEPTH = 16;
  logic [31:0] ras_stack [0:RAS_DEPTH-1];
  logic [3:0]  ras_ptr;
  logic        ras_empty;

  assign ras_valid_o = ~ras_empty;
  assign ras_top_o   = ras_stack[ras_ptr - 4'd1];

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      ras_ptr   <= '0;
      ras_empty <= 1'b1;
      for (int i = 0; i < RAS_DEPTH; i++) begin
        ras_stack[i] <= '0;
      end
    end else if (update_valid_i) begin
      if (update_is_call_i) begin
        // Push return address (PC+4)
        ras_stack[ras_ptr] <= update_pc_i + 32'd4;
        ras_ptr   <= ras_ptr + 4'd1;
        ras_empty <= 1'b0;
      end else if (update_is_ret_i && !ras_empty) begin
        // Pop return address
        ras_ptr <= ras_ptr - 4'd1;
        if (ras_ptr == 4'd1)
          ras_empty <= 1'b1;
      end
    end
  end

endmodule : tpt_branch_pred
