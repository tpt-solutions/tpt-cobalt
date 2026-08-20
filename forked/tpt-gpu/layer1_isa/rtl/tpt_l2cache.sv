//==============================================================================
// tpt_l2cache.sv — TPT L2 Unified Cache
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// 4-way set-associative, write-back L2 cache shared by all SMs.
// Services refill requests from L1 I-caches and D-caches and arbitrates
// access to the external memory controller (GDDR6/HBM2 bus).
//
// Parameters
//   SIZE_BYTES   Total cache capacity (default 2 MiB)
//   LINE_BYTES   Cache line size in bytes (must match L1, default 64)
//   WAYS         Set-associativity (default 4)
//   NUM_PORTS    Number of requestor ports (one per SM × 2 for I+D, default 2)
//==============================================================================

module tpt_l2cache #(
    parameter int SIZE_BYTES  = 2097152,  // 2 MiB
    parameter int LINE_BYTES  = 64,
    parameter int WAYS        = 4,
    parameter int NUM_PORTS   = 2         // L1-I fill + L1-D fill
) (
    input  logic              clk_i,
    input  logic              rst_n_i,

    //----------------------------------------------------------------------
    // L1 fill ports (one per SM I-cache + D-cache)
    //----------------------------------------------------------------------
    input  logic [NUM_PORTS-1:0]        l1_req_i,
    input  logic [NUM_PORTS-1:0]        l1_we_i,       // 0=fill-read, 1=writeback
    input  logic [NUM_PORTS-1:0][39:0]  l1_addr_i,     // physical cache-line address
    input  logic [NUM_PORTS-1:0][511:0] l1_wdata_i,    // writeback data (512b line)
    output logic [NUM_PORTS-1:0]        l1_ack_o,
    output logic [NUM_PORTS-1:0][511:0] l1_rdata_o,

    //----------------------------------------------------------------------
    // External memory interface (to tpt_mem_ctrl)
    //----------------------------------------------------------------------
    output logic [39:0]       mc_addr_o,
    output logic              mc_req_o,
    output logic              mc_we_o,
    output logic [511:0]      mc_wdata_o,
    input  logic              mc_ack_i,
    input  logic [511:0]      mc_rdata_i,

    //----------------------------------------------------------------------
    // Performance events
    //----------------------------------------------------------------------
    output logic              event_hit_o,
    output logic              event_miss_o,
    output logic              event_wb_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Derived parameters
  //--------------------------------------------------------------------------
  localparam int SETS       = SIZE_BYTES / (LINE_BYTES * WAYS);  // 8192
  localparam int SET_BITS   = $clog2(SETS);                       // 13
  localparam int OFFSET_BITS = $clog2(LINE_BYTES);               // 6
  localparam int TAG_BITS   = 40 - SET_BITS - OFFSET_BITS;       // 21

  //--------------------------------------------------------------------------
  // Cache arrays
  //--------------------------------------------------------------------------
  typedef struct packed {
    logic                 valid;
    logic                 dirty;
    logic [TAG_BITS-1:0]  tag;
  } meta_t;

  meta_t    meta  [0:SETS-1][0:WAYS-1];
  logic [511:0] data  [0:SETS-1][0:WAYS-1];

  // LRU counters (2-bit pseudo-LRU per set for 4 ways)
  logic [WAYS-1:0] lru [0:SETS-1];

  //--------------------------------------------------------------------------
  // Arbiter: round-robin across NUM_PORTS
  //--------------------------------------------------------------------------
  logic [$clog2(NUM_PORTS)-1:0] arb_sel;
  logic [NUM_PORTS-1:0]         grant;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) arb_sel <= '0;
    else if (|l1_req_i) begin
      // advance to next requesting port
      logic found;
      found = 1'b0;
      for (int p = 1; p <= NUM_PORTS; p++) begin
        logic [$clog2(NUM_PORTS)-1:0] np;
        np = (arb_sel + p[$clog2(NUM_PORTS)-1:0]);
        if (l1_req_i[np] && !found) begin
          arb_sel <= np;
          found = 1'b1;
        end
      end
    end
  end

  always_comb begin
    grant = '0;
    for (int p = 0; p < NUM_PORTS; p++)
      grant[p] = l1_req_i[p] && (p[$clog2(NUM_PORTS)-1:0] == arb_sel);
  end

  //--------------------------------------------------------------------------
  // Pipeline: single-cycle tag lookup → miss/hit
  //--------------------------------------------------------------------------
  logic [39:0]       req_addr;
  logic              req_we;
  logic [511:0]      req_wdata;
  logic [SET_BITS-1:0]  req_set;
  logic [TAG_BITS-1:0]  req_tag;

  always_comb begin
    req_addr  = l1_addr_i[arb_sel];
    req_we    = l1_we_i[arb_sel];
    req_wdata = l1_wdata_i[arb_sel];
    req_set   = req_addr[OFFSET_BITS +: SET_BITS];
    req_tag   = req_addr[39 -: TAG_BITS];
  end

  // Hit detection
  logic [WAYS-1:0] hit_way_oh;
  logic            hit;
  logic [$clog2(WAYS)-1:0] hit_way_idx;

  always_comb begin
    hit_way_oh = '0;
    for (int w = 0; w < WAYS; w++) begin
      if (meta[req_set][w].valid && meta[req_set][w].tag == req_tag)
        hit_way_oh[w] = 1'b1;
    end
    hit = |hit_way_oh;
    hit_way_idx = '0;
    for (int w = 0; w < WAYS; w++)
      if (hit_way_oh[w]) hit_way_idx = w[$clog2(WAYS)-1:0];
  end

  // Victim selection (LRU)
  logic [$clog2(WAYS)-1:0] victim_way;
  always_comb begin
    victim_way = '0;
    for (int w = WAYS-1; w >= 0; w--)
      if (!lru[req_set][w]) victim_way = w[$clog2(WAYS)-1:0];
  end

  //--------------------------------------------------------------------------
  // FSM
  //--------------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE,
    S_TAG,
    S_HIT_RESP,
    S_WB,       // write back dirty victim to MC
    S_FILL,     // fetch line from MC
    S_FILL_RESP
  } state_t;

  state_t state;
  logic [$clog2(NUM_PORTS)-1:0] saved_port;
  logic [39:0]       saved_addr;
  logic              saved_we;
  logic [511:0]      saved_wdata;
  logic [$clog2(WAYS)-1:0] saved_victim;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      state       <= S_IDLE;
      l1_ack_o    <= '0;
      l1_rdata_o  <= '0;
      mc_req_o    <= 1'b0;
      mc_we_o     <= 1'b0;
      mc_addr_o   <= '0;
      mc_wdata_o  <= '0;
      event_hit_o  <= 1'b0;
      event_miss_o <= 1'b0;
      event_wb_o   <= 1'b0;
      for (int s = 0; s < SETS; s++)
        for (int w = 0; w < WAYS; w++) begin
          meta[s][w].valid <= 1'b0;
          meta[s][w].dirty <= 1'b0;
          meta[s][w].tag   <= '0;
          lru[s]           <= '0;
        end
    end else begin
      // Default pulse-clears
      l1_ack_o    <= '0;
      event_hit_o  <= 1'b0;
      event_miss_o <= 1'b0;
      event_wb_o   <= 1'b0;

      unique case (state)
        // ----------------------------------------------------------------
        S_IDLE: begin
          if (|grant) begin
            saved_port   <= arb_sel;
            saved_addr   <= req_addr;
            saved_we     <= req_we;
            saved_wdata  <= req_wdata;
            saved_victim <= victim_way;
            state        <= S_TAG;
          end
        end

        // ----------------------------------------------------------------
        S_TAG: begin
          if (hit) begin
            // Cache hit
            event_hit_o <= 1'b1;
            if (saved_we) begin
              // L1 writeback: update line in-place (mark dirty)
              data[req_set][hit_way_idx] <= saved_wdata;
              meta[req_set][hit_way_idx].dirty <= 1'b1;
            end
            l1_rdata_o[saved_port] <= data[req_set][hit_way_idx];
            l1_ack_o[saved_port]   <= 1'b1;
            // Update LRU: mark hit way as most-recently-used
            lru[req_set] <= lru[req_set] | (1 << hit_way_idx);
            if (&(lru[req_set] | (1 << hit_way_idx)))
              lru[req_set] <= (1 << hit_way_idx);
            state <= S_IDLE;
          end else begin
            // Cache miss
            event_miss_o <= 1'b1;
            if (meta[req_set][saved_victim].valid &&
                meta[req_set][saved_victim].dirty) begin
              // Dirty victim — write back first
              mc_addr_o  <= {meta[req_set][saved_victim].tag,
                             saved_addr[OFFSET_BITS +: SET_BITS],
                             {OFFSET_BITS{1'b0}}};
              mc_wdata_o <= data[req_set][saved_victim];
              mc_req_o   <= 1'b1;
              mc_we_o    <= 1'b1;
              event_wb_o <= 1'b1;
              state      <= S_WB;
            end else begin
              // Clean victim — fetch directly
              mc_addr_o <= {saved_addr[39:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
              mc_req_o  <= 1'b1;
              mc_we_o   <= 1'b0;
              state     <= S_FILL;
            end
          end
        end

        // ----------------------------------------------------------------
        S_WB: begin
          if (mc_ack_i) begin
            mc_req_o <= 1'b0;
            mc_we_o  <= 1'b0;
            // Now fetch the new line
            mc_addr_o <= {saved_addr[39:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            mc_req_o  <= 1'b1;
            state     <= S_FILL;
          end
        end

        // ----------------------------------------------------------------
        S_FILL: begin
          if (mc_ack_i) begin
            mc_req_o <= 1'b0;
            // Install line in cache
            data[req_set][saved_victim]        <= mc_rdata_i;
            meta[req_set][saved_victim].valid  <= 1'b1;
            meta[req_set][saved_victim].dirty  <= saved_we;
            meta[req_set][saved_victim].tag    <=
                saved_addr[39 -: TAG_BITS];
            if (saved_we)
              data[req_set][saved_victim] <= saved_wdata;
            // Update LRU
            lru[req_set] <= lru[req_set] | (1 << saved_victim);
            if (&(lru[req_set] | (1 << saved_victim)))
              lru[req_set] <= (1 << saved_victim);
            state <= S_FILL_RESP;
          end
        end

        // ----------------------------------------------------------------
        S_FILL_RESP: begin
          l1_rdata_o[saved_port] <=
              data[saved_addr[OFFSET_BITS +: SET_BITS]][saved_victim];
          l1_ack_o[saved_port] <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule : tpt_l2cache
