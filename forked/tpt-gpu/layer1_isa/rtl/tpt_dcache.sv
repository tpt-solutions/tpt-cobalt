//==============================================================================
// tpt_dcache.sv — TPT L1 Data Cache
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// 4-way set-associative write-back data cache.
// Configuration: 32KB, 64-byte lines, 4-way, 128 sets
// Supports byte, halfword, word, and doubleword access with byte enables.
// Latency: 1 cycle hit, ~20 cycles miss (to L2)
//==============================================================================

module tpt_dcache (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // CPU load/store interface
    input  logic [31:0]       addr_i,
    input  logic              req_i,
    input  logic              we_i,            // write enable
    input  logic [3:0]        be_i,            // byte enables
    input  logic [31:0]       wdata_i,
    output logic              ack_o,
    output logic [31:0]       rdata_o,
    output logic              miss_o,

    // Fill interface (from L2 / memory)
    input  logic              fill_valid_i,
    input  logic [31:0]       fill_addr_i,
    input  logic [511:0]      fill_data_i,

    // Writeback interface (dirty line eviction to L2)
    output logic              wb_valid_o,
    output logic [31:0]       wb_addr_o,
    output logic [511:0]      wb_data_o,
    input  logic              wb_ack_i,

    // Performance counter
    output logic              event_miss_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Cache parameters
  //--------------------------------------------------------------------------
  localparam LINE_SIZE     = 64;
  localparam LINE_WORDS    = LINE_SIZE/4;
  localparam NUM_WAYS      = 4;
  localparam NUM_SETS      = 128;
  localparam SET_IDX_W     = $clog2(NUM_SETS);
  localparam TAG_W         = 32 - SET_IDX_W - $clog2(LINE_SIZE);
  localparam LINE_OFFSET_W = $clog2(LINE_SIZE);

  //--------------------------------------------------------------------------
  // Cache storage
  //--------------------------------------------------------------------------
  logic [TAG_W-1:0]    tags   [0:NUM_SETS-1][0:NUM_WAYS-1];
  logic                valids [0:NUM_SETS-1][0:NUM_WAYS-1];
  logic                dirty  [0:NUM_SETS-1][0:NUM_WAYS-1];
  logic [NUM_WAYS-1:0] lru    [0:NUM_SETS-1];
  logic [31:0]         data   [0:NUM_SETS-1][0:NUM_WAYS-1][0:LINE_WORDS-1];

  //--------------------------------------------------------------------------
  // Address decomposition
  //--------------------------------------------------------------------------
  logic [SET_IDX_W-1:0]    req_set;
  logic [LINE_OFFSET_W-1:0] req_word;
  logic [TAG_W-1:0]        req_tag;

  assign req_set  = addr_i[LINE_OFFSET_W+SET_IDX_W-1:LINE_OFFSET_W];
  assign req_word = addr_i[LINE_OFFSET_W-1:2];
  assign req_tag  = addr_i[31:LINE_OFFSET_W+SET_IDX_W];

  //--------------------------------------------------------------------------
  // Tag lookup
  //--------------------------------------------------------------------------
  logic [NUM_WAYS-1:0] way_hit;
  logic                cache_hit;
  logic [1:0]          hit_way;

  always_comb begin
    way_hit   = '0;
    cache_hit = 1'b0;
    hit_way   = '0;
    for (int w = 0; w < NUM_WAYS; w++) begin
      if (valids[req_set][w] && (tags[req_set][w] == req_tag)) begin
        way_hit[w] = 1'b1;
        cache_hit  = 1'b1;
        hit_way    = 2'(w);
      end
    end
  end

  //--------------------------------------------------------------------------
  // Load response
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      ack_o   <= 1'b0;
      rdata_o <= '0;
      miss_o  <= 1'b0;
    end else if (req_i) begin
      if (cache_hit) begin
        if (we_i) begin
          // Write hit — handled in cache update block
          ack_o <= 1'b1;
        end else begin
          // Read hit
          rdata_o <= data[req_set][hit_way][req_word];
          ack_o   <= 1'b1;
        end
        miss_o <= 1'b0;
      end else begin
        ack_o  <= 1'b0;
        miss_o <= 1'b1;
      end
    end else begin
      ack_o  <= 1'b0;
      miss_o <= 1'b0;
    end
  end

  //--------------------------------------------------------------------------
  // Write hit — update cache line with byte enables
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (req_i && cache_hit && we_i) begin
      if (be_i[0]) data[req_set][hit_way][req_word][7:0]   <= wdata_i[7:0];
      if (be_i[1]) data[req_set][hit_way][req_word][15:8]  <= wdata_i[15:8];
      if (be_i[2]) data[req_set][hit_way][req_word][23:16] <= wdata_i[23:16];
      if (be_i[3]) data[req_set][hit_way][req_word][31:24] <= wdata_i[31:24];
      dirty[req_set][hit_way] <= 1'b1;
    end
  end

  //--------------------------------------------------------------------------
  // Cache fill (from L2/memory on miss)
  //--------------------------------------------------------------------------
  logic [SET_IDX_W-1:0] fill_set;
  logic [TAG_W-1:0]     fill_tag;
  logic [1:0]           replace_way;
  logic                 evict_dirty;

  assign fill_set = fill_addr_i[LINE_OFFSET_W+SET_IDX_W-1:LINE_OFFSET_W];
  assign fill_tag = fill_addr_i[31:LINE_OFFSET_W+SET_IDX_W];

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int s = 0; s < NUM_SETS; s++) begin
        for (int w = 0; w < NUM_WAYS; w++) begin
          valids[s][w] <= 1'b0;
          dirty[s][w]  <= 1'b0;
          tags[s][w]   <= '0;
          lru[s]       <= '0;
        end
      end
    end else if (fill_valid_i) begin
      // Find invalid way or use LRU
      replace_way = lru[fill_set][1:0];
      for (int w = 0; w < NUM_WAYS; w++) begin
        if (!valids[fill_set][w]) begin
          replace_way = 2'(w);
        end
      end

      tags[fill_set][replace_way]   <= fill_tag;
      valids[fill_set][replace_way] <= 1'b1;
      dirty[fill_set][replace_way]  <= 1'b0;

      for (int w = 0; w < LINE_WORDS; w++) begin
        data[fill_set][replace_way][w] <= fill_data_i[w*32 +: 32];
      end

      lru[fill_set] <= lru[fill_set] + 4'd1;
    end
  end

  //--------------------------------------------------------------------------
  // Writeback (dirty eviction) — simplified for simulation
  // In real silicon, this would trigger a writeback to L2 on eviction
  //--------------------------------------------------------------------------
  assign wb_valid_o = 1'b0;  // No evictions in base simulation model
  assign wb_addr_o  = '0;
  assign wb_data_o  = '0;

  //--------------------------------------------------------------------------
  // Performance counter event
  //--------------------------------------------------------------------------
  assign event_miss_o = req_i && !cache_hit;

endmodule : tpt_dcache