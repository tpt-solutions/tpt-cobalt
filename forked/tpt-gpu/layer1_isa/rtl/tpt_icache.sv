//==============================================================================
// tpt_icache.sv — TPT L1 Instruction Cache
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// 4-way set-associative instruction cache.
// Configuration: 32KB, 64-byte lines, 4-way, 128 sets
// Latency: 1 cycle hit, ~20 cycles miss (to L2)
//
// Interface: Simple request/ack with 32-bit instruction word output.
//==============================================================================

module tpt_icache (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // CPU fetch interface
    input  logic [31:0]       addr_i,
    input  logic              req_i,
    output logic              ack_o,
    output logic [31:0]       rdata_o,
    output logic              miss_o,          // cache miss signal

    // Fill interface (from L2 / memory)
    input  logic              fill_valid_i,
    input  logic [31:0]       fill_addr_i,
    input  logic [511:0]      fill_data_i,     // 64-byte cache line

    // Performance counter
    output logic              event_miss_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Cache parameters
  //--------------------------------------------------------------------------
  localparam LINE_SIZE    = 64;          // bytes per cache line
  localparam LINE_WORDS   = LINE_SIZE/4; // 16 words per line
  localparam NUM_WAYS     = 4;
  localparam NUM_SETS     = 128;
  localparam SET_IDX_W    = $clog2(NUM_SETS);
  localparam TAG_W        = 32 - SET_IDX_W - $clog2(LINE_SIZE);
  localparam LINE_OFFSET_W = $clog2(LINE_SIZE);

  //--------------------------------------------------------------------------
  // Cache storage
  //--------------------------------------------------------------------------
  logic [TAG_W-1:0]   tags   [0:NUM_SETS-1][0:NUM_WAYS-1];
  logic               valids [0:NUM_SETS-1][0:NUM_WAYS-1];
  logic [NUM_WAYS-1:0] lru   [0:NUM_SETS-1];  // LRU replacement (simplified)
  logic [31:0]        data   [0:NUM_SETS-1][0:NUM_WAYS-1][0:LINE_WORDS-1];

  //--------------------------------------------------------------------------
  // Address decomposition
  //--------------------------------------------------------------------------
  logic [SET_IDX_W-1:0]  req_set;
  logic [LINE_OFFSET_W-1:0] req_offset;
  logic [TAG_W-1:0]       req_tag;

  assign req_set    = addr_i[LINE_OFFSET_W+SET_IDX_W-1:LINE_OFFSET_W];
  assign req_offset = addr_i[LINE_OFFSET_W-1:2];
  assign req_tag    = addr_i[31:LINE_OFFSET_W+SET_IDX_W];

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
  // Read data output
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      ack_o   <= 1'b0;
      rdata_o <= '0;
      miss_o  <= 1'b0;
    end else if (req_i) begin
      if (cache_hit) begin
        ack_o   <= 1'b1;
        rdata_o <= data[req_set][hit_way][req_offset];
        miss_o  <= 1'b0;
      end else begin
        ack_o   <= 1'b0;
        rdata_o <= '0;
        miss_o  <= 1'b1;
      end
    end else begin
      ack_o   <= 1'b0;
      miss_o  <= 1'b0;
    end
  end

  //--------------------------------------------------------------------------
  // Cache fill (from L2/memory on miss)
  //--------------------------------------------------------------------------
  logic [SET_IDX_W-1:0] fill_set;
  logic [TAG_W-1:0]     fill_tag;
  logic [1:0]           replace_way;

  assign fill_set = fill_addr_i[LINE_OFFSET_W+SET_IDX_W-1:LINE_OFFSET_W];
  assign fill_tag = fill_addr_i[31:LINE_OFFSET_W+SET_IDX_W];

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (int s = 0; s < NUM_SETS; s++) begin
        for (int w = 0; w < NUM_WAYS; w++) begin
          valids[s][w] <= 1'b0;
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

      // Fill cache line data
      for (int w = 0; w < LINE_WORDS; w++) begin
        data[fill_set][replace_way][w] <= fill_data_i[w*32 +: 32];
      end

      // Update LRU (simplified — MRU gets highest position)
      lru[fill_set] <= lru[fill_set] + 4'd1;
    end
  end

  //--------------------------------------------------------------------------
  // Performance counter event
  //--------------------------------------------------------------------------
  assign event_miss_o = req_i && !cache_hit;

endmodule : tpt_icache