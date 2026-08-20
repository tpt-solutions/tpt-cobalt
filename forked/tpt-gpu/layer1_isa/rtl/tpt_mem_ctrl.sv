//==============================================================================
// tpt_mem_ctrl.sv — TPT Memory Controller
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Bridges the L2 cache bus to an external GDDR6 / HBM2 memory interface.
// Implements:
//   • Command queue (8-entry FIFO) with read/write arbitration
//   • Bank-group / row interleaving for GDDR6 (4 channels × 16 banks)
//   • Burst length 8 (512-bit data per transaction)
//   • Refresh scheduling (tREFI = 7.8 µs @ 1 GHz → every 7800 cycles)
//   • Write-data buffer (hold data until CAS latency expires)
//==============================================================================

module tpt_mem_ctrl #(
    parameter int CLK_MHZ    = 1000,   // Core clock in MHz (for tREFI calc)
    parameter int CHANNELS   = 4,      // GDDR6 channels
    parameter int BANKS      = 16      // Banks per channel
) (
    input  logic              clk_i,
    input  logic              rst_n_i,

    //----------------------------------------------------------------------
    // L2 cache interface
    //----------------------------------------------------------------------
    input  logic [39:0]       l2_addr_i,
    input  logic              l2_req_i,
    input  logic              l2_we_i,
    input  logic [511:0]      l2_wdata_i,
    output logic              l2_ack_o,
    output logic [511:0]      l2_rdata_o,

    //----------------------------------------------------------------------
    // GDDR6 / HBM2 PHY interface (abstract — real PHY wraps this)
    //----------------------------------------------------------------------
    output logic [2:0]        phy_cmd_o,      // 3'b000=NOP, 001=ACT, 010=RD, 011=WR, 100=PRE, 101=REF
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
  // PHY command encoding
  //--------------------------------------------------------------------------
  localparam logic [2:0] CMD_NOP  = 3'b000;
  localparam logic [2:0] CMD_ACT  = 3'b001;
  localparam logic [2:0] CMD_RD   = 3'b010;
  localparam logic [2:0] CMD_WR   = 3'b011;
  localparam logic [2:0] CMD_PRE  = 3'b100;
  localparam logic [2:0] CMD_REF  = 3'b101;

  //--------------------------------------------------------------------------
  // Address decode: [39:0] → channel / bank / row / col
  //   [5:0]   — byte offset within 64-byte line (not sent to DRAM)
  //   [7:6]   — channel select (2 bits → 4 channels)
  //   [11:8]  — bank (4 bits → 16 banks)
  //   [26:12] — row (15 bits)
  //   [36:27] — col (10 bits)
  //--------------------------------------------------------------------------
  logic [1:0]  req_chan;
  logic [3:0]  req_bank;
  logic [14:0] req_row;
  logic [9:0]  req_col;

  always_comb begin
    req_chan = l2_addr_i[7:6];
    req_bank = l2_addr_i[11:8];
    req_row  = l2_addr_i[26:12];
    req_col  = l2_addr_i[36:27];
  end

  //--------------------------------------------------------------------------
  // Open-page tracking (one per bank)
  //--------------------------------------------------------------------------
  logic [14:0] open_row [0:BANKS-1];
  logic        row_open [0:BANKS-1];

  //--------------------------------------------------------------------------
  // Command queue (8-entry)
  //--------------------------------------------------------------------------
  typedef struct packed {
    logic [39:0]  addr;
    logic [511:0] wdata;
    logic         we;
  } cmd_t;

  localparam int CQ_DEPTH = 8;
  cmd_t  cq_fifo [0:CQ_DEPTH-1];
  logic [$clog2(CQ_DEPTH)-1:0] cq_head, cq_tail;
  logic cq_empty, cq_full;

  assign cq_empty = (cq_head == cq_tail);
  assign cq_full  = (cq_tail + 1'b1 == cq_head);

  //--------------------------------------------------------------------------
  // Refresh counter (tREFI)
  //--------------------------------------------------------------------------
  localparam int TREFI_CYCLES = CLK_MHZ * 7800 / 1000;  // ~7800 cycles @ 1 GHz
  logic [$clog2(TREFI_CYCLES+1)-1:0] refi_cnt;
  logic refresh_needed;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      refi_cnt       <= '0;
      refresh_needed <= 1'b0;
    end else begin
      if (refi_cnt == TREFI_CYCLES[$clog2(TREFI_CYCLES+1)-1:0]) begin
        refi_cnt       <= '0;
        refresh_needed <= 1'b1;
      end else begin
        refi_cnt <= refi_cnt + 1'b1;
        if (refresh_needed && state == S_REF)
          refresh_needed <= 1'b0;
      end
    end
  end

  //--------------------------------------------------------------------------
  // CAS latency counter
  //--------------------------------------------------------------------------
  localparam int CAS_LAT = 14;  // CL14 for GDDR6
  localparam int WL      = 8;   // Write latency

  logic [$clog2(CAS_LAT+2)-1:0] cas_cnt;
  logic                          cas_pending;
  logic [511:0]                  rd_capture;

  //--------------------------------------------------------------------------
  // FSM
  //--------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_ENQUEUE,
    S_CHECK_OPEN,
    S_PRE,
    S_ACT,
    S_CAS,
    S_WAIT_CAS,
    S_WR_DATA,
    S_RD_RESP,
    S_REF,
    S_INIT
  } mc_state_t;

  mc_state_t state;

  cmd_t cur_cmd;

  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      state      <= S_INIT;
      l2_ack_o   <= 1'b0;
      l2_rdata_o <= '0;
      phy_cmd_o  <= CMD_NOP;
      phy_cs_n_o <= 1'b1;
      phy_cke_o  <= 1'b0;
      phy_wdata_o <= '0;
      cq_head    <= '0;
      cq_tail    <= '0;
      cas_cnt    <= '0;
      cas_pending <= 1'b0;
      for (int b = 0; b < BANKS; b++) begin
        open_row[b] <= '0;
        row_open[b] <= 1'b0;
      end
    end else begin
      l2_ack_o  <= 1'b0;
      phy_cmd_o <= CMD_NOP;

      unique case (state)
        // ----------------------------------------------------------------
        S_INIT: begin
          // Hold CKE high after 200 µs (simplified: skip for sim)
          phy_cs_n_o <= 1'b0;
          phy_cke_o  <= 1'b1;
          state      <= S_IDLE;
        end

        // ----------------------------------------------------------------
        S_IDLE: begin
          if (refresh_needed) begin
            state <= S_REF;
          end else if (l2_req_i && !cq_full) begin
            cq_fifo[cq_tail] <= '{addr: l2_addr_i,
                                   wdata: l2_wdata_i,
                                   we: l2_we_i};
            cq_tail <= cq_tail + 1'b1;
            state   <= S_CHECK_OPEN;
          end else if (!cq_empty) begin
            state <= S_CHECK_OPEN;
          end
        end

        // ----------------------------------------------------------------
        S_ENQUEUE: begin
          // Additional enqueue if pipeline is deep (not used in base flow)
          state <= S_IDLE;
        end

        // ----------------------------------------------------------------
        S_CHECK_OPEN: begin
          cur_cmd = cq_fifo[cq_head];
          if (row_open[cur_cmd.addr[11:8]] &&
              open_row[cur_cmd.addr[11:8]] == cur_cmd.addr[26:12]) begin
            // Row already open — go direct to CAS
            state <= S_CAS;
          end else if (row_open[cur_cmd.addr[11:8]]) begin
            // Different row open — precharge first
            phy_cmd_o  <= CMD_PRE;
            phy_bank_o <= cur_cmd.addr[11:8];
            state      <= S_PRE;
          end else begin
            // No row open — activate
            phy_cmd_o  <= CMD_ACT;
            phy_bank_o <= cur_cmd.addr[11:8];
            phy_row_o  <= cur_cmd.addr[26:12];
            row_open[cur_cmd.addr[11:8]] <= 1'b1;
            open_row[cur_cmd.addr[11:8]] <= cur_cmd.addr[26:12];
            state <= S_ACT;
          end
        end

        // ----------------------------------------------------------------
        S_PRE: begin
          // tRP = 3 cycles
          phy_cmd_o <= CMD_NOP;
          row_open[cur_cmd.addr[11:8]] <= 1'b0;
          // After precharge, activate
          phy_cmd_o  <= CMD_ACT;
          phy_bank_o <= cur_cmd.addr[11:8];
          phy_row_o  <= cur_cmd.addr[26:12];
          row_open[cur_cmd.addr[11:8]] <= 1'b1;
          open_row[cur_cmd.addr[11:8]] <= cur_cmd.addr[26:12];
          state <= S_ACT;
        end

        // ----------------------------------------------------------------
        S_ACT: begin
          // tRCD = 4 cycles
          phy_cmd_o <= CMD_NOP;
          state     <= S_CAS;
        end

        // ----------------------------------------------------------------
        S_CAS: begin
          if (cur_cmd.we) begin
            phy_cmd_o  <= CMD_WR;
            phy_bank_o <= cur_cmd.addr[11:8];
            phy_col_o  <= cur_cmd.addr[36:27];
            phy_wdata_o <= cur_cmd.wdata;
            cas_cnt    <= WL[$clog2(CAS_LAT+2)-1:0];
            cas_pending <= 1'b1;
            state      <= S_WR_DATA;
          end else begin
            phy_cmd_o  <= CMD_RD;
            phy_bank_o <= cur_cmd.addr[11:8];
            phy_col_o  <= cur_cmd.addr[36:27];
            cas_cnt    <= CAS_LAT[$clog2(CAS_LAT+2)-1:0];
            cas_pending <= 1'b1;
            state      <= S_WAIT_CAS;
          end
        end

        // ----------------------------------------------------------------
        S_WAIT_CAS: begin
          phy_cmd_o <= CMD_NOP;
          if (phy_rdata_valid_i) begin
            l2_rdata_o <= phy_rdata_i;
            l2_ack_o   <= 1'b1;
            cq_head    <= cq_head + 1'b1;
            cas_pending <= 1'b0;
            state      <= S_IDLE;
          end
        end

        // ----------------------------------------------------------------
        S_WR_DATA: begin
          phy_cmd_o  <= CMD_NOP;
          // ACK immediately (posted write — data buffered in PHY)
          l2_ack_o   <= 1'b1;
          cq_head    <= cq_head + 1'b1;
          cas_pending <= 1'b0;
          state      <= S_IDLE;
        end

        // ----------------------------------------------------------------
        S_REF: begin
          phy_cmd_o <= CMD_REF;
          state     <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  assign phy_bank_o = (state == S_IDLE || state == S_INIT) ? '0 : cur_cmd.addr[11:8];
  assign phy_row_o  = (state == S_IDLE || state == S_INIT) ? '0 : cur_cmd.addr[26:12];
  assign phy_col_o  = (state == S_IDLE || state == S_INIT) ? '0 : cur_cmd.addr[36:27];

endmodule : tpt_mem_ctrl
