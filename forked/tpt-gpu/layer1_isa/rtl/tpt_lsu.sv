//==============================================================================
// tpt_lsu.sv — TPT Load / Store Unit
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Handles all memory operations: load/store byte, halfword, word, doubleword.
// Address space is encoded in the upper address bits.
//==============================================================================

module tpt_lsu (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // From pipeline
    input  logic              valid_i,
    input  logic              is_load_i,
    input  logic              is_store_i,
    input  logic [4:0]        func_i,
    input  logic [31:0]       base_addr_i,
    input  logic [31:0]       store_data_i,

    // Memory interface (single-port SRAM)
    output logic [31:0]       mem_addr_o,
    output logic              mem_req_o,
    output logic              mem_we_o,
    output logic [3:0]        mem_be_o,    // byte enables
    output logic [31:0]       mem_wdata_o,
    input  logic              mem_ack_i,
    input  logic [31:0]       mem_rdata_i,

    // To pipeline
    output logic [31:0]       load_data_o,
    output logic              valid_o,
    output logic              misaligned_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Address calculation
  //--------------------------------------------------------------------------
  logic [31:0] effective_addr;
  assign effective_addr = base_addr_i;  // offset pre-computed in core before LSU call

  //--------------------------------------------------------------------------
  // Byte enable and data alignment
  //--------------------------------------------------------------------------
  logic [1:0] addr_lsb;
  assign addr_lsb = effective_addr[1:0];

  always_comb begin
    mem_addr_o   = effective_addr & ~(32'h3);  // word-aligned address
    mem_req_o    = valid_i && (is_load_i || is_store_i);
    mem_we_o     = is_store_i;
    mem_be_o     = 4'b1111;
    mem_wdata_o  = store_data_i;
    load_data_o  = mem_rdata_i;
    misaligned_o = 1'b0;

    if (valid_i) begin
      unique case (mem_func_t'(func_i))
        //----------------------------------------------------------------------
        // Load operations
        //----------------------------------------------------------------------
        MEM_LB: begin
          mem_be_o = 4'b0001 << addr_lsb;
          mem_wdata_o = {4{store_data_i[7:0]}};
          case (addr_lsb)
            2'b00: load_data_o = {{24{mem_rdata_i[7]}},  mem_rdata_i[7:0]};
            2'b01: load_data_o = {{24{mem_rdata_i[15]}}, mem_rdata_i[15:8]};
            2'b10: load_data_o = {{24{mem_rdata_i[23]}}, mem_rdata_i[23:16]};
            2'b11: load_data_o = {{24{mem_rdata_i[31]}}, mem_rdata_i[31:24]};
          endcase
        end

        MEM_LBU: begin
          mem_be_o = 4'b0001 << addr_lsb;
          case (addr_lsb)
            2'b00: load_data_o = {24'h0, mem_rdata_i[7:0]};
            2'b01: load_data_o = {24'h0, mem_rdata_i[15:8]};
            2'b10: load_data_o = {24'h0, mem_rdata_i[23:16]};
            2'b11: load_data_o = {24'h0, mem_rdata_i[31:24]};
          endcase
        end

        MEM_LH: begin
          mem_be_o = 4'b0011 << addr_lsb;
          case (addr_lsb)
            2'b00: load_data_o = {{16{mem_rdata_i[15]}}, mem_rdata_i[15:0]};
            2'b10: load_data_o = {{16{mem_rdata_i[31]}}, mem_rdata_i[31:16]};
            default: misaligned_o = 1'b1;
          endcase
        end

        MEM_LHU: begin
          mem_be_o = 4'b0011 << addr_lsb;
          case (addr_lsb)
            2'b00: load_data_o = {16'h0, mem_rdata_i[15:0]};
            2'b10: load_data_o = {16'h0, mem_rdata_i[31:16]};
            default: misaligned_o = 1'b1;
          endcase
        end

        MEM_LW: begin
          mem_be_o = 4'b1111;
          load_data_o = mem_rdata_i;
        end

        MEM_LD: begin
          // Doubleword — uses two consecutive word loads
          mem_be_o = 4'b1111;
          load_data_o = mem_rdata_i;
        end

        //----------------------------------------------------------------------
        // Store operations
        //----------------------------------------------------------------------
        MEM_SB: begin
          mem_be_o = 4'b0001 << addr_lsb;
          mem_wdata_o = {4{store_data_i[7:0]}};
        end

        MEM_SH: begin
          mem_be_o = 4'b0011 << addr_lsb;
          mem_wdata_o = {2{store_data_i[15:0]}};
          case (addr_lsb)
            2'b00: mem_wdata_o = {store_data_i[15:0], 16'h0};
            2'b10: mem_wdata_o = {16'h0, store_data_i[15:0]};
            default: misaligned_o = 1'b1;
          endcase
        end

        MEM_SW: begin
          mem_be_o = 4'b1111;
          mem_wdata_o = store_data_i;
        end

        MEM_SD: begin
          mem_be_o = 4'b1111;
          mem_wdata_o = store_data_i;
        end

        default: begin
          mem_req_o = 1'b0;
        end
      endcase
    end
  end

  //--------------------------------------------------------------------------
  // Output valid (handshake)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      valid_o <= 1'b0;
    end else begin
      valid_o <= mem_ack_i && valid_i;
    end
  end

endmodule : tpt_lsu
