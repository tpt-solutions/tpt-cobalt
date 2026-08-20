//==============================================================================
// tpt_tb.sv — TPT Core Testbench
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Self-checking simulation testbench for the TPT core.
// Loads instruction and data memories from hex files,
// runs the simulation, and verifies results.
//==============================================================================

module tpt_tb;

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Clock and reset
  //--------------------------------------------------------------------------
  logic clk;
  logic rst_n;

  always #5 clk = ~clk;  // 100 MHz clock

  //--------------------------------------------------------------------------
  // DUT interface
  //--------------------------------------------------------------------------
  logic [31:0] imem_addr;
  logic [31:0] imem_rdata;
  logic        imem_req;
  logic        imem_ack;

  logic [31:0] dmem_addr;
  logic        dmem_req;
  logic        dmem_we;
  logic [3:0]  dmem_be;
  logic [31:0] dmem_wdata;
  logic        dmem_ack;
  logic [31:0] dmem_rdata;

  //--------------------------------------------------------------------------
  // Memory models
  //--------------------------------------------------------------------------
  logic [31:0] instr_mem [0:4095];  // 16 KB instruction memory
  logic [31:0] data_mem  [0:4095];  // 16 KB data memory

  // Initialize from hex file
  initial begin
    $readmemh("programs/simple_add.hex", instr_mem);
  end

  // Instruction memory read (single cycle)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      imem_ack <= 1'b0;
      imem_rdata <= '0;
    end else if (imem_req) begin
      imem_ack    <= 1'b1;
      imem_rdata  <= instr_mem[imem_addr[13:2]];
    end else begin
      imem_ack <= 1'b0;
    end
  end

  // Data memory (byte-addressable)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dmem_ack <= 1'b0;
      dmem_rdata <= '0;
    end else if (dmem_req) begin
      dmem_ack <= 1'b1;
      if (dmem_we) begin
        if (dmem_be[0]) data_mem[dmem_addr[13:2]][7:0]   <= dmem_wdata[7:0];
        if (dmem_be[1]) data_mem[dmem_addr[13:2]][15:8]  <= dmem_wdata[15:8];
        if (dmem_be[2]) data_mem[dmem_addr[13:2]][23:16] <= dmem_wdata[23:16];
        if (dmem_be[3]) data_mem[dmem_addr[13:2]][31:24] <= dmem_wdata[31:24];
      end
      dmem_rdata <= data_mem[dmem_addr[13:2]];
    end else begin
      dmem_ack <= 1'b0;
    end
  end

  //--------------------------------------------------------------------------
  // DUT instantiation
  //--------------------------------------------------------------------------
  tpt_core dut (
      .clk_i         (clk),
      .rst_n_i       (rst_n),

      .imem_addr_o   (imem_addr),
      .imem_rdata_i  (imem_rdata),
      .imem_req_o    (imem_req),
      .imem_ack_i    (imem_ack),

      .dmem_addr_o   (dmem_addr),
      .dmem_req_o    (dmem_req),
      .dmem_we_o     (dmem_we),
      .dmem_be_o     (dmem_be),
      .dmem_wdata_o  (dmem_wdata),
      .dmem_ack_i    (dmem_ack),
      .dmem_rdata_i  (dmem_rdata)
  );

  //--------------------------------------------------------------------------
  // Test stimulus and monitoring
  //--------------------------------------------------------------------------
  int cycle_count;

  initial begin
    // Dump waveforms
    $dumpfile("tpt_sim.vcd");
    $dumpvars(0, tpt_tb);

    // Initialize
    clk   = 0;
    rst_n = 0;
    cycle_count = 0;

    // Load data memory with test data
    for (int i = 0; i < 16; i++) begin
      data_mem[i] = 32'(i * 10);  // [0, 10, 20, ..., 150]
    end

    // Reset pulse
    #20 rst_n = 1;

    // Run simulation
    #5000;

    // Display final state
    $display("========================================");
    $display("TPT Core Simulation Results");
    $display("Cycles: %0d", cycle_count);
    $display("========================================");

    // Check data memory results
    for (int i = 0; i < 8; i++) begin
      $display("  data_mem[%0d] = 0x%08h (%0d)", i, data_mem[i], data_mem[i]);
    end

    $display("========================================");

    // Verify results
    if (data_mem[4] == 32'd120 && data_mem[5] == 32'd100) begin
      $display("PASS: All tests completed successfully.");
    end else begin
      $display("FAIL: Test results mismatch.");
      $display("  Expected data_mem[4] = 120, got %0d", data_mem[4]);
      $display("  Expected data_mem[5] = 100, got %0d", data_mem[5]);
    end

    $display("========================================");
    $finish;
  end

  // Cycle counter
  always @(posedge clk) begin
    if (rst_n) cycle_count <= cycle_count + 1;
  end

  // Watchdog timer
  initial begin
    #10000;
    $display("ERROR: Simulation timeout reached.");
    $finish;
  end

endmodule : tpt_tb
