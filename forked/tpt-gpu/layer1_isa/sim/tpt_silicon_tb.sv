//==============================================================================
// tpt_silicon_tb.sv — TPT GPU Silicon Verification Testbench
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Full-system testbench for the TPT GPU silicon integration.
// Tests: MMIO boot, warp dispatch, CSR reads/writes.
//==============================================================================

module tpt_silicon_tb;

  import tpt_pkg::*;

  logic clk;
  logic rst_n;
  logic [11:0] host_mmio_addr;
  logic        host_mmio_req;
  logic        host_mmio_we;
  logic [31:0] host_mmio_wdata;
  logic [31:0] host_mmio_rdata;
  logic        host_mmio_ack;
  logic        host_intr;
  logic [39:0] mem_addr;
  logic        mem_req, mem_we;
  logic [511:0] mem_wdata;
  logic        mem_ack;
  logic [511:0] mem_rdata;

  always #5 clk = ~clk;

  tpt_gpu_top #(.NUM_SM(1)) dut (
      .clk_i             (clk),
      .rst_n_i           (rst_n),
      .host_mmio_addr_i  (host_mmio_addr),
      .host_mmio_req_i   (host_mmio_req),
      .host_mmio_we_i    (host_mmio_we),
      .host_mmio_wdata_i (host_mmio_wdata),
      .host_mmio_rdata_o (host_mmio_rdata),
      .host_mmio_ack_o   (host_mmio_ack),
      .host_intr_o       (host_intr),
      .mem_addr_o        (mem_addr),
      .mem_req_o         (mem_req),
      .mem_we_o          (mem_we),
      .mem_wdata_o       (mem_wdata),
      .mem_ack_i         (mem_ack),
      .mem_rdata_i       (mem_rdata)
  );

  int pass_count, fail_count;

  // MMIO write task
  task mmio_write(input [11:0] addr, input [31:0] data);
    begin
      @(posedge clk);
      host_mmio_addr  <= addr;
      host_mmio_req   <= 1'b1;
      host_mmio_we    <= 1'b1;
      host_mmio_wdata <= data;
      @(posedge clk);
      host_mmio_req <= 1'b0;
      host_mmio_we  <= 1'b0;
      #10;
    end
  endtask

  // MMIO read task
  task mmio_read(input [11:0] addr, output [31:0] data);
    begin
      @(posedge clk);
      host_mmio_addr <= addr;
      host_mmio_req  <= 1'b1;
      host_mmio_we   <= 1'b0;
      @(posedge clk);
      host_mmio_req <= 1'b0;
      #10;
      data = host_mmio_rdata;
    end
  endtask

  // Main test sequence
  initial begin
    $dumpfile("tpt_silicon_sim.vcd");
    $dumpvars(0, tpt_silicon_tb);

    clk = 0; rst_n = 0;
    host_mmio_addr = '0; host_mmio_req = '0;
    host_mmio_we = '0; host_mmio_wdata = '0;
    mem_ack = '0; mem_rdata = '0;
    pass_count = 0; fail_count = 0;

    #20 rst_n = 1;
    #10;

    // Test 1: Read STATUS before boot
    $display("--- Test 1: STATUS before boot ---");
    mmio_read(12'h004, host_mmio_rdata);
    $display("  STATUS = 0x%08h", host_mmio_rdata);

    // Test 2: Boot the GPU
    $display("--- Test 2: Boot sequence ---");
    mmio_write(12'h000, 32'h00000001);  // BOOT=1
    #10;
    mmio_read(12'h004, host_mmio_rdata);
    if (host_mmio_rdata[0]) begin
      $display("  PASS: GPU ready after boot");
      pass_count++;
    end else begin
      $display("  FAIL: GPU not ready");
      fail_count++;
    end

    // Test 3: Enable scheduler
    $display("--- Test 3: Enable scheduler ---");
    mmio_write(12'h020, 32'h00000001);
    #10;

    // Test 4: Query VRAM
    $display("--- Test 4: Query VRAM ---");
    mmio_read(12'h030, host_mmio_rdata);
    $display("  VRAM = 0x%08h (%0d MB)", host_mmio_rdata, host_mmio_rdata/1024/1024);

    // Test 5: Query CTAs
    $display("--- Test 5: Query CTAS ---");
    mmio_read(12'h038, host_mmio_rdata);
    $display("  CTAs = %0d", host_mmio_rdata[5:0]);
    if (host_mmio_rdata[5:0] == 5'd16) begin
      $display("  PASS: CTAs = 16");
      pass_count++;
    end else begin
      $display("  FAIL: CTAs mismatch");
      fail_count++;
    end

    // Test 6: Version
    $display("--- Test 6: Version ---");
    mmio_read(12'h03C, host_mmio_rdata);
    $display("  Version: major=%0d minor=%0d", host_mmio_rdata[31:16], host_mmio_rdata[15:0]);

    // Test 7: Dispatch warp 0
    $display("--- Test 7: Dispatch warp 0 ---");
    mmio_write(12'h014, 32'h00000000);
    #20;
    mmio_read(12'h024, host_mmio_rdata);
    $display("  Active warps = %0d", host_mmio_rdata[5:0]);

    // Test 8: Interrupt mask
    $display("--- Test 8: Interrupt mask ---");
    mmio_write(12'h00C, 32'h000000FF);
    mmio_read(12'h00C, host_mmio_rdata);
    if (host_mmio_rdata == 32'hFF) begin
      $display("  PASS: Intr mask set");
      pass_count++;
    end else begin
      $display("  FAIL: Intr mask = 0x%08h", host_mmio_rdata);
      fail_count++;
    end

    // Test 9: CTRL register readback
    $display("--- Test 9: CTRL readback ---");
    mmio_read(12'h000, host_mmio_rdata);
    if (host_mmio_rdata[0]) begin
      $display("  PASS: BOOT bit set in CTRL");
      pass_count++;
    end else begin
      $display("  FAIL: BOOT bit not set");
      fail_count++;
    end

    // Summary
    #100;
    $display("========================================");
    $display("TPT Silicon Testbench Results");
    $display("  PASSED: %0d", pass_count);
    $display("  FAILED: %0d", fail_count);
    if (fail_count == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  SOME TESTS FAILED");
    $display("========================================");
    $finish;
  end

  // Watchdog
  initial begin
    #10000;
    $display("ERROR: Simulation timeout");
    $finish;
  end

endmodule : tpt_silicon_tb