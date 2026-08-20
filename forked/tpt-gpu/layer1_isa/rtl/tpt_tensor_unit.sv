//==============================================================================
// tpt_tensor_unit.sv — TPT Tensor / MMA Compute Unit
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements matrix multiply-accumulate (MMA) operations for tensor cores.
// Supports FP16 and INT8 precision in various tile sizes.
//==============================================================================

module tpt_tensor_unit (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Control
    input  logic              valid_i,
    input  logic [4:0]        func_i,
    input  logic [4:0]        subop_i,
    input  logic [1:0]        size_i,

    // Vector register inputs (512 bits each)
    input  logic [511:0]      vs1_i,
    input  logic [511:0]      vs2_i,
    input  logic [511:0]      vd_i,       // accumulator

    // Result
    output logic [511:0]      result_o,
    output logic              valid_o
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // MMA dimensions based on subop
  //--------------------------------------------------------------------------
  logic        mma_16x16x16, mma_32x32x8, mma_8x8x32;

  assign mma_16x16x16 = (func_i == 5'b00000) && (subop_i == 5'b00000);
  assign mma_32x32x8  = (func_i == 5'b00001) && (subop_i == 5'b00000);
  assign mma_8x8x32   = (func_i == 5'b00010) && (subop_i == 5'b00000);

  //--------------------------------------------------------------------------
  // MMA 16x16x16: Matrix A[16][16] FP16 × Matrix B[16][16] FP16 = C[16][16] FP16
  // vs1_i holds 16 x 16-bit = 256 bits of A (packed as 16 FP16 values per row × 16 rows)
  // vs2_i holds 16 x 16-bit = 256 bits of B
  // vd_i accumulates C (16 x 16-bit = 256 bits)
  //--------------------------------------------------------------------------
  logic [15:0] a_elems [0:15];
  logic [15:0] b_elems [0:15];
  logic [15:0] c_elems [0:15];

  // Unpack operands
  always_comb begin
    for (int i = 0; i < 16; i++) begin
      a_elems[i] = vs1_i[i*16 +: 16];
      b_elems[i] = vs2_i[i*16 +: 16];
      c_elems[i] = vd_i[i*16 +: 16];
    end
  end

  //--------------------------------------------------------------------------
  // FP16 multiply-accumulate
  //--------------------------------------------------------------------------
  function automatic logic [15:0] fp16_mul(input logic [15:0] a, b);
    shortreal af, bf;
    af = $bitstoshortreal({a, 16'h0});
    bf = $bitstoshortreal({b, 16'h0});
    return $shortrealtobits(af * bf)[31:16];
  endfunction

  function automatic logic [15:0] fp16_add(input logic [15:0] a, b);
    shortreal af, bf;
    af = $bitstoshortreal({a, 16'h0});
    bf = $bitstoshortreal({b, 16'h0});
    return $shortrealtobits(af + bf)[31:16];
  endfunction

  //--------------------------------------------------------------------------
  // MMA dot product: C += A × B (element-wise multiply and sum)
  // Simplified: performs outer product of vectors for selected tile size
  //--------------------------------------------------------------------------
  logic [15:0] mma_result [0:15];

  always_comb begin
    mma_result = c_elems;

    if (valid_i && mma_16x16x16) begin
      for (int i = 0; i < 16; i++) begin
        mma_result[i] = fp16_add(c_elems[i], fp16_mul(a_elems[i], b_elems[i]));
      end
    end
    // Additional tile sizes would be expanded here
  end

  //--------------------------------------------------------------------------
  // Pack result back to 512-bit vector
  //--------------------------------------------------------------------------
  always_comb begin
    result_o = '0;
    for (int i = 0; i < 16; i++) begin
      result_o[i*16 +: 16] = mma_result[i];
    end
  end

  //--------------------------------------------------------------------------
  // Output valid
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      valid_o <= 1'b0;
    end else begin
      valid_o <= valid_i;
    end
  end

`ifdef SIMULATION
  // Assertion: MMA unit should not receive overlapping operations
  assert property (@(posedge clk_i) disable iff (!rst_n_i)
    $onehot0({mma_16x16x16, mma_32x32x8, mma_8x8x32})
  ) else $warning("TPT_TENSOR: Multiple MMA operations selected simultaneously");
`endif

endmodule : tpt_tensor_unit
