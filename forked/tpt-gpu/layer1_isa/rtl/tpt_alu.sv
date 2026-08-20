//==============================================================================
// tpt_alu.sv — TPT ALU / Compute Unit
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements integer and floating-point ALU operations.
// Supports packed SIMD within 32-bit word for sub-word ops.
//==============================================================================

module tpt_alu (
    input  logic              clk_i,
    input  logic              rst_n_i,

    // Control
    input  logic              valid_i,
    input  logic              is_fp_i,
    input  logic [4:0]        func_i,

    // Operands
    input  logic [31:0]       rs1_i,
    input  logic [31:0]       rs2_i,
    input  logic [31:0]       imm_i,       // sign-extended immediate
    input  logic              use_imm_i,   // select immediate as src2

    // Results
    output logic [31:0]       result_o,
    output logic              valid_o,
    output logic              zero_o,      // result == 0
    output logic              negative_o,  // result[31] == 1
    output logic              overflow_o   // signed overflow
);

  import tpt_pkg::*;

  //--------------------------------------------------------------------------
  // Operand selection
  //--------------------------------------------------------------------------
  logic [31:0] src_a, src_b;
  assign src_a = rs1_i;
  assign src_b = use_imm_i ? imm_i : rs2_i;

  //--------------------------------------------------------------------------
  // ALU core
  //--------------------------------------------------------------------------
  logic [31:0] alu_result;
  logic        alu_zero, alu_negative, alu_overflow;
  logic        alu_valid;

  always_comb begin
    alu_result   = '0;
    alu_zero     = 1'b0;
    alu_negative = 1'b0;
    alu_overflow = 1'b0;
    alu_valid    = valid_i;

    if (valid_i) begin
      if (is_fp_i) begin
        // ---- Floating-Point Operations ----
        unique case (alu_fp_func_t'(func_i))
          FUNC_FADD:   alu_result = $shortrealtobits($bitstoshortreal(src_a) + $bitstoshortreal(src_b));
          FUNC_FSUB:   alu_result = $shortrealtobits($bitstoshortreal(src_a) - $bitstoshortreal(src_b));
          FUNC_FMUL:   alu_result = $shortrealtobits($bitstoshortreal(src_a) * $bitstoshortreal(src_b));
          FUNC_FDIV:   alu_result = $shortrealtobits($bitstoshortreal(src_a) / $bitstoshortreal(src_b));
          FUNC_FMA:    alu_result = $shortrealtobits($bitstoshortreal(src_a) * $bitstoshortreal(src_b)
                                                     + $bitstoshortreal(src_a));  // rd = rs1*rs2 + rs1
          FUNC_F2I:    alu_result = $rtoi($bitstoshortreal(src_a));
          FUNC_I2F:    alu_result = $shortrealtobits($itor($signed(src_a)));
          FUNC_FSQRT:  alu_result = $shortrealtobits($sqrt($bitstoshortreal(src_a)));
          default:     alu_result = src_a;
        endcase
      end else begin
        // ---- Integer Operations ----
        unique case (alu_int_func_t'(func_i))
          FUNC_ADD: begin
            {alu_overflow, alu_result} = {src_a[31], src_a} + {src_b[31], src_b};
          end
          FUNC_ADDI: begin
            {alu_overflow, alu_result} = {src_a[31], src_a} + {src_b[31], src_b};
          end
          FUNC_SUB: begin
            {alu_overflow, alu_result} = {src_a[31], src_a} - {src_b[31], src_b};
          end
          FUNC_SUBI: begin
            {alu_overflow, alu_result} = {src_a[31], src_a} - {src_b[31], src_b};
          end
          FUNC_MUL:    alu_result = src_a * src_b;
          FUNC_MULHI:  alu_result = ($signed(src_a) * $signed(src_b)) >> 32;
          FUNC_DIV:    alu_result = $signed(src_a) / $signed(src_b);
          FUNC_MOD:    alu_result = $signed(src_a) % $signed(src_b);
          FUNC_AND:    alu_result = src_a & src_b;
          FUNC_OR:     alu_result = src_a | src_b;
          FUNC_XOR:    alu_result = src_a ^ src_b;
          FUNC_SLL:    alu_result = src_a << src_b[4:0];
          FUNC_SRL:    alu_result = src_a >> src_b[4:0];
          FUNC_SRA:    alu_result = $signed(src_a) >>> src_b[4:0];
          FUNC_CLZ: begin
            alu_result = '0;
            for (int i = 31; i >= 0; i--) begin
              if (src_a[i]) begin
                alu_result = 31 - i;
                break;
              end
            end
          end
          FUNC_POPC: begin
            alu_result = '0;
            for (int i = 0; i < 32; i++) begin
              if (src_a[i]) alu_result = alu_result + 1;
            end
          end
          FUNC_MIN: alu_result = ($signed(src_a) < $signed(src_b)) ? src_a : src_b;
          FUNC_MAX: alu_result = ($signed(src_a) > $signed(src_b)) ? src_a : src_b;
          FUNC_ABS: alu_result = src_a[31] ? (~src_a + 1) : src_a;
          FUNC_NEG: alu_result = ~src_a + 1;
          default:  alu_result = src_a;
        endcase
      end
    end

    alu_zero     = (alu_result == '0);
    alu_negative = alu_result[31];
  end

  //--------------------------------------------------------------------------
  // Output registers
  //--------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      result_o   <= '0;
      valid_o    <= 1'b0;
      zero_o     <= 1'b0;
      negative_o <= 1'b0;
      overflow_o <= 1'b0;
    end else begin
      result_o   <= alu_result;
      valid_o    <= alu_valid;
      zero_o     <= alu_zero;
      negative_o <= alu_negative;
      overflow_o <= alu_overflow;
    end
  end

endmodule : tpt_alu
