//==============================================================================
// tpt_pkg.sv — TPT ISA Package: Constants, Types, and Opcode Definitions
//==============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//==============================================================================

package tpt_pkg;

  //--------------------------------------------------------------------------
  // Width / Size Parameters
  //--------------------------------------------------------------------------
  parameter int XLEN           = 32;        // Scalar data width (bits)
  parameter int VLEN           = 512;       // Vector data width (bits)
  parameter int PLEN           = 32;        // Predicate width (bits)
  parameter int NUM_SCALAR_REGS = 32;       // Scalar register count (R0-R31)
  parameter int NUM_VECTOR_REGS = 64;       // Vector register count (V0-V63)
  parameter int NUM_PRED_REGS   = 8;        // Predicate register count (P0-P7)
  parameter int NUM_SR_REGS     = 32;       // Special register count
  parameter int NUM_WARPS       = 64;       // Maximum warp pool size
  parameter int WARP_LANES      = 32;       // Lanes per warp (SIMD width)
  parameter int NUM_CTAS        = 16;       // Maximum concurrent CTAs

  //--------------------------------------------------------------------------
  // Silicon-specific Parameters
  //--------------------------------------------------------------------------
  parameter int ADDR_WIDTH      = 48;       // Virtual address width
  parameter int PHYS_ADDR_WIDTH = 40;       // Physical address width
  parameter int ICACHE_SIZE     = 32768;    // 32 KB I-Cache
  parameter int DCACHE_SIZE     = 32768;    // 32 KB D-Cache
  parameter int CACHE_LINE_SIZE = 64;       // 64-byte cache lines
  parameter int BTB_ENTRIES     = 4096;     // Branch Target Buffer entries
  parameter int RAS_DEPTH       = 16;       // Return Address Stack depth
  parameter int TLB_ENTRIES     = 64;       // ITLB/DTLB entries
  parameter int NUM_SM_DEFAULT  = 1;        // Default number of SMs

  //--------------------------------------------------------------------------
  // Instruction Format Bit Positions
  //--------------------------------------------------------------------------
  // Common fields
  parameter int OPCODE_LSB = 27;
  parameter int OPCODE_MSB = 31;
  parameter int OPCODE_W   = 5;

  // R-Type
  parameter int RD_LSB  = 22;
  parameter int RD_MSB  = 26;
  parameter int RS1_LSB = 17;
  parameter int RS1_MSB = 21;
  parameter int RS2_LSB = 12;
  parameter int RS2_MSB = 16;
  parameter int FUNC_LSB = 7;
  parameter int FUNC_MSB = 11;

  // I-Type
  parameter int IMM_I_LSB = 5;
  parameter int IMM_I_MSB = 16;

  // M-Type (Memory)
  parameter int OFFSET_LSB = 5;
  parameter int OFFSET_MSB = 16;

  // B-Type (Branch)
  parameter int BR_OFF_LSB  = 0;
  parameter int BR_OFF_MSB  = 11;

  // J-Type (Jump)
  parameter int JMP_TGT_LSB = 5;
  parameter int JMP_TGT_MSB = 26;

  // V-Type (Vector/Tensor)
  parameter int VD_LSB    = 22;
  parameter int VD_MSB    = 26;
  parameter int VS1_LSB   = 17;
  parameter int VS1_MSB   = 21;
  parameter int VS2_LSB   = 12;
  parameter int VS2_MSB   = 16;
  parameter int VSZ_LSB   = 10;
  parameter int VSZ_MSB   = 11;
  parameter int VDM_LSB   = 9;
  parameter int VDM_MSB   = 10;
  parameter int VSUBOP_LSB = 0;
  parameter int VSUBOP_MSB = 4;

  //--------------------------------------------------------------------------
  // Major Opcodes
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    OP_ALU_INT    = 5'b00000,
    OP_ALU_FP     = 5'b00001,
    OP_ALU_COMP   = 5'b00010,
    OP_ALU_LOG    = 5'b00011,
    OP_MEM_LD     = 5'b00100,
    OP_MEM_ST     = 5'b00101,
    OP_MEM_ATOM   = 5'b00110,
    OP_CTRL_BR    = 5'b00111,
    OP_CTRL_J     = 5'b01000,
    OP_CTRL_SYNC  = 5'b01001,
    OP_VEC        = 5'b01010,
    OP_TENSOR     = 5'b01011,
    OP_TEX        = 5'b01100,
    OP_CONV       = 5'b01101,
    OP_SYSTEM     = 5'b01110,
    OP_PRED       = 5'b01111,
    OP_MISC       = 5'b10000
  } opcode_t;

  //--------------------------------------------------------------------------
  // Integer ALU Functions
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    FUNC_ADD    = 5'b00000,
    FUNC_ADDI   = 5'b00001,
    FUNC_SUB    = 5'b00010,
    FUNC_SUBI   = 5'b00011,
    FUNC_MUL    = 5'b00100,
    FUNC_MULHI  = 5'b00101,
    FUNC_DIV    = 5'b00110,
    FUNC_MOD    = 5'b00111,
    FUNC_AND    = 5'b01000,
    FUNC_OR     = 5'b01001,
    FUNC_XOR    = 5'b01010,
    FUNC_SLL    = 5'b01011,
    FUNC_SRL    = 5'b01100,
    FUNC_SRA    = 5'b01101,
    FUNC_CLZ    = 5'b01110,
    FUNC_POPC   = 5'b01111,
    FUNC_MIN    = 5'b10000,
    FUNC_MAX    = 5'b10001,
    FUNC_ABS    = 5'b10010,
    FUNC_NEG    = 5'b10011
  } alu_int_func_t;

  //--------------------------------------------------------------------------
  // FP ALU Functions
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    FUNC_FADD    = 5'b00000,
    FUNC_FSUB    = 5'b00001,
    FUNC_FMUL    = 5'b00010,
    FUNC_FDIV    = 5'b00011,
    FUNC_FADD16  = 5'b00100,
    FUNC_FMUL16  = 5'b00101,
    FUNC_FADD64  = 5'b00110,
    FUNC_FMUL64  = 5'b00111,
    FUNC_F2I     = 5'b01000,
    FUNC_I2F     = 5'b01001,
    FUNC_F2F16   = 5'b01010,
    FUNC_F16TOF  = 5'b01011,
    FUNC_FMA     = 5'b01100,
    FUNC_FMA16   = 5'b01101,
    FUNC_FMA64   = 5'b01110,
    FUNC_FSQRT   = 5'b01111
  } alu_fp_func_t;

  //--------------------------------------------------------------------------
  // Memory Function Codes
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    MEM_LB   = 5'b00000,
    MEM_LBU  = 5'b00001,
    MEM_LH   = 5'b00010,
    MEM_LHU  = 5'b00011,
    MEM_LW   = 5'b00100,
    MEM_LD   = 5'b00101,
    MEM_LV   = 5'b00110,
    MEM_SB   = 5'b00111,
    MEM_SH   = 5'b01000,
    MEM_SW   = 5'b01001,
    MEM_SD   = 5'b01010,
    MEM_SV   = 5'b01011
  } mem_func_t;

  //--------------------------------------------------------------------------
  // Control / Branch Functions
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    CTRL_BEQ  = 5'b00000,
    CTRL_BNE  = 5'b00001,
    CTRL_BLT  = 5'b00010,
    CTRL_BGE  = 5'b00011,
    CTRL_BLTU = 5'b00100,
    CTRL_BGEU = 5'b00101,
    CTRL_JAL  = 5'b00110,
    CTRL_JALR = 5'b00111,
    CTRL_RET  = 5'b01000,
    CTRL_BAR  = 5'b01001
  } ctrl_func_t;

  //--------------------------------------------------------------------------
  // Special Register Addresses
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    SR_LANE_ID   = 5'd0,
    SR_WARP_ID   = 5'd1,
    SR_CTA_ID_X  = 5'd2,
    SR_CTA_ID_Y  = 5'd3,
    SR_CTA_ID_Z  = 5'd4,
    SR_NTID_X    = 5'd5,
    SR_NTID_Y    = 5'd6,
    SR_NTID_Z    = 5'd7,
    SR_CLOCK     = 5'd8,
    SR_STATUS    = 5'd9,
    SR_MASK      = 5'd10,
    SR_TVEC      = 5'd12,
    SR_INST_RET  = 5'd20,
    SR_CORE_CYCL = 5'd21,
    SR_L1D_MISS  = 5'd22,
    SR_L1I_MISS  = 5'd23,
    SR_BR_MISPRED = 5'd24,
    SR_WAR_STALL = 5'd25
  } special_reg_t;

  //--------------------------------------------------------------------------
  // Exception Codes
  //--------------------------------------------------------------------------
  typedef enum logic [4:0] {
    EXC_NONE           = 5'd0,
    EXC_INST_PF        = 5'd1,
    EXC_DATA_PF        = 5'd2,
    EXC_ILLEGAL_INST   = 5'd3,
    EXC_PRIVILEGE      = 5'd4,
    EXC_MISALIGNED     = 5'd5,
    EXC_DIV_BY_ZERO    = 5'd6,
    EXC_OVERFLOW       = 5'd7
  } exception_t;

  //--------------------------------------------------------------------------
  // Pipeline Stage Types
  //--------------------------------------------------------------------------
  typedef enum logic [3:0] {
    STAGE_F1  = 4'd0,
    STAGE_F2  = 4'd1,
    STAGE_D1  = 4'd2,
    STAGE_D2  = 4'd3,
    STAGE_E1  = 4'd4,
    STAGE_E2  = 4'd5,
    STAGE_E3  = 4'd6,
    STAGE_E4  = 4'd7,
    STAGE_W1  = 4'd8,
    STAGE_NOP = 4'd15
  } stage_t;

  //--------------------------------------------------------------------------
  // Memory Address Space Encoding
  //--------------------------------------------------------------------------
  typedef enum logic [1:0] {
    ASPACE_GLOBAL  = 2'b00,
    ASPACE_SHARED  = 2'b01,
    ASPACE_LOCAL   = 2'b10,
    ASPACE_CONSTANT = 2'b11
  } aspace_t;

  //--------------------------------------------------------------------------
  // Decoded Instruction Structure
  //--------------------------------------------------------------------------
  typedef struct packed {
    logic        valid;             // Instruction is valid
    opcode_t     opcode;            // Major opcode
    logic [4:0]  rd;                // Destination register
    logic [4:0]  rs1;               // Source register 1
    logic [4:0]  rs2;               // Source register 2
    logic [4:0]  func;              // Function selector
    logic [4:0]  subop;             // Sub-operation (vector/tensor)
    logic [11:0] imm;               // Immediate / offset
    logic [21:0] jump_target;       // Jump target
    logic [1:0]  vec_size;          // Vector data size
    logic [1:0]  vec_dest_mod;      // Vector destination modifier
    logic        is_r_type;         // R-type format
    logic        is_i_type;         // I-type format
    logic        is_m_type;         // M-type format
    logic        is_b_type;         // B-type format
    logic        is_j_type;         // J-type format
    logic        is_v_type;         // V-type format
    logic        is_load;           // Load operation
    logic        is_store;          // Store operation
    logic        is_branch;         // Branch operation
    logic        is_jump;           // Jump operation
    logic        is_alu;            // ALU operation
    logic        is_fp;             // Floating point operation
    logic        is_vector;         // Vector operation
    logic        is_tensor;         // Tensor operation
    logic        uses_predicate;    // Uses predicate register
    logic [4:0]  pred_reg;          // Predicate register index
  } decoded_instr_t;

endpackage : tpt_pkg
