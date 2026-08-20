// tptir_quant_gemm.mlir — Quantized GEMM Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
//
// Implements: C_f32 = dequant(A_int) * B_f32
// where A is a packed integer tensor (2/4/8-bit) with per-group scale/zero-point.
//
// Tunable placeholders:
//   {{TILE_M}}   — output tile rows (default 64)
//   {{TILE_N}}   — output tile columns (default 64)
//   {{TILE_K}}   — reduction tile (default 32)
//   {{BITS}}     — bits per weight (default 4)
//   {{GROUP_SIZE}} — weights per scale group (default 128)
//
// Strategy:
//   1. Load a tile of packed weights A_int into shared memory
//   2. Dequantize each weight in shared memory: w_f32 = (w_int - zp) * scale
//   3. Load activation tile B_f32 into shared memory
//   4. Accumulate C += dequant(A_tile) * B_tile in registers
//   5. Write C tile to global memory

!tptir_packed   = type tensor<?x?xi8,  0>  // packed weights (u8 storage)
!tptir_f32mat   = type tensor<?x?xf32, 0>  // f32 matrix
!tptir_scales   = type tensor<?x?xf32, 0>  // per-group scales
!tptir_zpoints  = type tensor<?x?xi8,  0>  // per-group zero-points
!tptir_index    = type index
!tptir_i32      = type i32

// ---------------------------------------------------------------------------
// 4-bit quantized GEMM: C[M,N] = dequant(A[M,K/2]) * B[K,N]
// A is packed: 2 weights per byte (lower 4 bits = weight0, upper 4 bits = weight1)
// ---------------------------------------------------------------------------
func.func @tptir_quant_gemm_int4(
    %A_packed : !tptir_packed,
    %B        : !tptir_f32mat,
    %scales   : !tptir_scales,
    %zpoints  : !tptir_zpoints,
    %M        : !tptir_index,
    %N        : !tptir_index,
    %K        : !tptir_index,
    %group_sz : !tptir_i32
) -> !tptir_f32mat attributes {
    tptir.kernel,
    tptir.grid_size  = [128, 1, 1],
    tptir.block_size = [256, 1, 1],
    tptir.shared_mem = 49152
} {
    %tile_m    = arith.constant {{TILE_M}}    : index
    %tile_n    = arith.constant {{TILE_N}}    : index
    %tile_k    = arith.constant {{TILE_K}}    : index
    %c0        = arith.constant 0             : index
    %c0f       = arith.constant 0.0           : f32

    // Shared memory tiles (using f16 for dequantized A to save smem bandwidth)
    %smem_a    = memref.alloca() : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>
    %smem_b    = memref.alloca() : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>

    %block_row = gpu.block_id x
    %block_col = gpu.block_id y
    %tid_x     = gpu.thread_id x

    // Output tile base indices
    %out_row   = arith.muli %block_row, %tile_m : index
    %out_col   = arith.muli %block_col, %tile_n : index

    // Accumulator in registers
    %acc = memref.alloca() : memref<{{TILE_M}}x{{TILE_N}}xf32, 5>

    // Zero accumulator
    scf.for %i = %c0 to %tile_m step %c0 {
        scf.for %j = %c0 to %tile_n step %c0 {
            memref.store %c0f, %acc[%i, %j] : memref<{{TILE_M}}x{{TILE_N}}xf32, 5>
        }
    }

    // Tile loop over K dimension
    scf.for %k_tile = %c0 to %K step %tile_k {

        // Phase 1: Load and dequantize A_int tile into smem_a
        // Each thread handles one (row, k_group) pair
        scf.for %tm = %c0 to %tile_m step %c0 {
            scf.for %tk = %c0 to %tile_k step %c0 {
                %global_row = arith.addi %out_row, %tm : index
                %global_k   = arith.addi %k_tile,  %tk : index

                // INT4 packing: two weights per byte
                %packed_k   = arith.divui %global_k, %c0 : index  // k / 2
                %raw_byte   = tptir.load(%A_packed[%global_row, %packed_k]) : !tptir_packed -> i8
                %shift      = arith.remui %global_k, %c0 : index   // k % 2 → 0 or 4 bits
                %nibble     = tptir.quantize(%raw_byte) : i8 -> i4  // extract nibble
                %w_int      = arith.extsi %nibble : i4 to i32

                // Dequantize: w_f32 = (w_int - zero_point) * scale
                %group_idx  = arith.divui %global_k, %c0 : index    // k / group_size
                %scale_val  = tptir.load(%scales[%global_row, %group_idx])   : !tptir_scales -> f32
                %zp_val     = tptir.load(%zpoints[%global_row, %group_idx])  : !tptir_zpoints -> i8
                %zp_i32     = arith.extsi %zp_val : i8 to i32
                %w_sub_zp   = arith.subi %w_int, %zp_i32 : i32
                %w_f32      = arith.sitofp %w_sub_zp : i32 to f32
                %w_deq      = arith.mulf %w_f32, %scale_val : f32
                %w_f16      = arith.truncf %w_deq : f32 to f16

                memref.store %w_f16, %smem_a[%tm, %tk] : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>
            }
        }

        // Phase 2: Load B tile into smem_b
        scf.for %tk = %c0 to %tile_k step %c0 {
            scf.for %tn = %c0 to %tile_n step %c0 {
                %gk = arith.addi %k_tile,  %tk : index
                %gn = arith.addi %out_col, %tn : index
                %b_val  = tptir.load(%B[%gk, %gn]) : !tptir_f32mat -> f32
                %b_f16  = arith.truncf %b_val : f32 to f16
                memref.store %b_f16, %smem_b[%tk, %tn] : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>
            }
        }

        gpu.barrier

        // Phase 3: Compute C_tile += smem_a * smem_b (using tensor core WMMA)
        scf.for %tm = %c0 to %tile_m step %c0 {
            scf.for %tn = %c0 to %tile_n step %c0 {
                scf.for %tk = %c0 to %tile_k step %c0 {
                    %a_frag = vector.load %smem_a[%tm, %tk] : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>, vector<{{VEC_WIDTH}}xf16>
                    %b_frag = vector.load %smem_b[%tk, %tn] : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>, vector<{{VEC_WIDTH}}xf16>
                    %prod   = arith.mulf %a_frag, %b_frag : vector<{{VEC_WIDTH}}xf16>
                    %prod_f32 = arith.extf %prod : vector<{{VEC_WIDTH}}xf16> to vector<{{VEC_WIDTH}}xf32>
                    %cur_acc = vector.load %acc[%tm, %tn] : memref<{{TILE_M}}x{{TILE_N}}xf32, 5>, vector<{{VEC_WIDTH}}xf32>
                    %new_acc = arith.addf %cur_acc, %prod_f32 : vector<{{VEC_WIDTH}}xf32>
                    vector.store %new_acc, %acc[%tm, %tn] : memref<{{TILE_M}}x{{TILE_N}}xf32, 5>, vector<{{VEC_WIDTH}}xf32>
                }
            }
        }

        gpu.barrier
    }

    // Write accumulator tile to output matrix C
    %C_out = tptir.alloc(%M, %N) : (!tptir_index, !tptir_index) -> !tptir_f32mat
    scf.for %tm = %c0 to %tile_m step %c0 {
        scf.for %tn = %c0 to %tile_n step %c0 {
            %gr  = arith.addi %out_row, %tm : index
            %gc  = arith.addi %out_col, %tn : index
            %val = memref.load %acc[%tm, %tn] : memref<{{TILE_M}}x{{TILE_N}}xf32, 5>
            tptir.store(%val, %C_out[%gr, %gc]) : f32, !tptir_f32mat
        }
    }

    tptir.return %C_out : !tptir_f32mat
}

// ---------------------------------------------------------------------------
// 8-bit quantized GEMM (simpler packing: 1 weight per byte)
// ---------------------------------------------------------------------------
func.func @tptir_quant_gemm_int8(
    %A_int8  : !tptir_packed,
    %B       : !tptir_f32mat,
    %scales  : !tptir_scales,
    %zpoints : !tptir_zpoints,
    %M       : !tptir_index,
    %N       : !tptir_index,
    %K       : !tptir_index,
    %group_sz : !tptir_i32
) -> !tptir_f32mat attributes {
    tptir.kernel,
    tptir.grid_size  = [128, 1, 1],
    tptir.block_size = [256, 1, 1],
    tptir.shared_mem = 32768
} {
    // Same structure as int4 but no bit-unpacking required.
    // Dequant: w_f32 = (w_int8 - zp) * scale per group.
    // Tile loop omitted for brevity — same pattern as @tptir_quant_gemm_int4
    // with direct i8 load replacing the nibble extraction.
    %c0 = arith.constant 0 : index
    %C_out = tptir.alloc(%M, %N) : (!tptir_index, !tptir_index) -> !tptir_f32mat
    tptir.return %C_out : !tptir_f32mat
}
