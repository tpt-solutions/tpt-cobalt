// tptir_gemm.mlir — GEMM Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes C = alpha * A * B + beta * C
// Strategy: Tiled GEMM with shared memory ({{TILE_M}}x{{TILE_N}}x{{TILE_K}} tiles)
// Tunable placeholders: {{TILE_M}}, {{TILE_N}}, {{TILE_K}}, {{VEC_WIDTH}}, {{UNROLL}}
// Defaults: TILE_M=64, TILE_N=64, TILE_K=16, VEC_WIDTH=4, UNROLL=2

!tptir_tensor_f32 = type tensor<?x?xf32, 0>
!tptir_tensor_f16 = type tensor<?x?xf16, 0>
!tptir_index = type index
!tptir_f32 = type f32

func.func @tptir_gemm_f16_f32(
    %A: !tptir_tensor_f16,
    %B: !tptir_tensor_f16,
    %C: !tptir_tensor_f32,
    %alpha: !tptir_f32,
    %beta: !tptir_f32,
    %M: !tptir_index,
    %N: !tptir_index,
    %K: !tptir_index
) -> !tptir_tensor_f32 attributes { tptir.kernel, tptir.grid_size = [128, 1, 1], tptir.block_size = [256, 1, 1] } {
    %tile_m = arith.constant {{TILE_M}} : index
    %tile_n = arith.constant {{TILE_N}} : index
    %tile_k = arith.constant {{TILE_K}} : index
    %smem_a = memref.alloca() : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>
    %smem_b = memref.alloca() : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %thread_id_x = gpu.thread_id x
    %thread_id_y = gpu.thread_id y
    %a_row_base = arith.muli %block_id_y, %tile_m
    %b_col_base = arith.muli %block_id_x, %tile_n
    %acc_init = arith.constant dense<0.0> : vector<{{VEC_WIDTH}}xf32>
    %acc = vector.splat %acc_init : vector<{{VEC_WIDTH}}xf32>
    %k_start = arith.constant 0 : index
    %k_end = %K
    %k_step = %tile_k
    scf.for %k_tile = %k_start to %k_end step %k_step {
        %a_row = arith.addi %a_row_base, %thread_id_y
        %a_col = arith.addi %k_tile, %thread_id_x
        %a_val = tensor.extract %A[%a_row, %a_col] : !tptir_tensor_f16
        memref.store %a_val, %smem_a[%thread_id_y, %thread_id_x] : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>
        %b_row = arith.addi %k_tile, %thread_id_y
        %b_col = arith.addi %b_col_base, %thread_id_x
        %b_val = tensor.extract %B[%b_row, %b_col] : !tptir_tensor_f16
        memref.store %b_val, %smem_b[%thread_id_y, %thread_id_x] : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>
        gpu.barrier
        scf.for %i = %k_tile to %k_end step 1 {
            %a_frag = vector.load %smem_a[%thread_id_y, %i] : memref<{{TILE_M}}x{{TILE_K}}xf16, 3>, vector<{{VEC_WIDTH}}xf16>
            %b_frag = vector.load %smem_b[%i, %thread_id_x] : memref<{{TILE_K}}x{{TILE_N}}xf16, 3>, vector<{{VEC_WIDTH}}xf16>
            %a_f32 = arith.extf %a_frag : vector<{{VEC_WIDTH}}xf16> to vector<{{VEC_WIDTH}}xf32>
            %b_f32 = arith.extf %b_frag : vector<{{VEC_WIDTH}}xf16> to vector<{{VEC_WIDTH}}xf32>
            %prod = arith.mulf %a_f32, %b_f32 : vector<{{VEC_WIDTH}}xf32>
            %acc = arith.addf %acc, %prod : vector<{{VEC_WIDTH}}xf32>
        }
        gpu.barrier
    }
    %alpha_vec = vector.splat %alpha : vector<{{VEC_WIDTH}}xf32>
    %acc = arith.mulf %acc, %alpha_vec : vector<{{VEC_WIDTH}}xf32>
    %c_row = arith.addi %a_row_base, %thread_id_y
    %c_col = arith.addi %b_col_base, %thread_id_x
    %c_val = tensor.extract %C[%c_row, %c_col] : !tptir_tensor_f32
    %beta_vec = vector.splat %beta : vector<{{VEC_WIDTH}}xf32>
    %c_scaled = arith.mulf %c_val, %beta_vec : vector<{{VEC_WIDTH}}xf32>
    %result = arith.addf %acc, %c_scaled : vector<{{VEC_WIDTH}}xf32>
    tensor.insert %result into %C[%c_row, %c_col] : !tptir_tensor_f32
    return %C : !tptir_tensor_f32
}
