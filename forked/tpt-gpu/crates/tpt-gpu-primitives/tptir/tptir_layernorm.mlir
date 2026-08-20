// tptir_layernorm.mlir — Layer Normalization Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes: y = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
// Normalized over the innermost axis (norm_size elements per row).
// Strategy: two-pass shared-memory reduction (sum + sum-of-squares → mean + inv_std),
//           then a vectorized normalize+scale+shift pass.
// Tunable placeholders: {{BLOCK_SIZE}}, {{BLOCK_SIZE_HALF}}, {{VEC_WIDTH}}
// Defaults: BLOCK_SIZE=256, BLOCK_SIZE_HALF=128, VEC_WIDTH=4

!tptir_tensor_f32    = type tensor<?x?xf32, 0>
!tptir_tensor_1d_f32 = type tensor<?xf32, 0>
!tptir_index         = type index
!tptir_f32           = type f32

func.func @tptir_layernorm_f32(
    %input:     !tptir_tensor_f32,       // [batch, norm_size] — overwritten in-place
    %gamma:     !tptir_tensor_1d_f32,    // [norm_size]
    %beta:      !tptir_tensor_1d_f32,    // [norm_size]
    %epsilon:   !tptir_f32,
    %batch:     !tptir_index,
    %norm_size: !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [256, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1],
        tptir.shared_mem = 4096
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %zero_f     = arith.constant 0.0 : f32
    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index

    // One block per row
    %row = %block_id

    // Shared memory: one f32 per thread for partial sum and partial sum-of-sq
    %smem_sum = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    %smem_sq  = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
    gpu.barrier

    // --- Pass 1: accumulate partial sums (strided over norm_size) ---
    scf.for %col = %thread_id to %norm_size step %block_size {
        %x      = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %s_cur  = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq_cur = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        %xsq    = arith.mulf %x, %x : f32
        memref.store (arith.addf %s_cur,  %x   : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq_cur, %xsq : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
    }
    gpu.barrier

    // --- Tree reduction: fold BLOCK_SIZE partial sums → lane 0 ---
    %half = arith.constant {{BLOCK_SIZE_HALF}} : index
    scf.for %stride = %half to %c1 step 2 {
        %partner  = arith.addi %thread_id, %stride
        %s0       = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %s1       = memref.load %smem_sum[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq0      = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq1      = memref.load %smem_sq[%partner]    : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %s0,  %s1  : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq0, %sq1 : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        gpu.barrier
    }

    // Compute mean and inv_std in a shared broadcast slot
    %smem_bc  = memref.alloca() : memref<2xf32, 3>
    %total_s  = memref.load %smem_sum[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
    %total_sq = memref.load %smem_sq[%c0]  : memref<{{BLOCK_SIZE}}xf32, 3>
    %n_f32    = arith.index_cast %norm_size : index to f32
    %mean     = arith.divf %total_s, %n_f32 : f32
    %e_xsq    = arith.divf %total_sq, %n_f32 : f32
    %mean_sq  = arith.mulf %mean, %mean : f32
    %var      = arith.subf %e_xsq, %mean_sq : f32
    %var_eps  = arith.addf %var, %epsilon : f32
    %inv_std  = math.rsqrt %var_eps : f32
    memref.store %mean,    %smem_bc[%c0] : memref<2xf32, 3>
    memref.store %inv_std, %smem_bc[%c1] : memref<2xf32, 3>
    gpu.barrier

    %mean_bc    = memref.load %smem_bc[%c0] : memref<2xf32, 3>
    %inv_std_bc = memref.load %smem_bc[%c1] : memref<2xf32, 3>

    // --- Pass 2: normalize + affine transform (vectorized, VEC_WIDTH elements/thread/step) ---
    scf.for %col = %thread_id to %norm_size step %block_size {
        %x       = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %x_shift = arith.subf %x, %mean_bc : f32
        %x_hat   = arith.mulf %x_shift, %inv_std_bc : f32
        %gamma_v = tensor.extract %gamma[%col] : !tptir_tensor_1d_f32
        %beta_v  = tensor.extract %beta[%col]  : !tptir_tensor_1d_f32
        %scaled  = arith.mulf %x_hat, %gamma_v : f32
        %out     = arith.addf %scaled, %beta_v : f32
        tensor.insert %out into %input[%row, %col] : !tptir_tensor_f32
    }

    return %input : !tptir_tensor_f32
}
