// tptir_rmsnorm.mlir — RMS Normalization Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes: y = x / rms(x) * gamma
//           where rms(x) = sqrt(mean(x^2) + epsilon)
// No mean subtraction (unlike LayerNorm); used in LLaMA, Mistral, Qwen, etc.
// Strategy: single-pass shared-memory reduction over x^2, then normalize+scale.
// Tunable placeholders: {{BLOCK_SIZE}}, {{BLOCK_SIZE_HALF}}, {{VEC_WIDTH}}
// Defaults: BLOCK_SIZE=256, BLOCK_SIZE_HALF=128, VEC_WIDTH=4

!tptir_tensor_f32    = type tensor<?x?xf32, 0>
!tptir_tensor_1d_f32 = type tensor<?xf32, 0>
!tptir_index         = type index
!tptir_f32           = type f32

func.func @tptir_rmsnorm_f32(
    %input:     !tptir_tensor_f32,       // [batch, norm_size]
    %gamma:     !tptir_tensor_1d_f32,    // [norm_size]
    %epsilon:   !tptir_f32,
    %batch:     !tptir_index,
    %norm_size: !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [256, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1],
        tptir.shared_mem = 2048
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %zero_f     = arith.constant 0.0 : f32
    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index

    // One block per row
    %row = %block_id

    // Shared memory: partial sum-of-squares per thread
    %smem_sq = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sq[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    gpu.barrier

    // --- Pass 1: accumulate x^2 partial sums (strided over norm_size) ---
    scf.for %col = %thread_id to %norm_size step %block_size {
        %x      = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %xsq    = arith.mulf %x, %x : f32
        %sq_cur = memref.load %smem_sq[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq_cur, %xsq : f32), %smem_sq[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    }
    gpu.barrier

    // --- Tree reduction: fold BLOCK_SIZE partial sums → lane 0 ---
    %half = arith.constant {{BLOCK_SIZE_HALF}} : index
    scf.for %stride = %half to %c1 step 2 {
        %partner = arith.addi %thread_id, %stride
        %sq0     = memref.load %smem_sq[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq1     = memref.load %smem_sq[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq0, %sq1 : f32), %smem_sq[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        gpu.barrier
    }

    // Compute inv_rms and broadcast
    %smem_bc  = memref.alloca() : memref<1xf32, 3>
    %total_sq = memref.load %smem_sq[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
    %n_f32    = arith.index_cast %norm_size : index to f32
    %mean_sq  = arith.divf %total_sq, %n_f32 : f32
    %ms_eps   = arith.addf %mean_sq, %epsilon : f32
    %inv_rms  = math.rsqrt %ms_eps : f32
    memref.store %inv_rms, %smem_bc[%c0] : memref<1xf32, 3>
    gpu.barrier

    %inv_rms_bc = memref.load %smem_bc[%c0] : memref<1xf32, 3>

    // --- Pass 2: normalize + scale (vectorized, VEC_WIDTH elements/thread/step) ---
    scf.for %col = %thread_id to %norm_size step %block_size {
        %x       = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %x_norm  = arith.mulf %x, %inv_rms_bc : f32
        %gamma_v = tensor.extract %gamma[%col] : !tptir_tensor_1d_f32
        %out     = arith.mulf %x_norm, %gamma_v : f32
        tensor.insert %out into %input[%row, %col] : !tptir_tensor_f32
    }

    return %input : !tptir_tensor_f32
}
