// tptir_groupnorm.mlir — Group Normalization Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
// Groups: C channels split into {{GROUPS}} groups of size C/{{GROUPS}}.
// Normalization is over C/G * H * W elements per (N, G) pair.
// Layout: input [N, C, S]  (S = H*W flattened)
// Strategy: one block per (batch, group); threads stride over group_channels*S.
// Tunable placeholders: {{GROUPS}}, {{BLOCK_SIZE}}, {{BLOCK_SIZE_HALF}}
// Defaults: GROUPS=32, BLOCK_SIZE=256, BLOCK_SIZE_HALF=128

!tptir_tensor_f32    = type tensor<?x?x?xf32, 0>   // [N, C, S]
!tptir_tensor_1d_f32 = type tensor<?xf32, 0>        // [C]
!tptir_index         = type index
!tptir_f32           = type f32

func.func @tptir_groupnorm_f32(
    %input:    !tptir_tensor_f32,      // [N, C, S]
    %gamma:    !tptir_tensor_1d_f32,   // [C]
    %beta:     !tptir_tensor_1d_f32,   // [C]
    %epsilon:  !tptir_f32,
    %batch:    !tptir_index,           // N
    %channels: !tptir_index,           // C
    %spatial:  !tptir_index            // S = H * W
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
    %num_groups = arith.constant {{GROUPS}} : index
    %zero_f     = arith.constant 0.0 : f32
    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index

    // Grid maps: block_id → (n, g) pair
    %n_idx  = arith.divsi %block_id, %num_groups : index
    %g_idx  = arith.remsi %block_id, %num_groups : index

    // Channels per group and the base channel index for this group
    %group_ch   = arith.divsi %channels, %num_groups : index
    %ch_base    = arith.muli %g_idx, %group_ch : index

    // Total elements to normalize over: group_ch * S
    %group_elem = arith.muli %group_ch, %spatial : index

    // Shared memory for partial sum and partial sum-of-sq
    %smem_sum = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    %smem_sq  = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
    gpu.barrier

    // --- Pass 1: accumulate sum and sum-of-sq over group_ch*S ---
    scf.for %elem = %thread_id to %group_elem step %block_size {
        %lc    = arith.divsi %elem, %spatial : index   // local channel within group
        %s_idx = arith.remsi %elem, %spatial : index   // spatial index
        %ch    = arith.addi %ch_base, %lc : index      // absolute channel
        %x     = tensor.extract %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
        %s_cur  = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq_cur = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        %xsq    = arith.mulf %x, %x : f32
        memref.store (arith.addf %s_cur,  %x   : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq_cur, %xsq : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
    }
    gpu.barrier

    // --- Tree reduction ---
    %half = arith.constant {{BLOCK_SIZE_HALF}} : index
    scf.for %stride = %half to %c1 step 2 {
        %partner = arith.addi %thread_id, %stride
        %s0  = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %s1  = memref.load %smem_sum[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq0 = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        %sq1 = memref.load %smem_sq[%partner]    : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %s0,  %s1  : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %sq0, %sq1 : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        gpu.barrier
    }

    // Compute mean and inv_std; broadcast via shared memory
    %smem_bc  = memref.alloca() : memref<2xf32, 3>
    %total_s  = memref.load %smem_sum[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
    %total_sq = memref.load %smem_sq[%c0]  : memref<{{BLOCK_SIZE}}xf32, 3>
    %ge_f32   = arith.index_cast %group_elem : index to f32
    %mean     = arith.divf %total_s, %ge_f32 : f32
    %e_xsq    = arith.divf %total_sq, %ge_f32 : f32
    %mean_sq  = arith.mulf %mean, %mean : f32
    %var      = arith.subf %e_xsq, %mean_sq : f32
    %var_eps  = arith.addf %var, %epsilon : f32
    %inv_std  = math.rsqrt %var_eps : f32
    memref.store %mean,    %smem_bc[%c0] : memref<2xf32, 3>
    memref.store %inv_std, %smem_bc[%c1] : memref<2xf32, 3>
    gpu.barrier

    %mean_bc    = memref.load %smem_bc[%c0] : memref<2xf32, 3>
    %inv_std_bc = memref.load %smem_bc[%c1] : memref<2xf32, 3>

    // --- Pass 2: normalize + per-channel affine transform ---
    scf.for %elem = %thread_id to %group_elem step %block_size {
        %lc      = arith.divsi %elem, %spatial : index
        %s_idx   = arith.remsi %elem, %spatial : index
        %ch      = arith.addi %ch_base, %lc : index
        %x       = tensor.extract %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
        %x_shift = arith.subf %x, %mean_bc : f32
        %x_hat   = arith.mulf %x_shift, %inv_std_bc : f32
        %gamma_v = tensor.extract %gamma[%ch] : !tptir_tensor_1d_f32
        %beta_v  = tensor.extract %beta[%ch]  : !tptir_tensor_1d_f32
        %scaled  = arith.mulf %x_hat, %gamma_v : f32
        %out     = arith.addf %scaled, %beta_v : f32
        tensor.insert %out into %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
    }

    return %input : !tptir_tensor_f32
}
