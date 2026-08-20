// tptir_batchnorm.mlir — Batch Normalization Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes (training): mean/var over N*S for each channel C,
//                      updates running stats, normalizes + affine.
// Computes (inference): normalize using running_mean/running_var.
// Layout: input [N, C, S]  (S = H*W for spatial; S=1 for 1-D inputs)
// Strategy: one block per channel; threads stride over N*S to accumulate.
// Tunable placeholders: {{BLOCK_SIZE}}, {{BLOCK_SIZE_HALF}}, {{MOMENTUM}}
// Defaults: BLOCK_SIZE=256, BLOCK_SIZE_HALF=128, MOMENTUM=0.1

!tptir_tensor_f32    = type tensor<?x?x?xf32, 0>   // [N, C, S]
!tptir_tensor_1d_f32 = type tensor<?xf32, 0>        // [C]
!tptir_index         = type index
!tptir_f32           = type f32
!tptir_i1            = type i1

func.func @tptir_batchnorm_f32(
    %input:       !tptir_tensor_f32,     // [N, C, S]
    %gamma:       !tptir_tensor_1d_f32,  // [C]
    %beta:        !tptir_tensor_1d_f32,  // [C]
    %running_mean: !tptir_tensor_1d_f32, // [C]  — updated when training
    %running_var:  !tptir_tensor_1d_f32, // [C]  — updated when training
    %epsilon:     !tptir_f32,
    %momentum:    !tptir_f32,            // override via {{MOMENTUM}} placeholder or runtime arg
    %batch:       !tptir_index,          // N
    %channels:    !tptir_index,          // C
    %spatial:     !tptir_index,          // S = H * W
    %is_training: !tptir_i1
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
    %one_f      = arith.constant 1.0 : f32
    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index

    // One block per channel
    %ch = %block_id

    // Shared memory for partial sum and partial sum-of-sq across the N*S axis
    %smem_sum = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    %smem_sq  = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f, %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
    gpu.barrier

    // Total elements over which mean/var are computed
    %ns = arith.muli %batch, %spatial : index

    // --- Pass 1 (training): accumulate sum and sum-of-sq over N*S ---
    scf.if %is_training {
        scf.for %ns_idx = %thread_id to %ns step %block_size {
            %n_idx = arith.divsi %ns_idx, %spatial : index
            %s_idx = arith.remsi %ns_idx, %spatial : index
            %x      = tensor.extract %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
            %s_cur  = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
            %sq_cur = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
            %xsq    = arith.mulf %x, %x : f32
            memref.store (arith.addf %s_cur,  %x   : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
            memref.store (arith.addf %sq_cur, %xsq : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
        }
        gpu.barrier

        // Tree reduction
        %half = arith.constant {{BLOCK_SIZE_HALF}} : index
        scf.for %stride = %half to %c1 step 2 {
            %partner = arith.addi %thread_id, %stride
            %s0  = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
            %s1  = memref.load %smem_sum[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
            %sq0 = memref.load %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
            %sq1 = memref.load %smem_sq[%partner]    : memref<{{BLOCK_SIZE}}xf32, 3>
            memref.store (arith.addf %s0, %s1   : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
            memref.store (arith.addf %sq0, %sq1 : f32), %smem_sq[%thread_id]  : memref<{{BLOCK_SIZE}}xf32, 3>
            gpu.barrier
        }
    }

    // Broadcast mean/inv_std to all threads
    %smem_bc = memref.alloca() : memref<2xf32, 3>
    scf.if %is_training {
        %total_s  = memref.load %smem_sum[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
        %total_sq = memref.load %smem_sq[%c0]  : memref<{{BLOCK_SIZE}}xf32, 3>
        %ns_f32   = arith.index_cast %ns : index to f32
        %mean_t   = arith.divf %total_s, %ns_f32 : f32
        %e_xsq    = arith.divf %total_sq, %ns_f32 : f32
        %mean_sq  = arith.mulf %mean_t, %mean_t : f32
        %var_t    = arith.subf %e_xsq, %mean_sq : f32
        %var_eps  = arith.addf %var_t, %epsilon : f32
        %inv_std  = math.rsqrt %var_eps : f32

        // Update running stats: running = (1 - momentum) * running + momentum * batch_stat
        %mom      = arith.constant {{MOMENTUM}} : f32
        %one_mom  = arith.subf %one_f, %mom : f32
        %r_mean   = tensor.extract %running_mean[%ch] : !tptir_tensor_1d_f32
        %r_var    = tensor.extract %running_var[%ch]  : !tptir_tensor_1d_f32
        %new_mean = arith.addf (arith.mulf %one_mom, %r_mean : f32),
                               (arith.mulf %mom, %mean_t : f32) : f32
        %new_var  = arith.addf (arith.mulf %one_mom, %r_var : f32),
                               (arith.mulf %mom, %var_t  : f32) : f32
        tensor.insert %new_mean into %running_mean[%ch] : !tptir_tensor_1d_f32
        tensor.insert %new_var  into %running_var[%ch]  : !tptir_tensor_1d_f32

        memref.store %mean_t,  %smem_bc[%c0] : memref<2xf32, 3>
        memref.store %inv_std, %smem_bc[%c1] : memref<2xf32, 3>
    } else {
        // Inference: use stored running statistics
        %r_mean   = tensor.extract %running_mean[%ch] : !tptir_tensor_1d_f32
        %r_var    = tensor.extract %running_var[%ch]  : !tptir_tensor_1d_f32
        %var_eps  = arith.addf %r_var, %epsilon : f32
        %inv_std  = math.rsqrt %var_eps : f32
        memref.store %r_mean,  %smem_bc[%c0] : memref<2xf32, 3>
        memref.store %inv_std, %smem_bc[%c1] : memref<2xf32, 3>
    }
    gpu.barrier

    %mean_bc    = memref.load %smem_bc[%c0] : memref<2xf32, 3>
    %inv_std_bc = memref.load %smem_bc[%c1] : memref<2xf32, 3>
    %gamma_ch   = tensor.extract %gamma[%ch] : !tptir_tensor_1d_f32
    %beta_ch    = tensor.extract %beta[%ch]  : !tptir_tensor_1d_f32

    // --- Pass 2: normalize + affine (all threads stride over N*S) ---
    scf.for %ns_idx = %thread_id to %ns step %block_size {
        %n_idx   = arith.divsi %ns_idx, %spatial : index
        %s_idx   = arith.remsi %ns_idx, %spatial : index
        %x       = tensor.extract %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
        %x_shift = arith.subf %x, %mean_bc : f32
        %x_hat   = arith.mulf %x_shift, %inv_std_bc : f32
        %scaled  = arith.mulf %x_hat, %gamma_ch : f32
        %out     = arith.addf %scaled, %beta_ch : f32
        tensor.insert %out into %input[%n_idx, %ch, %s_idx] : !tptir_tensor_f32
    }

    return %input : !tptir_tensor_f32
}
