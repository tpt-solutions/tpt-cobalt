// tptir_softmax.mlir — Softmax Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes numerically stable softmax along the last dimension:
//   max_x = max(x)
//   y_i   = exp(x_i - max_x) / sum(exp(x_j - max_x))
// Strategy: three-pass shared-memory reduction (max → exp+sum → normalize).
// Tunable placeholders: {{BLOCK_SIZE}}, {{BLOCK_SIZE_HALF}}
// Defaults: BLOCK_SIZE=256, BLOCK_SIZE_HALF=128

!tptir_tensor_f32 = type tensor<?x?xf32, 0>
!tptir_index      = type index
!tptir_f32        = type f32

func.func @tptir_softmax_f32(
    %input:    !tptir_tensor_f32,    // [batch, dim_size] — normalized over dim_size
    %batch:    !tptir_index,
    %dim_size: !tptir_index
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
    %neg_inf    = arith.constant 0xFF800000 : f32   // -inf as bit pattern
    %zero_f     = arith.constant 0.0 : f32
    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index

    %row = %block_id

    %smem_max = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    %smem_sum = memref.alloca() : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %neg_inf, %smem_max[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %zero_f,  %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    gpu.barrier

    // --- Pass 1: find per-thread partial max ---
    scf.for %col = %thread_id to %dim_size step %block_size {
        %x       = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %cur_max = memref.load %smem_max[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %new_max = arith.maximumf %x, %cur_max : f32
        memref.store %new_max, %smem_max[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    }
    gpu.barrier

    // --- Tree reduction: global max ---
    %half = arith.constant {{BLOCK_SIZE_HALF}} : index
    scf.for %stride = %half to %c1 step 2 {
        %partner  = arith.addi %thread_id, %stride
        %m0       = memref.load %smem_max[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %m1       = memref.load %smem_max[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.maximumf %m0, %m1 : f32), %smem_max[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        gpu.barrier
    }

    %smem_bc = memref.alloca() : memref<2xf32, 3>
    %row_max = memref.load %smem_max[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
    memref.store %row_max, %smem_bc[%c0] : memref<2xf32, 3>
    gpu.barrier
    %max_bc  = memref.load %smem_bc[%c0] : memref<2xf32, 3>

    // --- Pass 2: compute exp(x - max) and accumulate sum ---
    scf.for %col = %thread_id to %dim_size step %block_size {
        %x       = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %shifted = arith.subf %x, %max_bc : f32
        %ex      = math.exp %shifted : f32
        tensor.insert %ex into %input[%row, %col] : !tptir_tensor_f32
        %s_cur   = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %s_cur, %ex : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
    }
    gpu.barrier

    // --- Tree reduction: global exp-sum ---
    scf.for %stride = %half to %c1 step 2 {
        %partner = arith.addi %thread_id, %stride
        %s0      = memref.load %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        %s1      = memref.load %smem_sum[%partner]   : memref<{{BLOCK_SIZE}}xf32, 3>
        memref.store (arith.addf %s0, %s1 : f32), %smem_sum[%thread_id] : memref<{{BLOCK_SIZE}}xf32, 3>
        gpu.barrier
    }

    %total_sum  = memref.load %smem_sum[%c0] : memref<{{BLOCK_SIZE}}xf32, 3>
    %inv_sum    = arith.divf (arith.constant 1.0 : f32), %total_sum : f32
    memref.store %inv_sum, %smem_bc[%c1] : memref<2xf32, 3>
    gpu.barrier
    %inv_sum_bc = memref.load %smem_bc[%c1] : memref<2xf32, 3>

    // --- Pass 3: normalize ---
    scf.for %col = %thread_id to %dim_size step %block_size {
        %ex  = tensor.extract %input[%row, %col] : !tptir_tensor_f32
        %out = arith.mulf %ex, %inv_sum_bc : f32
        tensor.insert %out into %input[%row, %col] : !tptir_tensor_f32
    }

    return %input : !tptir_tensor_f32
}
