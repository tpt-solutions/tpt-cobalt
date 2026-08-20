// tptir_attention.mlir — Scaled Dot-Product Attention Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
// Strategy: Flash Attention-style tiling with online softmax

!tptir_tensor_f32 = type tensor<?x?xf32, 0>
!tptir_index = type index
!tptir_f32 = type f32
!tptir_mask = type tensor<?x?xf32, 0>

func.func @tptir_attention_f32(
    %Q: !tptir_tensor_f32,
    %K: !tptir_tensor_f32,
    %V: !tptir_tensor_f32,
    %mask: !tptir_mask,
    %scale: !tptir_f32,
    %seq_len: !tptir_index,
    %d_k: !tptir_index,
    %d_v: !tptir_index
) -> (!tptir_tensor_f32, !tptir_tensor_f32)
    attributes { tptir.kernel, tptir.grid_size = [32, 1, 1], tptir.block_size = [256, 1, 1], tptir.shared_mem = 16384 } {

    %tile_kv = arith.constant 64 : index
    %block_id = gpu.block_id x
    %thread_id = gpu.thread_id x
    %q_start = arith.muli %block_id, %tile_kv

    %smem_k = memref.alloca() : memref<64x128xf32, 3>
    %smem_v = memref.alloca() : memref<64x128xf32, 3>
    %m_old = memref.alloca() : memref<64xf32, 2>
    %l_old = memref.alloca() : memref<64xf32, 2>
    %acc_o = memref.alloca() : memref<64x128xf32, 2>

    %neg_inf = arith.constant dense<-1.0E30> : vector<128xf32>
    %zero = arith.constant dense<0.0> : vector<128xf32>
    %zero_scalar = arith.constant 0.0 : f32

    scf.for %i = %q_start to %seq_len step 1 {
        %row_idx = arith.subi %i, %q_start
        memref.store %zero_scalar, %m_old[%row_idx] : memref<64xf32, 2>
        memref.store %zero_scalar, %l_old[%row_idx] : memref<64xf32, 2>
    }

    scf.for %i = %q_start to %seq_len step 1 {
        %row_idx = arith.subi %i, %q_start
        vector.store %zero, %acc_o[%row_idx, %thread_id] : memref<64x128xf32, 2>, vector<128xf32>
    }

    %kv_start = arith.constant 0 : index
    %kv_end = %seq_len
    %kv_step = %tile_kv

    scf.for %kv_tile = %kv_start to %kv_end step %kv_step {
        %k_row = arith.addi %kv_tile, %thread_id
        %k_val = vector.load %K[%k_row, %thread_id] : !tptir_tensor_f32, vector<128xf32>
        vector.store %k_val, %smem_k[%thread_id, %thread_id] : memref<64x128xf32, 3>, vector<128xf32>
        %v_val = vector.load %V[%k_row, %thread_id] : !tptir_tensor_f32, vector<128xf32>
        vector.store %v_val, %smem_v[%thread_id, %thread_id] : memref<64x128xf32, 3>, vector<128xf32>
        gpu.barrier

        scf.for %q_idx = %q_start to %seq_len step 1 {
            %q_val = vector.load %Q[%q_idx, %thread_id] : !tptir_tensor_f32, vector<128xf32>
            %k_col = vector.load %smem_k[%thread_id, %thread_id] : memref<64x128xf32, 3>, vector<128xf32>
            %dot = arith.mulf %q_val, %k_col : vector<128xf32>
            %s_val = arith.divf %dot, %scale : vector<128xf32>
            %mask_val = vector.load %mask[%q_idx, %kv_tile] : !tptir_tensor_f32, vector<128xf32>
            %s_masked = arith.addf %s_val, %mask_val : vector<128xf32>
            %m_prev = memref.load %m_old[%thread_id] : memref<64xf32, 2>
            %l_prev = memref.load %l_old[%thread_id] : memref<64xf32, 2>
            %m_new = arith.maxf %m_prev, %s_masked : vector<128xf32>
            %exp_diff = arith.subf %m_prev, %m_new : vector<128xf32>
            %exp_val = math.exp2 %exp_diff : vector<128xf32>
            %l_new = arith.addf %l_prev, %exp_val : vector<128xf32>
            memref.store %m_new, %m_old[%thread_id] : memref<64xf32, 2>
            memref.store %l_new, %l_old[%thread_id] : memref<64xf32, 2>
            %rescale = arith.divf %l_prev, %l_new : vector<128xf32>
            %acc_val = vector.load %acc_o[%thread_id, %thread_id] : memref<64x128xf32, 2>, vector<128xf32>
            %acc_scaled = arith.mulf %acc_val, %rescale : vector<128xf32>
            %p_val = math.exp2 %s_masked : vector<128xf32>
            %v_col = vector.load %smem_v[%thread_id, %thread_id] : memref<64x128xf32, 3>, vector<128xf32>
            %pv = arith.mulf %p_val, %v_col : vector<128xf32>
            %acc_new = arith.addf %acc_scaled, %pv : vector<128xf32>
            vector.store %acc_new, %acc_o[%thread_id, %thread_id] : memref<64x128xf32, 2>, vector<128xf32>
            gpu.barrier
        }
        gpu.barrier
    }

    scf.for %i = %q_start to %seq_len step 1 {
        %l_final = memref.load %l_old[%thread_id] : memref<64xf32, 2>
        %acc_val = vector.load %acc_o[%thread_id, %thread_id] : memref<64x128xf32, 2>, vector<128xf32>
        %normalized = arith.divf %acc_val, %l_final : vector<128xf32>
        tensor.insert %normalized into %V[%i, %thread_id] : !tptir_tensor_f32
    }

    return %V, %V : !tptir_tensor_f32, !tptir_tensor_f32
}