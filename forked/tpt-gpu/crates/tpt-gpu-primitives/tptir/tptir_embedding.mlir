// tptir_embedding.mlir — Embedding Lookup Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes: output[b, s, :] = weight[indices[b, s], :]
// Input:  weight  [vocab_size, embed_dim] — embedding table
//         indices [batch, seq_len]        — integer token indices
// Output: output  [batch, seq_len, embed_dim]
// Strategy: 2-D grid (batch*seq_len rows × embed_dim columns);
//           each thread copies one f32 element from the weight row.
// Tunable placeholders: {{BLOCK_SIZE}}
// Defaults: BLOCK_SIZE=256

!tptir_tensor_f32  = type tensor<?x?xf32, 0>
!tptir_tensor_i32  = type tensor<?x?xi32, 0>
!tptir_tensor_3d   = type tensor<?x?x?xf32, 0>
!tptir_index       = type index

func.func @tptir_embedding_f32(
    %weight:    !tptir_tensor_f32,    // [vocab_size, embed_dim]
    %indices:   !tptir_tensor_i32,    // [batch, seq_len]
    %output:    !tptir_tensor_3d,     // [batch, seq_len, embed_dim]
    %batch:     !tptir_index,
    %seq_len:   !tptir_index,
    %embed_dim: !tptir_index
) -> !tptir_tensor_3d
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index

    // Each block handles one (batch, seq) position
    %total_tokens = arith.muli %batch, %seq_len : index
    %token_idx    = %block_id

    // Decompose token_idx → (b, s)
    %b = arith.divui %token_idx, %seq_len : index
    %s = arith.remui %token_idx, %seq_len : index

    // Fetch the vocabulary row index
    %vocab_idx_i32 = tensor.extract %indices[%b, %s] : !tptir_tensor_i32
    %vocab_idx     = arith.index_cast %vocab_idx_i32 : i32 to index

    // Copy embed_dim elements (thread-strided)
    scf.for %e = %thread_id to %embed_dim step %block_size {
        %v   = tensor.extract %weight[%vocab_idx, %e] : !tptir_tensor_f32
        tensor.insert %v into %output[%b, %s, %e] : !tptir_tensor_3d
    }

    return %output : !tptir_tensor_3d
}
