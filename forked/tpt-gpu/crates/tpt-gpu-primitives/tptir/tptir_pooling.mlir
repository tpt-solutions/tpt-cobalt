// tptir_pooling.mlir — 2D Pooling Kernels in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Implements MaxPool2D and AvgPool2D over [N, C, H, W] tensors.
// Strategy: one thread per output element; strided loop over the pooling window.
// Tunable placeholders: {{BLOCK_SIZE}}
// Defaults: BLOCK_SIZE=256

!tptir_tensor_4d = type tensor<?x?x?x?xf32, 0>
!tptir_index     = type index
!tptir_i32       = type i32
!tptir_f32       = type f32

// ─── MaxPool2D ────────────────────────────────────────────────────────────────
func.func @tptir_maxpool2d_f32(
    %input:     !tptir_tensor_4d,   // [N, C, H_in, W_in]
    %output:    !tptir_tensor_4d,   // [N, C, H_out, W_out]
    %n:         !tptir_index,
    %c:         !tptir_index,
    %h_out:     !tptir_index,
    %w_out:     !tptir_index,
    %kh:        !tptir_i32,         // kernel height
    %kw:        !tptir_i32,         // kernel width
    %stride_h:  !tptir_i32,
    %stride_w:  !tptir_i32,
    %pad_h:     !tptir_i32,
    %pad_w:     !tptir_i32
) -> !tptir_tensor_4d
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %neg_inf    = arith.constant 0xFF800000 : f32

    // Linearize [N, C, H_out, W_out] → flat thread index
    %total     = arith.muli (arith.muli (arith.muli %n, %c : index), %h_out : index), %w_out : index
    %flat_base = arith.muli %block_id, %block_size : index

    scf.for %i = %thread_id to %block_size step %block_size {
        %flat = arith.addi %flat_base, %i : index
        %in_bounds = arith.cmpi slt, %flat, %total : index
        scf.if %in_bounds {
            // Decompose flat → (ni, ci, oh, ow)
            %ncw    = arith.muli %n, (arith.muli %c, %w_out : index) : index
            %nch    = arith.muli %n, (arith.muli %c, %h_out : index) : index
            %ni     = arith.divui %flat, (arith.muli (arith.muli %c, %h_out : index), %w_out : index) : index
            %rem0   = arith.remui %flat, (arith.muli (arith.muli %c, %h_out : index), %w_out : index) : index
            %ci     = arith.divui %rem0, (arith.muli %h_out, %w_out : index) : index
            %rem1   = arith.remui %rem0, (arith.muli %h_out, %w_out : index) : index
            %oh     = arith.divui %rem1, %w_out : index
            %ow     = arith.remui %rem1, %w_out : index

            %sh     = arith.index_cast %stride_h : i32 to index
            %sw     = arith.index_cast %stride_w : i32 to index
            %ph     = arith.index_cast %pad_h    : i32 to index
            %pw     = arith.index_cast %pad_w    : i32 to index
            %khi    = arith.index_cast %kh       : i32 to index
            %kwi    = arith.index_cast %kw       : i32 to index

            %h_start = arith.subi (arith.muli %oh, %sh : index), %ph : index
            %w_start = arith.subi (arith.muli %ow, %sw : index), %pw : index

            %max_val = scf.for %ky = (arith.constant 0 : index) to %khi step (arith.constant 1 : index)
                           iter_args(%acc = %neg_inf) -> f32 {
                %ky_val = scf.for %kx = (arith.constant 0 : index) to %kwi step (arith.constant 1 : index)
                              iter_args(%acc2 = %acc) -> f32 {
                    %ih = arith.addi %h_start, %ky : index
                    %iw = arith.addi %w_start, %kx : index
                    %v  = tensor.extract %input[%ni, %ci, %ih, %iw] : !tptir_tensor_4d
                    %m  = arith.maximumf %acc2, %v : f32
                    scf.yield %m : f32
                }
                scf.yield %ky_val : f32
            }
            tensor.insert %max_val into %output[%ni, %ci, %oh, %ow] : !tptir_tensor_4d
        }
    }

    return %output : !tptir_tensor_4d
}

// ─── AvgPool2D ────────────────────────────────────────────────────────────────
func.func @tptir_avgpool2d_f32(
    %input:     !tptir_tensor_4d,   // [N, C, H_in, W_in]
    %output:    !tptir_tensor_4d,   // [N, C, H_out, W_out]
    %n:         !tptir_index,
    %c:         !tptir_index,
    %h_out:     !tptir_index,
    %w_out:     !tptir_index,
    %kh:        !tptir_i32,
    %kw:        !tptir_i32,
    %stride_h:  !tptir_i32,
    %stride_w:  !tptir_i32,
    %pad_h:     !tptir_i32,
    %pad_w:     !tptir_i32
) -> !tptir_tensor_4d
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %zero_f     = arith.constant 0.0 : f32

    %total     = arith.muli (arith.muli (arith.muli %n, %c : index), %h_out : index), %w_out : index
    %flat_base = arith.muli %block_id, %block_size : index

    scf.for %i = %thread_id to %block_size step %block_size {
        %flat = arith.addi %flat_base, %i : index
        %in_bounds = arith.cmpi slt, %flat, %total : index
        scf.if %in_bounds {
            %ni     = arith.divui %flat, (arith.muli (arith.muli %c, %h_out : index), %w_out : index) : index
            %rem0   = arith.remui %flat, (arith.muli (arith.muli %c, %h_out : index), %w_out : index) : index
            %ci     = arith.divui %rem0, (arith.muli %h_out, %w_out : index) : index
            %rem1   = arith.remui %rem0, (arith.muli %h_out, %w_out : index) : index
            %oh     = arith.divui %rem1, %w_out : index
            %ow     = arith.remui %rem1, %w_out : index

            %sh     = arith.index_cast %stride_h : i32 to index
            %sw     = arith.index_cast %stride_w : i32 to index
            %ph     = arith.index_cast %pad_h    : i32 to index
            %pw     = arith.index_cast %pad_w    : i32 to index
            %khi    = arith.index_cast %kh       : i32 to index
            %kwi    = arith.index_cast %kw       : i32 to index

            %h_start = arith.subi (arith.muli %oh, %sh : index), %ph : index
            %w_start = arith.subi (arith.muli %ow, %sw : index), %pw : index

            %window_area_i = arith.muli %khi, %kwi : index
            %window_area   = arith.uitofp %window_area_i : index to f32

            %sum = scf.for %ky = (arith.constant 0 : index) to %khi step (arith.constant 1 : index)
                       iter_args(%acc = %zero_f) -> f32 {
                %ky_val = scf.for %kx = (arith.constant 0 : index) to %kwi step (arith.constant 1 : index)
                              iter_args(%acc2 = %acc) -> f32 {
                    %ih = arith.addi %h_start, %ky : index
                    %iw = arith.addi %w_start, %kx : index
                    %v  = tensor.extract %input[%ni, %ci, %ih, %iw] : !tptir_tensor_4d
                    %s  = arith.addf %acc2, %v : f32
                    scf.yield %s : f32
                }
                scf.yield %ky_val : f32
            }
            %avg = arith.divf %sum, %window_area : f32
            tensor.insert %avg into %output[%ni, %ci, %oh, %ow] : !tptir_tensor_4d
        }
    }

    return %output : !tptir_tensor_4d
}
