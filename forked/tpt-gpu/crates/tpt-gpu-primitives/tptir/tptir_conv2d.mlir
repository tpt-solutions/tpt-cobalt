// tptir_conv2d.mlir — 2D Convolution Kernel in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Computes Output = conv2d(Input, Filter, strides, padding)
// Input: NCHW format, Filter: C_out x C_in x K_h x K_w

!tptir_tensor_f32 = type tensor<?x?x?x?xf32, 0>
!tptir_index = type index
!tptir_f32 = type f32

func.func @tptir_conv2d_f32(
    %input: !tptir_tensor_f32,
    %filter: !tptir_tensor_f32,
    %stride_h: i32,
    %stride_w: i32,
    %padding_h: i32,
    %padding_w: i32,
    %dilation_h: i32,
    %dilation_w: i32,
    %groups: i32
) -> !tptir_tensor_f32
    attributes { tptir.kernel, tptir.grid_size = [32, 32, 1], tptir.block_size = [16, 16, 1], tptir.shared_mem = 8192 } {

    // Extract dimensions
    %N = tensor.dim %input, 0 : !tptir_tensor_f32
    %C_in = tensor.dim %input, 1 : !tptir_tensor_f32
    %H = tensor.dim %input, 2 : !tptir_tensor_f32
    %W = tensor.dim %input, 3 : !tptir_tensor_f32

    %C_out = tensor.dim %filter, 0 : !tptir_tensor_f32
    %K_h = tensor.dim %filter, 2 : !tptir_tensor_f32
    %K_w = tensor.dim %filter, 3 : !tptir_tensor_f32

    // Compute output dimensions
    %H_out = arith.subi %H, %K_h
    %H_out_p1 = arith.addi %H_out, %padding_h
    %H_out_final = arith.divui %H_out_p1, %stride_h

    %W_out = arith.subi %W, %K_w
    %W_out_p1 = arith.addi %W_out, %padding_w
    %W_out_final = arith.divui %W_out_p1, %stride_w

    // Shared memory for input tile (includes padding for filter overlap)
    %tile_h = arith.constant 16 : index
    %tile_w = arith.constant 16 : index
    %smem_input = memref.alloca() : memref<16x16xf32, 3>

    // Block indices map to output spatial position
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %thread_id_x = gpu.thread_id x
    %thread_id_y = gpu.thread_id y

    // Compute output position for this thread
    %out_h = arith.muli %block_id_y, %tile_h
    %out_w = arith.muli %block_id_x, %tile_w
    %local_h = arith.addi %out_h, %thread_id_y
    %local_w = arith.addi %out_w, %thread_id_x

    // Initialize accumulator
    %acc_init = arith.constant 0.0 : f32
    %acc = memref.alloca() : memref<f32, 2>

    // Channel loop (simplified: single channel per CTA)
    %c_in_idx = arith.constant 0 : index
    %c_out_idx = arith.constant 0 : index

    memref.store %acc_init, %acc[] : memref<f32, 2>

    // Filter loop
    %kh_start = arith.constant 0 : index
    %kh_end = %K_h
    %kw_start = arith.constant 0 : index
    %kw_end = %K_w

    scf.for %kh = %kh_start to %kh_end step 1 {
        scf.for %kw = %kw_start to %kw_end step 1 {
            // Compute input position
            %in_h = arith.muli %local_h, %stride_h
            %in_h_offset = arith.addi %in_h, %kh
            %in_h_padded = arith.subi %in_h_offset, %padding_h

            %in_w = arith.muli %local_w, %stride_w
            %in_w_offset = arith.addi %in_w, %kw
            %in_w_padded = arith.subi %in_w_offset, %padding_w>

            // Bounds check
            %h_valid = arith.cmpi sge, %in_h_padded, %H
            %w_valid = arith.cmpi sge, %in_w_padded, %W

            scf.if %h_valid {
                scf.if %w_valid {
                    // Load input value
                    %in_val = tensor.extract %input[%N, %c_in_idx, %in_h_padded, %in_w_padded] : !tptir_tensor_f32
                    // Load filter value
                    %filt_val = tensor.extract %filter[%c_out_idx, %c_in_idx, %kh, %kw] : !tptir_tensor_f32
                    // Multiply-accumulate
                    %product = arith.mulf %in_val, %filt_val : f32
                    %prev_acc = memref.load %acc[] : memref<f32, 2>
                    %new_acc = arith.addf %prev_acc, %product : f32
                    memref.store %new_acc, %acc[] : memref<f32, 2>
                }
            }
        }
    }

    // Store result
    %result = memref.load %acc[] : memref<f32, 2>
    tensor.insert %result into %input[%N, %c_out_idx, %local_h, %local_w] : !tptir_tensor_f32

    return %input : !tptir_tensor_f32
}