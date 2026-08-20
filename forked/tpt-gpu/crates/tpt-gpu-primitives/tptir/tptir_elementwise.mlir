// tptir_elementwise.mlir — Elementwise Activation Kernels in TPTIR
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// Implements: ReLU, GELU (tanh approximation), SiLU/Swish, Sigmoid
// Strategy: flat 1-D view; each thread handles VEC_WIDTH elements.
// Tunable placeholders: {{BLOCK_SIZE}}, {{VEC_WIDTH}}
// Defaults: BLOCK_SIZE=256, VEC_WIDTH=4

!tptir_tensor_f32 = type tensor<?xf32, 0>
!tptir_index      = type index
!tptir_f32        = type f32

// ─── ReLU: y = max(0, x) ─────────────────────────────────────────────────────
func.func @tptir_relu_f32(
    %input: !tptir_tensor_f32,
    %n:     !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %zero_f     = arith.constant 0.0 : f32

    %base = arith.muli %block_id, %block_size : index
    scf.for %i = %thread_id to (arith.constant {{VEC_WIDTH}} : index) step %block_size {
        %idx = arith.addi %base, %i : index
        %in_bounds = arith.cmpi slt, %idx, %n : index
        scf.if %in_bounds {
            %x   = tensor.extract %input[%idx] : !tptir_tensor_f32
            %out = arith.maximumf %x, %zero_f : f32
            tensor.insert %out into %input[%idx] : !tptir_tensor_f32
        }
    }
    return %input : !tptir_tensor_f32
}

// ─── GELU (tanh approximation): y = 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³))) ──
func.func @tptir_gelu_f32(
    %input: !tptir_tensor_f32,
    %n:     !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %half       = arith.constant 0.5 : f32
    %one        = arith.constant 1.0 : f32
    %sqrt2_pi   = arith.constant 0.7978845608 : f32   // sqrt(2/pi)
    %c          = arith.constant 0.044715 : f32

    %base = arith.muli %block_id, %block_size : index
    scf.for %i = %thread_id to (arith.constant {{VEC_WIDTH}} : index) step %block_size {
        %idx = arith.addi %base, %i : index
        %in_bounds = arith.cmpi slt, %idx, %n : index
        scf.if %in_bounds {
            %x    = tensor.extract %input[%idx] : !tptir_tensor_f32
            %x3   = arith.mulf (arith.mulf %x, %x : f32), %x : f32
            %cx3  = arith.mulf %c, %x3 : f32
            %inner = arith.addf %x, %cx3 : f32
            %arg   = arith.mulf %sqrt2_pi, %inner : f32
            %th    = math.tanh %arg : f32
            %t1    = arith.addf %one, %th : f32
            %out   = arith.mulf (arith.mulf %half, %x : f32), %t1 : f32
            tensor.insert %out into %input[%idx] : !tptir_tensor_f32
        }
    }
    return %input : !tptir_tensor_f32
}

// ─── SiLU/Swish: y = x * sigmoid(x) ─────────────────────────────────────────
func.func @tptir_silu_f32(
    %input: !tptir_tensor_f32,
    %n:     !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %one        = arith.constant 1.0 : f32
    %neg_one    = arith.constant -1.0 : f32

    %base = arith.muli %block_id, %block_size : index
    scf.for %i = %thread_id to (arith.constant {{VEC_WIDTH}} : index) step %block_size {
        %idx = arith.addi %base, %i : index
        %in_bounds = arith.cmpi slt, %idx, %n : index
        scf.if %in_bounds {
            %x    = tensor.extract %input[%idx] : !tptir_tensor_f32
            %negx = arith.mulf %neg_one, %x : f32
            %ex   = math.exp %negx : f32
            %denom = arith.addf %one, %ex : f32
            %sig  = arith.divf %one, %denom : f32
            %out  = arith.mulf %x, %sig : f32
            tensor.insert %out into %input[%idx] : !tptir_tensor_f32
        }
    }
    return %input : !tptir_tensor_f32
}

// ─── Sigmoid: y = 1 / (1 + exp(-x)) ─────────────────────────────────────────
func.func @tptir_sigmoid_f32(
    %input: !tptir_tensor_f32,
    %n:     !tptir_index
) -> !tptir_tensor_f32
    attributes {
        tptir.kernel,
        tptir.grid_size  = [65536, 1, 1],
        tptir.block_size = [{{BLOCK_SIZE}}, 1, 1]
    } {

    %block_id   = gpu.block_id x
    %thread_id  = gpu.thread_id x
    %block_size = arith.constant {{BLOCK_SIZE}} : index
    %one        = arith.constant 1.0 : f32
    %neg_one    = arith.constant -1.0 : f32

    %base = arith.muli %block_id, %block_size : index
    scf.for %i = %thread_id to (arith.constant {{VEC_WIDTH}} : index) step %block_size {
        %idx = arith.addi %base, %i : index
        %in_bounds = arith.cmpi slt, %idx, %n : index
        scf.if %in_bounds {
            %x    = tensor.extract %input[%idx] : !tptir_tensor_f32
            %negx = arith.mulf %neg_one, %x : f32
            %ex   = math.exp %negx : f32
            %denom = arith.addf %one, %ex : f32
            %out  = arith.divf %one, %denom : f32
            tensor.insert %out into %input[%idx] : !tptir_tensor_f32
        }
    }
    return %input : !tptir_tensor_f32
}
