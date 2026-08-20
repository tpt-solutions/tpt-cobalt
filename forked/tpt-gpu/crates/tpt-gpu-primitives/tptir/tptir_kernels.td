// tptir_kernels.td — TPTIR Kernel Operation Definitions
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// TPTIR GPU Kernel operations for GEMM, Attention, Conv2D, and ML primitives.

#ifndef TPTIR_KERNELS_TD
#define TPTIR_KERNELS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def TPTIR_Kernel_Dialect : Dialect {
  let name = "tptir_kernel";
  let summary = "TPTIR GPU Kernel Operations";
  let cppNamespace = "tptir::kernel";
}

def TensorType : TypeConstraint<CPred<"$_self.isa<TensorType>()">, "tensor type">;
def Float32Type : TypeConstraint<CPred<"$_self.isa<Float32Type>()">, "f32 type">;
def IndexType : TypeConstraint<CPred<"$_self.isa<IndexType>()">, "index type">;
def BoolAttr : Attr<CPred<"$_self.isa<BoolAttr>()">, "bool attribute">;

// GEMM: C = alpha * A * B + beta * C
def TPTIR_GemmOp : Op<TPTIR_Kernel_Dialect, "gemm", [Pure]> {
  let summary = "General Matrix Multiply";
  let arguments = (ins TensorType:$A, TensorType:$B, TensorType:$C,
                   Float32Type:$alpha, Float32Type:$beta,
                   IndexType:$M, IndexType:$N, IndexType:$K);
  let results = (outs TensorType:$result);
  let assemblyFormat = "`ins` `(` $A `,` $B `,` $C `:` type($A) `,` type($B) `,` type($C) `)` `attrs` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

// Attention: softmax(Q * K^T / scale) * V
def TPTIR_AttentionOp : Op<TPTIR_Kernel_Dialect, "attention", [Pure]> {
  let summary = "Scaled Dot-Product Attention";
  let arguments = (ins TensorType:$Q, TensorType:$K, TensorType:$V,
                   Float32Type:$scale, TensorType:$mask,
                   IndexType:$seq_len, IndexType:$d_k, IndexType:$d_v,
                   BoolAttr:$causal);
  let results = (outs TensorType:$output, TensorType:$attention_weights);
  let hasVerifier = 1;
}

// Conv2D: output = conv2d(input, filter, strides, padding)
def TPTIR_Conv2DOp : Op<TPTIR_Kernel_Dialect, "conv2d", [Pure]> {
  let summary = "2D Convolution";
  let arguments = (ins TensorType:$input, TensorType:$filter,
                   I32Attr:$stride_h, I32Attr:$stride_w,
                   I32Attr:$padding_h, I32Attr:$padding_w,
                   I32Attr:$dilation_h, I32Attr:$dilation_w,
                   I32Attr:$groups);
  let results = (outs TensorType:$output);
  let assemblyFormat = "`ins` `(` $input `,` $filter `:` type($input) `,` type($filter) `)` `attrs` attr-dict `->` type($output)";
  let hasVerifier = 1;
}

// Softmax: softmax along last dimension
def TPTIR_SoftmaxOp : Op<TPTIR_Kernel_Dialect, "softmax", [Pure]> {
  let summary = "Softmax kernel";
  let arguments = (ins TensorType:$input, IndexType:$dim);
  let results = (outs TensorType:$output);
}

// LayerNorm: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta, normalized over last axis
def TPTIR_LayerNormOp : Op<TPTIR_Kernel_Dialect, "layer_norm", [Pure]> {
  let summary = "Layer Normalization";
  let arguments = (ins TensorType:$input, TensorType:$gamma, TensorType:$beta,
                   Float32Type:$epsilon, IndexType:$batch, IndexType:$norm_size);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// BatchNorm: normalize over N*S for each channel; updates running stats during training
def TPTIR_BatchNormOp : Op<TPTIR_Kernel_Dialect, "batch_norm", [Pure]> {
  let summary = "Batch Normalization";
  let arguments = (ins TensorType:$input, TensorType:$gamma, TensorType:$beta,
                   TensorType:$running_mean, TensorType:$running_var,
                   Float32Type:$epsilon, Float32Type:$momentum,
                   IndexType:$batch, IndexType:$channels, IndexType:$spatial,
                   I1Attr:$is_training);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// GroupNorm: channels split into G groups; normalize over C/G * S per (N, G) pair
def TPTIR_GroupNormOp : Op<TPTIR_Kernel_Dialect, "group_norm", [Pure]> {
  let summary = "Group Normalization";
  let arguments = (ins TensorType:$input, TensorType:$gamma, TensorType:$beta,
                   Float32Type:$epsilon,
                   IndexType:$batch, IndexType:$channels, IndexType:$spatial,
                   I32Attr:$num_groups);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// RMSNorm: y = x / rms(x) * gamma; rms(x) = sqrt(mean(x^2) + eps) — no mean subtraction
def TPTIR_RMSNormOp : Op<TPTIR_Kernel_Dialect, "rms_norm", [Pure]> {
  let summary = "RMS Normalization (LLaMA-style)";
  let arguments = (ins TensorType:$input, TensorType:$gamma,
                   Float32Type:$epsilon,
                   IndexType:$batch, IndexType:$norm_size);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// Elementwise activations: ReLU, GELU (tanh approx), SiLU/Swish, Sigmoid
def TPTIR_ElementwiseOp : Op<TPTIR_Kernel_Dialect, "elementwise", [Pure]> {
  let summary = "Elementwise Activation (relu/gelu/silu/sigmoid)";
  let arguments = (ins TensorType:$input, IndexType:$n,
                   StrAttr:$kind);        // "relu" | "gelu" | "silu" | "sigmoid"
  let results = (outs TensorType:$output);
}

// Embedding: output[b,s,:] = weight[indices[b,s],:]
def TPTIR_EmbeddingOp : Op<TPTIR_Kernel_Dialect, "embedding", [Pure]> {
  let summary = "Embedding Table Lookup";
  let arguments = (ins TensorType:$weight, TensorType:$indices,
                   IndexType:$batch, IndexType:$seq_len, IndexType:$embed_dim,
                   I32Attr:$padding_idx);  // -1 means disabled
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// MaxPool2D: sliding window maximum over [N, C, H, W]
def TPTIR_MaxPool2DOp : Op<TPTIR_Kernel_Dialect, "max_pool2d", [Pure]> {
  let summary = "2D Max Pooling";
  let arguments = (ins TensorType:$input,
                   I32Attr:$kernel_h, I32Attr:$kernel_w,
                   I32Attr:$stride_h, I32Attr:$stride_w,
                   I32Attr:$padding_h, I32Attr:$padding_w);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

// AvgPool2D: sliding window average over [N, C, H, W]
def TPTIR_AvgPool2DOp : Op<TPTIR_Kernel_Dialect, "avg_pool2d", [Pure]> {
  let summary = "2D Average Pooling";
  let arguments = (ins TensorType:$input,
                   I32Attr:$kernel_h, I32Attr:$kernel_w,
                   I32Attr:$stride_h, I32Attr:$stride_w,
                   I32Attr:$padding_h, I32Attr:$padding_w);
  let results = (outs TensorType:$output);
  let hasVerifier = 1;
}

#endif // TPTIR_KERNELS_TD