use crate::ast::PrimitiveType;

use super::types::{DimVal, TptType};

// ---------------------------------------------------------------------------
// Builtin return-type inference
//
// Rather than encoding full parametric signatures (which would need a
// constraint solver), we implement `infer_builtin` as a function that
// inspects the *concrete argument types* provided at a call site and
// synthesises the output type.  This is sufficient for the shape-inference
// goals of the compiler frontend.
// ---------------------------------------------------------------------------

/// Attempt to infer the return type of a built-in `tpt.*` or method call.
///
/// `name` is the full dotted name after `tpt.` (e.g. `"zeros"`, `"matmul"`).
/// `args` are the positional argument types; `named` are the named argument
/// types in order of appearance.
///
/// Returns `TptType::Unknown` when the builtin is not recognised or when
/// inference cannot be performed without more information.
pub fn infer_builtin(name: &str, args: &[TptType], named: &[(&str, TptType)]) -> TptType {
    // Helper: resolve a `dtype=` named argument to a TptType.
    let dtype_arg = || -> TptType {
        named
            .iter()
            .find(|(k, _)| *k == "dtype")
            .map(|(_, v)| v.clone())
            .unwrap_or(TptType::F32) // default dtype if not specified
    };

    // Helper: resolve a shape from the first positional argument ([m, n, ...]).
    // The shape arg is expected to be an ArrayLit of Symbolic / Concrete dims.
    // At this stage we receive it as TptType::Slice/Array/Unknown, so we
    // cannot recover the actual dims—use Dynamic placeholders.
    // (Real shape inference from array literals is handled by the caller.)

    match name {
        // ---- Tensor creation (shape comes from arg[0], dtype from named) ----
        "zeros" | "ones" | "empty" | "full" | "random" | "randn" => TptType::Tensor {
            dtype: Box::new(dtype_arg()),
            shape: vec![DimVal::Dynamic],
        },
        "eye" => {
            let dtype = dtype_arg();
            TptType::Tensor {
                dtype: Box::new(dtype),
                shape: vec![DimVal::Dynamic, DimVal::Dynamic],
            }
        }
        "arange" => TptType::Tensor {
            dtype: Box::new(TptType::I64),
            shape: vec![DimVal::Dynamic],
        },
        "linspace" => TptType::Tensor {
            dtype: Box::new(dtype_arg()),
            shape: vec![DimVal::Dynamic],
        },
        "from_list" => TptType::Tensor {
            dtype: Box::new(dtype_arg()),
            shape: vec![DimVal::Dynamic],
        },

        // ---- Shape-preserving unary tensor ops ----
        "relu" | "gelu" | "silu" | "sigmoid" | "tanh" | "leaky_relu" | "elu" | "sqrt" | "abs"
        | "neg" | "exp" | "log" | "log2" | "floor" | "ceil" | "round" | "contiguous"
        | "to_host" | "is_nan" | "is_inf" => {
            // Output shape mirrors input.
            if let Some(TptType::Tensor { dtype, shape }) = args.first() {
                let out_dtype = match name {
                    "is_nan" | "is_inf" => Box::new(TptType::Bool),
                    _ => dtype.clone(),
                };
                return TptType::Tensor {
                    dtype: out_dtype,
                    shape: shape.clone(),
                };
            }
            TptType::Unknown
        }

        // ---- Softmax / log_softmax (shape-preserving, same dtype) ----
        "softmax" | "log_softmax" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Type conversion ----
        "cast" => {
            // `tpt.cast(x, dtype=D)` → Tensor[D, ...same shape...]
            if let Some(TptType::Tensor { shape, .. }) = args.first() {
                TptType::Tensor {
                    dtype: Box::new(dtype_arg()),
                    shape: shape.clone(),
                }
            } else {
                TptType::Unknown
            }
        }
        "to_device" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Linear algebra ----
        "matmul" => {
            // (Tensor[T, m, k], Tensor[T, k, n]) -> Tensor[T, m, n]
            match (args.first(), args.get(1)) {
                (
                    Some(TptType::Tensor { dtype, shape: s1 }),
                    Some(TptType::Tensor { shape: s2, .. }),
                ) if s1.len() >= 2 && s2.len() >= 2 => {
                    let m = s1[s1.len() - 2].clone();
                    let n = s2[s2.len() - 1].clone();
                    let mut shape = s1[..s1.len() - 2].to_vec();
                    shape.push(m);
                    shape.push(n);
                    TptType::Tensor {
                        dtype: dtype.clone(),
                        shape,
                    }
                }
                _ => TptType::Unknown,
            }
        }
        "bmm" => {
            // (Tensor[T, b, m, k], Tensor[T, b, k, n]) -> Tensor[T, b, m, n]
            match (args.first(), args.get(1)) {
                (
                    Some(TptType::Tensor { dtype, shape: s1 }),
                    Some(TptType::Tensor { shape: s2, .. }),
                ) if s1.len() == 4 && s2.len() == 4 => {
                    let shape = vec![s1[0].clone(), s1[1].clone(), s1[2].clone(), s2[3].clone()];
                    TptType::Tensor {
                        dtype: dtype.clone(),
                        shape,
                    }
                }
                _ => TptType::Unknown,
            }
        }
        "gemm" => TptType::Unit, // in-place, no return value
        "dot" => {
            // Tensor[T, n] × Tensor[T, n] -> T (scalar)
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                *dtype.clone()
            } else {
                TptType::Unknown
            }
        }
        "outer" => match (args.first(), args.get(1)) {
            (
                Some(TptType::Tensor { dtype, shape: s1 }),
                Some(TptType::Tensor { shape: s2, .. }),
            ) if s1.len() == 1 && s2.len() == 1 => TptType::Tensor {
                dtype: dtype.clone(),
                shape: vec![s1[0].clone(), s2[0].clone()],
            },
            _ => TptType::Unknown,
        },
        "det" | "trace" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                *dtype.clone()
            } else {
                TptType::Unknown
            }
        }
        "inv" => args.first().cloned().unwrap_or(TptType::Unknown),
        "svd" => {
            // Returns (U, S, V) — tuple of tensors
            if let Some(TptType::Tensor { dtype, shape: _ }) = args.first() {
                let u = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic, DimVal::Dynamic],
                };
                let s = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                };
                let v = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic, DimVal::Dynamic],
                };
                TptType::Tuple(vec![u, s, v])
            } else {
                TptType::Unknown
            }
        }
        "qr" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                let q = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic, DimVal::Dynamic],
                };
                let r = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic, DimVal::Dynamic],
                };
                TptType::Tuple(vec![q, r])
            } else {
                TptType::Unknown
            }
        }

        // ---- Reduction ops ----
        "sum" | "mean" | "max" | "min" | "prod" | "norm" => {
            // If `dim` arg present, reduce along that dim (shape inferred as Dynamic).
            // If `dim` absent, result is scalar-ish (rank-0 tensor).
            if let Some(TptType::Tensor { dtype, shape: _ }) = args.first() {
                let has_dim = named.iter().any(|(k, _)| *k == "dim") || args.len() > 1;
                if has_dim {
                    // We can't easily infer the new shape without knowing which dim.
                    TptType::Tensor {
                        dtype: dtype.clone(),
                        shape: vec![DimVal::Dynamic],
                    }
                } else {
                    TptType::Tensor {
                        dtype: dtype.clone(),
                        shape: vec![],
                    }
                }
            } else {
                TptType::Unknown
            }
        }
        "argmax" | "argmin" => TptType::Tensor {
            dtype: Box::new(TptType::I64),
            shape: vec![DimVal::Dynamic],
        },
        "any" | "all" | "none" => {
            // Global reductions collapse the entire tensor to a scalar bool.
            TptType::Bool
        }

        // ---- Comparison / masking ----
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
            if let Some(TptType::Tensor { shape, .. }) = args.first() {
                TptType::Tensor {
                    dtype: Box::new(TptType::Bool),
                    shape: shape.clone(),
                }
            } else {
                TptType::Unknown
            }
        }
        "where" | "masked_fill" => {
            // Output shape follows x (arg 1).
            args.get(1).cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Element-wise binary (add/sub/mul/div/pow/clip) ----
        "add" | "sub" | "mul" | "div" | "pow" | "clip" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Shape manipulation ----
        "reshape" | "expand" | "flatten" | "squeeze" | "unsqueeze" | "permute" | "transpose" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }
        "concat" | "stack" => {
            // Output shape: same dtype, one dynamic dim
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }
        "split" | "chunk" => {
            // Returns a slice of tensors
            if let Some(t) = args.first() {
                TptType::Slice(Box::new(t.clone()))
            } else {
                TptType::Unknown
            }
        }
        "slice" | "pad" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Normalisation ----
        "normalize" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Convolution / pooling ----
        "conv1d" | "conv2d" | "conv3d" | "depthwise_conv2d" | "conv_transpose2d" | "pool2d" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }

        // ---- Attention ----
        "attention" | "flash_attention" => {
            // Output shape matches `v` (arg 2).
            args.get(2).cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Loss functions (return scalar-ish tensor) ----
        "cross_entropy" | "mse" | "mae" | "bce" | "kl_div" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![],
                }
            } else {
                TptType::Tensor {
                    dtype: Box::new(TptType::F32),
                    shape: vec![],
                }
            }
        }

        // ---- Model utilities ----
        "load_model" => TptType::Model,
        "save_model" => TptType::Unit,
        "freeze" => TptType::Unit,
        "unfreeze" => TptType::Unit,
        "count_params" => TptType::I64,
        "data_loader" => TptType::DataLoader,

        // ---- Distributed ----
        "all_reduce" | "all_gather" | "broadcast" | "scatter" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }
        "barrier" => TptType::Unit,

        // ---- Utility ----
        "print" | "sync" | "seed" => TptType::Unit,
        "shape" => TptType::Slice(Box::new(TptType::Index)),
        "dtype" => TptType::Unknown, // returns a dtype token
        "device" => TptType::Index,
        "numel" => TptType::I64,
        "benchmark" => TptType::F64,
        "grad" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Tensor creation from existing tensor ----
        "zeros_like" | "ones_like" | "full_like" | "rand_like" | "randn_like" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Scalar broadcast ops (tensor op scalar → same tensor) ----
        "add_scalar" | "sub_scalar" | "mul_scalar" | "div_scalar" | "pow_scalar"
        | "rsub_scalar" | "rdiv_scalar" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Broadcast binary ops ----
        "broadcast_add" | "broadcast_sub" | "broadcast_mul" | "broadcast_div" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Specialized matmul ----
        "matmul_2d" => match (args.first(), args.get(1)) {
            (
                Some(TptType::Tensor { dtype, shape: s1 }),
                Some(TptType::Tensor { shape: s2, .. }),
            ) if s1.len() >= 2 && s2.len() >= 2 => {
                let m = s1[s1.len() - 2].clone();
                let n = s2[s2.len() - 1].clone();
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![m, n],
                }
            }
            _ => TptType::Unknown,
        },

        // ---- Transpose variants ----
        "transpose_last2" | "t" => {
            if let Some(TptType::Tensor { dtype, shape }) = args.first() {
                let mut new_shape = shape.clone();
                let n = new_shape.len();
                if n >= 2 {
                    new_shape.swap(n - 2, n - 1);
                }
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: new_shape,
                }
            } else {
                TptType::Unknown
            }
        }

        // ---- Additional activations ----
        "mish" | "hardswish" | "hardsigmoid" | "hardtanh" | "softsign" | "softplus"
        | "log_sigmoid" | "gelu_tanh" | "gelu_new" | "selu" | "celu" | "prelu" | "rrelu"
        | "threshold" => {
            if let Some(TptType::Tensor { dtype, shape }) = args.first() {
                return TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                };
            }
            TptType::Unknown
        }

        // ---- SwiGLU: gate activation used in LLaMA/PaLM FFN ----
        "swiglu" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Dropout (shape-preserving) ----
        "dropout" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Gradient utilities ----
        "clip_grad_norm" | "clip_grad_value" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Initializers ----
        "kaiming_uniform" | "kaiming_normal" | "xavier_uniform" | "xavier_normal"
        | "trunc_normal" | "orthogonal" => TptType::Tensor {
            dtype: Box::new(dtype_arg()),
            shape: vec![DimVal::Dynamic],
        },

        // ---- Causal / attention masks ----
        "causal_mask" | "attention_mask" => TptType::Tensor {
            dtype: Box::new(dtype_arg()),
            shape: vec![DimVal::Dynamic, DimVal::Dynamic],
        },

        // ---- Loss: NLL loss ----
        "nll_loss" | "smooth_l1" | "huber" | "hinge" | "margin_ranking" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![],
                }
            } else {
                TptType::Tensor {
                    dtype: Box::new(TptType::F32),
                    shape: vec![],
                }
            }
        }

        // ---- One-hot encoding ----
        "one_hot" => TptType::Tensor {
            dtype: Box::new(TptType::F32),
            shape: vec![DimVal::Dynamic, DimVal::Dynamic],
        },

        // ---- Interpolation / upsampling ----
        "interpolate" | "upsample" | "pixel_shuffle" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }

        // ---- Cumulative ops ----
        "cumsum" | "cumprod" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Sorting ----
        "sort" => args.first().cloned().unwrap_or(TptType::Unknown),
        "argsort" => TptType::Tensor {
            dtype: Box::new(TptType::I64),
            shape: vec![DimVal::Dynamic],
        },
        "topk" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                let values = TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                };
                let indices = TptType::Tensor {
                    dtype: Box::new(TptType::I64),
                    shape: vec![DimVal::Dynamic],
                };
                TptType::Tuple(vec![values, indices])
            } else {
                TptType::Unknown
            }
        }

        // ---- Gather / scatter ops ----
        "gather" | "scatter_add" | "index_select" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }

        // ---- Statistical ops ----
        "var" | "std" | "median" | "mode" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }

        // ---- Elementwise ternary / clamp alias ----
        "clamp" | "clip_by_value" => args.first().cloned().unwrap_or(TptType::Unknown),

        // ---- Batch extraction helpers ----
        "batch_x" | "batch_y" => TptType::Tensor {
            dtype: Box::new(TptType::F32),
            shape: vec![DimVal::Dynamic],
        },

        // ---- Model/checkpoint utilities ----
        "pack_model" | "load_checkpoint" => TptType::Model,
        "save_checkpoint" => TptType::Unit,

        // ---- String/path utilities ----
        "path_join" | "format" | "to_string" => TptType::Unknown,

        // ---- Misc tensor ops ----
        "flip" | "roll" | "tril" | "triu" | "diag" | "diagonal" => {
            args.first().cloned().unwrap_or(TptType::Unknown)
        }
        "repeat" | "repeat_interleave" => args.first().cloned().unwrap_or(TptType::Unknown),
        "unfold" | "fold" => {
            if let Some(TptType::Tensor { dtype, .. }) = args.first() {
                TptType::Tensor {
                    dtype: dtype.clone(),
                    shape: vec![DimVal::Dynamic],
                }
            } else {
                TptType::Unknown
            }
        }

        // ---- Autodiff methods (called on a tensor value) ----
        "backward" | "step" | "no_grad" => TptType::Unit,
        "forward" => TptType::Unknown, // output depends on model

        _ => TptType::Unknown,
    }
}

/// Look up a named argument value by key.
pub fn named_arg_type<'a>(named: &'a [(&str, TptType)], key: &str) -> Option<&'a TptType> {
    named.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

/// Convert an identifier used as a dtype keyword (`f32`, `i64`, ...) into a
/// TptType. Used when a dtype is passed as a named argument like `dtype=f32`.
pub fn ident_as_dtype(name: &str) -> Option<TptType> {
    PrimitiveType::from_str(name).map(|p| TptType::from_primitive(&p))
}
