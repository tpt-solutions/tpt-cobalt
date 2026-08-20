//! Arch-template dispatch layer.
//!
//! Maps a model architecture string (from the GGUF `general.architecture` key)
//! to a typed sequence of `ForwardOp`s that fully describes the model's
//! forward pass. Adding a new architecture requires one new function and one
//! new match arm in `template_for_arch` — no changes to the dispatch loop.

use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::inference::ModelInfo;

/// A single operation in the transformer forward pass.
#[derive(Debug, Clone)]
pub enum ForwardOp {
    /// Token → hidden state lookup via an embedding table.
    Embed { vocab_size: u32, hidden_dim: u32 },
    /// RMS Layer Normalisation: y = x / rms(x) * gamma.
    RmsNorm { hidden_dim: u32, eps: f32 },
    /// Scaled dot-product attention with optional grouped-query (GQA).
    Attention {
        num_heads: u32,
        num_kv_heads: u32,
        hidden_dim: u32,
        head_dim: u32,
    },
    /// SwiGLU feed-forward block: FFN(x) = down(silu(gate(x)) * up(x)).
    FfnSwiGlu { hidden_dim: u32, ffn_dim: u32 },
    /// Final vocabulary projection (lm_head). `tied = true` for Qwen2/Gemma2
    /// where the embedding table and lm_head share weights.
    LinearOut {
        hidden_dim: u32,
        vocab_size: u32,
        tied: bool,
    },
    /// Softmax over the vocab dimension.
    Softmax { dim: i32 },
    /// Sample the next token from a probability distribution.
    Sampling { temperature: f32, top_k: u32 },
    /// Apply Rotary Position Embedding to Q and K projections.
    ///
    /// `pos` is a placeholder (0) in the template; the runtime substitutes the
    /// real decode position from `kv_cache.seq_len` at forward-pass time.
    ApplyRope {
        head_dim: u32,
        base: f32,
        pos: usize,
    },
}

/// The complete op sequence for one model architecture.
#[derive(Debug, Clone)]
pub struct ArchTemplate {
    /// Number of transformer layers (blocks).
    pub num_layers: u32,
    /// Ops run once before the first layer (typically: Embed).
    pub pre_ops: Vec<ForwardOp>,
    /// Ops run once per transformer layer, repeated `num_layers` times.
    pub per_layer_ops: Vec<ForwardOp>,
    /// Ops run once after the last layer (final norm, lm_head, softmax, sample).
    pub post_ops: Vec<ForwardOp>,
}

/// Build an `ArchTemplate` from the `general.architecture` GGUF key.
///
/// Returns `UnsupportedFeature` for unrecognised architectures.
pub fn template_for_arch(arch: &str, info: &ModelInfo) -> TptrResult<ArchTemplate> {
    match arch {
        "llama3" | "llama" => Ok(llama3(info)),
        "mistral"          => Ok(mistral(info)),
        "qwen2" | "qwen"   => Ok(qwen2(info)),
        "phi3"             => Ok(phi3(info)),
        "gemma2" | "gemma" => Ok(gemma2(info)),
        other => Err(TptrError::new(
            ErrorCode::UnsupportedFeature,
            format!(
                "unsupported model architecture '{}'; supported: llama3, mistral, qwen2, phi3, gemma2",
                other
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Arch implementations
// ---------------------------------------------------------------------------

fn llama3(info: &ModelInfo) -> ArchTemplate {
    let num_heads = info.num_heads.max(1);
    let head_dim = info.hidden_dim / num_heads;
    let num_kv_heads = if info.num_kv_heads > 0 {
        info.num_kv_heads
    } else {
        num_heads / 4
    };
    ArchTemplate {
        num_layers: info.num_layers,
        pre_ops: vec![ForwardOp::Embed {
            vocab_size: info.vocab_size,
            hidden_dim: info.hidden_dim,
        }],
        per_layer_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::ApplyRope {
                head_dim,
                base: 500_000.0,
                pos: 0,
            },
            ForwardOp::Attention {
                num_heads,
                num_kv_heads,
                hidden_dim: info.hidden_dim,
                head_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::FfnSwiGlu {
                hidden_dim: info.hidden_dim,
                ffn_dim: info.ffn_dim,
            },
        ],
        post_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::LinearOut {
                hidden_dim: info.hidden_dim,
                vocab_size: info.vocab_size,
                tied: false,
            },
            ForwardOp::Softmax { dim: -1 },
            ForwardOp::Sampling {
                temperature: 1.0,
                top_k: 1,
            },
        ],
    }
}

fn mistral(info: &ModelInfo) -> ArchTemplate {
    let num_heads = info.num_heads.max(1);
    let head_dim = info.hidden_dim / num_heads;
    // Mistral 7B uses 8 KV heads by default; fall back if not in GGUF.
    let num_kv_heads = if info.num_kv_heads > 0 {
        info.num_kv_heads
    } else {
        8
    };
    ArchTemplate {
        num_layers: info.num_layers,
        pre_ops: vec![ForwardOp::Embed {
            vocab_size: info.vocab_size,
            hidden_dim: info.hidden_dim,
        }],
        per_layer_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::ApplyRope {
                head_dim,
                base: 10_000.0,
                pos: 0,
            },
            ForwardOp::Attention {
                num_heads,
                num_kv_heads,
                hidden_dim: info.hidden_dim,
                head_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::FfnSwiGlu {
                hidden_dim: info.hidden_dim,
                ffn_dim: info.ffn_dim,
            },
        ],
        post_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::LinearOut {
                hidden_dim: info.hidden_dim,
                vocab_size: info.vocab_size,
                tied: false,
            },
            ForwardOp::Softmax { dim: -1 },
            ForwardOp::Sampling {
                temperature: 1.0,
                top_k: 1,
            },
        ],
    }
}

fn qwen2(info: &ModelInfo) -> ArchTemplate {
    let num_heads = info.num_heads.max(1);
    let head_dim = info.hidden_dim / num_heads;
    // Qwen2-7B uses 4 KV heads; 72B uses 8.
    let num_kv_heads = if info.num_kv_heads > 0 {
        info.num_kv_heads
    } else {
        4
    };
    ArchTemplate {
        num_layers: info.num_layers,
        pre_ops: vec![ForwardOp::Embed {
            vocab_size: info.vocab_size,
            hidden_dim: info.hidden_dim,
        }],
        per_layer_ops: vec![
            // Qwen2 uses eps = 1e-6 (tighter than Llama3/Mistral)
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            },
            ForwardOp::ApplyRope {
                head_dim,
                base: 1_000_000.0,
                pos: 0,
            },
            ForwardOp::Attention {
                num_heads,
                num_kv_heads,
                hidden_dim: info.hidden_dim,
                head_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            },
            ForwardOp::FfnSwiGlu {
                hidden_dim: info.hidden_dim,
                ffn_dim: info.ffn_dim,
            },
        ],
        // Qwen2 ties input embeddings to lm_head (vocab_size = 151936 typically)
        post_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            },
            ForwardOp::LinearOut {
                hidden_dim: info.hidden_dim,
                vocab_size: info.vocab_size,
                tied: true,
            },
            ForwardOp::Softmax { dim: -1 },
            ForwardOp::Sampling {
                temperature: 1.0,
                top_k: 1,
            },
        ],
    }
}

fn phi3(info: &ModelInfo) -> ArchTemplate {
    let num_heads = info.num_heads.max(1);
    let head_dim = info.hidden_dim / num_heads;
    let num_kv_heads = if info.num_kv_heads > 0 {
        info.num_kv_heads
    } else {
        num_heads
    };
    // Phi-3 uses ffn_dim = 3× hidden_dim when not specified in GGUF.
    let ffn_dim = if info.ffn_dim > 0 {
        info.ffn_dim
    } else {
        info.hidden_dim * 3
    };
    ArchTemplate {
        num_layers: info.num_layers,
        pre_ops: vec![ForwardOp::Embed {
            vocab_size: info.vocab_size,
            hidden_dim: info.hidden_dim,
        }],
        per_layer_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::ApplyRope {
                head_dim,
                base: 10_000.0,
                pos: 0,
            },
            ForwardOp::Attention {
                num_heads,
                num_kv_heads,
                hidden_dim: info.hidden_dim,
                head_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::FfnSwiGlu {
                hidden_dim: info.hidden_dim,
                ffn_dim,
            },
        ],
        post_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-5,
            },
            ForwardOp::LinearOut {
                hidden_dim: info.hidden_dim,
                vocab_size: info.vocab_size,
                tied: false,
            },
            ForwardOp::Softmax { dim: -1 },
            ForwardOp::Sampling {
                temperature: 1.0,
                top_k: 1,
            },
        ],
    }
}

fn gemma2(info: &ModelInfo) -> ArchTemplate {
    let num_heads = info.num_heads.max(1);
    let head_dim = info.hidden_dim / num_heads;
    let num_kv_heads = if info.num_kv_heads > 0 {
        info.num_kv_heads
    } else {
        num_heads / 2
    };
    // Gemma2 adds a post-attention norm and a post-FFN norm (7 ops per layer with RoPE).
    ArchTemplate {
        num_layers: info.num_layers,
        pre_ops: vec![ForwardOp::Embed {
            vocab_size: info.vocab_size,
            hidden_dim: info.hidden_dim,
        }],
        per_layer_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            }, // pre-attention
            ForwardOp::ApplyRope {
                head_dim,
                base: 10_000.0,
                pos: 0,
            },
            ForwardOp::Attention {
                num_heads,
                num_kv_heads,
                hidden_dim: info.hidden_dim,
                head_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            }, // post-attention
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            }, // pre-FFN
            ForwardOp::FfnSwiGlu {
                hidden_dim: info.hidden_dim,
                ffn_dim: info.ffn_dim,
            },
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            }, // post-FFN
        ],
        // Gemma2 also ties lm_head with the embedding table.
        post_ops: vec![
            ForwardOp::RmsNorm {
                hidden_dim: info.hidden_dim,
                eps: 1e-6,
            },
            ForwardOp::LinearOut {
                hidden_dim: info.hidden_dim,
                vocab_size: info.vocab_size,
                tied: true,
            },
            ForwardOp::Softmax { dim: -1 },
            ForwardOp::Sampling {
                temperature: 1.0,
                top_k: 1,
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelInfo;
    use std::path::PathBuf;

    fn stub_info(arch: &str, num_layers: u32) -> ModelInfo {
        ModelInfo {
            path: PathBuf::from("stub.gguf"),
            arch: arch.to_string(),
            context_len: 4096,
            vocab_size: 32000,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            ffn_dim: 11008,
            num_layers,
            per_layer_bits: Vec::new(),
            pruning_mask: None,
            rope_freq_base: None,
            tokenizer: None,
        }
    }

    #[test]
    fn llama3_layer_count() {
        let info = stub_info("llama3", 32);
        let t = template_for_arch("llama3", &info).unwrap();
        assert_eq!(t.num_layers, 32);
        // RmsNorm + ApplyRope + Attention + RmsNorm + FfnSwiGlu = 5
        assert_eq!(t.per_layer_ops.len(), 5);
    }

    #[test]
    fn qwen2_tied_lm_head() {
        let info = stub_info("qwen2", 28);
        let t = template_for_arch("qwen2", &info).unwrap();
        let tied = t
            .post_ops
            .iter()
            .any(|op| matches!(op, ForwardOp::LinearOut { tied: true, .. }));
        assert!(tied, "Qwen2 must have tied lm_head");
    }

    #[test]
    fn gemma2_has_seven_per_layer_ops() {
        let info = stub_info("gemma2", 46);
        let t = template_for_arch("gemma2", &info).unwrap();
        assert_eq!(
            t.per_layer_ops.len(),
            7,
            "Gemma2 uses 7 ops/layer (pre-attn norm + RoPE + attn + post-attn norm + pre-FFN norm + FFN + post-FFN norm)"
        );
    }

    #[test]
    fn unknown_arch_is_unsupported_feature_error() {
        let info = stub_info("gpt2-xl", 48);
        let err = template_for_arch("gpt2-xl", &info).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedFeature);
    }

    #[test]
    fn phi3_ffn_fallback_to_3x_hidden() {
        let mut info = stub_info("phi3", 32);
        info.ffn_dim = 0; // not set in GGUF
        let t = template_for_arch("phi3", &info).unwrap();
        let ffn = t
            .per_layer_ops
            .iter()
            .find_map(|op| {
                if let ForwardOp::FfnSwiGlu {
                    hidden_dim,
                    ffn_dim,
                } = op
                {
                    Some((*hidden_dim, *ffn_dim))
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(ffn.1, ffn.0 * 3);
    }

    #[test]
    fn mistral_has_correct_kv_heads_fallback() {
        let mut info = stub_info("mistral", 32);
        info.num_kv_heads = 0;
        let t = template_for_arch("mistral", &info).unwrap();
        let kv = t
            .per_layer_ops
            .iter()
            .find_map(|op| {
                if let ForwardOp::Attention { num_kv_heads, .. } = op {
                    Some(*num_kv_heads)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(kv, 8);
    }
}
