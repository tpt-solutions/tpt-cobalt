//! Rotary Position Embedding (RoPE).
//!
//! Applied to Q and K tensors before attention to inject position information.
//! The rotation is applied pair-wise: for each consecutive pair `(x[2i], x[2i+1])`
//! in a head, a 2-D rotation by angle `θ_i = pos / base^(2i / head_dim)` is applied:
//!
//! ```text
//!   x'[2i]   =  x[2i]   * cos(θ_i) − x[2i+1] * sin(θ_i)
//!   x'[2i+1] =  x[2i]   * sin(θ_i) + x[2i+1] * cos(θ_i)
//! ```
//!
//! Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//! (Su et al., 2021).

/// Configuration for Rotary Position Embedding.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Per-head feature dimension.
    pub head_dim: usize,
    /// θ base frequency.  Common values:
    /// - LLaMA 1/2, Mistral, Phi-3, Gemma: 10 000.0
    /// - LLaMA 3: 500 000.0
    /// - Qwen2: 1 000 000.0
    pub base: f32,
    /// Maximum sequence length this config is designed for.
    /// Stored for informational purposes; not used in on-the-fly computation.
    pub max_seq_len: usize,
}

impl RopeConfig {
    /// LLaMA 1/2 defaults (7B–65B family).
    pub fn llama() -> Self {
        Self {
            head_dim: 128,
            base: 10_000.0,
            max_seq_len: 4096,
        }
    }

    /// LLaMA 3 defaults (extended context with higher base frequency).
    pub fn llama3() -> Self {
        Self {
            head_dim: 128,
            base: 500_000.0,
            max_seq_len: 8192,
        }
    }

    /// Mistral 7B defaults.
    pub fn mistral() -> Self {
        Self {
            head_dim: 128,
            base: 10_000.0,
            max_seq_len: 32768,
        }
    }

    /// Qwen2 defaults (very high base for extended context).
    pub fn qwen2() -> Self {
        Self {
            head_dim: 128,
            base: 1_000_000.0,
            max_seq_len: 32768,
        }
    }

    /// Phi-3 defaults (smaller head_dim than the LLaMA family).
    pub fn phi3() -> Self {
        Self {
            head_dim: 96,
            base: 10_000.0,
            max_seq_len: 4096,
        }
    }

    /// Gemma 2 defaults.
    pub fn gemma2() -> Self {
        Self {
            head_dim: 256,
            base: 10_000.0,
            max_seq_len: 8192,
        }
    }
}

/// Rotate a flat slice of head vectors by a signed decode position.
///
/// The slice is treated as a sequence of `cfg.head_dim`-sized chunks (one per
/// head).  Within each chunk, pairs `(x[2i], x[2i+1])` are rotated by
/// `θ_i = pos / base^(2i / head_dim)`.  Passing a negative position applies
/// the inverse rotation, which is the basis of the round-trip guarantee.
pub(crate) fn rotate_slice(x: &mut [f32], pos: i64, cfg: &RopeConfig) {
    let hd = cfg.head_dim;
    if hd == 0 {
        return;
    }
    let half = hd / 2;

    for chunk in x.chunks_mut(hd) {
        for i in 0..half {
            let idx1 = 2 * i;
            let idx2 = 2 * i + 1;
            if idx2 >= chunk.len() {
                break;
            }
            // Use f64 for the angle to avoid accumulated rounding on high bases.
            let exponent = 2.0_f64 * i as f64 / hd as f64;
            let freq = pos as f64 / (cfg.base as f64).powf(exponent);
            let cos_v = freq.cos() as f32;
            let sin_v = freq.sin() as f32;
            let x0 = chunk[idx1];
            let x1 = chunk[idx2];
            chunk[idx1] = x0 * cos_v - x1 * sin_v;
            chunk[idx2] = x0 * sin_v + x1 * cos_v;
        }
    }
}

/// Apply Rotary Position Embedding to Q and K tensors at decode position `pos`.
///
/// Both `q` and `k` are flat, row-major arrays with layout
/// `[num_heads × head_dim]` and `[num_kv_heads × head_dim]` respectively.
/// RoPE is applied independently to each head's slice.
///
/// Call this after the Q/K projections and before appending to the KV cache
/// so that cached K values are already position-encoded.
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, cfg: &RopeConfig) {
    rotate_slice(q, pos as i64, cfg);
    rotate_slice(k, pos as i64, cfg);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// At position 0 every angle is 0 → cos = 1, sin = 0 → output equals input.
    #[test]
    fn test_rope_identity_at_pos_0() {
        let cfg = RopeConfig::llama3();
        let original: Vec<f32> = (0..cfg.head_dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let mut q = original.clone();
        let mut k = original.clone();

        apply_rope(&mut q, &mut k, 0, &cfg);

        for (i, (got, &want)) in q.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "q[{i}] mismatch at pos=0: got {got}, expected {want}"
            );
        }
        for (i, (got, &want)) in k.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "k[{i}] mismatch at pos=0: got {got}, expected {want}"
            );
        }
    }

    /// Verify a known (q, k) pair at pos = 1, head_dim = 2, base = 10 000.
    ///
    /// With `head_dim = 2` there is exactly one frequency index (i = 0):
    ///   `θ_0 = 1 / 10000^(0/2) = 1.0 rad`
    ///   `cos(1.0) ≈ 0.5403023`, `sin(1.0) ≈ 0.8414710`
    ///
    /// For `q = [1.0, 0.0]`:
    ///   `q' = [cos(1), sin(1)]`
    ///
    /// For `k = [0.0, 1.0]`:
    ///   `k' = [−sin(1), cos(1)]`
    #[test]
    fn test_rope_rotates_pos_1() {
        let cfg = RopeConfig {
            head_dim: 2,
            base: 10_000.0,
            max_seq_len: 4096,
        };
        let mut q = vec![1.0f32, 0.0];
        let mut k = vec![0.0f32, 1.0];

        apply_rope(&mut q, &mut k, 1, &cfg);

        let cos1 = (1.0_f64).cos() as f32;
        let sin1 = (1.0_f64).sin() as f32;

        assert!(
            (q[0] - cos1).abs() < 1e-5,
            "q[0] = {}, expected cos(1) ≈ {cos1}",
            q[0]
        );
        assert!(
            (q[1] - sin1).abs() < 1e-5,
            "q[1] = {}, expected sin(1) ≈ {sin1}",
            q[1]
        );
        assert!(
            (k[0] - (-sin1)).abs() < 1e-5,
            "k[0] = {}, expected -sin(1) ≈ {}",
            k[0],
            -sin1
        );
        assert!(
            (k[1] - cos1).abs() < 1e-5,
            "k[1] = {}, expected cos(1) ≈ {cos1}",
            k[1]
        );
    }

    /// Applying RoPE at pos = P then at pos = -P restores the original values.
    ///
    /// This holds because `R(-θ) · R(θ) = I` for any 2-D rotation matrix.
    /// We use the `pub(crate)` `rotate_slice` helper to pass a negative position.
    #[test]
    fn test_rope_round_trip() {
        let cfg = RopeConfig {
            head_dim: 4,
            base: 10_000.0,
            max_seq_len: 4096,
        };
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();

        // Forward rotation at pos = 7
        rotate_slice(&mut q, 7, &cfg);
        rotate_slice(&mut k, 7, &cfg);

        // Inverse rotation at pos = -7
        rotate_slice(&mut q, -7, &cfg);
        rotate_slice(&mut k, -7, &cfg);

        for (i, (got, &want)) in q.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "q[{i}] round-trip failed: got {got}, expected {want}"
            );
        }
        for (i, (got, &want)) in k.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "k[{i}] round-trip failed: got {got}, expected {want}"
            );
        }
    }
}
