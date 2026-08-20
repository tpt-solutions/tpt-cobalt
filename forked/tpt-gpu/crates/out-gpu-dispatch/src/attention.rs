// Attention dispatch — scaled dot-product attention.
//
// Layout: (batch, heads, seq, d) row-major.

use crate::gemm::fallback_gemm;

/// Dispatch: softmax(Q @ K^T * scale) @ V → (batch × heads × seq_q × d_v)
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_q: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
) -> Vec<f32> {
    let head_size_q = seq_q * d_k;
    let head_size_v = seq_q * d_v;
    let seq_k = k.len() / (batch * heads * d_k); // K is (batch, heads, seq_k, d_k)

    let mut out = vec![0.0f32; batch * heads * seq_q * d_v];

    for b in 0..batch {
        for h in 0..heads {
            let base_q = (b * heads + h) * head_size_q;
            let base_k = (b * heads + h) * seq_k * d_k;
            let base_v = (b * heads + h) * seq_k * d_v;
            let base_o = (b * heads + h) * head_size_v;

            // scores = Q @ K^T  (seq_q × seq_k)
            let q_slice = &q[base_q..base_q + seq_q * d_k];
            let k_slice = &k[base_k..base_k + seq_k * d_k];

            // K^T: (d_k × seq_k)
            let kt = transpose(k_slice, seq_k, d_k);
            let zero_c = vec![0.0f32; seq_q * seq_k];
            let mut scores = fallback_gemm(q_slice, &kt, &zero_c, seq_q, d_k, seq_k, scale, 0.0);

            // softmax over last dim
            softmax_inplace(&mut scores, seq_q, seq_k);

            // out = scores @ V  (seq_q × d_v)
            let v_slice = &v[base_v..base_v + seq_k * d_v];
            let zero_o = vec![0.0f32; seq_q * d_v];
            let head_out = fallback_gemm(&scores, v_slice, &zero_o, seq_q, seq_k, d_v, 1.0, 0.0);

            out[base_o..base_o + seq_q * d_v].copy_from_slice(&head_out);
        }
    }
    out
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

fn softmax_inplace(m: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut m[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_identity() {
        // batch=1, heads=1, seq=2, d=2
        // Q = K = V = I(2) → softmax(I @ I^T / sqrt(2)) @ I = softmax scores @ I
        let q = vec![1.0_f32, 0.0, 0.0, 1.0];
        let k = q.clone();
        let v = q.clone();
        let scale = (2.0_f32).powf(-0.5);
        let out = dispatch(&q, &k, &v, 1, 1, 2, 2, 2, scale);
        assert_eq!(out.len(), 4);
        // Each output row should sum to 1 (convex combination of V rows)
        let r0_sum = out[0] + out[1];
        let r1_sum = out[2] + out[3];
        assert!((r0_sum - 1.0).abs() < 1e-5, "row0 sum={r0_sum}");
        assert!((r1_sum - 1.0).abs() < 1e-5, "row1 sum={r1_sum}");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut m = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut m, 2, 2);
        let s0 = m[0] + m[1];
        let s1 = m[2] + m[3];
        assert!((s0 - 1.0).abs() < 1e-6);
        assert!((s1 - 1.0).abs() < 1e-6);
    }
}
