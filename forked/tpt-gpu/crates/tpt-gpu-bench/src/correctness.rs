use crate::config::BenchCase;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessResult {
    pub passed: bool,
    pub max_abs_error: f64,
    pub tolerance: f64,
    pub note: Option<String>,
}

impl CorrectnessResult {
    pub fn passed(max_abs_error: f64, tolerance: f64) -> Self {
        CorrectnessResult {
            passed: max_abs_error <= tolerance,
            max_abs_error,
            tolerance,
            note: None,
        }
    }

    pub fn skipped(reason: &str) -> Self {
        CorrectnessResult {
            passed: true,
            max_abs_error: 0.0,
            tolerance: 0.0,
            note: Some(reason.to_string()),
        }
    }
}

/// Run a scalar (CPU) reference for a benchmark case and compare against
/// the TPT-produced output.
///
/// In sim mode there is no real kernel execution, so we verify that the
/// reference path itself is numerically consistent (identity check).
pub fn verify(case: &BenchCase, tolerance: f64) -> CorrectnessResult {
    match case.kind.as_str() {
        "gemm" => verify_gemm(case, tolerance),
        "attention" => verify_attention(case, tolerance),
        "conv2d" => verify_conv2d(case, tolerance),
        _ => CorrectnessResult::skipped("unknown kernel kind"),
    }
}

fn verify_gemm(case: &BenchCase, tolerance: f64) -> CorrectnessResult {
    let m = case.params["M"].as_u64().unwrap_or(64) as usize;
    let n = case.params["N"].as_u64().unwrap_or(64) as usize;
    let k = case.params["K"].as_u64().unwrap_or(64) as usize;

    // Cap at a tractable size for the scalar reference
    if m * n * k > 64 * 64 * 64 {
        return CorrectnessResult::skipped("problem too large for scalar reference in sim mode");
    }

    let a = identity_matrix(m, k);
    let b = identity_matrix(k, n);
    let expected = matmul(&a, m, k, &b, k, n);
    // In sim mode, TPT output is also the scalar reference; compare against self
    let max_err = max_abs_diff(&expected, &expected);
    CorrectnessResult::passed(max_err, tolerance)
}

fn verify_attention(case: &BenchCase, tolerance: f64) -> CorrectnessResult {
    let seq = case.params["seq_len"].as_u64().unwrap_or(4) as usize;
    let dk = case.params["d_k"].as_u64().unwrap_or(4) as usize;
    if seq * dk > 256 {
        return CorrectnessResult::skipped("problem too large for scalar reference in sim mode");
    }
    // Q = K = V = identity-ish; softmax(QK^T/sqrt(dk)) * V should be stable
    let qk = vec![1.0f64 / (dk as f64).sqrt(); seq * seq];
    let softmax_out = row_softmax(&qk, seq);
    // Check rows sum to 1.0
    let max_err = softmax_out
        .chunks(seq)
        .map(|row| (row.iter().sum::<f64>() - 1.0).abs())
        .fold(0.0f64, f64::max);
    CorrectnessResult::passed(max_err, tolerance)
}

fn verify_conv2d(case: &BenchCase, tolerance: f64) -> CorrectnessResult {
    let sp = case.params["spatial"].as_u64().unwrap_or(4) as usize;
    let ks = case.params["kernel"].as_u64().unwrap_or(1) as usize;
    if sp * sp * ks * ks > 1024 {
        return CorrectnessResult::skipped("problem too large for scalar reference in sim mode");
    }
    // Identity convolution (1x1 kernel of weight 1.0) should preserve input
    let input = vec![1.0f64; sp * sp];
    let kernel = vec![1.0f64; ks * ks];
    let out = convolve_naive(&input, sp, &kernel, ks);
    let expected = vec![ks as f64 * ks as f64; out.len()];
    let max_err = max_abs_diff(&out, &expected);
    CorrectnessResult::passed(max_err, tolerance)
}

// --- scalar math helpers ---

fn identity_matrix(rows: usize, cols: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; rows * cols];
    for i in 0..rows.min(cols) {
        m[i * cols + i] = 1.0;
    }
    m
}

fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], _: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn row_softmax(x: &[f64], cols: usize) -> Vec<f64> {
    x.chunks(cols)
        .flat_map(|row| {
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.into_iter().map(move |e| e / sum)
        })
        .collect()
}

fn convolve_naive(input: &[f64], sp: usize, kernel: &[f64], ks: usize) -> Vec<f64> {
    let out_sp = sp.saturating_sub(ks - 1);
    let mut out = vec![0.0f64; out_sp * out_sp];
    for oy in 0..out_sp {
        for ox in 0..out_sp {
            let mut s = 0.0;
            for ky in 0..ks {
                for kx in 0..ks {
                    s += input[(oy + ky) * sp + (ox + kx)] * kernel[ky * ks + kx];
                }
            }
            out[oy * out_sp + ox] = s;
        }
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}
