// GEMM dispatch — routes to hardware or pure-Rust fallback.

/// Compute alpha * A @ B + beta * C and return the (M×N) result.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    #[cfg(feature = "hardware")]
    {
        if let Some(result) = hardware_gemm(a, b, c, m, k, n, alpha, beta) {
            return result;
        }
    }
    fallback_gemm(a, b, c, m, k, n, alpha, beta)
}

/// Pure-Rust GEMM (row-major, no SIMD — correctness reference).
#[allow(clippy::too_many_arguments)]
pub fn fallback_gemm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
    }
    out
}

#[cfg(feature = "hardware")]
fn hardware_gemm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Option<Vec<f32>> {
    // tpt-gpu-runtime integration point — enabled only with --features hardware
    use tpt_gpu_runtime::kernel::{ArgumentBuffer, KernelConfig};
    let _ = (a, b, c, m, k, n, alpha, beta); // suppress unused warnings during stub
    None // return None to fall through to fallback until runtime is wired up
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_gemm() {
        // A = I(2), B = [[1,2],[3,4]] → result = B
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0; 4];
        let r = fallback_gemm(&a, &b, &c, 2, 2, 2, 1.0, 0.0);
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6);
        assert!((r[2] - 3.0).abs() < 1e-6);
        assert!((r[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_beta() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I
        let b = vec![1.0, 0.0, 0.0, 1.0]; // I
        let c = vec![2.0, 2.0, 2.0, 2.0];
        // alpha=2, beta=0.5 → 2*I + 0.5*[[2,2],[2,2]] = [[3,1],[1,3]]
        let r = fallback_gemm(&a, &b, &c, 2, 2, 2, 2.0, 0.5);
        assert!((r[0] - 3.0).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_rect() {
        // (2×3) @ (3×2) → (2×2)
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2×3
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // 3×2
        let c = vec![0.0; 4];
        let r = fallback_gemm(&a, &b, &c, 2, 3, 2, 1.0, 0.0);
        // Row 0: [1,0,0] @ [[5,6],[7,8],[9,10]] = [5,6]
        assert!((r[0] - 5.0).abs() < 1e-6);
        assert!((r[1] - 6.0).abs() < 1e-6);
        // Row 1: [0,1,0] @ ... = [7,8]
        assert!((r[2] - 7.0).abs() < 1e-6);
        assert!((r[3] - 8.0).abs() < 1e-6);
    }
}
