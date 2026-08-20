//! Runtime support for `#[derive_jit]`.
//!
//! "JIT" here is a **fusion hint**, not a machine-code compiler: the macro
//! rewrites an element-wise tensor body into a scalar kernel and [`fuse`]
//! evaluates it in a *single* pass over the (broadcast) output, allocating no
//! intermediate tensors. With one input of the output shape, the pass is
//! handed to [`tpt_omni::Tensor::par_map`]; with several (broadcast) inputs the
//! output buffer is filled by a rayon parallel iterator. Either way the pass
//! runs in parallel.

use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;
use tpt_omni::tensor::broadcast_shapes;
use tpt_omni::Tensor;

/// Output elements handled by one rayon task; keeps the per-task scratch
/// buffers out of the inner loop.
const CHUNK: usize = 1024;

fn bc_get(t: &Tensor<f64>, out_idx: &[usize]) -> f64 {
    let sh = t.shape();
    let off = out_idx.len() - sh.len();
    let idx: Vec<usize> = sh
        .iter()
        .enumerate()
        .map(|(i, &s)| if s == 1 { 0 } else { out_idx[off + i] })
        .collect();
    t.inner()[IxDyn(&idx)]
}

/// Write the row-major multi-index of the flat index `flat` into `idx`.
fn unravel(flat: usize, shape: &[usize], idx: &mut [usize]) {
    let mut rem = flat;
    for k in (0..shape.len()).rev() {
        let s = shape[k].max(1);
        idx[k] = rem % s;
        rem /= s;
    }
}

/// Evaluate `f` once per output element, with the element of each input in
/// argument order. Inputs are broadcast together (NumPy rules).
pub fn fuse<F>(inputs: &[&Tensor<f64>], f: F) -> Tensor<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    assert!(!inputs.is_empty(), "fuse() needs at least one input");
    let mut shape: Vec<usize> = inputs[0].shape().to_vec();
    for t in &inputs[1..] {
        shape = broadcast_shapes(&shape, t.shape()).expect("jit: inputs are not broadcastable");
    }
    if inputs.len() == 1 && inputs[0].shape() == shape.as_slice() {
        // Single input, no broadcasting: use the parallel element-wise path.
        return inputs[0].par_map(|v| f(&[v]));
    }
    // Several (broadcast) inputs: fill the flat output buffer in parallel and
    // reshape once at the end.
    let n: usize = shape.iter().product();
    let rank = shape.len();
    let mut out = vec![0.0f64; n];
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk, slot)| {
            let mut idx = vec![0usize; rank];
            let mut buf = vec![0.0f64; inputs.len()];
            for (j, o) in slot.iter_mut().enumerate() {
                unravel(chunk * CHUNK + j, &shape, &mut idx);
                for (k, t) in inputs.iter().enumerate() {
                    buf[k] = bc_get(t, &idx);
                }
                *o = f(&buf);
            }
        });
    Tensor::new(ArrayD::from_shape_vec(IxDyn(&shape), out).expect("jit: output shape mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{scalar, tensor};

    #[test]
    fn fuse_single_input_keeps_the_par_map_path() {
        let x = tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let out = fuse(&[&x], |v| v[0] * v[0]);
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.to_vec(), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn fuse_broadcasts_a_row_vector() {
        // [2, 3] against a [3] row vector.
        let x = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor(&[3], &[10.0, 20.0, 30.0]);
        let out = fuse(&[&x, &b], |v| v[0] + v[1]);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn fuse_broadcasts_a_rank0_input() {
        let x = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let k = scalar(0.5);
        let out = fuse(&[&x, &k], |v| v[0] * v[1]);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.to_vec(), vec![0.5, 1.0, 1.5, 2.0]);

        // Two rank-0 inputs: the degenerate shape still round-trips.
        let out = fuse(&[&k, &k], |v| v[0] + v[1]);
        assert!(out.shape().is_empty());
        assert_eq!(out.to_vec(), vec![1.0]);
    }

    #[test]
    fn fuse_parallel_path_spans_several_chunks() {
        // More elements than one rayon chunk, so ordering across chunks and the
        // flat-index unravelling both get exercised.
        let n = CHUNK * 3 + 7;
        let a = tensor(&[n, 1], &(0..n).map(|i| i as f64).collect::<Vec<_>>());
        let b = tensor(&[2], &[1.0, -1.0]);
        let out = fuse(&[&a, &b], |v| v[0] * v[1]);
        assert_eq!(out.shape(), &[n, 2]);
        let got = out.to_vec();
        for i in 0..n {
            assert_eq!(got[2 * i], i as f64);
            assert_eq!(got[2 * i + 1], -(i as f64));
        }
    }

    #[test]
    fn unravel_matches_row_major_order() {
        let shape = [2usize, 3, 4];
        let mut idx = vec![0usize; 3];
        unravel(0, &shape, &mut idx);
        assert_eq!(idx, vec![0, 0, 0]);
        unravel(5, &shape, &mut idx);
        assert_eq!(idx, vec![0, 1, 1]);
        unravel(23, &shape, &mut idx);
        assert_eq!(idx, vec![1, 2, 3]);
    }
}
