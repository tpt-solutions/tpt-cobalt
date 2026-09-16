//! # tpt-ml — Data loading (Phase 2, spec §5.3)
//!
//! A minimal `Dataset` abstraction plus an in-memory `TensorDataset` and a
//! `DataLoader` that yields stacked batches with optional epoch shuffling.
//! Multi-threaded / Arrow-backed prefetching is deferred (see todo.md, Phase 2);
//! this scaffold provides the core training-loop contract without external deps.

use tpt_tensor::Tensor;

/// A source of `(input, target)` samples, indexed by integer.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The `idx`-th sample. Panics if `idx >= len()`.
    fn get(&self, idx: usize) -> (Tensor, Tensor);
}

/// An in-memory dataset backed by parallel `Vec`s of tensors (one per sample).
pub struct TensorDataset {
    inputs: Vec<Tensor>,
    targets: Vec<Tensor>,
}

impl TensorDataset {
    pub fn new(inputs: Vec<Tensor>, targets: Vec<Tensor>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "TensorDataset: inputs/targets length mismatch"
        );
        TensorDataset { inputs, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.inputs[idx].clone(), self.targets[idx].clone())
    }
}

/// Stack `tensors` into a single `[B, ...]` batch tensor (each sample keeps its
/// shape as the trailing dims). All samples must share the same shape.
pub fn stack(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "stack: empty input");
    let sample_shape = tensors[0].shape().to_vec();
    let per = tensors[0].numel();
    let mut flat = Vec::with_capacity(per * tensors.len());
    for t in tensors {
        assert_eq!(
            t.shape(),
            sample_shape.as_slice(),
            "stack: sample shapes differ"
        );
        flat.extend_from_slice(&t.to_vec::<f64>().unwrap());
    }
    let mut batch_shape = vec![tensors.len()];
    batch_shape.extend_from_slice(&sample_shape);
    Tensor::from_typed(flat).reshape(&batch_shape).unwrap()
}

/// Deterministic Fisher–Yates shuffle (LCG-seeded; no external RNG dependency).
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in (1..n).rev() {
        let j = (rng() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// An iterator over a [`Dataset`], yielding `(inputs, targets)` batches where
/// each is a stacked `[B, ...]` tensor.
pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    order: Vec<usize>,
    pos: usize,
    epoch: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let order = if shuffle {
            shuffled_indices(dataset.len(), 0)
        } else {
            (0..dataset.len()).collect()
        };
        DataLoader {
            dataset,
            batch_size: batch_size.max(1),
            shuffle,
            order,
            pos: 0,
            epoch: 0,
        }
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }
}

impl<'a, D: Dataset> Iterator for DataLoader<'a, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.order.len());
        let idxs = &self.order[self.pos..end];
        let mut xs = Vec::with_capacity(idxs.len());
        let mut ys = Vec::with_capacity(idxs.len());
        for &i in idxs {
            let (x, y) = self.dataset.get(i);
            xs.push(x);
            ys.push(y);
        }
        self.pos = end;
        Some((stack(&xs), stack(&ys)))
    }
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// Advance to the next epoch, reshuffling the order if requested.
    pub fn reset(&mut self) {
        self.epoch += 1;
        self.pos = 0;
        if self.shuffle {
            self.order = shuffled_indices(self.dataset.len(), self.epoch as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_tensor::Tensor;

    fn make_dataset(n: usize) -> TensorDataset {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            xs.push(Tensor::from_typed(vec![i as f64]).reshape(&[1]).unwrap());
            ys.push(
                Tensor::from_typed(vec![(i * 2) as f64])
                    .reshape(&[1])
                    .unwrap(),
            );
        }
        TensorDataset::new(xs, ys)
    }

    #[test]
    fn batches_have_correct_shape_and_count() {
        let ds = make_dataset(5);
        let dl = DataLoader::new(&ds, 2, false);
        assert_eq!(dl.num_batches(), 3);
        let batches: Vec<_> = DataLoader::new(&ds, 2, false).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.shape(), &[2, 1]);
        assert_eq!(batches[2].0.shape(), &[1, 1]);
        // no shuffle -> first batch is samples 0,1
        assert_eq!(batches[0].0.to_vec::<f64>().unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn shuffle_covers_all_indices() {
        let ds = make_dataset(6);
        let mut dl = DataLoader::new(&ds, 2, true);
        let mut seen = std::collections::HashSet::new();
        for (x, _) in &mut dl {
            for v in x.to_vec::<f64>().unwrap() {
                seen.insert(v as usize);
            }
        }
        assert_eq!(seen.len(), 6);
        dl.reset();
        // after reset a new shuffled epoch is available
        assert!(dl.num_batches() >= 1);
    }

    #[test]
    fn stack_preserves_values() {
        let ts = vec![
            Tensor::from_typed(vec![1.0_f64, 2.0])
                .reshape(&[2])
                .unwrap(),
            Tensor::from_typed(vec![3.0_f64, 4.0])
                .reshape(&[2])
                .unwrap(),
        ];
        let s = stack(&ts);
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
