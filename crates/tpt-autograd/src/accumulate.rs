//! Cross-device gradient accumulation (Phase 4, spec §5.5).
//!
//! Data-parallel training produces one gradient vector per device per step;
//! the host must sum them into a single vector before the optimizer applies
//! it. [`GradAccumulator`] models exactly that hand-off: workers call
//! [`GradAccumulator::accumulate`] with their device and gradient vector (in
//! the model's stable parameter order), and [`GradAccumulator::reduce`] sums
//! across devices per parameter index.
//!
//! On the current host-only runtime every `Device` maps to CPU memory, so
//! `reduce` is a plain element-wise sum; when real device backends land, the
//! same API becomes the place where device-to-device copies are inserted.

use std::collections::BTreeMap;

use tpt_tensor::{Device, Tensor};

use crate::add;

/// Accumulated, per-device gradient sets.
#[derive(Default)]
pub struct GradAccumulator {
    /// device label -> gradient vector (stable parameter order)
    per_device: BTreeMap<String, Vec<Tensor>>,
}

impl GradAccumulator {
    pub fn new() -> Self {
        GradAccumulator {
            per_device: BTreeMap::new(),
        }
    }

    /// Record one worker's gradients. `grads` must be in the same parameter
    /// order for every device that participates in the reduce.
    pub fn accumulate(&mut self, device: Device, grads: Vec<Tensor>) {
        self.per_device.insert(format!("{device}"), grads);
    }

    /// Number of devices that have contributed gradients.
    pub fn device_count(&self) -> usize {
        self.per_device.len()
    }

    /// Sum across devices per parameter index. The result lives on the host
    /// and carries no tape (gradients are leaves by the time they are reduced).
    ///
    /// # Panics
    /// Panics if contributing devices supplied different parameter counts.
    pub fn reduce(&self) -> Vec<Tensor> {
        let mut iter = self.per_device.values();
        let mut acc: Vec<Tensor> = match iter.next() {
            Some(v) => v.clone(),
            None => return Vec::new(),
        };
        for grads in iter {
            assert_eq!(
                acc.len(),
                grads.len(),
                "cross-device reduce: parameter count mismatch"
            );
            for (a, g) in acc.iter_mut().zip(grads) {
                *a = add(a, g);
            }
        }
        acc
    }

    /// Drop all accumulated gradients (call after the optimizer step).
    pub fn clear(&mut self) {
        self.per_device.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_sums_across_devices() {
        let mut acc = GradAccumulator::new();
        acc.accumulate(Device::Cpu, vec![Tensor::from_typed(vec![1.0_f64, 2.0])]);
        acc.accumulate(Device::Wgpu, vec![Tensor::from_typed(vec![10.0_f64, 20.0])]);
        assert_eq!(acc.device_count(), 2);
        let reduced = acc.reduce();
        assert_eq!(reduced.len(), 1);
        assert_eq!(reduced[0].to_vec::<f64>().unwrap(), vec![11.0, 22.0]);
    }

    #[test]
    fn reduce_with_single_device_is_identity() {
        let mut acc = GradAccumulator::new();
        acc.accumulate(
            Device::Cpu,
            vec![
                Tensor::from_typed(vec![3.0_f64]),
                Tensor::from_typed(vec![4.0_f64]),
            ],
        );
        let reduced = acc.reduce();
        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced[0].to_vec::<f64>().unwrap(), vec![3.0]);
        assert_eq!(reduced[1].to_vec::<f64>().unwrap(), vec![4.0]);
    }

    #[test]
    fn clear_resets_and_reduce_empty_is_empty() {
        let mut acc = GradAccumulator::new();
        assert!(acc.reduce().is_empty());
        acc.accumulate(Device::Cpu, vec![Tensor::from_typed(vec![1.0_f64])]);
        acc.clear();
        assert_eq!(acc.device_count(), 0);
        assert!(acc.reduce().is_empty());
    }

    #[test]
    #[should_panic(expected = "parameter count mismatch")]
    fn reduce_rejects_mismatched_param_counts() {
        let mut acc = GradAccumulator::new();
        acc.accumulate(Device::Cpu, vec![Tensor::from_typed(vec![1.0_f64])]);
        acc.accumulate(
            Device::Wgpu,
            vec![
                Tensor::from_typed(vec![1.0_f64]),
                Tensor::from_typed(vec![2.0_f64]),
            ],
        );
        acc.reduce();
    }
}
