//! Surgical Pruner — zeros FFN neurons associated with unwanted domains.
//!
//! Uses structural pruning (whole neurons = full rows/columns of FFN weight
//! matrices) so downstream tensor shapes remain valid. Records zeroed indices
//! in a `PruningMask` that is embedded in the output `.tptf` file.

use crate::domain_mapper::DomainMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse record of which neurons were zeroed, one Vec per layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PruningMask {
    /// layer_idx → sorted list of zeroed neuron indices (in FFN intermediate dim)
    pub zeroed: HashMap<usize, Vec<usize>>,
    pub num_layers: usize,
}

impl PruningMask {
    pub fn total_pruned(&self) -> usize {
        self.zeroed.values().map(|v| v.len()).sum()
    }

    pub fn is_neuron_pruned(&self, layer: usize, neuron: usize) -> bool {
        self.zeroed
            .get(&layer)
            .map(|v| v.binary_search(&neuron).is_ok())
            .unwrap_or(false)
    }
}

/// Performs domain-targeted structural pruning.
pub struct SurgicalPruner {
    /// Domains to remove (e.g. `["sql", "typescript"]`).
    domains_to_drop: Vec<String>,
    /// Minimum importance score for a neuron to be considered domain-specific.
    importance_threshold: f32,
}

impl SurgicalPruner {
    pub fn new(domains_to_drop: Vec<String>) -> Self {
        SurgicalPruner {
            domains_to_drop,
            importance_threshold: 0.05,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.importance_threshold = threshold;
        self
    }

    /// Build a pruning mask from a domain map.
    ///
    /// For each layer, marks neurons where:
    /// - Their primary domain is in `domains_to_drop`.
    /// - Their importance score exceeds `importance_threshold`.
    ///
    /// In production: after building the mask, the caller applies it to the
    /// weight tensors by zeroing the corresponding rows in `gate_proj`,
    /// `up_proj`, and the corresponding columns in `down_proj`.
    pub fn build_mask(&self, domain_map: &DomainMap) -> Result<PruningMask> {
        let mut zeroed: HashMap<usize, Vec<usize>> = HashMap::new();

        for (layer_idx, layer_scores) in &domain_map.scores {
            let mut pruned: Vec<usize> = layer_scores
                .iter()
                .filter(|s| {
                    self.domains_to_drop.contains(&s.domain)
                        && s.importance >= self.importance_threshold
                })
                .map(|s| s.neuron_idx)
                .collect();
            pruned.sort_unstable();
            pruned.dedup();
            if !pruned.is_empty() {
                zeroed.insert(*layer_idx, pruned);
            }
        }

        Ok(PruningMask {
            zeroed,
            num_layers: domain_map.num_layers,
        })
    }

    /// Apply the pruning mask to a flat weight buffer (in-place).
    ///
    /// For FFN gate/up projections `[hidden_dim, ffn_dim]`: zeros column `col`
    /// for each pruned neuron index. For down projection `[ffn_dim, hidden_dim]`:
    /// zeros row `row` for each pruned neuron index.
    pub fn apply_to_weights(
        mask: &PruningMask,
        layer: usize,
        weights: &mut [f32],
        rows: usize,
        cols: usize,
        pruned_dim: PrunedDim,
    ) {
        let Some(pruned_indices) = mask.zeroed.get(&layer) else {
            return;
        };
        for &idx in pruned_indices {
            match pruned_dim {
                PrunedDim::Column => {
                    // Zero column `idx` across all rows
                    for row in 0..rows {
                        let flat = row * cols + idx;
                        if flat < weights.len() {
                            weights[flat] = 0.0;
                        }
                    }
                }
                PrunedDim::Row => {
                    // Zero row `idx` across all columns
                    let start = idx * cols;
                    let end = (start + cols).min(weights.len());
                    if start < weights.len() {
                        weights[start..end].fill(0.0);
                    }
                }
            }
        }
    }
}

/// Whether pruning targets a column (gate/up) or row (down) in the FFN.
#[derive(Debug, Clone, Copy)]
pub enum PrunedDim {
    Column,
    Row,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_mapper::DomainMapper;

    #[test]
    fn mask_zeros_target_domain() {
        let mapper = DomainMapper::new(vec!["sql".to_string(), "python".to_string()]);
        let domain_map = mapper.build(4, 32).unwrap();

        let pruner = SurgicalPruner::new(vec!["sql".to_string()]);
        let mask = pruner.build_mask(&domain_map).unwrap();

        // Some neurons should be marked for pruning
        let total = mask.total_pruned();
        assert!(total > 0, "expected some neurons pruned, got 0");
    }

    #[test]
    fn apply_zeros_column() {
        let mask = {
            let mut m = PruningMask::default();
            m.zeroed.insert(0, vec![1]); // prune neuron index 1
            m.num_layers = 1;
            m
        };
        let mut w = vec![1.0f32; 4 * 4]; // 4×4 matrix
        SurgicalPruner::apply_to_weights(&mask, 0, &mut w, 4, 4, PrunedDim::Column);
        // Column 1 should be zero: indices 1, 5, 9, 13
        assert_eq!(w[1], 0.0);
        assert_eq!(w[5], 0.0);
        assert_eq!(w[9], 0.0);
        assert_eq!(w[13], 0.0);
        // Column 0 should be untouched
        assert_eq!(w[0], 1.0);
    }
}
