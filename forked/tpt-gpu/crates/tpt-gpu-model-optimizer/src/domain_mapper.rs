//! Domain Knowledge Mapper — identifies which neurons handle specific domains.
//!
//! Algorithm (Wanda-style, gradient-free):
//! 1. Run calibration prompts from each domain through the model.
//! 2. Capture per-layer FFN activation magnitudes.
//! 3. Score each neuron: importance = |weight| × mean(|activation|).
//! 4. Cluster neuron importance scores by domain using cosine similarity.
//! 5. Produce a DomainMap: layer → [(neuron_idx, domain, importance_score)].

use crate::activation_capture::ActivationMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// Supported analysis domains.
pub const KNOWN_DOMAINS: &[&str] = &[
    "python",
    "typescript",
    "sql",
    "math",
    "reasoning",
    "code",
    "general",
    "science",
    "creative",
];

/// Importance score for a single neuron in a specific domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronDomainScore {
    pub neuron_idx: usize,
    pub domain: String,
    /// Combined importance: |weight_norm| × mean(|activation|)
    pub importance: f32,
}

/// Full domain map: layer_idx → ranked neuron scores per domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMap {
    /// layer_idx → Vec<NeuronDomainScore>
    pub scores: HashMap<usize, Vec<NeuronDomainScore>>,
    pub num_layers: usize,
}

impl DomainMap {
    /// Return neuron indices in `layer` that are primarily associated with `domain`.
    ///
    /// Returns neurons where the given domain's importance exceeds `threshold`
    /// AND is the dominant domain for that neuron.
    pub fn domain_neurons(&self, layer: usize, domain: &str, threshold: f32) -> Vec<usize> {
        self.scores
            .get(&layer)
            .map(|scores| {
                scores
                    .iter()
                    .filter(|s| s.domain == domain && s.importance >= threshold)
                    .map(|s| s.neuron_idx)
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Runs domain analysis on a model.
pub struct DomainMapper {
    domains: Vec<String>,
}

impl DomainMapper {
    pub fn new(domains: Vec<String>) -> Self {
        DomainMapper { domains }
    }

    pub fn with_default_domains() -> Self {
        Self::new(KNOWN_DOMAINS.iter().map(|s| s.to_string()).collect())
    }

    /// Analyze model activations and build a domain map.
    ///
    /// `num_layers` — transformer layer count from `ModelInfo`.
    /// `ffn_dim`    — FFN intermediate dimension (neurons per layer to analyze).
    ///
    /// In production: loads the model, runs domain-specific calibration prompts,
    /// hooks into the forward pass to capture activation tensors, then clusters
    /// by domain using cosine similarity of importance vectors.
    /// This implementation produces a heuristic map for scaffold purposes.
    pub fn build(&self, num_layers: usize, ffn_dim: usize) -> Result<DomainMap> {
        let mut scores: HashMap<usize, Vec<NeuronDomainScore>> = HashMap::new();

        for layer in 0..num_layers {
            let mut layer_scores: Vec<NeuronDomainScore> = Vec::new();
            for neuron in 0..ffn_dim {
                // Heuristic: assign domain based on a simple deterministic pattern.
                // Production: use real activation statistics from calibration runs.
                let domain_idx = (layer * ffn_dim + neuron) % self.domains.len();
                let domain = self.domains[domain_idx].clone();
                let importance = 0.1 + (neuron % 10) as f32 * 0.01;
                layer_scores.push(NeuronDomainScore {
                    neuron_idx: neuron,
                    domain,
                    importance,
                });
            }
            scores.insert(layer, layer_scores);
        }

        Ok(DomainMap { scores, num_layers })
    }

    /// Build domain map from real captured activations.
    ///
    /// This is the production path that uses actual activation magnitudes
    /// from the model to score neurons.
    pub fn build_from_activations(
        activation_map: &ActivationMap,
        weight_importance: &[Vec<f32>], // per-layer weight magnitudes
    ) -> Result<DomainMap> {
        let mut scores: HashMap<usize, Vec<NeuronDomainScore>> = HashMap::new();
        let domains = KNOWN_DOMAINS
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        for (layer_idx, layer_acts) in &activation_map.layers {
            let mut layer_scores: Vec<NeuronDomainScore> = Vec::new();

            for neuron_idx in 0..layer_acts.mean_magnitudes.len() {
                let weight_mag = weight_importance
                    .get(*layer_idx)
                    .map(|w| w.get(neuron_idx).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let act_mag = layer_acts
                    .mean_abs()
                    .get(neuron_idx)
                    .copied()
                    .unwrap_or(0.0);

                // Wanda-style importance: weight magnitude × activation magnitude
                let importance = weight_mag * act_mag;

                // Assign to domain with highest similarity (simplified: modulo)
                let domain_idx = neuron_idx % domains.len();
                let domain = domains[domain_idx].clone();

                layer_scores.push(NeuronDomainScore {
                    neuron_idx,
                    domain,
                    importance,
                });
            }

            scores.insert(*layer_idx, layer_scores);
        }

        Ok(DomainMap {
            scores,
            num_layers: activation_map.layers.len(),
        })
    }

    /// Build domain map from per-domain captured activations (production path).
    ///
    /// For each neuron in each layer, computes a Wanda-style importance score
    /// `|weight| × mean(|activation|)` per domain and assigns the neuron to the
    /// domain that activates it most strongly (argmax). Neurons with negligible
    /// importance in every domain are assigned to `"general"` so they survive
    /// surgical pruning.
    ///
    /// `weight_importance` — optional per-layer per-neuron weight magnitudes
    /// (`weight_importance[layer][neuron]`). When `None`, activation magnitudes
    /// alone are used (weight factor 1.0). Layer/FFN dimensions are inferred
    /// from the activation maps themselves.
    pub fn build_from_domain_activations(
        domain_activations: &HashMap<String, ActivationMap>,
        weight_importance: Option<&[Vec<f32>]>,
    ) -> Result<DomainMap> {
        let ffn_dim = domain_activations
            .values()
            .map(|m| m.ffn_dim)
            .max()
            .unwrap_or(0);

        // All layers observed across every domain map.
        let mut layer_set: BTreeSet<usize> = BTreeSet::new();
        for map in domain_activations.values() {
            layer_set.extend(map.layers.keys().copied());
        }
        if let Some(w) = weight_importance {
            layer_set.extend(0..w.len());
        }

        let mut scores: HashMap<usize, Vec<NeuronDomainScore>> = HashMap::new();

        for layer_idx in &layer_set {
            // neuron_idx → (domain, importance) accumulation
            let mut per_neuron: HashMap<usize, Vec<(String, f32)>> = HashMap::new();

            for (domain, map) in domain_activations {
                let Some(layer_acts) = map.layers.get(layer_idx) else {
                    continue;
                };
                for (neuron_idx, act_mag) in layer_acts.mean_abs().iter().enumerate() {
                    let weight_mag = weight_importance
                        .and_then(|w| w.get(*layer_idx))
                        .and_then(|layer_w| layer_w.get(neuron_idx))
                        .copied()
                        .unwrap_or(1.0);
                    let importance = weight_mag * act_mag;
                    per_neuron
                        .entry(neuron_idx)
                        .or_default()
                        .push((domain.clone(), importance));
                }
            }

            let neuron_count = per_neuron
                .keys()
                .max()
                .map(|&m| m + 1)
                .unwrap_or(0)
                .max(ffn_dim);

            let mut layer_scores: Vec<NeuronDomainScore> = Vec::with_capacity(neuron_count);
            for neuron_idx in 0..neuron_count {
                let domain_scores = per_neuron.remove(&neuron_idx).unwrap_or_default();
                let best = domain_scores
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let (domain, importance) = match best {
                    Some((d, imp)) if imp.is_finite() && *imp > 0.0 => (d.clone(), *imp),
                    _ => ("general".to_string(), 0.0),
                };
                layer_scores.push(NeuronDomainScore {
                    neuron_idx,
                    domain,
                    importance,
                });
            }

            scores.insert(*layer_idx, layer_scores);
        }

        Ok(DomainMap {
            scores,
            num_layers: layer_set.len(),
        })
    }

    /// Build domain map from tensor weights using heuristic.
    /// Used when no activation data is available.
    pub fn build_from_weights(
        &self,
        weights_by_layer: &[Vec<f32>],
        ffn_dim: usize,
    ) -> Result<DomainMap> {
        let mut scores: HashMap<usize, Vec<NeuronDomainScore>> = HashMap::new();

        for (layer_idx, weights) in weights_by_layer.iter().enumerate() {
            let mut layer_scores: Vec<NeuronDomainScore> = Vec::new();

            for neuron_idx in 0..ffn_dim {
                // Find weight magnitude for this neuron
                let weight_mag =
                    weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len().max(1) as f32;

                // Heuristic domain assignment
                let domain_idx = (layer_idx * ffn_dim + neuron_idx) % self.domains.len();
                let domain = self.domains[domain_idx].clone();

                // Importance combines weight and heuristic activation
                let importance = weight_mag * (0.1 + (neuron_idx % 10) as f32 * 0.01);

                layer_scores.push(NeuronDomainScore {
                    neuron_idx,
                    domain,
                    importance,
                });
            }

            scores.insert(layer_idx, layer_scores);
        }

        Ok(DomainMap {
            num_layers: weights_by_layer.len(),
            scores,
        })
    }
}

/// Compute weight magnitude importance for Wanda-style scoring
pub fn compute_weight_importance(weights: &[f32], ffn_dim: usize) -> Vec<f32> {
    // For FFN gate_proj weights [hidden_dim, ffn_dim]:
    // We want per-neuron importance: sum of absolute values across input dim
    let hidden_dim = weights.len() / ffn_dim.max(1);
    let mut importance = vec![0.0f32; ffn_dim];

    for (col, imp) in importance.iter_mut().enumerate() {
        for row in 0..hidden_dim {
            let idx = row * ffn_dim + col;
            if idx < weights.len() {
                *imp += weights[idx].abs();
            }
        }
    }

    // Normalize
    let max_imp: f32 = importance.iter().copied().fold(0.0f32, f32::max);
    if max_imp > 0.0 {
        for imp in &mut importance {
            *imp /= max_imp;
        }
    }

    importance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_domain_map() {
        let mapper = DomainMapper::with_default_domains();
        let map = mapper.build(4, 64).unwrap();
        assert_eq!(map.num_layers, 4);
        assert!(map.scores.contains_key(&0));
        assert_eq!(map.scores[&0].len(), 64);
    }

    #[test]
    fn domain_neurons_filters_by_threshold() {
        let mapper = DomainMapper::new(vec!["sql".to_string()]);
        let map = mapper.build(2, 16).unwrap();
        let neurons = map.domain_neurons(0, "sql", 0.0);
        assert!(!neurons.is_empty());
    }

    #[test]
    fn build_from_activations() {
        use crate::activation_capture::LayerActivations;

        let mut act_map = ActivationMap {
            ffn_dim: 4,
            ..Default::default()
        };
        act_map.layers.insert(
            0,
            LayerActivations {
                layer_idx: 0,
                mean_magnitudes: vec![1.0, 2.0, 3.0, 4.0],
                stddev_magnitudes: vec![0.1, 0.2, 0.3, 0.4],
                sample_count: 10,
            },
        );

        let weight_importance = vec![vec![0.5, 0.6, 0.7, 0.8]];
        let domain_map =
            DomainMapper::build_from_activations(&act_map, &weight_importance).unwrap();

        assert!(domain_map.scores.contains_key(&0));
    }

    #[test]
    fn build_from_domain_activations_assigns_argmax_domain() {
        use crate::activation_capture::LayerActivations;

        let mut sql = ActivationMap {
            ffn_dim: 4,
            ..Default::default()
        };
        sql.layers.insert(
            0,
            LayerActivations {
                layer_idx: 0,
                mean_magnitudes: vec![0.0, 2.0, 0.0, 0.0],
                stddev_magnitudes: vec![0.0, 0.1, 0.0, 0.0],
                sample_count: 10,
            },
        );

        let mut python = ActivationMap {
            ffn_dim: 4,
            ..Default::default()
        };
        python.layers.insert(
            0,
            LayerActivations {
                layer_idx: 0,
                mean_magnitudes: vec![0.0, 0.0, 3.0, 0.0],
                stddev_magnitudes: vec![0.0, 0.0, 0.1, 0.0],
                sample_count: 10,
            },
        );

        let mut domain_activations = HashMap::new();
        domain_activations.insert("sql".to_string(), sql);
        domain_activations.insert("python".to_string(), python);

        let weight_importance = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let map = DomainMapper::build_from_domain_activations(
            &domain_activations,
            Some(&weight_importance),
        )
        .unwrap();

        assert_eq!(map.num_layers, 1);
        let layer = &map.scores[&0];
        assert_eq!(layer.len(), 4);
        // Neuron 1 fires on sql prompts → sql; neuron 2 fires on python → python.
        assert_eq!(layer[1].domain, "sql");
        assert_eq!(layer[2].domain, "python");
        // Inactive neurons fall back to general.
        assert_eq!(layer[0].domain, "general");
        assert_eq!(layer[3].domain, "general");
        // Neuron 3 (python) has the highest importance.
        assert!(layer[2].importance > layer[1].importance);
    }

    #[test]
    fn build_from_domain_activations_empty_input_is_safe() {
        let domain_activations = HashMap::new();
        let map = DomainMapper::build_from_domain_activations(&domain_activations, None).unwrap();
        assert_eq!(map.num_layers, 0);
        assert!(map.scores.is_empty());
    }

    #[test]
    fn weight_importance_computation() {
        let weights = vec![-1.0, 0.0, 2.0, 1.0, 0.5, -0.5, 1.5, 2.5];
        let importance = compute_weight_importance(&weights, 4);
        assert_eq!(importance.len(), 4);
        // Column sums: 1.5, 0.5, 3.5, 3.5, normalized by max=3.5
        // importance[0] = 1.5 / 3.5 ≈ 0.43
        assert!((importance[0] - 1.5 / 3.5).abs() < 0.01);
        // importance[2] = 3.5 / 3.5 = 1.0 (max)
        assert!((importance[2] - 1.0).abs() < 0.01);
    }
}
