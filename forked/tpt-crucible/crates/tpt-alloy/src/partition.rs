use crate::topology::Topology;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PartitionStrategy {
    /// Standard layer-wise round-robin across nodes
    Layer,
    /// Distribute attention heads across nodes; each node computes a subset of heads
    HeadParallel,
    /// Head-parallel for attention sublayers, layer-serial for FFN sublayers
    Hybrid,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::Layer
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PartitionConfig {
    pub topology: Topology,
    pub max_layers_per_node: usize,
    pub minimize_cross_node_edges: bool,
    pub strategy: PartitionStrategy,
    /// Number of attention heads in the model (used for head-parallel / hybrid)
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Op types that identify transformer attention sublayers in TPT-IR
    pub attention_op_types: Vec<String>,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            topology: Topology::Grid2D { rows: 4, cols: 4 },
            max_layers_per_node: 4,
            minimize_cross_node_edges: true,
            strategy: PartitionStrategy::Layer,
            num_heads: 32,
            head_dim: 64,
            attention_op_types: vec![
                "attention".into(),
                "self_attn".into(),
                "multi_head_attention".into(),
                "mha".into(),
                "qkv".into(),
                "attn".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Partition {
    pub node_id: usize,
    pub assigned_layers: Vec<usize>,
    pub cross_node_edges: Vec<CrossEdge>,
    /// For head-parallel / hybrid: which head indices this node is responsible for
    pub assigned_heads: Vec<usize>,
    /// Whether this node performs sum-reduce aggregation for head outputs
    pub is_aggregator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrossEdge {
    pub from_node: usize,
    pub to_node: usize,
    pub tensor_name: String,
    /// Communication protocol for this edge (e.g. "sum_reduce" for head aggregation)
    pub protocol: String,
}

/// Analyze a list of op_type strings to detect which layer indices are attention sublayers.
pub fn detect_attention_layers(
    op_types: &[String],
    attention_keywords: &[String],
) -> Vec<bool> {
    op_types
        .iter()
        .map(|op| {
            let lower = op.to_lowercase();
            attention_keywords
                .iter()
                .any(|kw| lower.contains(&kw.to_lowercase()))
        })
        .collect()
}

/// Partition a model across swarm nodes according to the configured strategy.
pub fn partition_model(
    layer_count: usize,
    config: &PartitionConfig,
    op_types: Option<&[String]>,
) -> Vec<Partition> {
    match config.strategy {
        PartitionStrategy::Layer => partition_layer_serial(layer_count, config),
        PartitionStrategy::HeadParallel => {
            partition_head_parallel(layer_count, config, op_types)
        }
        PartitionStrategy::Hybrid => partition_hybrid(layer_count, config, op_types),
    }
}

/// Standard layer-serial (round-robin) partitioning.
fn partition_layer_serial(layer_count: usize, config: &PartitionConfig) -> Vec<Partition> {
    let node_count = config.topology.node_count();
    let mut partitions: Vec<Partition> = (0..node_count)
        .map(|id| Partition {
            node_id: id,
            assigned_layers: Vec::new(),
            cross_node_edges: Vec::new(),
            assigned_heads: Vec::new(),
            is_aggregator: false,
        })
        .collect();

    for layer_id in 0..layer_count {
        let node_idx = layer_id % node_count;
        partitions[node_idx].assigned_layers.push(layer_id);
    }

    partitions
}

/// Head-parallel partitioning: distribute attention heads across nodes.
/// Each node gets a subset of heads for every attention layer.
/// Non-attention layers are assigned round-robin.
fn partition_head_parallel(
    layer_count: usize,
    config: &PartitionConfig,
    op_types: Option<&[String]>,
) -> Vec<Partition> {
    let node_count = config.topology.node_count();
    let is_attention = match op_types {
        Some(types) => detect_attention_layers(types, &config.attention_op_types),
        None => vec![false; layer_count],
    };

    let mut partitions: Vec<Partition> = (0..node_count)
        .map(|id| Partition {
            node_id: id,
            assigned_layers: Vec::new(),
            cross_node_edges: Vec::new(),
            assigned_heads: Vec::new(),
            is_aggregator: id == 0, // node 0 is the default aggregator
        })
        .collect();

    // Distribute heads across nodes
    let heads_per_node = if node_count > 0 {
        (config.num_heads + node_count - 1) / node_count
    } else {
        config.num_heads
    };

    for node_id in 0..node_count {
        let start = node_id * heads_per_node;
        let end = (start + heads_per_node).min(config.num_heads);
        for h in start..end {
            partitions[node_id].assigned_heads.push(h);
        }
    }

    // Assign layers: attention layers go to all nodes (head-parallel),
    // non-attention layers are round-robin
    for layer_id in 0..layer_count {
        if layer_id < is_attention.len() && is_attention[layer_id] {
            // Attention layer: all nodes participate with their assigned heads
            for node_id in 0..node_count {
                partitions[node_id].assigned_layers.push(layer_id);
            }
        } else {
            // Non-attention layer: round-robin
            let node_idx = layer_id % node_count;
            partitions[node_idx].assigned_layers.push(layer_id);
        }
    }

    // Add sum-reduce cross edges for attention layers
    for layer_id in 0..layer_count {
        if layer_id < is_attention.len() && is_attention[layer_id] {
            for node_id in 1..node_count {
                if !partitions[node_id].assigned_heads.is_empty() {
                    partitions[node_id]
                        .cross_node_edges
                        .push(CrossEdge {
                            from_node: node_id,
                            to_node: 0,
                            tensor_name: format!("sum_reduce_attn_{}", layer_id),
                            protocol: "sum_reduce".into(),
                        });
                }
            }
        }
    }

    partitions
}

/// Hybrid partitioning: head-parallel for attention sublayers,
/// layer-serial for FFN sublayers.
fn partition_hybrid(
    layer_count: usize,
    config: &PartitionConfig,
    op_types: Option<&[String]>,
) -> Vec<Partition> {
    let node_count = config.topology.node_count();
    let is_attention = match op_types {
        Some(types) => detect_attention_layers(types, &config.attention_op_types),
        None => vec![false; layer_count],
    };

    let mut partitions: Vec<Partition> = (0..node_count)
        .map(|id| Partition {
            node_id: id,
            assigned_layers: Vec::new(),
            cross_node_edges: Vec::new(),
            assigned_heads: Vec::new(),
            is_aggregator: id == 0,
        })
        .collect();

    // Distribute heads across nodes
    let heads_per_node = if node_count > 0 {
        (config.num_heads + node_count - 1) / node_count
    } else {
        config.num_heads
    };

    for node_id in 0..node_count {
        let start = node_id * heads_per_node;
        let end = (start + heads_per_node).min(config.num_heads);
        for h in start..end {
            partitions[node_id].assigned_heads.push(h);
        }
    }

    for layer_id in 0..layer_count {
        if layer_id < is_attention.len() && is_attention[layer_id] {
            // Attention layer: head-parallel — all nodes participate
            for node_id in 0..node_count {
                partitions[node_id].assigned_layers.push(layer_id);
            }
        } else {
            // FFN / other layer: layer-serial — round-robin to one node
            let node_idx = layer_id % node_count;
            partitions[node_idx].assigned_layers.push(layer_id);
        }
    }

    // Add sum-reduce cross edges for attention layers
    for layer_id in 0..layer_count {
        if layer_id < is_attention.len() && is_attention[layer_id] {
            for node_id in 1..node_count {
                if !partitions[node_id].assigned_heads.is_empty() {
                    partitions[node_id]
                        .cross_node_edges
                        .push(CrossEdge {
                            from_node: node_id,
                            to_node: 0,
                            tensor_name: format!("sum_reduce_attn_{}", layer_id),
                            protocol: "sum_reduce".into(),
                        });
                }
            }
        }
    }

    partitions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_attention_layers() {
        let op_types = vec![
            "self_attention".into(),
            "linear".into(),
            "multi_head_attention".into(),
            "gelu".into(),
            "attention".into(),
        ];
        let keywords = vec![
            "attention".into(),
            "self_attn".into(),
            "mha".into(),
        ];
        let result = detect_attention_layers(&op_types, &keywords);
        assert_eq!(result, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_partition_layer_serial() {
        let config = PartitionConfig {
            topology: Topology::Grid2D { rows: 2, cols: 2 },
            strategy: PartitionStrategy::Layer,
            ..Default::default()
        };
        let partitions = partition_model(6, &config, None);
        assert_eq!(partitions.len(), 4);
        // 6 layers round-robin across 4 nodes: 0,1,2,3,0,1
        assert_eq!(partitions[0].assigned_layers, vec![0, 4]);
        assert_eq!(partitions[1].assigned_layers, vec![1, 5]);
        assert_eq!(partitions[2].assigned_layers, vec![2]);
        assert_eq!(partitions[3].assigned_layers, vec![3]);
    }

    #[test]
    fn test_partition_head_parallel() {
        let config = PartitionConfig {
            topology: Topology::Grid2D { rows: 1, cols: 4 },
            strategy: PartitionStrategy::HeadParallel,
            num_heads: 8,
            head_dim: 64,
            ..Default::default()
        };
        let op_types: Vec<String> = (0..4)
            .map(|i| {
                if i % 2 == 0 {
                    "self_attention".into()
                } else {
                    "linear".into()
                }
            })
            .collect();

        let partitions = partition_model(4, &config, Some(&op_types));
        assert_eq!(partitions.len(), 4);

        // Heads: 8 heads across 4 nodes = 2 per node
        assert_eq!(partitions[0].assigned_heads, vec![0, 1]);
        assert_eq!(partitions[1].assigned_heads, vec![2, 3]);
        assert_eq!(partitions[2].assigned_heads, vec![4, 5]);
        assert_eq!(partitions[3].assigned_heads, vec![6, 7]);

        // Attention layers (0, 2) should be on all nodes
        assert!(partitions[0].assigned_layers.contains(&0));
        assert!(partitions[1].assigned_layers.contains(&0));
        assert!(partitions[2].assigned_layers.contains(&0));
        assert!(partitions[3].assigned_layers.contains(&0));

        // Non-attention layers (1, 3) should be round-robin
        assert!(partitions[1].assigned_layers.contains(&1));
        assert!(partitions[3].assigned_layers.contains(&3));

        // Sum-reduce edges should exist from non-aggregator nodes to node 0
        assert!(partitions[1].cross_node_edges.iter().any(|e| e.protocol == "sum_reduce"));
        assert!(partitions[2].cross_node_edges.iter().any(|e| e.protocol == "sum_reduce"));
        assert!(partitions[3].cross_node_edges.iter().any(|e| e.protocol == "sum_reduce"));
    }

    #[test]
    fn test_partition_hybrid() {
        let config = PartitionConfig {
            topology: Topology::Grid2D { rows: 1, cols: 2 },
            strategy: PartitionStrategy::Hybrid,
            num_heads: 4,
            head_dim: 64,
            ..Default::default()
        };
        let op_types: Vec<String> = vec![
            "self_attention".into(),
            "linear".into(),
            "attention".into(),
            "gelu".into(),
        ];

        let partitions = partition_model(4, &config, Some(&op_types));
        assert_eq!(partitions.len(), 2);

        // Attention layers (0, 2) on all nodes
        assert!(partitions[0].assigned_layers.contains(&0));
        assert!(partitions[1].assigned_layers.contains(&0));
        assert!(partitions[0].assigned_layers.contains(&2));
        assert!(partitions[1].assigned_layers.contains(&2));

        // Non-attention layers (1, 3) round-robin across 2 nodes:
        // layer 1 -> 1 % 2 = 1 (node 1)
        // layer 3 -> 3 % 2 = 1 (node 1)
        assert!(partitions[1].assigned_layers.contains(&1));
        assert!(partitions[1].assigned_layers.contains(&3));
    }
}