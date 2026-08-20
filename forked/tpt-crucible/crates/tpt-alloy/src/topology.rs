use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Topology {
    Grid2D { rows: usize, cols: usize },
    Star { center: usize, leaves: usize },
    Ring { size: usize },
    Custom { adjacency: Vec<Vec<usize>> },
}

impl Topology {
    pub fn node_count(&self) -> usize {
        match self {
            Topology::Grid2D { rows, cols } => rows * cols,
            Topology::Star { leaves, .. } => leaves + 1,
            Topology::Ring { size } => *size,
            Topology::Custom { adjacency } => adjacency.len(),
        }
    }

    pub fn neighbors(&self, node_id: usize) -> Vec<usize> {
        match self {
            Topology::Grid2D { rows, cols } => {
                let mut neighbors = Vec::new();
                let row = node_id / cols;
                let col = node_id % cols;
                if row > 0 {
                    neighbors.push(node_id - cols);
                }
                if row + 1 < *rows {
                    neighbors.push(node_id + cols);
                }
                if col > 0 {
                    neighbors.push(node_id - 1);
                }
                if col + 1 < *cols {
                    neighbors.push(node_id + 1);
                }
                neighbors
            }
            Topology::Star { center, .. } => {
                if node_id == *center {
                    (0..self.node_count()).filter(|&i| i != *center).collect()
                } else {
                    vec![*center]
                }
            }
            Topology::Ring { size } => {
                let prev = if node_id == 0 { size - 1 } else { node_id - 1 };
                let next = if node_id + 1 == *size { 0 } else { node_id + 1 };
                vec![prev, next]
            }
            Topology::Custom { adjacency } => adjacency.get(node_id).cloned().unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid2d_topology() {
        let topo = Topology::Grid2D { rows: 2, cols: 3 };
        assert_eq!(topo.node_count(), 6);
        assert_eq!(topo.neighbors(0), vec![3, 1]);
        assert_eq!(topo.neighbors(4), vec![1, 3, 5]);
    }

    #[test]
    fn test_star_topology() {
        let topo = Topology::Star {
            center: 0,
            leaves: 4,
        };
        assert_eq!(topo.node_count(), 5);
        assert_eq!(topo.neighbors(0), vec![1, 2, 3, 4]);
        assert_eq!(topo.neighbors(2), vec![0]);
    }
}
