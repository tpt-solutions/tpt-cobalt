//! # tpt-eng-geo-topology
//!
//! Directional infrastructure graphs for pipes / wires / ducts, built on top of
//! [`tpt_eng_geo_asset`] node identities.
//!
//! Each [`Edge`] has a direction (`from → to`) representing the nominal flow
//! direction, enabling upstream/downstream traversal and flow-direction logic
//! (e.g. "what is fed by node X?", "what feeds node Y?"). Node ids correspond
//! to [`tpt_eng_geo_asset::Asset`] ids in the wider model.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};
//!
//! let mut topo = Topology::new();
//! topo.add_node("tank");
//! topo.add_node("pump");
//! topo.add_node("valve");
//! topo.add_edge(Edge::new("e1", "tank", "pump", EdgeKind::Pipe, 1.0));
//! topo.add_edge(Edge::new("e2", "pump", "valve", EdgeKind::Pipe, 1.0));
//!
//! // Downstream of the tank is the pump and then the valve.
//! let mut down = topo.downstream("tank");
//! down.sort();
//! assert_eq!(down, vec!["pump", "valve"]);
//! // Upstream of the valve is the pump and the tank.
//! let mut up = topo.upstream("valve");
//! up.sort();
//! assert_eq!(up, vec!["pump", "tank"]);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

pub use tpt_eng_geo_asset as asset;

/// The medium carried by an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// A fluid pipe.
    Pipe,
    /// An electrical conductor.
    Wire,
    /// An air/vent duct.
    Duct,
}

/// A directed infrastructure connection.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Unique edge id.
    pub id: String,
    /// Tail node (flow source).
    pub from: String,
    /// Head node (flow target).
    pub to: String,
    /// What the edge carries.
    pub kind: EdgeKind,
    /// A nominal capacity / rating (e.g. kW, L/s).
    pub capacity: f64,
}

impl Edge {
    /// Construct an edge.
    pub fn new(id: &str, from: &str, to: &str, kind: EdgeKind, capacity: f64) -> Self {
        Edge {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            kind,
            capacity,
        }
    }
}

/// A directional infrastructure graph.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    nodes: HashSet<String>,
    edges: Vec<Edge>,
    /// Adjacency: node -> outgoing edge indices (along flow direction).
    out_edges: HashMap<String, Vec<usize>>,
    /// Adjacency: node -> incoming edge indices (against flow direction).
    in_edges: HashMap<String, Vec<usize>>,
}

impl Topology {
    /// An empty topology.
    pub fn new() -> Self {
        Topology::default()
    }

    /// Add a node (idempotent).
    pub fn add_node(&mut self, id: &str) {
        self.nodes.insert(id.to_string());
    }

    /// Add an edge, ensuring its endpoint nodes exist.
    pub fn add_edge(&mut self, e: Edge) {
        self.nodes.insert(e.from.clone());
        self.nodes.insert(e.to.clone());
        let idx = self.edges.len();
        self.out_edges.entry(e.from.clone()).or_default().push(idx);
        self.in_edges.entry(e.to.clone()).or_default().push(idx);
        self.edges.push(e);
    }

    /// All node ids.
    pub fn nodes(&self) -> &HashSet<String> {
        &self.nodes
    }

    /// All edges.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Outgoing (downstream) edges from `node`.
    pub fn outgoing(&self, node: &str) -> Vec<&Edge> {
        self.out_edges
            .get(node)
            .map(|v| v.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Incoming (upstream) edges into `node`.
    pub fn incoming(&self, node: &str) -> Vec<&Edge> {
        self.in_edges
            .get(node)
            .map(|v| v.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// All nodes reachable from `start` by following flow direction
    /// (excluding `start` itself).
    pub fn downstream(&self, start: &str) -> Vec<String> {
        bfs(start, |n| {
            self.outgoing(n)
                .iter()
                .map(|e| e.to.clone())
                .collect::<Vec<_>>()
        })
    }

    /// All nodes that can reach `start` by following flow direction
    /// (excluding `start` itself).
    pub fn upstream(&self, start: &str) -> Vec<String> {
        bfs(start, |n| {
            self.incoming(n)
                .iter()
                .map(|e| e.from.clone())
                .collect::<Vec<_>>()
        })
    }

    /// Whether a directed flow path exists from `src` to `dst`.
    pub fn reaches(&self, src: &str, dst: &str) -> bool {
        self.downstream(src).iter().any(|n| n == dst)
    }
}

/// Generic breadth-first traversal over a successor function.
fn bfs(start: &str, succ: impl Fn(&str) -> Vec<String>) -> Vec<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut result = Vec::new();
    visited.insert(start.to_string());
    queue.push_back(start.to_string());
    while let Some(cur) = queue.pop_front() {
        for next in succ(&cur) {
            if visited.insert(next.clone()) {
                result.push(next.clone());
                queue.push_back(next);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Topology {
        let mut t = Topology::new();
        for n in ["tank", "pump", "valve", "sink"] {
            t.add_node(n);
        }
        t.add_edge(Edge::new("e1", "tank", "pump", EdgeKind::Pipe, 1.0));
        t.add_edge(Edge::new("e2", "pump", "valve", EdgeKind::Pipe, 1.0));
        t.add_edge(Edge::new("e3", "valve", "sink", EdgeKind::Pipe, 1.0));
        t
    }

    #[test]
    fn downstream_chain() {
        let t = sample();
        let mut d = t.downstream("tank");
        d.sort();
        assert_eq!(d, vec!["pump", "sink", "valve"]);
    }

    #[test]
    fn upstream_chain() {
        let t = sample();
        let mut u = t.upstream("sink");
        u.sort();
        assert_eq!(u, vec!["pump", "tank", "valve"]);
    }

    #[test]
    fn reachability() {
        let t = sample();
        assert!(t.reaches("tank", "sink"));
        assert!(!t.reaches("sink", "tank"));
    }

    #[test]
    fn degree_counts() {
        let t = sample();
        assert_eq!(t.outgoing("pump").len(), 1);
        assert_eq!(t.incoming("valve").len(), 1);
    }
}
