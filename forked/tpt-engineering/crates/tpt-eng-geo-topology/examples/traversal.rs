//! Runnable example: directional infrastructure-graph traversal.

use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};

fn main() {
    let mut topo = Topology::new();
    topo.add_node("source");
    topo.add_node("junction");
    topo.add_node("sink");
    topo.add_edge(Edge::new("e1", "source", "junction", EdgeKind::Pipe, 100.0));
    topo.add_edge(Edge::new("e2", "junction", "sink", EdgeKind::Pipe, 100.0));

    let down = topo.downstream("source");
    println!("downstream of `source`: {:?}", down);

    assert!(topo.reaches("source", "sink"));
    assert!(!topo.reaches("sink", "source"));
    println!("topology traversal example passed");
}
