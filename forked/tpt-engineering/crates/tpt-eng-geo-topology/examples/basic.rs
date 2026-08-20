//! Runnable example: core `tpt-eng-geo-topology` API.
//!
//! Builds a directed infrastructure graph (a pump drawing from a tank and feeding
//! a valve → sink) and exercises adjacency and upstream/downstream traversal.

use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};

fn main() {
    let mut topo = Topology::new();

    // A simple water line: tank --(pipe)--> pump --(pipe)--> valve --(pipe)--> sink.
    topo.add_node("tank");
    topo.add_node("pump");
    topo.add_node("valve");
    topo.add_node("sink");
    topo.add_edge(Edge::new("e1", "tank", "pump", EdgeKind::Pipe, 100.0));
    topo.add_edge(Edge::new("e2", "pump", "valve", EdgeKind::Pipe, 100.0));
    topo.add_edge(Edge::new("e3", "valve", "sink", EdgeKind::Pipe, 100.0));

    println!("nodes: {}", topo.nodes().len());
    println!("edges: {}", topo.edges().len());

    // Adjacency: outgoing (downstream) and incoming (upstream) edges.
    println!(
        "outgoing from pump: {}",
        topo.outgoing("pump")
            .iter()
            .map(|e| format!("{} -> {}", e.from, e.to))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "incoming into valve: {}",
        topo.incoming("valve")
            .iter()
            .map(|e| format!("{} -> {}", e.from, e.to))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Reachability along flow direction.
    let mut downstream: Vec<_> = topo.downstream("tank");
    downstream.sort();
    println!("downstream of tank: {:?}", downstream);

    let mut upstream: Vec<_> = topo.upstream("sink");
    upstream.sort();
    println!("upstream of sink: {:?}", upstream);

    println!("reaches(tank, sink)? {}", topo.reaches("tank", "sink"));
    println!("reaches(sink, tank)? {}", topo.reaches("sink", "tank"));
}
