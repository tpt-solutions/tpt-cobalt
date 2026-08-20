//! Runnable example: flow-direction reasoning with `tpt-eng-geo-topology`.
//!
//! Models a branching water network and classifies each node by its role in the
//! nominal flow direction (source / junction / sink), then reports what feeds a
//! node and what it feeds. Also shows that flipping an edge's direction inverts
//! the upstream/downstream reachability.

use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};

fn main() {
    let mut topo = Topology::new();

    // reservoir -> main, main branches to two districts, each district fills a tank.
    topo.add_edge(Edge::new("e1", "reservoir", "main", EdgeKind::Pipe, 500.0));
    topo.add_edge(Edge::new("e2", "main", "district_a", EdgeKind::Pipe, 250.0));
    topo.add_edge(Edge::new("e3", "main", "district_b", EdgeKind::Pipe, 250.0));
    topo.add_edge(Edge::new(
        "e4",
        "district_a",
        "tank_a",
        EdgeKind::Pipe,
        250.0,
    ));
    topo.add_edge(Edge::new(
        "e5",
        "district_b",
        "tank_b",
        EdgeKind::Pipe,
        250.0,
    ));

    // Classify each node by flow direction.
    for node in topo.nodes() {
        let nin = topo.incoming(node).len();
        let nout = topo.outgoing(node).len();
        let role = if nin == 0 && nout > 0 {
            "SOURCE (flow originates here)"
        } else if nin > 0 && nout == 0 {
            "SINK (flow terminates here)"
        } else {
            "JUNCTION (flow passes through)"
        };
        println!("{}: in={} out={} -> {}", node, nin, nout, role);
    }

    // What feeds `main`, and what `main` feeds.
    let mut feeds: Vec<_> = topo.upstream("main");
    feeds.sort();
    let mut fed: Vec<_> = topo.downstream("main");
    fed.sort();
    println!("feeds(main): {:?}", feeds);
    println!("fed by(main): {:?}", fed);

    // Capacity on the trunk edge.
    let trunk = &topo.outgoing("reservoir")[0];
    println!(
        "trunk flow direction: {} -> {} @ {:.3} L/s",
        trunk.from, trunk.to, trunk.capacity
    );

    // Flipping an edge inverts reachability: reverse e1 so the reservoir is now
    // downstream of `main`.
    let mut flipped = Topology::new();
    for e in topo.edges() {
        if e.id == "e1" {
            flipped.add_edge(Edge::new(&e.id, &e.to, &e.from, e.kind, e.capacity));
        } else {
            flipped.add_edge(e.clone());
        }
    }
    println!(
        "after reversing e1, reaches(reservoir, main)? {}",
        flipped.reaches("reservoir", "main")
    );
    println!(
        "after reversing e1, reaches(main, reservoir)? {}",
        flipped.reaches("main", "reservoir")
    );
}
