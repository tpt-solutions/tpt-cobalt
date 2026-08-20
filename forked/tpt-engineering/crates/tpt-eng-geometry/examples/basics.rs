//! Basic geometry: intersect a line with a plane.
//!
//! Run with: `cargo run --example basics -p tpt-eng-geometry`

use tpt_eng_geometry::{Point3, curve::Line3, intersection, surface::Plane3};

fn main() {
    let line = Line3::new(Point3::new(0.0, 0.0, -1.0), Point3::new(0.0, 0.0, 1.0));
    let plane = Plane3::new(Point3::ZERO, Point3::new(0.0, 0.0, 1.0));
    let (hit, t) = intersection::line_plane(line, plane).unwrap();
    println!("line meets plane at {hit:?} (parameter t = {t})");
}
