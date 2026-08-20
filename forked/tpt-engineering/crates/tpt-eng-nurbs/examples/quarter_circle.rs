//! NURBS: a rational quadratic Bézier quarter circle, tessellated to a polyline.
//!
//! Run with: `cargo run --example quarter_circle -p tpt-eng-nurbs`

use std::f32::consts::FRAC_1_SQRT_2;

use tpt_eng_geometry::Point3;
use tpt_eng_nurbs::{KnotVector, NurbsCurve};

fn main() {
    let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let arc = NurbsCurve::new(
        2,
        vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        vec![1.0, FRAC_1_SQRT_2, 1.0],
        kv,
    )
    .unwrap();

    let pts = arc.tessellate(32);
    let mid = pts[pts.len() / 2];
    println!("quarter circle tessellated to {} points", pts.len());
    println!("midpoint ~ ({:.4}, {:.4})", mid.x, mid.y);
}
