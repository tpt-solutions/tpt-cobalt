//! Geometry: sample and query a sphere surface, plus curve and intersection use.
//!
//! Run with: `cargo run --example surface -p tpt-eng-geometry`

use std::f32::consts::{PI, TAU};

use tpt_eng_geometry::{
    Point3,
    curve::{Arc3, Circle3, Curve3, Line3},
    intersection,
    query::{Aabb, distance, triangle_area},
    surface::{Plane3, Sphere3, Surface3},
};

fn main() {
    // --- Surface: sample a sphere grid and report a few points/normals. ---
    let sphere = Sphere3 {
        center: Point3::ZERO,
        radius: 2.0,
    };
    let mut samples: Vec<Point3> = Vec::new();
    let mut max_area = 0.0_f32;
    for i in 0..=8 {
        let u = TAU * (i as f32 / 8.0);
        for j in 0..=4 {
            let v = -PI / 2.0 + PI * (j as f32 / 4.0);
            let p = sphere.eval(u, v);
            let n = sphere.normal(u, v);
            samples.push(p);
            // The outward normal equals (p - center) normalized.
            let outward = (p - sphere.center).normalize();
            assert!((n - outward).length() < 1e-3);
            if i < 8 && j < 4 {
                let du = TAU / 8.0;
                let dv = PI / 4.0;
                let tri = [
                    sphere.eval(u, v),
                    sphere.eval(u + du, v),
                    sphere.eval(u, v + dv),
                ];
                max_area = max_area.max(triangle_area(tri[0], tri[1], tri[2]));
            }
        }
    }
    println!(
        "sphere: sampled {} points; largest grid triangle area = {:.3}",
        samples.len(),
        max_area
    );

    let bbox = Aabb::from_points(&samples);
    println!(
        "bounding box: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2})",
        bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z
    );
    assert!((bbox.extent() - Point3::splat(4.0)).length() < 1e-2);

    // --- Curve: a quarter-circle arc and its closing chord. ---
    let arc = Arc3 {
        circle: Circle3 {
            center: Point3::ZERO,
            normal: Point3::Z,
            radius: 1.0,
        },
        start_angle: 0.0,
        end_angle: PI / 2.0,
    };
    let start = arc.eval(0.0);
    let end = arc.eval(1.0);
    println!(
        "arc start=({:.3},{:.3}) end=({:.3},{:.3})",
        start.x, start.y, end.x, end.y
    );
    let chord = distance(start, end);
    println!(
        "chord length = {:.3} (expected ~{:.3})",
        chord,
        2.0_f32.sqrt()
    );

    // --- Intersection: a ray through the sphere vs. the center plane. ---
    let ray = Line3::new(Point3::new(0.0, 0.0, -3.0), Point3::Z);
    let hit = intersection::line_sphere(ray, sphere.center, sphere.radius).unwrap();
    println!(
        "ray hits sphere at t0={:.3} (expected 1.0) and t1={:.3} (expected 5.0)",
        hit.t0, hit.t1
    );

    let plane = Plane3::new(Point3::ZERO, Point3::Z);
    let (p, _t) = intersection::line_plane(ray, plane).unwrap();
    println!("ray meets z=0 plane at z={:.3}", p.z);
}
