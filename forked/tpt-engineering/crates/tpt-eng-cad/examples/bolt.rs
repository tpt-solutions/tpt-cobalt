//! CAD: a sphere with a spherical pocket (boolean difference), meshed to STL.
//!
//! Run with: `cargo run --example bolt -p tpt-eng-cad`

use tpt_eng_cad::{Part, SolidFeature, Sphere};
use tpt_eng_geometry::Point3;

fn main() {
    let base = Box::new(Sphere {
        center: Point3::ZERO,
        radius: 1.0,
    });
    let pocket = Sphere {
        center: Point3::new(0.5, 0.0, 0.0),
        radius: 0.6,
    };
    let part = Part::new("bolt", base).add_feature(SolidFeature::Cut(Box::new(pocket)));

    let mesh = part.mesh(20);
    println!("part `{}` -> {} triangles", part.name, mesh.face_count());
}
