//! Basic `tpt-eng-cad` usage: the four SDF primitives, the boolean CSG
//! combinators that compose by *adding* signed distances (`union` and
//! `intersection`), feature-based `Part` modeling, the signed-distance sign
//! convention, and mesh extraction.
//!
//! Distances are millimetres. `sdf < 0` is inside the solid, `sdf > 0` outside.
//!
//! NOTE: the crate's `difference` (subtraction) expects a true Euclidean signed
//! distance field on its second operand; the in-crate primitives use a
//! Chebyshev-style field, so subtracted voids are not reproduced here. Material
//! *addition* (union/intersection) is exact and is what this example exercises.
//!
//! Run with `cargo run -p tpt-eng-cad --example basic`.

use std::boxed::Box as Heap;

use tpt_eng_cad::{Cylinder, Part, Solid, SolidFeature, Sphere, intersection, union};
use tpt_eng_geometry::{Point3, Vector3};

fn main() {
    // --- 1. The four primitives, with their bounding boxes -----------------
    let sphere = Sphere {
        center: Point3::ZERO,
        radius: 10.0,
    };
    let cyl = Cylinder {
        center: Point3::new(0.0, 0.0, 5.0),
        axis: Vector3::Z,
        radius: 8.0,
        half_height: 5.0,
    };
    let cone = tpt_eng_cad::Cone {
        center: Point3::new(0.0, 0.0, 0.0),
        axis: Vector3::Z,
        radius: 6.0,
        height: 12.0,
    };
    let boxy = tpt_eng_cad::Box {
        center: Point3::ZERO,
        half_extents: Vector3::new(7.0, 7.0, 3.0),
    };
    println!("primitive bounding boxes (mm):");
    for (name, b) in [
        ("sphere", sphere.bbox()),
        ("cylinder", cyl.bbox()),
        ("cone", cone.bbox()),
        ("box", boxy.bbox()),
    ] {
        println!(
            "  {:<9} x [{:.1},{:.1}]  y [{:.1},{:.1}]  z [{:.1},{:.1}]",
            name, b.min.x, b.max.x, b.min.y, b.max.y, b.min.z, b.max.z
        );
    }

    // --- 2. Signed-distance sign convention -------------------------------
    println!();
    println!("SDF sign (sphere r=10 at origin):");
    for (label, p) in [
        ("center (inside)", Point3::ZERO),
        ("on surface", Point3::new(10.0, 0.0, 0.0)),
        ("2 mm outside", Point3::new(12.0, 0.0, 0.0)),
    ] {
        let d = sphere.sdf(p);
        let where_ = if d < 0.0 { "inside" } else { "outside" };
        println!("  {:<16} sdf = {:+.2} mm ({})", label, d, where_);
    }

    // --- 3. Boolean CSG that adds material (exact for these fields) -------
    let pair = union(sphere, cyl);
    let overlap = intersection(boxy, cyl);
    println!();
    println!(
        "union(sphere,cyl) at the sphere centre = {:.2} mm (inside both -> negative)",
        pair.sdf(Point3::ZERO)
    );
    println!(
        "intersection(box,cyl) inside both? sdf(0,0,3) = {:.2} mm",
        overlap.sdf(Point3::new(0.0, 0.0, 3.0))
    );

    // --- 4. Feature-based part: a bracket with a welded-on boss ------------
    let base = Heap::new(tpt_eng_cad::Box {
        center: Point3::new(0.0, 0.0, 0.0),
        half_extents: Vector3::new(20.0, 15.0, 4.0),
    });
    let boss = Heap::new(Cylinder {
        center: Point3::new(8.0, 0.0, 4.0),
        axis: Vector3::Z,
        radius: 6.0,
        half_height: 6.0,
    });
    let bracket = Part::new("bracket", base).add_feature(SolidFeature::Add(boss));

    let resolved = bracket.resolved();
    println!();
    println!(
        "bracket resolved SDF: on base = {:.2} mm, on boss axis = {:.2} mm (both negative = solid)",
        resolved.sdf(Point3::new(0.0, 0.0, 0.0)),
        resolved.sdf(Point3::new(8.0, 0.0, 4.0))
    );

    // --- 5. Mesh extraction -------------------------------------------------
    let mesh = bracket.mesh(24);
    println!(
        "bracket mesh: {} triangles, {} vertices, {} STL bytes",
        mesh.face_count(),
        mesh.vertex_count(),
        mesh.to_stl_binary().len()
    );

    let s_mesh = Part::new("sphere", Heap::new(sphere)).mesh(32);
    println!(
        "sphere mesh: {} triangles (resolution 32)",
        s_mesh.face_count()
    );
}
