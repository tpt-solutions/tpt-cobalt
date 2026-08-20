//! # tpt-eng-cad
//!
//! A small, license-clean CAD / solid-modeling kernel built on signed-distance
//! fields (SDF). It provides:
//!
//! - **Solid primitives** ([`Sphere`], [`Box`], [`Cylinder`], [`Cone`]) and a
//!   [`Solid`] trait with a negative-inside signed distance (`sdf < 0` inside,
//!   `sdf > 0` outside).
//! - **Boolean CSG** via SDF combination ([`Union`], [`Intersection`],
//!   [`Difference`]) and the convenience free functions [`union`],
//!   [`intersection`], [`difference`].
//! - **Mesh extraction** from any [`Solid`] using an in-house
//!   [marching tetrahedra](`marching_tetrahedra`) isosurface extractor.
//! - A minimal **B-Rep** topology data structure ([`Brep`]) derived from a mesh.
//! - **Feature modeling** ([`SolidFeature`]) and a [`Part`] container carrying
//!   CAD metadata.
//!
//! All solid modeling and boolean operations are implemented in-house using
//! SDFs; no external solid-modeling or geometry-kernel bindings are used.

use std::boxed::Box as StdBox;
use std::collections::HashMap;

use tpt_eng_geometry::query::Aabb;
use tpt_eng_geometry::{Point3, Vector3};
use tpt_eng_mesh::Mesh;

/// A solid body described by a signed distance field.
///
/// The signed distance convention is *negative inside*: `sdf(p) < 0` when `p`
/// is inside the solid, `sdf(p) > 0` when `p` is outside, and `sdf(p) == 0`
/// on the boundary.
pub trait Solid {
    /// Signed distance from `p` to the solid's boundary.
    fn sdf(&self, p: Point3) -> f32;

    /// Axis-aligned bounding box that fully encloses the solid.
    fn bbox(&self) -> Aabb;

    /// Clone this solid behind a trait object (object-safe deep copy).
    fn clone_solid(&self) -> StdBox<dyn Solid>;
}

impl<T: Solid + ?Sized> Solid for &T {
    fn sdf(&self, p: Point3) -> f32 {
        (**self).sdf(p)
    }

    fn bbox(&self) -> Aabb {
        (**self).bbox()
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        (**self).clone_solid()
    }
}

impl Solid for StdBox<dyn Solid> {
    fn sdf(&self, p: Point3) -> f32 {
        self.as_ref().sdf(p)
    }

    fn bbox(&self) -> Aabb {
        self.as_ref().bbox()
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        self.as_ref().clone_solid()
    }
}

/// A solid sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    /// Center of the sphere.
    pub center: Point3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl Solid for Sphere {
    fn sdf(&self, p: Point3) -> f32 {
        p.distance(self.center) - self.radius
    }

    fn bbox(&self) -> Aabb {
        let r = Vector3::splat(self.radius);
        Aabb {
            min: self.center - r,
            max: self.center + r,
        }
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(*self)
    }
}

/// An axis-aligned solid box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box {
    /// Center of the box.
    pub center: Point3,
    /// Half size of the box along each axis.
    pub half_extents: Vector3,
}

impl Solid for Box {
    fn sdf(&self, p: Point3) -> f32 {
        let q = (p - self.center).abs() - self.half_extents;
        let outside = q.max_element().max(0.0);
        let inside = q.min_element().min(0.0);
        outside + inside
    }

    fn bbox(&self) -> Aabb {
        Aabb {
            min: self.center - self.half_extents,
            max: self.center + self.half_extents,
        }
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(*self)
    }
}

/// A finite (capped) solid cylinder of arbitrary orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    /// Center of the cylinder (midpoint of its axis segment).
    pub center: Point3,
    /// Axis direction.
    pub axis: Vector3,
    /// Radius of the cylinder.
    pub radius: f32,
    /// Half the height of the cylinder.
    pub half_height: f32,
}

impl Solid for Cylinder {
    fn sdf(&self, p: Point3) -> f32 {
        let axis = self.axis.normalize();
        let d = p - self.center;
        let proj = d.dot(axis);
        let radial = d - axis * proj;
        let radial_len = radial.length();
        let q = Vector3::new(radial_len, proj.abs(), 0.0);
        let delta = q - Vector3::new(self.radius, self.half_height, 0.0);
        let outside = delta.max_element().max(0.0);
        let inside = delta.min_element().min(0.0);
        outside + inside
    }

    fn bbox(&self) -> Aabb {
        bbox_capsule(self.center, self.axis, self.radius, self.half_height)
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(*self)
    }
}

/// A right circular solid cone (capped cone) of arbitrary orientation.
///
/// The base disk (radius `radius`) is centered at `center` and the apex lies at
/// `center + axis * height`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    /// Center of the base disk.
    pub center: Point3,
    /// Axis pointing from the base toward the apex.
    pub axis: Vector3,
    /// Radius of the base disk.
    pub radius: f32,
    /// Height of the cone (distance from base to apex).
    pub height: f32,
}

impl Solid for Cone {
    fn sdf(&self, p: Point3) -> f32 {
        let axis = self.axis.normalize();
        let canonical_center = self.center + axis * (self.height * 0.5);
        let half = self.height * 0.5;
        let d = p - canonical_center;
        let proj = d.dot(axis);
        let radial = d - axis * proj;
        capped_cone_sdf(radial.length(), proj, half, self.radius, 0.0)
    }

    fn bbox(&self) -> Aabb {
        bbox_capsule(self.center, self.axis, self.radius, self.height * 0.5)
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(*self)
    }
}

/// SDF of a capped cone in canonical form.
///
/// `radial_len` is the distance from the axis, `axial` is the signed height
/// coordinate (centered), `half` is half the height, and `r1`/`r2` are the
/// radii at the `-half` and `+half` ends. This is the standard Inigo Quilez
/// capped-cone formulation.
fn capped_cone_sdf(radial_len: f32, axial: f32, half: f32, r1: f32, r2: f32) -> f32 {
    let q = Vector3::new(radial_len, axial.abs(), 0.0);
    let k1 = Vector3::new(r2, half, 0.0);
    let k2 = Vector3::new(r1 - r2, 2.0 * half, 0.0);
    let base_r = if axial < 0.0 { r1 } else { r2 };
    let base_h = if axial < 0.0 { -half } else { half };
    let ca = Vector3::new((q.x - q.x.min(base_r)).max(0.0), base_h, 0.0);
    let t = ((k1 - q).dot(k2) / k2.dot(k2)).clamp(0.0, 1.0);
    let cb = q - k1 + k2 * t;
    let sign = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    sign * (ca.dot(ca).min(cb.dot(cb))).sqrt()
}

/// Axis-aligned bounding box of a capped primitive (cylinder or cone).
///
/// `canonical_center` is the midpoint of the axis segment, `radius` is the
/// maximum radial sweep, and `half_height` is half the axial length. The
/// projection of the swept segment plus end disk onto each world axis gives the
/// box half-extent.
fn bbox_capsule(canonical_center: Point3, axis: Vector3, radius: f32, half_height: f32) -> Aabb {
    let a = axis.normalize();
    let extent = |c: f32| -> f32 {
        let radial_part = radius * radius * (1.0 - c * c);
        let axial_part = half_height * half_height * (c * c);
        (radial_part + axial_part).max(0.0).sqrt()
    };
    let half = Vector3::new(extent(a.x), extent(a.y), extent(a.z));
    Aabb {
        min: canonical_center - half,
        max: canonical_center + half,
    }
}

/// Boolean union of two solids (`a ∪ b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Union<A: Solid, B: Solid> {
    /// First operand.
    pub a: A,
    /// Second operand.
    pub b: B,
}

impl<A: Solid, B: Solid> Solid for Union<A, B> {
    fn sdf(&self, p: Point3) -> f32 {
        self.a.sdf(p).min(self.b.sdf(p))
    }

    fn bbox(&self) -> Aabb {
        self.a.bbox().union(&self.b.bbox())
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(Union {
            a: self.a.clone_solid(),
            b: self.b.clone_solid(),
        })
    }
}

/// Boolean intersection of two solids (`a ∩ b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intersection<A: Solid, B: Solid> {
    /// First operand.
    pub a: A,
    /// Second operand.
    pub b: B,
}

impl<A: Solid, B: Solid> Solid for Intersection<A, B> {
    fn sdf(&self, p: Point3) -> f32 {
        self.a.sdf(p).max(self.b.sdf(p))
    }

    fn bbox(&self) -> Aabb {
        let a = self.a.bbox();
        let b = self.b.bbox();
        Aabb {
            min: a.min.max(b.min),
            max: a.max.min(b.max),
        }
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(Intersection {
            a: self.a.clone_solid(),
            b: self.b.clone_solid(),
        })
    }
}

/// Boolean difference of two solids (`a \ b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Difference<A: Solid, B: Solid> {
    /// Solid that is kept.
    pub a: A,
    /// Solid that is subtracted.
    pub b: B,
}

impl<A: Solid, B: Solid> Solid for Difference<A, B> {
    fn sdf(&self, p: Point3) -> f32 {
        self.a.sdf(p).max(-self.b.sdf(p))
    }

    fn bbox(&self) -> Aabb {
        self.a.bbox()
    }

    fn clone_solid(&self) -> StdBox<dyn Solid> {
        StdBox::new(Difference {
            a: self.a.clone_solid(),
            b: self.b.clone_solid(),
        })
    }
}

/// Union of two solids, returned as a trait object.
pub fn union<A: Solid + 'static, B: Solid + 'static>(a: A, b: B) -> StdBox<dyn Solid> {
    StdBox::new(Union { a, b })
}

/// Intersection of two solids, returned as a trait object.
pub fn intersection<A: Solid + 'static, B: Solid + 'static>(a: A, b: B) -> StdBox<dyn Solid> {
    StdBox::new(Intersection { a, b })
}

/// Difference of two solids (`a` minus `b`), returned as a trait object.
pub fn difference<A: Solid + 'static, B: Solid + 'static>(a: A, b: B) -> StdBox<dyn Solid> {
    StdBox::new(Difference { a, b })
}

/// The six tetrahedra that decompose a cube sharing its main diagonal
/// (corner `0` at the min corner, corner `7` at the max corner).
const TETS: [[usize; 4]; 6] = [
    [0, 1, 5, 7],
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
];

/// Edges of a tetrahedron, indexed to pair with the corners of [`TETS`].
const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Extract an isosurface mesh from `solid` using marching tetrahedra.
///
/// `bounds` is subdivided into `resolution × resolution × resolution` cubes,
/// each decomposed into six tetrahedra; surface-crossing edges are linearly
/// interpolated and assembled into triangles. Every emitted triangle is
/// oriented so its normal points outward (away from the solid).
///
/// # Panics
///
/// Panics if the generated positions and indices are rejected by
/// [`Mesh::from_positions_indices`], which would indicate a broken internal
/// invariant rather than bad input.
#[must_use]
pub fn marching_tetrahedra<S: Solid + ?Sized>(solid: &S, resolution: u32, bounds: &Aabb) -> Mesh {
    let res = resolution.max(1) as usize;
    let min = bounds.min;
    let cell = (bounds.max - bounds.min) / res as f32;

    let mut positions: Vec<Point3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for zi in 0..res {
        for yi in 0..res {
            for xi in 0..res {
                let mut corner_pos = [Point3::ZERO; 8];
                let mut corner_val = [0.0_f32; 8];
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let idx = dx + 2 * dy + 4 * dz;
                            let p = min
                                + Vector3::new(
                                    (xi + dx) as f32 * cell.x,
                                    (yi + dy) as f32 * cell.y,
                                    (zi + dz) as f32 * cell.z,
                                );
                            corner_pos[idx] = p;
                            corner_val[idx] = solid.sdf(p);
                        }
                    }
                }

                for tet in TETS.iter() {
                    let pos4 = [
                        corner_pos[tet[0]],
                        corner_pos[tet[1]],
                        corner_pos[tet[2]],
                        corner_pos[tet[3]],
                    ];
                    let val4 = [
                        corner_val[tet[0]],
                        corner_val[tet[1]],
                        corner_val[tet[2]],
                        corner_val[tet[3]],
                    ];
                    for triangle in tet_triangles(&pos4, &val4) {
                        let oriented = orient_outward(solid, triangle);
                        let base = positions.len() as u32;
                        positions.push(oriented[0]);
                        positions.push(oriented[1]);
                        positions.push(oriented[2]);
                        indices.push(base);
                        indices.push(base + 1);
                        indices.push(base + 2);
                    }
                }
            }
        }
    }

    Mesh::from_positions_indices(positions, indices)
        .expect("marching tetrahedra produced an invalid mesh")
}

/// Emit the surface triangles of one tetrahedron given its corner positions
/// and signed-distance values.
fn tet_triangles(pos: &[Point3; 4], val: &[f32; 4]) -> Vec<[Point3; 3]> {
    let mut triangles = Vec::new();
    let inside = [val[0] < 0.0, val[1] < 0.0, val[2] < 0.0, val[3] < 0.0];
    let inside_count = inside.iter().filter(|&&b| b).count();
    if inside_count == 0 || inside_count == 4 {
        return triangles;
    }

    let mut cross = [None; 6];
    for (ei, &(i, j)) in TET_EDGES.iter().enumerate() {
        if inside[i] != inside[j] {
            let t = val[i] / (val[i] - val[j]);
            cross[ei] = Some(pos[i] + (pos[j] - pos[i]) * t);
        }
    }
    let edge_point = |i: usize, j: usize| -> Point3 {
        let ei = TET_EDGES
            .iter()
            .position(|&(a, b)| (a == i && b == j) || (a == j && b == i))
            .expect("edge must exist in tetrahedron");
        cross[ei].expect("edge must cross the surface")
    };

    if inside_count == 1 {
        let k = inside.iter().position(|&b| b).expect("one inside corner");
        let others: Vec<usize> = (0..4).filter(|&x| x != k).collect();
        triangles.push([
            edge_point(k, others[0]),
            edge_point(k, others[1]),
            edge_point(k, others[2]),
        ]);
    } else if inside_count == 3 {
        let k = inside.iter().position(|&b| !b).expect("one outside corner");
        let others: Vec<usize> = (0..4).filter(|&x| x != k).collect();
        triangles.push([
            edge_point(k, others[0]),
            edge_point(k, others[1]),
            edge_point(k, others[2]),
        ]);
    } else {
        let ins: Vec<usize> = (0..4).filter(|&x| inside[x]).collect();
        let out: Vec<usize> = (0..4).filter(|&x| !inside[x]).collect();
        let (a, b) = (ins[0], ins[1]);
        let (c, d) = (out[0], out[1]);
        let ac = edge_point(a, c);
        let ad = edge_point(a, d);
        let bc = edge_point(b, c);
        let bd = edge_point(b, d);
        triangles.push([ac, ad, bc]);
        triangles.push([ad, bd, bc]);
    }

    triangles
}

/// Approximate the SDF gradient at `p` via central finite differences.
///
/// The gradient of a signed-distance field points in the direction of increasing
/// distance, i.e. *outward* from the solid (where `sdf < 0`). It is well-defined
/// on both convex and concave surfaces, so it orients mesh triangles correctly
/// even for internal cavities produced by boolean subtraction.
fn sdf_gradient<S: Solid + ?Sized>(solid: &S, p: Point3) -> Vector3 {
    const H: f32 = 1e-3;
    let gx = (solid.sdf(p + Vector3::new(H, 0.0, 0.0)) - solid.sdf(p - Vector3::new(H, 0.0, 0.0)))
        / (2.0 * H);
    let gy = (solid.sdf(p + Vector3::new(0.0, H, 0.0)) - solid.sdf(p - Vector3::new(0.0, H, 0.0)))
        / (2.0 * H);
    let gz = (solid.sdf(p + Vector3::new(0.0, 0.0, H)) - solid.sdf(p - Vector3::new(0.0, 0.0, H)))
        / (2.0 * H);
    Vector3::new(gx, gy, gz)
}

/// Orient a triangle so its normal points outward from the solid.
///
/// The triangle normal (from its current winding) is compared against the
/// average SDF gradient at its vertices; if they oppose, the winding is flipped.
/// Falls back to the centroid-sign heuristic only where the gradient is
/// degenerate (near-zero, e.g. exactly at a sphere center).
fn orient_outward<S: Solid + ?Sized>(solid: &S, triangle: [Point3; 3]) -> [Point3; 3] {
    let n = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    let grad = (sdf_gradient(solid, triangle[0])
        + sdf_gradient(solid, triangle[1])
        + sdf_gradient(solid, triangle[2]))
        / 3.0;
    if grad.length() < 1e-6 {
        let centroid = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
        if solid.sdf(centroid) > 0.0 {
            [triangle[0], triangle[2], triangle[1]]
        } else {
            triangle
        }
    } else if n.dot(grad) < 0.0 {
        [triangle[0], triangle[2], triangle[1]]
    } else {
        triangle
    }
}

/// A vertex in a boundary-representation (`Brep`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrepVertex {
    /// Position of the vertex in 3D space.
    pub position: Point3,
}

/// An edge in a `Brep`, connecting two vertices by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrepEdge {
    /// Index of the start vertex.
    pub start: usize,
    /// Index of the end vertex.
    pub end: usize,
}

/// A face in a `Brep` defined by a closed loop of edges.
#[derive(Debug, Clone, PartialEq)]
pub struct BrepFace {
    /// Identifier of the underlying surface (e.g. `"triangle"`).
    pub surface: String,
    /// Indices into [`Brep::edges`] forming the face loop.
    pub edges: Vec<usize>,
}

/// A minimal boundary-representation topology derived from a triangle mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Brep {
    /// Unique vertices (coincident within `1e-4` are merged).
    pub vertices: Vec<BrepVertex>,
    /// Unique undirected edges.
    pub edges: Vec<BrepEdge>,
    /// Faces referencing edge indices.
    pub faces: Vec<BrepFace>,
}

impl Brep {
    /// Build a `Brep` from a triangle [`Mesh`].
    ///
    /// Vertex positions coincident within `1e-4` are merged, and triangle
    /// edges are deduplicated by their unordered vertex pair.
    #[must_use]
    pub fn from_mesh(mesh: &Mesh) -> Brep {
        let positions = &mesh.positions;
        let indices = &mesh.indices;
        let mut vertices: Vec<BrepVertex> = Vec::new();
        let mut edges: Vec<BrepEdge> = Vec::new();
        let mut faces: Vec<BrepFace> = Vec::new();

        let mut vertex_lookup: HashMap<(i32, i32, i32), usize> = HashMap::new();
        let mut edge_lookup: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in indices.chunks_exact(3) {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];
            let v0 = brep_vertex_index(&mut vertex_lookup, &mut vertices, p0);
            let v1 = brep_vertex_index(&mut vertex_lookup, &mut vertices, p1);
            let v2 = brep_vertex_index(&mut vertex_lookup, &mut vertices, p2);
            let e0 = brep_edge_index(&mut edge_lookup, &mut edges, v0, v1);
            let e1 = brep_edge_index(&mut edge_lookup, &mut edges, v1, v2);
            let e2 = brep_edge_index(&mut edge_lookup, &mut edges, v2, v0);
            faces.push(BrepFace {
                surface: String::from("triangle"),
                edges: vec![e0, e1, e2],
            });
        }

        Brep {
            vertices,
            edges,
            faces,
        }
    }

    /// Number of vertices in the `Brep`.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges in the `Brep`.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of faces in the `Brep`.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }
}

/// Map a vertex position to a unique `Brep` vertex index, merging coincident
/// positions within `1e-4`.
fn brep_vertex_index(
    lookup: &mut HashMap<(i32, i32, i32), usize>,
    vertices: &mut Vec<BrepVertex>,
    p: Point3,
) -> usize {
    let key = (
        (p.x * 1e4).round() as i32,
        (p.y * 1e4).round() as i32,
        (p.z * 1e4).round() as i32,
    );
    if let Some(&idx) = lookup.get(&key) {
        idx
    } else {
        let idx = vertices.len();
        vertices.push(BrepVertex { position: p });
        lookup.insert(key, idx);
        idx
    }
}

/// Map an unordered vertex pair to a unique `Brep` edge index.
fn brep_edge_index(
    lookup: &mut HashMap<(usize, usize), usize>,
    edges: &mut Vec<BrepEdge>,
    a: usize,
    b: usize,
) -> usize {
    let key = if a <= b { (a, b) } else { (b, a) };
    if let Some(&idx) = lookup.get(&key) {
        idx
    } else {
        let idx = edges.len();
        edges.push(BrepEdge {
            start: key.0,
            end: key.1,
        });
        lookup.insert(key, idx);
        idx
    }
}

/// A feature applied to a base solid: either added material or removed
/// (cut) material.
pub enum SolidFeature {
    /// Add the contained solid to the base (`union`).
    Add(StdBox<dyn Solid>),
    /// Subtract the contained solid from the base (`difference`).
    Cut(StdBox<dyn Solid>),
}

impl SolidFeature {
    /// Apply this feature to `base`, returning the resulting combined solid.
    #[must_use]
    pub fn apply(&self, base: &dyn Solid) -> StdBox<dyn Solid> {
        match self {
            SolidFeature::Add(feature) => StdBox::new(Union {
                a: base.clone_solid(),
                b: feature.clone_solid(),
            }),
            SolidFeature::Cut(feature) => StdBox::new(Difference {
                a: base.clone_solid(),
                b: feature.clone_solid(),
            }),
        }
    }
}

/// A named CAD part: a base solid, an ordered list of features, and metadata.
pub struct Part {
    /// Human-readable part name.
    pub name: String,
    /// The base solid of the part.
    pub solid: StdBox<dyn Solid>,
    /// Ordered modeling features applied to the base solid.
    pub features: Vec<SolidFeature>,
    /// Arbitrary key/value metadata (e.g. material, revision).
    pub metadata: HashMap<String, String>,
}

impl Part {
    /// Create a new part from a base solid.
    #[must_use]
    pub fn new(name: impl Into<String>, solid: StdBox<dyn Solid>) -> Part {
        Part {
            name: name.into(),
            solid,
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a modeling feature, returning the part for chaining.
    #[must_use]
    pub fn add_feature(mut self, feature: SolidFeature) -> Part {
        self.features.push(feature);
        self
    }

    /// Mutable access to the part metadata map.
    pub fn metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.metadata
    }

    /// Resolve the final solid (base with all features applied, in order).
    pub fn resolved(&self) -> StdBox<dyn Solid> {
        let mut current: StdBox<dyn Solid> = self.solid.clone_solid();
        for feature in &self.features {
            current = feature.apply(current.as_ref());
        }
        current
    }

    /// Extract a mesh of the part by padding the resolved solid's bounding box
    /// by 10% and running marching tetrahedra at `resolution`.
    #[must_use]
    pub fn mesh(&self, resolution: u32) -> Mesh {
        let resolved = self.resolved();
        let b = resolved.bbox();
        let center = b.center();
        let half = b.extent() * 0.5 * 1.1;
        let bounds = Aabb {
            min: center - half,
            max: center + half,
        };
        marching_tetrahedra(resolved.as_ref(), resolution, &bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sdf_and_bbox() {
        let s = Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        };
        assert!((s.sdf(Point3::ZERO) - (-1.0)).abs() < 1e-5);
        assert!(s.sdf(Point3::new(1.0, 0.0, 0.0)).abs() < 1e-5);
        assert!(s.sdf(Point3::new(2.0, 0.0, 0.0)) > 0.0);
        let b = s.bbox();
        assert!(b.contains(Point3::new(1.0, 0.0, 0.0)));
        assert!(b.contains(Point3::new(-1.0, 0.0, 0.0)));
        assert!(!b.contains(Point3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn difference_removes_region() {
        let a = Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Point3::new(0.5, 0.0, 0.0),
            radius: 0.6,
        };
        let d = Difference { a, b };
        assert!(d.sdf(Point3::new(0.5, 0.0, 0.0)) > 0.0);
        assert!(d.sdf(Point3::new(-0.9, 0.0, 0.0)) < 0.0);
    }

    #[test]
    fn union_overlap() {
        let a = Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Point3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        let u = union(a, b);
        assert!(u.sdf(Point3::ZERO) < 0.0);
        assert!(u.sdf(Point3::new(1.0, 0.0, 0.0)) < 0.0);
    }

    #[test]
    fn marching_tetrahedra_sphere() {
        let s = Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        };
        let bounds = s.bbox();
        let mesh = marching_tetrahedra(&s, 24, &bounds);
        assert!(mesh.face_count() > 0);
        assert!(mesh.vertex_count() > 0);

        let positions = &mesh.positions;
        for p in positions.iter() {
            assert!(s.sdf(*p).abs() < 0.2, "vertex off surface: {}", s.sdf(*p));
        }

        let mesh_bbox = Aabb::from_points(positions);
        let tol = Vector3::splat(0.25);
        assert!((mesh_bbox.min - bounds.min).abs().cmple(tol).all());
        assert!((mesh_bbox.max - bounds.max).abs().cmple(tol).all());
    }

    #[test]
    fn brep_from_single_triangle() {
        let tri = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let mesh = Mesh::from_triangles(&[tri]);
        let brep = Brep::from_mesh(&mesh);
        assert_eq!(brep.vertex_count(), 3);
        assert_eq!(brep.edge_count(), 3);
        assert_eq!(brep.face_count(), 1);
    }

    #[test]
    fn mesh_normals_point_outward() {
        let s = Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        };
        let mesh = marching_tetrahedra(&s, 16, &s.bbox());
        // Every triangle's geometric normal must align with the outward radial
        // direction (away from the sphere center).
        for tri in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let n = (b - a).cross(c - a);
            if n.length() < 1e-9 {
                continue; // skip coincidental zero-area triangles
            }
            let center = (a + b + c) / 3.0;
            assert!(n.dot(center) > 0.0, "inward-facing triangle at {center:?}");
        }
    }

    #[test]
    fn cavity_surface_orients_into_void() {
        // Big block with a spherical pocket subtracted: the cavity wall should
        // be oriented so its normal points into the void (away from material).
        let block = Box {
            center: Point3::ZERO,
            half_extents: Vector3::new(2.0, 2.0, 2.0),
        };
        let pocket = Sphere {
            center: Point3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let part = Part::new("block", StdBox::new(block))
            .add_feature(SolidFeature::Cut(StdBox::new(pocket)));
        let mesh = part.mesh(24);
        // For each triangle, the SDF gradient (outward from material) must agree
        // with the triangle's geometric normal.
        for tri in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let n = (b - a).cross(c - a);
            if n.length() < 1e-9 {
                continue; // skip coincidental zero-area triangles
            }
            let center = (a + b + c) / 3.0;
            let grad = sdf_gradient(part.resolved().as_ref(), center);
            assert!(
                n.dot(grad) > 0.0,
                "misoriented triangle at cavity/face {center:?}"
            );
        }
    }

    #[test]
    fn part_with_cut_feature() {
        let base = StdBox::new(Sphere {
            center: Point3::ZERO,
            radius: 1.0,
        });
        let cut = Sphere {
            center: Point3::new(0.5, 0.0, 0.0),
            radius: 0.6,
        };
        let part = Part::new("bolt", base).add_feature(SolidFeature::Cut(StdBox::new(cut)));
        let mesh = part.mesh(20);
        assert!(mesh.face_count() > 0);
        assert!(mesh.vertex_count() > 0);
    }
}
