//! # tpt-eng-mesh
//!
//! A license-clean triangle-mesh crate for engineering/CAD workloads.
//!
//! It provides a simple indexed [`Mesh`] data model together with:
//!
//! - **Normals** — per-face and area-weighted smooth vertex normals.
//! - **Quality metrics** — triangle angles, aspect ratios, edge lengths, and
//!   degenerate-face counts.
//! - **Refinement & repair** — midpoint subdivision, degenerate-face removal,
//!   and vertex welding.
//! - **Format conversion (in-house)** — binary/ASCII STL and Wavefront OBJ,
//!   implemented from scratch with no third-party format crates.
//!
//! `Point3` and `Vector3` are re-exported aliases over [`glam::Vec3`] (positions
//! vs. directions); the crate builds on [`tpt_eng_geometry`] for the underlying
//! geometric primitives.

use std::collections::HashMap;

use tpt_eng_geometry::{EPSILON, Point3, Vector3, point, query, vector};

/// An indexed triangle mesh.
///
/// `positions` holds the unique (or duplicated) vertex coordinates and
/// `indices` holds vertex indices grouped in triples, each triple forming one
/// triangle. `normals`, when present, stores one normal per vertex.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Vertex positions (one entry per vertex).
    pub positions: Vec<Point3>,
    /// Triangle vertex indices, grouped in triples (`[a, b, c, a, b, c, ...]`).
    pub indices: Vec<u32>,
    /// Optional per-vertex normals (one entry per vertex).
    pub normals: Option<Vec<Vector3>>,
    /// Optional texture coordinates (`vt`), referenced by `tex_indices`.
    pub tex_coords: Option<Vec<[f32; 2]>>,
    /// Optional per-corner texture-coordinate indices, parallel to `indices`
    /// (one entry per triangle corner). Present only when `tex_coords` is.
    pub tex_indices: Option<Vec<u32>>,
    /// Optional per-corner normal indices, parallel to `indices` (one entry per
    /// triangle corner). When present, `normals` is treated as the OBJ `vn`
    /// pool rather than per-vertex data.
    pub normal_indices: Option<Vec<u32>>,
}

impl Mesh {
    /// Build a mesh from explicit positions and triangle indices.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a descriptive message if `indices` is not a multiple
    /// of 3 or if any index is out of bounds for `positions`.
    pub fn from_positions_indices(
        positions: Vec<Point3>,
        indices: Vec<u32>,
    ) -> Result<Mesh, String> {
        if indices.len() % 3 != 0 {
            return Err(format!(
                "index count {} is not a multiple of 3",
                indices.len()
            ));
        }
        let n = positions.len() as u64;
        for &idx in &indices {
            if u64::from(idx) >= n {
                return Err(format!("vertex index {idx} out of bounds ({} vertices)", n));
            }
        }
        Ok(Mesh {
            positions,
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        })
    }

    /// Build a mesh from a slice of triangles, concatenating their positions and
    /// assigning sequential indices.
    ///
    /// Duplicate vertices are *not* merged; call [`Mesh::weld_vertices`] if a
    /// welded mesh is desired.
    #[must_use]
    pub fn from_triangles(tris: &[[Point3; 3]]) -> Mesh {
        let mut positions = Vec::with_capacity(tris.len() * 3);
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for tri in tris {
            for v in tri {
                indices.push(positions.len() as u32);
                positions.push(*v);
            }
        }
        Mesh {
            positions,
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        }
    }

    /// Number of vertices (length of `positions`).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (length of `indices` divided by 3).
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of triangles (alias of [`Mesh::face_count`], matching the STL
    /// terminology where each face is a triangle).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.face_count()
    }

    /// The three vertex positions of triangle `i`.
    #[must_use]
    pub fn face(&self, i: usize) -> [Point3; 3] {
        let [a, b, c] = self.face_indices(i);
        [
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        ]
    }

    /// The three vertex indices of triangle `i`.
    #[must_use]
    fn face_indices(&self, i: usize) -> [u32; 3] {
        let b = i * 3;
        [self.indices[b], self.indices[b + 1], self.indices[b + 2]]
    }

    /// Unit normal of triangle `i` (normalized cross product of its edges).
    ///
    /// Returns the zero vector for a degenerate (zero-area) triangle.
    #[must_use]
    pub fn face_normal(&self, i: usize) -> Vector3 {
        let [a, b, c] = self.face(i);
        let n = vector::cross(b - a, c - a);
        let len = n.length();
        if len > EPSILON {
            n / len
        } else {
            Vector3::ZERO
        }
    }

    /// One unit normal per face.
    #[must_use]
    pub fn compute_face_normals(&self) -> Vec<Vector3> {
        (0..self.face_count())
            .map(|i| self.face_normal(i))
            .collect()
    }

    /// Area-weighted, normalized per-vertex normals.
    ///
    /// Each vertex normal is the area-weighted sum of the (unit) normals of all
    /// adjacent faces, then normalized. Vertices with no contributing area keep
    /// the zero vector.
    #[must_use]
    pub fn compute_vertex_normals(&self) -> Vec<Vector3> {
        let mut vn = vec![Vector3::ZERO; self.positions.len()];
        let face_normals = self.compute_face_normals();
        for (i, &n) in face_normals.iter().enumerate() {
            let [a, b, c] = self.face(i);
            let area = query::triangle_area(a, b, c);
            let [ia, ib, ic] = self.face_indices(i);
            vn[ia as usize] += n * area;
            vn[ib as usize] += n * area;
            vn[ic as usize] += n * area;
        }
        for v in &mut vn {
            let len = v.length();
            if len > EPSILON {
                *v /= len;
            }
        }
        vn
    }

    /// Return a copy of this mesh whose `normals` are set to smooth vertex
    /// normals (see [`Mesh::compute_vertex_normals`]).
    #[must_use]
    pub fn with_smooth_normals(mut self) -> Mesh {
        self.normals = Some(self.compute_vertex_normals());
        self
    }

    /// The smallest interior angle (radians) across all triangles.
    #[must_use]
    pub fn min_triangle_angle(&self) -> f32 {
        (0..self.face_count())
            .flat_map(|i| {
                let [a, b, c] = self.face(i);
                [
                    query::triangle_angle_at(a, b, c),
                    query::triangle_angle_at(b, a, c),
                    query::triangle_angle_at(c, a, b),
                ]
            })
            .fold(f32::INFINITY, f32::min)
    }

    /// The largest interior angle (radians) across all triangles.
    #[must_use]
    pub fn max_triangle_angle(&self) -> f32 {
        (0..self.face_count())
            .flat_map(|i| {
                let [a, b, c] = self.face(i);
                [
                    query::triangle_angle_at(a, b, c),
                    query::triangle_angle_at(b, a, c),
                    query::triangle_angle_at(c, a, b),
                ]
            })
            .fold(0.0_f32, f32::max)
    }

    /// Aspect ratio of triangle `i`, defined as `circumradius / (2 * inradius)`.
    ///
    /// `1.0` indicates an equilateral triangle; larger values indicate worse
    /// quality. Degenerate triangles (vanishing inradius) return
    /// [`f32::INFINITY`].
    #[must_use]
    pub fn aspect_ratio(&self, i: usize) -> f32 {
        let [a, b, c] = self.face(i);
        let e1 = query::distance(a, b);
        let e2 = query::distance(b, c);
        let e3 = query::distance(c, a);
        let area = query::triangle_area(a, b, c);
        let perimeter = e1 + e2 + e3;
        if area < EPSILON {
            return f32::INFINITY;
        }
        let circumradius = (e1 * e2 * e3) / (4.0 * area);
        let inradius = (2.0 * area) / perimeter;
        if inradius < EPSILON {
            f32::INFINITY
        } else {
            circumradius / (2.0 * inradius)
        }
    }

    /// Shortest edge length across all triangles.
    #[must_use]
    pub fn min_edge_length(&self) -> f32 {
        self.edge_lengths().fold(f32::INFINITY, f32::min)
    }

    /// Longest edge length across all triangles.
    #[must_use]
    pub fn max_edge_length(&self) -> f32 {
        self.edge_lengths().fold(0.0_f32, f32::max)
    }

    /// Iterator over all triangle edge lengths.
    fn edge_lengths(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.face_count()).flat_map(|i| {
            let [a, b, c] = self.face(i);
            [
                query::distance(a, b),
                query::distance(b, c),
                query::distance(c, a),
            ]
        })
    }

    /// Number of faces whose area is below `threshold`.
    #[must_use]
    pub fn degenerate_face_count(&self, threshold: f32) -> usize {
        (0..self.face_count())
            .filter(|&i| {
                let [a, b, c] = self.face(i);
                query::triangle_area(a, b, c) < threshold
            })
            .count()
    }

    /// Mean aspect ratio across all triangles.
    ///
    /// Returns `0.0` for an empty mesh.
    #[must_use]
    pub fn average_aspect_ratio(&self) -> f32 {
        let n = self.face_count();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = (0..n).map(|i| self.aspect_ratio(i)).sum();
        sum / n as f32
    }

    /// Subdivide every triangle into four by inserting edge midpoints.
    ///
    /// Shared midpoints are welded (by exact position) so the resulting mesh
    /// stays connected. Returns a new mesh; the original is unchanged.
    #[must_use]
    pub fn subdivide_midpoint(&self) -> Mesh {
        let mut positions = self.positions.clone();
        let mut map: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (idx, &p) in self.positions.iter().enumerate() {
            map.insert(key_of(p), idx as u32);
        }
        let mut indices = Vec::with_capacity(self.indices.len() * 4);

        let add_vertex = |map: &mut HashMap<(u32, u32, u32), u32>,
                          positions: &mut Vec<Point3>,
                          p: Point3|
         -> u32 {
            let k = key_of(p);
            if let Some(&idx) = map.get(&k) {
                return idx;
            }
            let idx = positions.len() as u32;
            positions.push(p);
            map.insert(k, idx);
            idx
        };

        for i in 0..self.face_count() {
            let [a, b, c] = self.face(i);
            let ia = self.indices[i * 3];
            let ib = self.indices[i * 3 + 1];
            let ic = self.indices[i * 3 + 2];
            let mab = add_vertex(&mut map, &mut positions, point::midpoint(a, b));
            let mbc = add_vertex(&mut map, &mut positions, point::midpoint(b, c));
            let mca = add_vertex(&mut map, &mut positions, point::midpoint(c, a));
            indices.extend_from_slice(&[ia, mab, mca]);
            indices.extend_from_slice(&[mab, ib, mbc]);
            indices.extend_from_slice(&[mca, mbc, ic]);
            indices.extend_from_slice(&[mab, mbc, mca]);
        }

        Mesh {
            positions,
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        }
    }

    /// Remove faces whose area is below `threshold`, keeping the (possibly
    /// unused) vertex positions and re-emitting valid triangle indices.
    #[must_use]
    pub fn remove_degenerate_faces(&self, threshold: f32) -> Mesh {
        let mut indices = Vec::with_capacity(self.indices.len());
        for i in 0..self.face_count() {
            let [a, b, c] = self.face(i);
            if query::triangle_area(a, b, c) >= threshold {
                indices.extend_from_slice(&self.face_indices(i));
            }
        }
        Mesh {
            positions: self.positions.clone(),
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        }
    }

    /// Merge vertices closer than `threshold`, remapping indices accordingly.
    ///
    /// Uses a simple O(n²) nearest-existing-vertex compare, which is adequate
    /// for moderate meshes and avoids any external spatial-index dependency.
    #[must_use]
    pub fn weld_vertices(&self, threshold: f32) -> Mesh {
        let mut new_positions: Vec<Point3> = Vec::new();
        let mut remap = vec![0u32; self.positions.len()];
        for (i, &p) in self.positions.iter().enumerate() {
            let mut merged: Option<u32> = None;
            for (j, &q) in new_positions.iter().enumerate() {
                if query::distance(p, q) < threshold {
                    merged = Some(j as u32);
                    break;
                }
            }
            match merged {
                Some(j) => remap[i] = j,
                None => {
                    remap[i] = new_positions.len() as u32;
                    new_positions.push(p);
                }
            }
        }
        let indices = self
            .indices
            .iter()
            .map(|&idx| remap[idx as usize])
            .collect();
        Mesh {
            positions: new_positions,
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        }
    }

    /// Serialize the mesh to a binary STL buffer.
    ///
    /// Layout: an 80-byte (zero) header, a little-endian `u32` triangle count,
    /// then per triangle a 3-float normal, three 3-float vertices, and a
    /// little-endian `u16` attribute byte count of `0`.
    #[must_use]
    pub fn to_stl_binary(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(84 + 50 * self.face_count());
        out.extend_from_slice(&[0u8; 80]);
        out.extend_from_slice(&(self.face_count() as u32).to_le_bytes());
        for i in 0..self.face_count() {
            let n = self.face_normal(i);
            let [a, b, c] = self.face(i);
            push_f32(&mut out, n.x);
            push_f32(&mut out, n.y);
            push_f32(&mut out, n.z);
            for v in [a, b, c] {
                push_f32(&mut out, v.x);
                push_f32(&mut out, v.y);
                push_f32(&mut out, v.z);
            }
            out.extend_from_slice(&0u16.to_le_bytes());
        }
        out
    }

    /// Parse a binary STL buffer into a mesh, welding identical vertices.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `data` is shorter than the 84-byte header, if the
    /// declared triangle count overflows addressable size, or if the buffer is
    /// shorter than the `84 + 50 * count` bytes the header declares.
    // The `try_into().unwrap()` calls below cannot fail: every slice is exactly
    // 4 bytes long and all offsets are pre-validated against `needed`.
    #[allow(clippy::missing_panics_doc)]
    pub fn from_stl_binary(data: &[u8]) -> Result<Mesh, String> {
        if data.len() < 84 {
            return Err("STL binary buffer shorter than 84-byte header".to_string());
        }
        let count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
        let needed = count
            .checked_mul(50)
            .and_then(|n| n.checked_add(84))
            .ok_or_else(|| "STL triangle count overflows addressable size".to_string())?;
        if data.len() < needed {
            return Err(format!(
                "STL binary buffer too short: need {needed} bytes, have {}",
                data.len()
            ));
        }

        let mut positions: Vec<Point3> = Vec::with_capacity(count * 3);
        let mut map: HashMap<(u32, u32, u32), u32> = HashMap::new();
        let mut indices = Vec::with_capacity(count * 3);

        for i in 0..count {
            let o = 84 + 50 * i;
            let verts = [
                Point3::new(
                    f32::from_le_bytes(data[o + 12..o + 16].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 16..o + 20].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 20..o + 24].try_into().unwrap()),
                ),
                Point3::new(
                    f32::from_le_bytes(data[o + 24..o + 28].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 28..o + 32].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 32..o + 36].try_into().unwrap()),
                ),
                Point3::new(
                    f32::from_le_bytes(data[o + 36..o + 40].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 40..o + 44].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 44..o + 48].try_into().unwrap()),
                ),
            ];
            for v in verts {
                let k = key_of(v);
                let idx = match map.get(&k) {
                    Some(&idx) => idx,
                    None => {
                        let idx = positions.len() as u32;
                        positions.push(v);
                        map.insert(k, idx);
                        idx
                    }
                };
                indices.push(idx);
            }
        }

        Ok(Mesh {
            positions,
            indices,
            normals: None,
            tex_coords: None,
            tex_indices: None,
            normal_indices: None,
        })
    }

    /// Parse an ASCII STL string into a mesh, welding identical vertices.
    ///
    /// Each `facet normal ...`/`vertex ...` triple becomes one triangle. The
    /// facet normal is ignored in favour of the geometric normal (consistency
    /// with [`Mesh::from_stl_binary`]).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a `vertex` line has fewer than 3 parseable
    /// coordinates, if the text ends with an incomplete facet (fewer than 3
    /// vertices collected), or if the assembled positions and indices fail
    /// [`Mesh::from_positions_indices`] validation.
    pub fn from_stl_ascii(text: &str) -> Result<Mesh, String> {
        let mut positions: Vec<Point3> = Vec::new();
        let mut map: HashMap<(u32, u32, u32), u32> = HashMap::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut verts: [Point3; 3] = [Point3::ZERO; 3];
        let mut got = 0usize;

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            if let Some("vertex") = parts.next() {
                let coords: Vec<f32> = parts.filter_map(|t| t.parse().ok()).collect();
                if coords.len() < 3 {
                    return Err("ASCII STL `vertex` needs 3 coordinates".to_string());
                }
                verts[got] = Point3::new(coords[0], coords[1], coords[2]);
                got += 1;
                if got == 3 {
                    for v in verts {
                        let k = key_of(v);
                        let idx = match map.get(&k) {
                            Some(&idx) => idx,
                            None => {
                                let idx = positions.len() as u32;
                                positions.push(v);
                                map.insert(k, idx);
                                idx
                            }
                        };
                        indices.push(idx);
                    }
                    got = 0;
                }
            }
        }

        if got != 0 {
            return Err("ASCII STL ended with an incomplete facet".to_string());
        }

        Mesh::from_positions_indices(positions, indices)
    }

    /// Serialize the mesh to an ASCII STL string.
    #[must_use]
    pub fn to_stl_ascii(&self) -> String {
        let mut s = String::new();
        s.push_str("solid tpt_eng_mesh\n");
        for i in 0..self.face_count() {
            let n = self.face_normal(i);
            let [a, b, c] = self.face(i);
            s.push_str(&format!("facet normal {} {} {}\n", n.x, n.y, n.z));
            s.push_str("  outer loop\n");
            for v in [a, b, c] {
                s.push_str(&format!("    vertex {} {} {}\n", v.x, v.y, v.z));
            }
            s.push_str("  endloop\n");
            s.push_str("endfacet\n");
        }
        s.push_str("endsolid tpt_eng_mesh\n");
        s
    }

    /// Serialize the mesh to a Wavefront OBJ string.
    ///
    /// Emits `v` lines, and (when present) `vt` texture coordinates, `vn`
    /// normals, and `f` lines carrying per-corner vertex/texture/normal indices
    /// in the OBJ `v/t/n` form. Faces are 1-indexed triangles.
    #[must_use]
    pub fn to_obj(&self) -> String {
        let mut s = String::new();
        for p in &self.positions {
            s.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
        }
        if let Some(ref tc) = self.tex_coords {
            for t in tc {
                s.push_str(&format!("vt {} {}\n", t[0], t[1]));
            }
        }
        if self.normal_indices.is_some() {
            if let Some(ref ns) = self.normals {
                for n in ns {
                    s.push_str(&format!("vn {} {} {}\n", n.x, n.y, n.z));
                }
            }
        }
        for i in 0..self.face_count() {
            let base = i * 3;
            let mut line = String::from("f");
            for k in 0..3 {
                let vi = self.indices[base + k] + 1;
                let vt = self.tex_indices.as_ref().map(|t| t[base + k] + 1);
                let vn = self.normal_indices.as_ref().map(|n| n[base + k] + 1);
                match (vt, vn) {
                    (Some(vt), Some(vn)) => line.push_str(&format!(" {}/{}/{}", vi, vt, vn)),
                    (Some(vt), None) => line.push_str(&format!(" {}/{}", vi, vt)),
                    (None, Some(vn)) => line.push_str(&format!(" {}//{}", vi, vn)),
                    (None, None) => line.push_str(&format!(" {}", vi)),
                }
            }
            s.push_str(&line);
            s.push('\n');
        }
        s
    }

    /// Parse a Wavefront OBJ string into a mesh.
    ///
    /// `v` lines define positions; `vt` define texture coordinates; `vn` define
    /// normals. `f` lines are triangulated as a fan (so quads and higher
    /// polygons are supported) and may use the `i/j/k` form, in which `i` is the
    /// vertex index, `j` the (optional) texture-coordinate index, and `k` the
    /// (optional) normal index. Indices are 1-indexed; negative (relative)
    /// indices are supported. Other lines are ignored.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a `v`/`vn` line has fewer than 3 coordinates, a `vt`
    /// line fewer than 2, a face index is unparseable or out of bounds, or the
    /// assembled positions and indices fail
    /// [`Mesh::from_positions_indices`] validation.
    pub fn from_obj(text: &str) -> Result<Mesh, String> {
        let mut positions: Vec<Point3> = Vec::new();
        let mut tex_coords: Vec<[f32; 2]> = Vec::new();
        let mut normals: Vec<Vector3> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut tex_indices: Vec<u32> = Vec::new();
        let mut normal_indices: Vec<u32> = Vec::new();

        let resolve = |raw: &str, len: usize| -> Result<u32, String> {
            let idx: i64 = raw
                .parse()
                .map_err(|_| format!("invalid OBJ index `{raw}`"))?;
            let resolved = if idx > 0 { idx - 1 } else { len as i64 + idx };
            if resolved < 0 || resolved >= len as i64 {
                return Err(format!("OBJ index {idx} out of bounds"));
            }
            Ok(resolved as u32)
        };

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let coords: Vec<f32> = parts.filter_map(|t| t.parse().ok()).collect();
                    if coords.len() < 3 {
                        return Err("OBJ `v` line needs at least 3 coordinates".to_string());
                    }
                    positions.push(Point3::new(coords[0], coords[1], coords[2]));
                }
                Some("vt") => {
                    let coords: Vec<f32> = parts.filter_map(|t| t.parse().ok()).collect();
                    if coords.len() < 2 {
                        return Err("OBJ `vt` line needs at least 2 coordinates".to_string());
                    }
                    tex_coords.push([coords[0], coords[1]]);
                }
                Some("vn") => {
                    let coords: Vec<f32> = parts.filter_map(|t| t.parse().ok()).collect();
                    if coords.len() < 3 {
                        return Err("OBJ `vn` line needs at least 3 coordinates".to_string());
                    }
                    normals.push(Vector3::new(coords[0], coords[1], coords[2]));
                }
                Some("f") => {
                    let mut fv: Vec<u32> = Vec::new();
                    let mut fvt: Vec<Option<u32>> = Vec::new();
                    let mut fvn: Vec<Option<u32>> = Vec::new();
                    for tok in parts {
                        let mut it = tok.split('/');
                        let v = it.next().unwrap_or("");
                        let vt = it.next().unwrap_or("");
                        let vn = it.next().unwrap_or("");
                        fv.push(resolve(v, positions.len())?);
                        fvt.push(if vt.is_empty() {
                            None
                        } else {
                            Some(resolve(vt, tex_coords.len())?)
                        });
                        fvn.push(if vn.is_empty() {
                            None
                        } else {
                            Some(resolve(vn, normals.len())?)
                        });
                    }
                    if fv.len() >= 3 {
                        for k in 1..fv.len() - 1 {
                            for &c in &[fv[0], fv[k], fv[k + 1]] {
                                indices.push(c);
                            }
                            for &c in &[fvt[0], fvt[k], fvt[k + 1]] {
                                tex_indices.push(c.unwrap_or(0));
                            }
                            for &c in &[fvn[0], fvn[k], fvn[k + 1]] {
                                normal_indices.push(c.unwrap_or(0));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let mut mesh = Mesh::from_positions_indices(positions, indices)?;
        if !tex_coords.is_empty() {
            mesh.tex_coords = Some(tex_coords);
            mesh.tex_indices = Some(tex_indices);
        }
        if !normals.is_empty() {
            mesh.normals = Some(normals);
            mesh.normal_indices = Some(normal_indices);
        }
        Ok(mesh)
    }
}

/// Append a little-endian `f32` to a byte buffer.
fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Exact, bit-precise position key used for welding.
fn key_of(p: Point3) -> (u32, u32, u32) {
    (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube built from 12 triangles (2 per face).
    fn cube() -> Mesh {
        let v = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let tris: [[Point3; 3]; 12] = [
            [v[0], v[1], v[2]],
            [v[0], v[2], v[3]],
            [v[4], v[6], v[5]],
            [v[4], v[7], v[6]],
            [v[0], v[4], v[5]],
            [v[0], v[5], v[1]],
            [v[1], v[5], v[6]],
            [v[1], v[6], v[2]],
            [v[2], v[6], v[7]],
            [v[2], v[7], v[3]],
            [v[3], v[7], v[4]],
            [v[3], v[4], v[0]],
        ];
        Mesh::from_triangles(&tris)
    }

    #[test]
    fn cube_has_twelve_faces_and_eight_vertices_after_weld() {
        let m = cube();
        assert_eq!(m.face_count(), 12);
        assert_eq!(m.weld_vertices(1e-6).vertex_count(), 8);
        assert_eq!(m.degenerate_face_count(1e-6), 0);
    }

    #[test]
    fn stl_binary_round_trips() {
        let m = cube().weld_vertices(1e-6);
        let bytes = m.to_stl_binary();
        let parsed = Mesh::from_stl_binary(&bytes).unwrap();
        let welded = parsed.weld_vertices(1e-6);
        assert_eq!(welded.face_count(), m.face_count());
        assert_eq!(welded.vertex_count(), m.vertex_count());
    }

    #[test]
    fn subdivide_single_triangle() {
        let tri = Mesh::from_triangles(&[[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]]);
        let sub = tri.subdivide_midpoint();
        assert_eq!(sub.face_count(), 4);
        assert_eq!(sub.vertex_count(), 6);
    }

    #[test]
    fn weld_reduces_duplicate_vertices() {
        let dup = Mesh::from_triangles(&[
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
        ]);
        assert_eq!(dup.vertex_count(), 6);
        assert_eq!(dup.weld_vertices(1e-6).vertex_count(), 3);
    }

    #[test]
    fn equilateral_aspect_ratio_is_one() {
        let s = 2.0_f32;
        let tri = Mesh::from_triangles(&[[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(s, 0.0, 0.0),
            Point3::new(s / 2.0, s * (3.0_f32.sqrt()) / 2.0, 0.0),
        ]]);
        assert!((tri.average_aspect_ratio() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn obj_round_trips_vertex_and_face_count() {
        let m = cube().weld_vertices(1e-6);
        let obj = m.to_obj();
        let parsed = Mesh::from_obj(&obj).unwrap().weld_vertices(1e-6);
        assert_eq!(parsed.face_count(), m.face_count());
        assert_eq!(parsed.vertex_count(), m.vertex_count());
    }

    #[test]
    fn obj_round_trips_texcoords_and_normals() {
        // Build a tiny mesh with explicit texture coordinates and per-corner
        // normal indices, exercising the extended OBJ codec.
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];
        let mut mesh = Mesh::from_positions_indices(positions, indices).unwrap();
        mesh.tex_coords = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        mesh.tex_indices = Some(vec![0, 1, 2]);
        mesh.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        mesh.normal_indices = Some(vec![0, 1, 2]);

        let obj = mesh.to_obj();
        let parsed = Mesh::from_obj(&obj).unwrap();
        assert_eq!(parsed.tex_coords, mesh.tex_coords);
        assert_eq!(parsed.tex_indices, mesh.tex_indices);
        assert_eq!(parsed.normals, mesh.normals);
        assert_eq!(parsed.normal_indices, mesh.normal_indices);

        // The `f` line must carry the per-corner indices (v/t/n), here
        // v=1..3, t=1..3 (matching `tex_indices`), n=1..3 (matching
        // `normal_indices`).
        assert!(obj.contains("f 1/1/1 2/2/2 3/3/3"));
    }

    #[test]
    fn stl_ascii_round_trips() {
        let m = cube().weld_vertices(1e-6);
        let text = m.to_stl_ascii();
        let parsed = Mesh::from_stl_ascii(&text).unwrap().weld_vertices(1e-6);
        assert_eq!(parsed.face_count(), m.face_count());
        assert_eq!(parsed.vertex_count(), m.vertex_count());
    }
}
