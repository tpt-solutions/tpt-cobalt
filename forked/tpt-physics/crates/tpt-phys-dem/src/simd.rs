//! SIMD-accelerated narrow-phase contact resolution.
//!
//! The per-pair Hertz normal force is a scalar map of the overlap `δ`. For a
//! monodisperse particle cloud (all radii equal, hence a single reduced radius
//! `r*`) the force on four contact pairs can be evaluated simultaneously with
//! `wide::f64x4` SIMD vectors, packing the four `δ`, `√δ`, and normal
//! components into 128-bit lanes.

use wide::f64x4;

/// Normal contact force on four particle-`i` instances from four particle-`j`
/// instances, evaluated with SIMD.
///
/// `r_star` is the reduced radius (for monodisperse particles of radius `r`,
/// `r* = r/2`); `e_star` is the reduced modulus. Only the Hertz normal force is
/// computed here (the scalar [`crate::contact::contact_force`] adds damping and
/// tangential friction on top).
pub fn normal_forces_simd(
    pos_i: &[[f64; 3]; 4],
    pos_j: &[[f64; 3]; 4],
    r_star: f64,
    e_star: f64,
) -> [[f64; 3]; 4] {
    let dx = f64x4::from([pos_i[0][0], pos_i[1][0], pos_i[2][0], pos_i[3][0]])
        - f64x4::from([pos_j[0][0], pos_j[1][0], pos_j[2][0], pos_j[3][0]]);
    let dy = f64x4::from([pos_i[0][1], pos_i[1][1], pos_i[2][1], pos_i[3][1]])
        - f64x4::from([pos_j[0][1], pos_j[1][1], pos_j[2][1], pos_j[3][1]]);
    let dz = f64x4::from([pos_i[0][2], pos_i[1][2], pos_i[2][2], pos_i[3][2]])
        - f64x4::from([pos_j[0][2], pos_j[1][2], pos_j[2][2], pos_j[3][2]]);

    let d2 = dx * dx + dy * dy + dz * dz;
    let d = d2.sqrt();
    // Overlap δ = (r1 + r2) - d, clamped at 0. For a monodisperse cloud of
    // radius `r`, r* = r/2, so r1 + r2 = 2r = 4 r*.
    let r_sum = f64x4::splat(4.0 * r_star);
    let delta = (r_sum - d).max(f64x4::splat(0.0));

    // F_n = (4/3) E* √R* δ^{3/2} = (4/3) E* √R* δ √δ.
    let k = f64x4::splat(4.0 / 3.0) * f64x4::splat(e_star) * f64x4::splat(r_star.sqrt());
    let f_n = k * delta * delta.sqrt();

    // Unit normal (guard against d == 0).
    let safe_d = d.max(f64x4::splat(1e-12));
    let nx = dx / safe_d;
    let ny = dy / safe_d;
    let nz = dz / safe_d;

    let fx = (f_n * nx).to_array();
    let fy = (f_n * ny).to_array();
    let fz = (f_n * nz).to_array();

    let mut out = [[0.0_f64; 3]; 4];
    for i in 0..4 {
        out[i] = [fx[i], fy[i], fz[i]];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::{hertz_normal_force, reduced_modulus, reduced_radius};

    #[test]
    fn simd_matches_scalar_hertz() {
        let ri = 0.5;
        let rj = 0.5;
        let e_star = reduced_modulus(1e9, 0.3, 1e9, 0.3);
        let r_star = reduced_radius(ri, rj);

        // Four overlapping pairs with varying separations.
        let pos_i = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let pos_j = [
            [0.6, 0.0, 0.0], // δ = 0.4
            [0.3, 0.0, 0.0], // δ = 0.7
            [1.0, 0.0, 0.0], // touching, δ = 0
            [5.0, 0.0, 0.0], // far, δ < 0
        ];
        let f = normal_forces_simd(&pos_i, &pos_j, r_star, e_star);

        // Pair 0: `j` lies at +x, so the force on `i` points -x with magnitude
        // = hertz(δ).
        let expected0 = hertz_normal_force(e_star, r_star, 0.4);
        assert!(
            (f[0][0] + expected0).abs() < 1e-6,
            "{} vs {}",
            f[0][0],
            -expected0
        );
        // Pair 2: no contact.
        assert!(f[2].iter().all(|v| v.abs() < 1e-9));
        // Pair 3: separated, no force.
        assert!(f[3].iter().all(|v| v.abs() < 1e-9));
    }
}
