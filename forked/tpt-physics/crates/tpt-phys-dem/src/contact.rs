//! Hertz–Mindlin contact law for two rigid spheres.

/// Reduced (harmonic) modulus `E*` for two spheres with Young's moduli
/// `e1`, `e2` and Poisson ratios `nu1`, `nu2`.
pub fn reduced_modulus(e1: f64, nu1: f64, e2: f64, nu2: f64) -> f64 {
    let a = (1.0 - nu1 * nu1) / e1;
    let b = (1.0 - nu2 * nu2) / e2;
    1.0 / (a + b)
}

/// Reduced radius `R* = (1/R1 + 1/R2)⁻¹` for spheres of `r1`, `r2`.
pub fn reduced_radius(r1: f64, r2: f64) -> f64 {
    1.0 / (1.0 / r1 + 1.0 / r2)
}

/// Normal overlap `δ` of two spheres (positive when penetrating).
pub fn overlap(ri: &crate::particle::Particle, rj: &crate::particle::Particle) -> f64 {
    let dx = ri.position[0] - rj.position[0];
    let dy = ri.position[1] - rj.position[1];
    let dz = ri.position[2] - rj.position[2];
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    (ri.radius + rj.radius) - d
}

/// Hertz normal contact force magnitude
/// `F_n = (4/3) E* √R* δ^{3/2}` for overlap `delta > 0`, else `0`.
pub fn hertz_normal_force(e_star: f64, r_star: f64, delta: f64) -> f64 {
    if delta <= 0.0 {
        0.0
    } else {
        (4.0 / 3.0) * e_star * r_star.sqrt() * delta.powf(1.5)
    }
}

/// Map a coefficient of restitution `e ∈ [0, 1)` to the critical-damping
/// ratio `ζ` used for the (linearised) Hertzian contact normal damping:
/// `ζ = -ln(e) / √(π² + (ln e)²)`.
///
/// Returns `1.0` (critically/over-damped) for `e → 0`.
pub(crate) fn restitution_to_zeta(restitution: f64) -> f64 {
    if restitution < 1e-6 {
        1.0
    } else {
        let le = -restitution.ln();
        le / (std::f64::consts::PI * std::f64::consts::PI + le * le).sqrt()
    }
}

/// Contact force on particle `i` from particle `j`.
///
/// Returns the force vector `[fx, fy, fz]` to add to particle `i` (the equal
/// and opposite force applies to `j`). Combines the Hertz normal force with a
/// normal damping term and a Coulomb-capped Mindlin tangential friction force.
///
/// `friction` is the coefficient of friction `μ`; `restitution` `e` sets the
/// normal damping via the critical-damping approximation.
pub fn contact_force(
    ri: &crate::particle::Particle,
    rj: &crate::particle::Particle,
    e_star: f64,
    mu: f64,
    restitution: f64,
) -> [f64; 3] {
    let delta = overlap(ri, rj);
    if delta <= 0.0 {
        return [0.0; 3];
    }
    let r_star = reduced_radius(ri.radius, rj.radius);
    let n = normal_unit(ri, rj);
    let m_eff = reduced_mass(ri.mass, rj.mass);

    // Normal force (Hertz) + damping.
    let fn_hertz = hertz_normal_force(e_star, r_star, delta);
    // Relative normal velocity.
    let rv = [
        ri.velocity[0] - rj.velocity[0],
        ri.velocity[1] - rj.velocity[1],
        ri.velocity[2] - rj.velocity[2],
    ];
    let vn = rv[0] * n[0] + rv[1] * n[1] + rv[2] * n[2];
    // Tangent normal contact stiffness dF_n/dδ = 2 E* √R* √δ (Hertzian), used
    // for the critical-damping coefficient below.
    let kn = 2.0 * e_star * r_star.sqrt() * delta.sqrt();
    let zeta = restitution_to_zeta(restitution);
    let cn = 2.0 * zeta * (kn * m_eff).sqrt();
    let f_n = fn_hertz - cn * vn; // repulsive when overlapping & approaching

    let mut f = [0.0; 3];
    for k in 0..3 {
        f[k] += f_n * n[k];
    }

    // Tangential friction (Mindlin elastic stiffness + Coulomb cap).
    let vt = [rv[0] - vn * n[0], rv[1] - vn * n[1], rv[2] - vn * n[2]];
    let vt_mag = (vt[0] * vt[0] + vt[1] * vt[1] + vt[2] * vt[2]).sqrt();
    if vt_mag > 1e-12 {
        let kt = 8.0 * (e_star / (2.0 * (1.0 + 0.3))) * r_star.sqrt() * delta.sqrt();
        let ft_mag = (kt * vt_mag).min(mu * f_n.abs());
        let inv = 1.0 / vt_mag;
        for k in 0..3 {
            f[k] -= ft_mag * vt[k] * inv; // opposes tangential relative motion
        }
    }
    f
}

/// Reduced mass `m* = (1/m1 + 1/m2)⁻¹`.
pub fn reduced_mass(m1: f64, m2: f64) -> f64 {
    1.0 / (1.0 / m1 + 1.0 / m2)
}

/// Unit normal `n̂` pointing from `j` to `i`.
fn normal_unit(ri: &crate::particle::Particle, rj: &crate::particle::Particle) -> [f64; 3] {
    let dx = ri.position[0] - rj.position[0];
    let dy = ri.position[1] - rj.position[1];
    let dz = ri.position[2] - rj.position[2];
    let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
    [dx / d, dy / d, dz / d]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    #[test]
    fn no_force_when_separated() {
        let a = Particle::new([0.0, 0.0, 0.0], [0.0; 3], 0.5, 1000.0);
        let b = Particle::new([2.0, 0.0, 0.0], [0.0; 3], 0.5, 1000.0);
        let f = contact_force(&a, &b, 1e9, 0.5, 0.3);
        assert!(f.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn repulsive_when_overlapping() {
        let a = Particle::new([0.0, 0.0, 0.0], [0.0; 3], 0.5, 1000.0);
        let b = Particle::new([0.6, 0.0, 0.0], [0.0; 3], 0.5, 1000.0); // overlap 0.4
        let e_star = reduced_modulus(1e9, 0.3, 1e9, 0.3);
        let f = contact_force(&a, &b, e_star, 0.5, 0.3);
        // `b` lies at +x, so the repulsive force on `a` points -x (away from `b`).
        assert!(f[0] < 0.0, "fx = {}", f[0]);
    }

    #[test]
    fn restitution_to_zeta_matches_closed_form() {
        // ζ = -ln(e) / √(π² + (ln e)²) (mass-spring-dashpot relation).
        let cases = [
            (0.0, 1.0), // fully inelastic ⇒ critical damping
            (0.2, 0.4560),
            (0.5, 0.2156),
            (0.9, 0.0335),
        ];
        for (e, expected) in cases {
            let z = restitution_to_zeta(e);
            assert!(
                (z - expected).abs() < 1e-3,
                "ζ(e={e}) = {z:.4}, expected {expected}"
            );
        }
    }
}
