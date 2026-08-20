//! Static obstacle boundaries for the DEM [`World`](crate::world::World).
//!
//! The base [`World`](crate::world::World) only ships a planar floor. Real
//! validation scenarios (soil–structure interaction around a cylindrical
//! spacer, aggregate flow through a pile cage, hopper discharge, packed-bed
//! containers) need *fixed* geometry that granular particles cannot penetrate.
//!
//! [`Obstacle`] supplies two such boundaries:
//!
//! * [`Obstacle::Cylinder`] — a capped (smooth) cylinder, used as the
//!   3D-printed spacer column and as the pile cage;
//! * [`Obstacle::Plane`] — a half-space wall (oriented by an outward normal),
//!   used to build hopper funnels and rectangular containments.

use crate::contact::restitution_to_zeta;
use crate::particle::Particle;
use serde::{Deserialize, Serialize};

/// A fixed, immovable boundary that particles bounce off / settle against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Obstacle {
    /// A capped cylinder. `center` is the midpoint of the axis segment,
    /// `axis` a unit direction, `radius` the cylinder radius, and
    /// `half_height` half the axial length.
    Cylinder {
        center: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        half_height: f64,
    },
    /// A solid half-space wall. `point` lies on the wall and `normal` is the
    /// unit **outward** normal (pointing away from the region particles must
    /// stay inside). A particle is kept on the interior side
    /// (`(p − point)·normal ≤ −radius`). `y_range`, when set, limits the wall
    /// to a finite vertical band `[lo, hi]` (used for hopper funnels so that
    /// particles which have discharged below the orifice are no longer
    /// constrained).
    Plane {
        point: [f64; 3],
        normal: [f64; 3],
        y_range: Option<[f64; 2]>,
    },
}

/// Result of resolving one particle against one obstacle for one step.
struct Resolution {
    /// Contact force to add to the particle.
    force: [f64; 3],
    /// Position correction (de-penetration) to apply to the particle.
    correction: [f64; 3],
}

impl Obstacle {
    /// Resolve `p` against this obstacle, returning the contact force and any
    /// positional de-penetration. `e_star` is the contact modulus, `mu` the
    /// friction coefficient and `restitution` sets normal damping. The obstacle
    /// is treated as rigid (`m → ∞`), so the particle's own radius sets the
    /// contact curvature.
    pub fn resolve(
        &self,
        p: &Particle,
        e_star: f64,
        mu: f64,
        restitution: f64,
    ) -> Option<([f64; 3], [f64; 3])> {
        let r = Resolution::from(self, p, e_star, mu, restitution)?;
        if r.correction.iter().all(|c| *c == 0.0) && r.force.iter().all(|c| *c == 0.0) {
            None
        } else {
            Some((r.force, r.correction))
        }
    }
}

impl Resolution {
    fn from(
        obs: &Obstacle,
        p: &Particle,
        e_star: f64,
        mu: f64,
        restitution: f64,
    ) -> Option<Resolution> {
        match obs {
            Obstacle::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => Self::cylinder(
                p,
                center,
                axis,
                *radius,
                *half_height,
                e_star,
                mu,
                restitution,
            ),
            Obstacle::Plane {
                point,
                normal,
                y_range,
            } => Self::plane(p, point, normal, *y_range, e_star, mu, restitution),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cylinder(
        p: &Particle,
        center: &[f64; 3],
        axis: &[f64; 3],
        radius: f64,
        half_height: f64,
        e_star: f64,
        mu: f64,
        restitution: f64,
    ) -> Option<Resolution> {
        // Closest point on the (capped) axis segment to the particle centre.
        let w = sub(p.position, *center);
        let t = clamp(dot(w, *axis), -half_height, half_height);
        let c = add(*center, scale(axis, t));
        let r_vec = sub(p.position, c);
        let rd = norm(r_vec).max(1e-12);
        let pen = (radius + p.radius) - rd;
        if pen <= 0.0 {
            return None;
        }
        let n = scale(&r_vec, 1.0 / rd); // outward (from axis to particle)
        Some(contact(p, n, pen, e_star, mu, restitution))
    }

    #[allow(clippy::too_many_arguments)]
    fn plane(
        p: &Particle,
        point: &[f64; 3],
        normal: &[f64; 3],
        y_range: Option<[f64; 2]>,
        e_star: f64,
        mu: f64,
        restitution: f64,
    ) -> Option<Resolution> {
        // Finite-band walls (e.g. hopper funnels): only constrain particles
        // within the active vertical band; discharged particles below the
        // orifice are left free.
        if let Some([lo, hi]) = y_range {
            if p.position[1] < lo || p.position[1] > hi {
                return None;
            }
        }
        // Signed distance; interior side is negative (normal points outward).
        let sd = dot(sub(p.position, *point), *normal);
        let pen = sd + p.radius; // > 0 ⇒ particle has breached the wall
        if pen <= 0.0 {
            return None;
        }
        // Push back toward the interior (−normal).
        let n = scale(normal, -1.0);
        Some(contact(p, n, pen, e_star, mu, restitution))
    }
}

/// Build the contact force + de-penetration for a sphere against a rigid
/// surface with unit outward normal `n` and penetration depth `pen`.
#[allow(clippy::too_many_arguments)]
fn contact(
    p: &Particle,
    n: [f64; 3],
    pen: f64,
    e_star: f64,
    mu: f64,
    restitution: f64,
) -> Resolution {
    let pr = p.radius;
    // Hertz normal stiffness (tangent, dF_n/dδ = 2 E* √r √δ) and critical
    // damping (rigid obstacle ⇒ m_eff = m).
    let kn = 2.0 * e_star * pr.sqrt() * pen.sqrt();
    let fn_hertz = (4.0 / 3.0) * e_star * pr.sqrt() * pen.powf(1.5);
    let zeta = restitution_to_zeta(restitution);
    let cn = 2.0 * zeta * (kn * p.mass).sqrt();
    let vn = dot(p.velocity, n);
    let f_n = fn_hertz - cn * vn; // repulsive when overlapping & approaching

    let mut force = scale(&n, f_n);

    // Coulomb-capped Mindlin tangential friction.
    let vt = sub(p.velocity, scale(&n, vn));
    let vt_mag = norm(vt);
    if vt_mag > 1e-12 {
        let kt = 8.0 * (e_star / (2.0 * (1.0 + 0.3))) * pr.sqrt() * pen.sqrt();
        let ft_mag = (kt * vt_mag).min(mu * f_n.abs());
        force = sub(force, scale(&vt, ft_mag / vt_mag));
    }

    let correction = scale(&n, pen);
    Resolution { force, correction }
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scale(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt().max(1e-12)
}
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylinder_keeps_particle_outside() {
        let obs = Obstacle::Cylinder {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            radius: 1.0,
            half_height: 5.0,
        };
        let p = Particle::new([0.5, 0.0, 0.0], [0.0; 3], 0.2, 1000.0);
        let r = obs.resolve(&p, 1e9, 0.5, 0.2).unwrap();
        // Penetration pushes particle further from the axis (positive x).
        assert!(r.1[0] > 0.0, "correction = {:?}", r.1);
    }

    #[test]
    fn plane_keeps_particle_on_interior_side() {
        let obs = Obstacle::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            y_range: None,
        };
        // Particle just below the floor (sd = -0.1, but radius 0.2 ⇒ penetrates).
        let p = Particle::new([0.0, -0.1, 0.0], [0.0; 3], 0.2, 1000.0);
        let r = obs.resolve(&p, 1e9, 0.5, 0.2).unwrap();
        // Outward normal is +y, so push is toward interior (−y).
        assert!(r.1[1] < 0.0, "correction = {:?}", r.1);
    }
}
