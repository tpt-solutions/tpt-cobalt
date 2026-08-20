//! # tpt-eng-biomech
//!
//! Biomechanics primitives for the TPT engineering ecosystem.
//!
//! This crate provides two coherent, real subspaces:
//!
//! * **Hyperelastic constitutive models** for soft tissue under incompressible,
//!   uniaxial extension parameterized by the stretch ratio `λ` (ratio of
//!   deformed to reference length, dimensionless). Two models are provided:
//!   * the [`mooney_rivlin_stress`](crate::constitutive::mooney_rivlin_stress)
//!     two-parameter model, and
//!   * the [`ogden_stress`](crate::constitutive::ogden_stress)
//!     sum-of-terms model (which reduces to the neo-Hookean form for a single
//!     `α = 2` term).
//! * **Implant geometry primitives** ([`Stem`](crate::implant::Stem),
//!   [`Cup`](crate::implant::Cup)) built on the
//!   [`tpt_eng_geometry`] frame/point types, with closed-form volume and
//!   surface-area approximations.
//!
//! ## Units
//!
//! All scalar quantities are `f64` in SI units: stresses in pascals (Pa),
//! lengths in metres (m), and angles in degrees at the API boundary (converted
//! to radians internally). No unit library is used; units are documented on
//! each item.
//!
//! #![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use tpt_eng_geometry::frame::Frame3;
pub use tpt_eng_geometry::{Point3, Vector3};

/// Incompressible hyperelastic constitutive models.
///
/// All models here assume incompressibility (volume-preserving deformation,
/// `J = λ1·λ2·λ3 = 1`) and report the principal (true/Cauchy) stress for
/// uniaxial extension in the stretch direction, in pascals.
pub mod constitutive {
    /// True (Cauchy) stress, in pascals (Pa), for incompressible Mooney-Rivlin
    /// rubber-like material under uniaxial extension.
    ///
    /// Given the stretch ratio `λ = L / L0` (dimensionless, `λ > 0`), the
    /// principal stress is
    ///
    /// ```text
    /// σ = 2·(c1 + c2/λ)·(λ − 1/λ²)
    /// ```
    ///
    /// where `c1` and `c2` are the Mooney-Rivlin material constants (Pa). The
    /// form is valid for uniaxial tension with the lateral stretches fixed by
    /// incompressibility (`λ2 = λ3 = 1/√λ`).
    ///
    /// # Arguments
    ///
    /// * `lambda` — uniaxial stretch ratio `λ > 0` (dimensionless).
    /// * `c1` — first Mooney-Rivlin constant (Pa).
    /// * `c2` — second Mooney-Rivlin constant (Pa).
    ///
    /// # Panics
    ///
    /// Panics if `lambda <= 0`, which is outside the physically meaningful
    /// domain.
    #[must_use]
    pub fn mooney_rivlin_stress(lambda: f64, c1: f64, c2: f64) -> f64 {
        assert!(lambda > 0.0, "stretch ratio lambda must be positive");
        // (λ − 1/λ²) is the uniaxial stretch term; divide is exact for λ > 0.
        let stretch_term = lambda - (lambda * lambda).recip();
        let factor = c1 + c2 / lambda;
        2.0 * factor * stretch_term
    }

    /// True (Cauchy) stress, in pascals (Pa), for the incompressible Ogden
    /// model under uniaxial extension.
    ///
    /// The Ogden strain-energy density is
    ///
    /// ```text
    /// W = Σ_i μ_i/α_i·(λ1^αi + λ2^αi + λ3^αi − 3)
    /// ```
    ///
    /// with incompressibility `λ1·λ2·λ3 = 1`. For uniaxial extension
    /// `λ1 = λ` and the lateral stretches satisfy `λ2 = λ3 = 1/√λ`. The
    /// returned principal stress is the derivative of the reduced strain energy
    /// with respect to `λ1`:
    ///
    /// ```text
    /// σ = dW/dλ = Σ_i μ_i·(λ^(αi−1) − λ^(−αi/2 − 1))
    /// ```
    ///
    /// # Consistency
    ///
    /// A single term with `α = 2` reduces to the neo-Hookean form
    /// `μ·(λ − 1/λ²)`; see [`crate::constitutive::neo_hookean_stress`].
    ///
    /// # Arguments
    ///
    /// * `lambda` — uniaxial stretch ratio `λ > 0` (dimensionless).
    /// * `params` — slice of `(μ_i, α_i)` Ogden parameter pairs. `μ_i` is a
    ///   shear-like modulus (Pa) and `α_i` is the dimensionless exponent.
    ///
    /// # Panics
    ///
    /// Panics if `lambda <= 0`, which is outside the physically meaningful
    /// domain.
    #[must_use]
    pub fn ogden_stress(lambda: f64, params: &[(f64, f64)]) -> f64 {
        assert!(lambda > 0.0, "stretch ratio lambda must be positive");
        let mut sigma = 0.0;
        for &(mu, alpha) in params {
            let term1 = lambda.powf(alpha - 1.0);
            let term2 = lambda.powf(-alpha / 2.0 - 1.0);
            sigma += mu * (term1 - term2);
        }
        sigma
    }

    /// True (Cauchy) stress, in pascals (Pa), for the incompressible
    /// neo-Hookean model under uniaxial extension.
    ///
    /// This is the simplest hyperelastic form,
    ///
    /// ```text
    /// σ = μ·(λ − 1/λ²)
    /// ```
    ///
    /// where `μ` is the shear modulus (Pa). It is recovered from the Ogden
    /// model with a single term `(μ, α = 2)`, and from the Mooney-Rivlin model
    /// with `c2 = 0` (and `μ = 2·c1`).
    ///
    /// # Arguments
    ///
    /// * `lambda` — uniaxial stretch ratio `λ > 0` (dimensionless).
    /// * `mu` — shear modulus (Pa).
    ///
    /// # Panics
    ///
    /// Panics if `lambda <= 0`, which is outside the physically meaningful
    /// domain.
    #[must_use]
    pub fn neo_hookean_stress(lambda: f64, mu: f64) -> f64 {
        assert!(lambda > 0.0, "stretch ratio lambda must be positive");
        mu * (lambda - (lambda * lambda).recip())
    }
}

/// Implant geometry primitives built on [`tpt_eng_geometry`] frames and points.
///
/// Lengths are stored in metres (m) and angles in degrees at the API boundary.
/// Each implant carries a [`Frame3`] describing its pose in a world frame: the
/// frame's local `+Z` axis is the implant's long/symmetry axis.
pub mod implant {
    use crate::{Frame3, Point3, Vector3};

    /// A tapered cylindrical hip-implant stem.
    ///
    /// Modeled as a truncated cone (frustum) of revolution about the local `+Z`
    /// axis. All lengths are in metres (m).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Stem {
        /// Overall stem length along the long axis (m).
        pub length: f64,
        /// Diameter at the proximal (top) end (m).
        pub diameter_proximal: f64,
        /// Diameter at the distal (bottom) end (m).
        pub diameter_distal: f64,
        /// Pose of the stem in the world frame. The local `+Z` axis is the stem
        /// long axis; the proximal end is at local `+Z = +length/2`.
        pub placement: Frame3,
    }

    impl Stem {
        /// Construct a stem with the given dimensions (metres) and pose.
        ///
        /// # Panics
        ///
        /// Panics if `length`, `diameter_proximal`, or `diameter_distal` is not
        /// strictly positive.
        #[must_use]
        pub fn new(
            length: f64,
            diameter_proximal: f64,
            diameter_distal: f64,
            placement: Frame3,
        ) -> Self {
            assert!(length > 0.0, "stem length must be positive");
            assert!(
                diameter_proximal > 0.0,
                "proximal diameter must be positive"
            );
            assert!(diameter_distal > 0.0, "distal diameter must be positive");
            Self {
                length,
                diameter_proximal,
                diameter_distal,
                placement,
            }
        }

        /// Approximate volume (m³) of the stem as a truncated cone (frustum of
        /// revolution).
        ///
        /// ```text
        /// V = (π/3)·h·(R² + R·r + r²)
        /// ```
        ///
        /// where `h = length`, `R = diameter_proximal/2`, and `r =
        /// diameter_distal/2`. Reduces to the cylinder volume `π·R²·h` when the
        /// two diameters are equal.
        #[must_use]
        pub fn volume_approx(&self) -> f64 {
            let r = self.diameter_proximal / 2.0;
            let s = self.diameter_distal / 2.0;
            let h = self.length;
            std::f64::consts::PI / 3.0 * h * (r * r + r * s + s * s)
        }

        /// World-space direction of the stem's long axis (unit vector).
        #[must_use]
        pub fn axis_world(&self) -> Vector3 {
            self.placement.z_axis()
        }

        /// World-space position of the proximal (top) end centre (m).
        #[must_use]
        pub fn proximal_center(&self) -> Point3 {
            let z = self.length as f32 / 2.0;
            self.placement.to_world_point(Point3::new(0.0, 0.0, z))
        }

        /// World-space position of the distal (bottom) end centre (m).
        #[must_use]
        pub fn distal_center(&self) -> Point3 {
            let z = -(self.length as f32) / 2.0;
            self.placement.to_world_point(Point3::new(0.0, 0.0, z))
        }
    }

    /// A hemispherical acetabular cup implant.
    ///
    /// Modeled as a spherical cap (the bearing surface) of the inner radius.
    /// All lengths are in metres (m) and angles in degrees at the API boundary.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Cup {
        /// Inner (bearing) radius of the cup (m).
        pub inner_radius: f64,
        /// Outer radius of the cup shell (m). Must exceed `inner_radius`.
        pub outer_radius: f64,
        /// Opening half-angle from the symmetry axis to the rim (degrees, in
        /// `(0, 180]`). `180°` describes a full sphere; `90°` a hemisphere.
        pub opening_angle_deg: f64,
        /// Pose of the cup in the world frame. The local `+Z` axis is the cup
        /// symmetry axis (pointing out of the opening).
        pub placement: Frame3,
    }

    impl Cup {
        /// Construct a cup with the given dimensions and pose.
        ///
        /// # Panics
        ///
        /// Panics if `inner_radius` is not strictly positive, if
        /// `outer_radius <= inner_radius`, or if `opening_angle_deg` is not in
        /// `(0, 180]`.
        #[must_use]
        pub fn new(
            inner_radius: f64,
            outer_radius: f64,
            opening_angle_deg: f64,
            placement: Frame3,
        ) -> Self {
            assert!(inner_radius > 0.0, "inner radius must be positive");
            assert!(
                outer_radius > inner_radius,
                "outer radius must exceed inner radius"
            );
            assert!(
                opening_angle_deg > 0.0 && opening_angle_deg <= 180.0,
                "opening angle must be in (0, 180] degrees"
            );
            Self {
                inner_radius,
                outer_radius,
                opening_angle_deg,
                placement,
            }
        }

        /// Inner bearing-surface area (m²) of the cup as a spherical cap.
        ///
        /// ```text
        /// A = 2·π·R·h,  h = R·(1 − cos θ)
        /// ```
        ///
        /// where `R = inner_radius` and `θ` is the opening half-angle in
        /// radians. Equivalently `A = 2·π·R²·(1 − cos θ)`. For `θ = 90°` this
        /// gives the hemisphere area `2·π·R²`; for `θ = 180°` the full sphere
        /// area `4·π·R²`.
        #[must_use]
        pub fn surface_area(&self) -> f64 {
            let r = self.inner_radius;
            let theta = self.opening_angle_deg.to_radians();
            let h = r * (1.0 - theta.cos());
            2.0 * std::f64::consts::PI * r * h
        }

        /// Radial shell thickness (m): `outer_radius − inner_radius`.
        #[must_use]
        pub fn wall_thickness(&self) -> f64 {
            self.outer_radius - self.inner_radius
        }

        /// World-space direction of the cup symmetry axis (unit vector).
        #[must_use]
        pub fn axis_world(&self) -> Vector3 {
            self.placement.z_axis()
        }
    }
}
