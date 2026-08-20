//! # tpt-eng-geotech
//!
//! Soil mechanics primitives for the TPT engineering ecosystem: shear-strength
//! failure criteria ([`mohr_coulomb`]), a reduced critical-state (Cam-Clay)
//! model ([`cam_clay`]), shallow-foundation bearing capacity
//! ([`bearing_capacity`]), 1-D consolidation settlement and time-rate
//! ([`consolidation`]), lateral earth pressure ([`lateral_earth_pressure`]),
//! Atterberg limits and USCS classification ([`atterberg`]), and borehole
//! stratigraphy with provenance tracking.
//!
//! ## Units
//!
//! All stress, pressure, and strength quantities are in **pascals (Pa)** unless
//! noted otherwise, depths are in **metres (m)**, angles in **degrees**, and
//! void ratios are dimensionless. No unit library (e.g. `uom`) is used; SI units
//! are documented at each public item and are the caller's responsibility.
//!
//! ## Provenance
//!
//! Soil layers record where their data came from via the
//! [`tpt_eng_materials::DataSource`] type, shared with the materials crate so
//! that geotechnical records can participate in the same data-policy checks.
#![forbid(unsafe_code)]

use tpt_eng_materials::DataSource;

/// A single stratigraphic unit recovered or inferred from a borehole.
///
/// Depths are measured downwards from ground level (0.0 at the surface) in
/// metres. Strength parameters follow the Mohr-Coulomb model documented in the
/// [`mohr_coulomb`] module.
#[derive(Debug, Clone, PartialEq)]
pub struct SoilLayer {
    /// Top depth of the layer below ground level, in metres.
    pub depth_top: f64,
    /// Bottom depth of the layer below ground level, in metres.
    ///
    /// Must satisfy `depth_bottom > depth_top`; the crate does not enforce this
    /// but downstream summaries assume it.
    pub depth_bottom: f64,
    /// Free-text classification of the soil (e.g. `"sand"`, `"clay"`).
    pub soil_type: String,
    /// Mohr-Coulomb friction angle, in degrees.
    pub friction_angle_deg: f64,
    /// Mohr-Coulomb cohesion, in pascals.
    pub cohesion: f64,
    /// Bulk unit weight γ, in newtons per cubic metre (N/m³).
    pub unit_weight: f64,
    /// Provenance of this layer's data.
    pub source: DataSource,
}

impl SoilLayer {
    /// Thickness of the layer in metres (`depth_bottom - depth_top`).
    #[must_use]
    pub fn thickness(&self) -> f64 {
        self.depth_bottom - self.depth_top
    }
}

/// A vertical soil investigation (borehole) at a single location.
///
/// Layers are expected to be ordered from shallow to deep and (typically)
/// contiguous, but the summaries in this crate work with any ordering of
/// non-overlapping layers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Borehole {
    /// Human-readable location label for the borehole.
    pub location: String,
    /// The stratigraphic layers, shallowest first.
    pub layers: Vec<SoilLayer>,
}

impl Borehole {
    /// Create an empty borehole at the given location.
    #[must_use]
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            layers: Vec::new(),
        }
    }

    /// Total investigated depth in metres, summed over all layer thicknesses.
    ///
    /// This is the cumulative thickness of the logged layers, not necessarily
    /// the depth of the deepest layer boundary if layers are non-contiguous.
    #[must_use]
    pub fn total_depth(&self) -> f64 {
        self.layers.iter().map(SoilLayer::thickness).sum()
    }

    /// Return the layer that contains depth `d` (metres below ground level).
    ///
    /// A layer contains `d` when `layer.depth_top <= d < layer.depth_bottom`.
    /// Returns `None` if `d` lies above the shallowest layer, below the deepest
    /// layer, or within a gap between non-contiguous layers.
    #[must_use]
    pub fn layer_at_depth(&self, d: f64) -> Option<&SoilLayer> {
        self.layers
            .iter()
            .find(|l| d >= l.depth_top && d < l.depth_bottom)
    }
}

/// Mohr-Coulomb and reduced critical-state soil models.
pub mod mohr_coulomb {
    /// Mohr-Coulomb shear strength (failure envelope): `τ_f = c + σ_n · tan φ`.
    ///
    /// # Parameters
    ///
    /// * `cohesion` — cohesion `c` in pascals.
    /// * `friction_angle_deg` — friction angle `φ` in degrees (`[0, 90)`).
    /// * `normal_stress` — total or effective normal stress `σ_n` in pascals.
    ///
    /// # Returns
    ///
    /// The shear strength `τ_f` in pascals at the given normal stress.
    #[must_use]
    pub fn shear_strength(cohesion: f64, friction_angle_deg: f64, normal_stress: f64) -> f64 {
        let phi = friction_angle_deg.to_radians();
        cohesion + normal_stress * phi.tan()
    }

    /// Factor of safety against Mohr-Coulomb shear failure.
    ///
    /// Defined as `FoS = τ_f / τ`, where `τ_f` is the [`shear_strength`] and
    /// `τ` is the applied shear stress. A value of `1.0` means the soil is
    /// exactly at failure; `FoS > 1` is stable and `FoS < 1` has failed.
    ///
    /// # Parameters
    ///
    /// * `shear_stress` — applied shear stress `τ` in pascals.
    /// * `cohesion` — cohesion `c` in pascals.
    /// * `friction_angle_deg` — friction angle `φ` in degrees.
    /// * `normal_stress` — normal stress `σ_n` in pascals.
    ///
    /// # Returns
    ///
    /// The dimensionless factor of safety. If `shear_stress` is `0.0` the result
    /// is `±∞` (or `NaN` for `0.0` strength), since this is ordinary `f64`
    /// division and no panic occurs.
    #[must_use]
    pub fn factor_of_safety(
        shear_stress: f64,
        cohesion: f64,
        friction_angle_deg: f64,
        normal_stress: f64,
    ) -> f64 {
        shear_strength(cohesion, friction_angle_deg, normal_stress) / shear_stress
    }
}

/// Reduced critical-state (Cam-Clay) soil model.
pub mod cam_clay {
    /// Evaluate the modified Cam-Clay yield surface.
    ///
    /// The surface is `f(p, q) = q² + M² · p · (p − p_c)`, where `p` is the mean
    /// effective stress, `q` is the deviatoric stress, `M` (`m`) is the slope of
    /// the critical-state line, and `p_c` is the pre-consolidation pressure.
    ///
    /// # Returns
    ///
    /// * `0.0` exactly on the yield surface,
    /// * `< 0.0` for an elastic (interior) state,
    /// * `> 0.0` for a state outside the yield surface (plastic over-stress).
    ///
    /// # Parameters
    ///
    /// * `p` — mean effective stress in pascals (should be `> 0`).
    /// * `q` — deviatoric stress in pascals.
    /// * `m` — critical-state line slope `M` (dimensionless, `> 0`).
    /// * `pc` — pre-consolidation pressure in pascals (`> 0`).
    #[must_use]
    pub fn cam_clay_yield(p: f64, q: f64, m: f64, pc: f64) -> f64 {
        q * q + m * m * p * (p - pc)
    }

    /// Whether a void-ratio change follows normal consolidation or swelling.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ConsolidationMode {
        /// Virgin loading: use the compression index `λ`.
        Normal,
        /// Unloading/reloading: use the swelling index `κ`.
        Swelling,
    }

    /// Update the void ratio `e` under a change in mean effective stress.
    ///
    /// For normal consolidation `e = e0 − λ · ln(p1 / p0)` and for swelling
    /// `e = e0 − κ · ln(p1 / p0)`, where `λ` (`lambda`) is the compression index
    /// and `κ` (`kappa`) is the swelling index.
    ///
    /// # Parameters
    ///
    /// * `e0` — initial void ratio (dimensionless).
    /// * `lambda` — compression index `λ` (dimensionless, normal consolidation).
    /// * `kappa` — swelling index `κ` (dimensionless, swelling).
    /// * `p0` — initial mean effective stress in pascals.
    /// * `p1` — final mean effective stress in pascals.
    /// * `mode` — [`ConsolidationMode`] selecting `λ` or `κ`.
    ///
    /// # Panics
    ///
    /// Panics if `p0 <= 0` or `p1 <= 0`, since the logarithm of a non-positive
    /// pressure is undefined.
    #[must_use]
    pub fn void_ratio_update(
        e0: f64,
        lambda: f64,
        kappa: f64,
        p0: f64,
        p1: f64,
        mode: ConsolidationMode,
    ) -> f64 {
        assert!(p0 > 0.0, "initial pressure p0 must be positive");
        assert!(p1 > 0.0, "final pressure p1 must be positive");
        let index = match mode {
            ConsolidationMode::Normal => lambda,
            ConsolidationMode::Swelling => kappa,
        };
        e0 - index * (p1 / p0).ln()
    }
}

pub use cam_clay::{ConsolidationMode, cam_clay_yield, void_ratio_update};
pub use mohr_coulomb::{factor_of_safety, shear_strength};

/// Shallow-foundation ultimate bearing capacity (Terzaghi / Meyerhof).
pub mod bearing_capacity {
    /// Foundation plan-shape used for bearing-capacity shape factors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FoundationShape {
        /// Continuous (strip) footing: length ≫ width, no end effects.
        Strip,
        /// Square footing: length == width.
        Square,
        /// Circular footing (treated as `B == L`): 1-D end-bearing.
        Circular,
        /// Rectangular footing: explicit length `L` (uses `B / L`).
        Rectangle,
    }

    /// Terzaghi / Meyerhof bearing-capacity factors `(Nc, Nq, Nγ)` for a
    /// friction angle `φ` (degrees).
    ///
    /// Uses the standard forms `Nq = e^{π·tanφ}·tan²(45° + φ/2)`,
    /// `Nc = (Nq − 1)/tanφ`, `Nγ = 2·(Nq + 1)·tanφ` (the common Vesic/Meyerhof
    /// `Nγ`). For `φ = 0` (pure clay) it returns `(5.14, 1.0, 0.0)`.
    #[must_use]
    pub fn bearing_capacity_factors(friction_angle_deg: f64) -> (f64, f64, f64) {
        if friction_angle_deg <= 0.0 {
            return (5.14, 1.0, 0.0);
        }
        let phi = friction_angle_deg.to_radians();
        let sin_phi = phi.sin();
        let nq = ((1.0 + sin_phi) / (1.0 - sin_phi)) * (std::f64::consts::PI * phi.tan()).exp();
        let nc = (nq - 1.0) / phi.tan();
        let ngamma = 2.0 * (nq + 1.0) * phi.tan();
        (nc, nq, ngamma)
    }

    /// Terzaghi shape factors `(sc, sq, sγ)` for the given footing shape.
    ///
    /// `b_over_l` is the width/length ratio (ignored for `Strip`, `Square`,
    /// `Circular`). Strip uses `1.0` for all three; square/circular use the
    /// classic Terzaghi values `(1.3, 1.2, 0.8)` / `(1.3, 1.2, 0.6)`; rectangle
    /// interpolates with the `B / L` ratio.
    #[must_use]
    pub fn terzaghi_shape_factors(shape: FoundationShape, b_over_l: f64) -> (f64, f64, f64) {
        match shape {
            FoundationShape::Strip => (1.0, 1.0, 1.0),
            FoundationShape::Square => (1.3, 1.2, 0.8),
            FoundationShape::Circular => (1.3, 1.2, 0.6),
            FoundationShape::Rectangle => {
                let r = b_over_l.clamp(0.0, 1.0);
                (1.0 + 0.2 * r, 1.0 + 0.1 * r, (1.0 - 0.2 * r).max(0.0))
            }
        }
    }

    /// Terzaghi ultimate bearing capacity `q_ult` (Pa) for a shallow footing.
    ///
    /// `q_ult = c·Nc·sc + q'·Nq·sq + ½·γ·B·Nγ·sγ` where the effective
    /// overburden at the base is `q' = γ·depth` (water table assumed deep),
    /// `B` is `width`, `γ` is `unit_weight`, and `c`/`φ` are `cohesion` /
    /// `friction_angle_deg`. `length` is only used for the rectangular shape
    /// factor (pass `width` for square/circular).
    #[must_use]
    pub fn terzaghi_ultimate_bearing_capacity(
        cohesion: f64,
        friction_angle_deg: f64,
        unit_weight: f64,
        width: f64,
        depth: f64,
        shape: FoundationShape,
        length: f64,
    ) -> f64 {
        let (nc, nq, ngamma) = bearing_capacity_factors(friction_angle_deg);
        let (sc, sq, sgamma) =
            terzaghi_shape_factors(shape, if length > 0.0 { width / length } else { 1.0 });
        let q_prime = unit_weight * depth;
        cohesion * nc * sc + q_prime * nq * sq + 0.5 * unit_weight * width * ngamma * sgamma
    }

    /// Net ultimate bearing capacity: `q_ult − γ·depth` (the stress in excess of
    /// the removed overburden).
    #[must_use]
    pub fn net_ultimate_bearing_capacity(q_ult: f64, unit_weight: f64, depth: f64) -> f64 {
        q_ult - unit_weight * depth
    }

    /// Allowable bearing capacity for a given factor of safety `fs`.
    #[must_use]
    pub fn allowable_bearing_capacity(q_ult: f64, fs: f64) -> f64 {
        q_ult / fs
    }

    /// Meyerhof ultimate bearing capacity `q_ult` (Pa) with explicit shape and
    /// depth factors (`depth / width ≤ 1`).
    ///
    /// Uses Meyerhof's shape factors `F_qs = 1 + (B/L)(Nq/Nc)·tanφ`,
    /// `F_cs = (F_qs·Nq − 1)/(Nq − 1)`, `F_γs = 1 − 0.4·(B/L)` and depth factors
    /// `F_qd = 1 + 0.1(D/B)(Nq/Nc)·tanφ`, `F_cd = 1 + 0.2(D/B)(Nc/Nq)·tanφ`
    /// (the `Nγ` term uses no depth factor). For `depth / width > 1` the depth
    /// factors are held at their `D/B = 1` value.
    #[must_use]
    pub fn meyerhof_ultimate_bearing_capacity(
        cohesion: f64,
        friction_angle_deg: f64,
        unit_weight: f64,
        width: f64,
        depth: f64,
        length: f64,
    ) -> f64 {
        let (nc, nq, ngamma) = bearing_capacity_factors(friction_angle_deg);
        let phi = friction_angle_deg.to_radians();
        let b_over_l = if length > 0.0 { width / length } else { 1.0 }.clamp(0.0, 1.0);
        let d_over_b = if width > 0.0 {
            (depth / width).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let fqs = 1.0 + b_over_l * (nq / nc) * phi.tan();
        let fcs = if (nq - 1.0).abs() > 1e-9 {
            (fqs * nq - 1.0) / (nq - 1.0)
        } else {
            1.0 + 0.2 * b_over_l
        };
        let fgs = (1.0 - 0.4 * b_over_l).max(0.0);
        let fqd = 1.0 + 0.1 * d_over_b * (nq / nc) * phi.tan();
        let fcd = 1.0 + 0.2 * d_over_b * (nc / nq) * phi.tan();

        let q_prime = unit_weight * depth;
        cohesion * nc * fcs * fcd
            + q_prime * nq * fqs * fqd
            + 0.5 * unit_weight * width * ngamma * fgs
    }
}

/// One-dimensional primary consolidation: settlement and time-rate (Terzaghi).
pub mod consolidation {
    /// Coefficient of consolidation `c_v = k·(1 + e)/(a_v·γ_w)` (m²/s), from
    /// permeability `k` (m/s), void ratio `e`, compressibility `a_v` (1/Pa) and
    /// water unit weight `γ_w` (N/m³).
    #[must_use]
    pub fn coeff_consolidation(
        permeability: f64,
        void_ratio: f64,
        av: f64,
        gamma_water: f64,
    ) -> f64 {
        permeability * (1.0 + void_ratio) / (av * gamma_water)
    }

    /// 1-D primary consolidation settlement `S = (Cc/(1 + e₀))·H·log₁₀((σ₀ + Δσ)/σ₀)`
    /// (metres), for compression index `Cc`, initial void ratio `e0`, layer
    /// thickness `H` and effective stress change `σ₀ → σ₀ + Δσ`.
    #[must_use]
    pub fn consolidation_settlement(
        cc: f64,
        e0: f64,
        h: f64,
        sigma0: f64,
        delta_sigma: f64,
    ) -> f64 {
        (cc / (1.0 + e0)) * h * ((sigma0 + delta_sigma) / sigma0).log10()
    }

    /// Time factor `T_v` from the average degree of consolidation `U` (percent,
    /// `0…100`). Uses the closed-form small-strain relation for `U < 60%` and
    /// Taylor's logarithmic relation for `U ≥ 60%`.
    #[must_use]
    pub fn time_factor_from_degree(u_percent: f64) -> f64 {
        let u = (u_percent / 100.0).clamp(0.0, 0.9999);
        if u_percent < 60.0 {
            std::f64::consts::FRAC_PI_4 * u * u
        } else {
            -0.933 * (1.0 - u).log10() - 0.085
        }
    }

    /// Average degree of consolidation (percent, `0…100`) from the time factor
    /// `T_v`. Inverse of [`time_factor_from_degree`].
    #[must_use]
    pub fn degree_from_time_factor(tv: f64) -> f64 {
        if tv <= 0.2827 {
            100.0 * (4.0 * tv / std::f64::consts::PI).sqrt()
        } else {
            100.0 * (1.0 - 10f64.powf(-(tv + 0.085) / 0.933))
        }
    }

    /// Time `t = T_v·H_dr² / c_v` (seconds) to reach `u_percent` consolidation,
    /// for coefficient of consolidation `c_v` and drainage half-height `H_dr`
    /// (full layer thickness for double drainage, half thickness for single).
    #[must_use]
    pub fn consolidation_time(cv: f64, drain_half_height: f64, u_percent: f64) -> f64 {
        time_factor_from_degree(u_percent) * drain_half_height * drain_half_height / cv
    }
}

/// Lateral earth-pressure coefficients and resultants (Rankine / Coulomb).
pub mod lateral_earth_pressure {
    /// Rankine active earth-pressure coefficient `K_a = tan²(45° − φ/2)`.
    #[must_use]
    pub fn rankine_ka(friction_angle_deg: f64) -> f64 {
        let a = (45.0 - friction_angle_deg / 2.0).to_radians();
        a.tan() * a.tan()
    }

    /// Rankine passive earth-pressure coefficient `K_p = tan²(45° + φ/2)`.
    #[must_use]
    pub fn rankine_kp(friction_angle_deg: f64) -> f64 {
        let a = (45.0 + friction_angle_deg / 2.0).to_radians();
        a.tan() * a.tan()
    }

    /// Rankine active resultant force per unit wall length (N/m), dry backfill
    /// with uniform surcharge `q` (Pa): `K_a·(q·H + ½·γ·H²)`.
    #[must_use]
    pub fn rankine_active_force(unit_weight: f64, height: f64, surcharge: f64, ka: f64) -> f64 {
        ka * (surcharge * height + 0.5 * unit_weight * height * height)
    }

    /// Rankine passive resultant force per unit wall length (N/m): `K_p·(q·H +
    /// ½·γ·H²)`.
    #[must_use]
    pub fn rankine_passive_force(unit_weight: f64, height: f64, surcharge: f64, kp: f64) -> f64 {
        kp * (surcharge * height + 0.5 * unit_weight * height * height)
    }

    /// Coulomb active earth-pressure coefficient `K_a`.
    ///
    /// Angles (degrees): `phi` soil friction, `delta` wall friction, `beta`
    /// backfill inclination from horizontal, `omega` wall-face inclination from
    /// vertical (0 for a vertical wall). Uses the standard Coulomb expression.
    #[must_use]
    pub fn coulomb_ka(phi_deg: f64, delta_deg: f64, beta_deg: f64, omega_deg: f64) -> f64 {
        let phi = phi_deg.to_radians();
        let delta = delta_deg.to_radians();
        let beta = beta_deg.to_radians();
        let omega = omega_deg.to_radians();
        let num = (phi - omega).cos() * (phi - omega).cos();
        let den = omega.cos()
            * omega.cos()
            * (delta + omega).cos()
            * (1.0
                + ((delta + phi).sin() * (phi - beta).sin()
                    / ((delta + omega).cos() * (beta - omega).cos()))
                .sqrt())
            .powi(2);
        num / den
    }
}

/// Atterberg limits and USCS index-property classification for fine-grained
/// soils.
pub mod atterberg {
    /// Consistency state of a fine-grained soil from its liquidity index.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ConsistencyState {
        /// Liquid (liquidity index > 1.0).
        Liquid,
        /// Plastic (0.0 ≤ liquidity index ≤ 1.0).
        Plastic,
        /// Semi-solid (−0.25 < liquidity index < 0.0).
        SemiSolid,
        /// Solid (liquidity index ≤ −0.25).
        Solid,
    }

    /// Plasticity index `PI = max(0, LL − PL)`.
    #[must_use]
    pub fn plasticity_index(liquid_limit: f64, plastic_limit: f64) -> f64 {
        (liquid_limit - plastic_limit).max(0.0)
    }

    /// Liquidity index `LI = (w − PL)/(LL − PL)` for water content `w`.
    #[must_use]
    pub fn liquidity_index(water_content: f64, liquid_limit: f64, plastic_limit: f64) -> f64 {
        (water_content - plastic_limit) / (liquid_limit - plastic_limit)
    }

    /// Consistency state from a liquidity index `LI`.
    #[must_use]
    pub fn consistency_state(liquid_index: f64) -> ConsistencyState {
        if liquid_index > 1.0 {
            ConsistencyState::Liquid
        } else if liquid_index >= 0.0 {
            ConsistencyState::Plastic
        } else if liquid_index > -0.25 {
            ConsistencyState::SemiSolid
        } else {
            ConsistencyState::Solid
        }
    }

    /// USCS fine-grained group symbol from the liquid limit `LL` and plastic
    /// limit `PL`. Returns `"CL"`, `"CH"` (clays), `"ML"`, `"MH"` (silts), or
    /// `"ML"` for a non-plastic (PI ≤ 0) fine-grained soil. Uses the A-line
    /// `PI = 0.73·(LL − 20)` to split clay vs. silt, and `LL = 50` to split
    /// low (`L`) vs. high (`H`) plasticity.
    #[must_use]
    pub fn uscs_fine_grained(liquid_limit: f64, plastic_limit: f64) -> &'static str {
        let pi = plasticity_index(liquid_limit, plastic_limit);
        if pi <= 0.0 {
            return "ML";
        }
        let a_line = 0.73 * (liquid_limit - 20.0);
        let is_clay = pi > a_line;
        if liquid_limit < 50.0 {
            if is_clay { "CL" } else { "ML" }
        } else if is_clay {
            "CH"
        } else {
            "MH"
        }
    }

    /// Soil activity `A = PI / (clay fraction, % passing 75 µm)`.
    #[must_use]
    pub fn activity(pi: f64, clay_fraction_pct: f64) -> f64 {
        pi / clay_fraction_pct
    }
}

pub use atterberg::{
    ConsistencyState, activity, consistency_state, liquidity_index, plasticity_index,
    uscs_fine_grained,
};
pub use bearing_capacity::{
    FoundationShape, allowable_bearing_capacity, bearing_capacity_factors,
    meyerhof_ultimate_bearing_capacity, net_ultimate_bearing_capacity, terzaghi_shape_factors,
    terzaghi_ultimate_bearing_capacity,
};
pub use consolidation::{
    coeff_consolidation, consolidation_settlement, consolidation_time, degree_from_time_factor,
    time_factor_from_degree,
};
pub use lateral_earth_pressure::{
    coulomb_ka, rankine_active_force, rankine_ka, rankine_kp, rankine_passive_force,
};
