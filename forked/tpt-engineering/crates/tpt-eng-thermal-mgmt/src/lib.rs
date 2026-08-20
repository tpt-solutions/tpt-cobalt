//! # tpt-eng-thermal-mgmt
//!
//! Electronics thermal-management primitives: straight-fin efficiency,
//! extruded heat-sink resistance, fan-curve operating points, and
//! junction-to-ambient thermal-resistance networks.
//!
//! All quantities are SI: lengths in metres (m), areas in square metres
//! (m²), temperatures in kelvin (K) or degrees Celsius (°C) as noted,
//! thermal resistances in kelvin per watt (K/W), heat-transfer
//! coefficients in watts per square metre per kelvin (W/(m²·K)), and
//! volumetric flow rates in cubic metres per second (m³/s).

use tpt_eng_heat_transfer::{parallel_resistances, series_resistances};

/// Fin parameter `m = sqrt(2·h / (k·t))` (1/m) for a straight rectangular
/// fin of thickness `t`, convective coefficient `h` (W/(m²·K)) and
/// conductivity `k` (W/(m·K)).
///
/// # Panics
///
/// Does not panic; returns a non-finite value when `k` or `t` is not
/// strictly positive, or when `h` is negative.
pub fn fin_parameter(h: f64, k: f64, thickness: f64) -> f64 {
    (2.0 * h / (k * thickness)).sqrt()
}

/// Straight-fin (rectangular, adiabatic-tip approximation) efficiency
/// `η = tanh(m·L) / (m·L)`, where `m` is the fin parameter (see
/// [`fin_parameter`]) and `length` is the fin height `L` (m) — the
/// conduction length from the base to the tip.
///
/// Returns `1.0` for the degenerate case `m·L → 0` (a very short or very
/// conductive fin, which is essentially isothermal with the base).
///
/// # Panics
///
/// Does not panic; returns a non-finite value when `m` or `length` is
/// negative (the magnitude is used, so a negative sign does not change the
/// result).
pub fn fin_efficiency(m: f64, length: f64) -> f64 {
    let ml = m * length;
    if ml.abs() < 1e-12 {
        1.0
    } else {
        ml.tanh() / ml
    }
}

/// Geometry of an extruded straight-fin heat sink.
///
/// The sink carries `fin_count` identical rectangular fins standing on a
/// flat base of total footprint `base_area`. Each fin has width
/// `fin_length` (m, measured along the base, perpendicular to
/// `fin_height`), thickness `fin_thickness` (m) and height `fin_height`
/// (m, the conduction length from base to tip).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatSink {
    /// Total base footprint area (m²).
    pub base_area: f64,
    /// Number of fins.
    pub fin_count: usize,
    /// Fin width along the base (m).
    pub fin_length: f64,
    /// Fin thickness (m).
    pub fin_thickness: f64,
    /// Fin height / conduction length (m).
    pub fin_height: f64,
}

impl HeatSink {
    /// Convective surface area of a single fin (m²): two broad faces plus
    /// the tip.
    fn per_fin_area(&self) -> f64 {
        2.0 * self.fin_length * self.fin_height + self.fin_length * self.fin_thickness
    }

    /// Effective sink-to-ambient thermal resistance `θ_sa` (K/W).
    ///
    /// The sink is modelled as two parallel heat paths:
    /// 1. the **fin array** — each fin's convective resistance
    ///    `1/(η·h·A_fin)` (with fin efficiency `η` from [`fin_efficiency`]
    ///    capturing the distributed conduction/convection coupling), with
    ///    all `fin_count` fins in parallel; and
    /// 2. the **exposed base** of area `A_base − N·L·t` convecting
    ///    directly to the fluid.
    ///
    /// The two branches are combined with
    /// [`tpt_eng_heat_transfer::parallel_resistances`]. The result is always
    /// positive for positive inputs; if the fins over-cover the base the
    /// exposed area is clamped to a small positive value.
    ///
    /// # Panics
    ///
    /// Does not panic; returns a non-finite value for non-positive `h`, `k`,
    /// `fin_count`, or for non-positive base dimensions.
    pub fn thermal_resistance(&self, h: f64, k: f64) -> f64 {
        let m = fin_parameter(h, k, self.fin_thickness);
        let eta = fin_efficiency(m, self.fin_height);
        let a_fin = self.per_fin_area();
        let r_fin = 1.0 / (eta * h * a_fin);
        let r_fins = r_fin / self.fin_count as f64;

        let base_exposed = (self.base_area
            - self.fin_count as f64 * self.fin_length * self.fin_thickness)
            .max(1e-9);
        let r_base = 1.0 / (h * base_exposed);

        // Parallel combination of the fin-array branch and the exposed-base
        // branch. Both resistances are finite and positive for valid inputs.
        parallel_resistances(&[r_fins, r_base]).expect(
            "fin-array and exposed-base resistances are finite and positive, \
             so the parallel combination always exists",
        )
    }
}

/// Quadratic fan (or pump) pressure curve `Δp(q) = a − b·q²`, where `q` is
/// the volumetric flow rate (m³/s) and `Δp` is the developed pressure (Pa).
///
/// `a` is the shut-off pressure (Pa) and `b` (Pa·s²/m⁶) is the shape
/// coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FanCurve {
    /// Shut-off pressure `a` (Pa).
    pub a: f64,
    /// Quadratic shape coefficient `b` (Pa·s²/m⁶).
    pub b: f64,
}

impl FanCurve {
    /// Developed pressure (Pa) at volumetric flow `q` (m³/s).
    pub fn pressure(&self, q: f64) -> f64 {
        self.a - self.b * q * q
    }

    /// Flow rate (m³/s) at the intersection of this fan curve with the
    /// system resistance curve `Δp = R·q²`, i.e. `q = sqrt(a / (b + R))`.
    ///
    /// # Errors
    ///
    /// Returns `None` (rather than panicking) when there is no positive-flow
    /// solution:
    /// * `a <= 0` — the fan develops no shut-off pressure; or
    /// * `b + R <= 0` — the fan curve lies above the system curve for every
    ///   `q > 0` and the two never meet at a real flow.
    pub fn operating_point(&self, system_resistance_coeff: f64) -> Option<f64> {
        let denom = self.b + system_resistance_coeff;
        if self.a <= 0.0 || denom <= 0.0 {
            return None;
        }
        Some((self.a / denom).sqrt())
    }
}

/// A composable thermal-resistance network used to build a
/// junction-to-ambient path out of case, sink and ambient resistances.
///
/// Branches are reduced exactly as in `tpt_eng_heat_transfer`: `Series`
/// sums its children and `Parallel` takes their reciprocal sum.
#[derive(Debug, Clone)]
pub enum ThermalPath {
    /// A single lumped resistance (K/W).
    Resistance(f64),
    /// Sub-paths in series (sum of resistances).
    Series(Vec<ThermalPath>),
    /// Sub-paths in parallel (reciprocal sum of resistances).
    Parallel(Vec<ThermalPath>),
}

impl ThermalPath {
    /// Resolve the total resistance (K/W) of the network.
    ///
    /// # Errors
    ///
    /// Returns `None` when a `Parallel` node is empty (a parallel network
    /// with no branches is undefined) or when any branch resistance is
    /// non-finite.
    pub fn total_resistance(&self) -> Option<f64> {
        match self {
            ThermalPath::Resistance(r) => r.is_finite().then_some(*r),
            ThermalPath::Series(paths) => {
                let vals: Vec<f64> = paths
                    .iter()
                    .filter_map(ThermalPath::total_resistance)
                    .collect();
                if vals.len() != paths.len() {
                    return None;
                }
                Some(series_resistances(&vals))
            }
            ThermalPath::Parallel(paths) => {
                if paths.is_empty() {
                    return None;
                }
                let vals: Vec<f64> = paths
                    .iter()
                    .filter_map(ThermalPath::total_resistance)
                    .collect();
                if vals.len() != paths.len() {
                    return None;
                }
                parallel_resistances(&vals)
            }
        }
    }
}

/// Assemble the classic junction-to-ambient path `case → sink → air` as a
/// series network of three resistances (K/W): `θ_jc` (junction-to-case),
/// `θ_cs` (case-to-sink / interface) and `θ_sa` (sink-to-ambient).
pub fn junction_to_ambient(theta_jc: f64, theta_cs: f64, theta_sa: f64) -> ThermalPath {
    ThermalPath::Series(vec![
        ThermalPath::Resistance(theta_jc),
        ThermalPath::Resistance(theta_cs),
        ThermalPath::Resistance(theta_sa),
    ])
}

/// Junction temperature (same unit as `t_ambient`, typically °C or K) for a
/// device dissipating `power` (W) through a junction-to-ambient resistance
/// `theta_ja` (K/W): `T_j = T_ambient + P·θ_ja`.
///
/// # Panics
///
/// Does not panic; returns a non-finite value for any non-finite input.
pub fn junction_temperature(power: f64, theta_ja: f64, t_ambient: f64) -> f64 {
    t_ambient + power * theta_ja
}
