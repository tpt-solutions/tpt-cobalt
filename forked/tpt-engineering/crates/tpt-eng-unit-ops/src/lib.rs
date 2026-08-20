//! # tpt-eng-unit-ops
//!
//! Process-engineering unit-operations primitives for chemical / mechanical
//! plant design, built on [`tpt_eng_heat_transfer`] and [`tpt_eng_props`].
//!
//! * **Distillation (McCabe–Thiele)** — minimum theoretical stages at total
//!   reflux ([`fenske_min_stages`]), the minimum reflux ratio
//!   ([`underwood_rmin`]), the Gilliland estimate of actual stages
//!   ([`gilliland_stages`]) and a full constant-relative-volatility stage
//!   count with feed-stage location ([`mccabe_thiele_stages`]).
//! * **Heat exchangers** — log-mean temperature difference ([`lmtd`]),
//!   the ε-NTU relations for counterflow / parallel-flow / phase-change duty
//!   ([`epsilon_ntu_counterflow`], [`epsilon_ntu_parallel`],
//!   [`epsilon_ntu_phase_change`]) and their inverses, the heat rate
//!   ([`effectiveness_to_q`]) and overall coefficient assembly from series
//!   and parallel resistance paths ([`overall_u`], [`overall_u_parallel`]).
//! * **Pumps & compressors** — hydraulic / shaft power
//!   ([`pump_power`], [`compressor_isentropic_power`]), the quadratic pump
//!   curve and its system operating point ([`PumpCurve`]) and net-positive
//!   suction head ([`npsh_available`]).
//!
//! All quantities are plain `f64` in **SI** units; the unit of each argument
//! is documented on the relevant item (no `uom` is used). Where a property is
//! required, relative volatility and the inside-film convective coefficient
//! are pulled from [`tpt_eng_props`] and [`tpt_eng_heat_transfer`] so the
//! crate stays dependency-light.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_unit_ops::{fenske_min_stages, lmtd, epsilon_ntu_counterflow};
//!
//! // Minimum stages to split 0.95 overhead / 0.05 bottoms at α = 2.5.
//! let n_min = fenske_min_stages(0.95, 0.05, 2.5).unwrap();
//! assert!(n_min > 6.0 && n_min < 7.0);
//!
//! // Counterflow ε-NTU with NTU=2, capacity ratio 0.5.
//! let eps = epsilon_ntu_counterflow(2.0, 0.5).unwrap();
//! assert!(eps > 0.5 && eps < 0.8);
//! ```

use tpt_eng_heat_transfer::{
    convection_coefficient, nusselt_internal_pipe, parallel_grey_plates_flux, parallel_resistances,
};
use tpt_eng_props::mixture::{Component, pr_saturation_pressure};

/// Gravitational acceleration (m/s²), SI value.
pub const G: f64 = 9.806_65;

/// Returns `true` when `x` lies in the open unit interval `(0, 1)`.
///
/// Composition and quality arguments in this crate are physically restricted
/// to the open unit interval; this guards the few divisions that would
/// otherwise produce infinities or NaNs.
fn in_open_unit(x: f64) -> bool {
    x > 0.0 && x < 1.0
}

/// Relative volatility `α = P_sat,light / P_sat,heavy` of a binary at
/// temperature `t_k` (K), from Peng–Robinson saturation pressures.
///
/// The relative volatility is the equilibrium constant ratio
/// `K_light / K_heavy ≈ P_sat,light / P_sat,heavy` for ideal (Raoult-law)
/// behaviour; it appears in every distillation correlation below.
///
/// # Errors
///
/// Returns `None` when either Peng–Robinson saturation pressure is
/// non-finite (e.g. a `NaN` component key) so the caller is not handed a
/// spurious `NaN` volatility.
pub fn relative_volatility(t_k: f64, light: Component, heavy: Component) -> Option<f64> {
    let pl = pr_saturation_pressure(t_k, light);
    let ph = pr_saturation_pressure(t_k, heavy);
    if !pl.is_finite() || !ph.is_finite() || ph <= 0.0 {
        return None;
    }
    Some(pl / ph)
}

/// Minimum theoretical stages at total reflux (Fenske equation):
/// `N_min = ln[(x_d/(1−x_d)) / (x_b/(1−x_b))] / ln α`.
///
/// `xd` is the overhead (distillate) mole fraction of the light key, `xb` the
/// bottoms mole fraction of the light key, and `alpha` the relative volatility
/// (use [`relative_volatility`] or assume a constant value). A larger `alpha`
/// or a sharper split (high `xd`, low `xb`) reduces `N_min`. `N_min` excludes
/// the reboiler and condenser.
///
/// # Errors
///
/// Returns `None` when the inputs are physically invalid: any of `xd`, `xb`
/// outside `(0, 1)`, `xd <= xb` (no separation) or `alpha <= 1` (no
/// selectivity), or when any quantity is non-finite.
pub fn fenske_min_stages(xd: f64, xb: f64, alpha: f64) -> Option<f64> {
    if !in_open_unit(xd) || !in_open_unit(xb) || xd <= xb || alpha <= 1.0 || !alpha.is_finite() {
        return None;
    }
    let ratio: f64 = (xd / (1.0 - xd)) / (xb / (1.0 - xb));
    let s = ratio.ln() / alpha.ln();
    Some(s)
}

/// Separation sharpness ratio `S = (x_d/(1−x_d)) / (x_b/(1−x_b))` used by the
/// Fenske equation. Useful on its own for quick split screening.
///
/// # Errors
///
/// Returns `None` when `xd` or `xb` is not in `(0, 1)`, `xd <= xb`, or either
/// quantity is non-finite.
pub fn separation_factor(xd: f64, xb: f64) -> Option<f64> {
    if !in_open_unit(xd) || !in_open_unit(xb) || xd <= xb {
        return None;
    }
    let ratio: f64 = (xd / (1.0 - xd)) / (xb / (1.0 - xb));
    Some(ratio)
}

/// Underwood parameter `θ` for a binary: the root in `(1, α)` of the
/// relative-volatility feed equation `α·x_f / (α − θ) + (1 − x_f) / (1 − θ) =
/// 1 − q` (light-key volatility `alpha`, heavy-key volatility `1`).
///
/// This is the quantity needed by [`underwood_rmin`]; it is exposed so callers
/// can re-use it (for instance in the multi-component Underwood sum) or test
/// it directly. The equation reduces to a quadratic in `θ` solved in closed
/// form; the physically meaningful root is the one with `1 < θ < α`.
///
/// # Errors
///
/// Returns `None` when `xf` is not in `(0, 1)`, `alpha <= 1`, `q` is
/// non-finite, or no root lies strictly inside `(1, α)` (the `q = 1` saturated
/// liquid-feed pinch sends `θ → 1`, which makes `R_min` singular).
pub fn underwood_theta(xf: f64, q: f64, alpha: f64) -> Option<f64> {
    if !in_open_unit(xf) || alpha <= 1.0 || !q.is_finite() {
        return None;
    }
    let u = 1.0 - q;
    let a = alpha * xf;
    let b = 1.0 - xf;
    // Multiplying the feed equation by (α − θ)(1 − θ) and collecting powers of
    // θ yields `ao·θ² + bo·θ + co = 0` with a = α·xf, b = 1 − xf, u = 1 − q:
    //   ao = u
    //   bo = (a + b) − u·(α + 1)
    //   co = α·(u − 1)
    let ao = u;
    let bo = (a + b) - u * (alpha + 1.0);
    let co = alpha * (u - 1.0);
    let disc = bo * bo - 4.0 * ao * co;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    if ao.abs() < 1e-12 {
        // Degenerate (q → 1): the quadratic collapses to a linear `bo·θ + co = 0`.
        let cand = -co / bo;
        if cand > 1.0 && cand < alpha && cand.is_finite() {
            return Some(cand);
        }
        return None;
    }
    let candidates = [(-bo + sq) / (2.0 * ao), (-bo - sq) / (2.0 * ao)];
    candidates
        .into_iter()
        .find(|&cand| cand > 1.0 && cand < alpha && cand.is_finite())
}

/// Minimum reflux ratio `R_min` for a binary (Underwood):
/// `R_min + 1 = α·x_d / (α − θ)`, with `θ` from [`underwood_theta`].
///
/// Assumptions: constant relative volatility `alpha`; binary system; the feed
/// composition `xf` and thermal condition `q` describe a single feed; `xb` is
/// supplied for a validity check (`xb < xf < xd`) but does not enter the
/// Underwood sum. The result excludes the reboiler and condenser.
///
/// # Errors
///
/// Returns `None` when the inputs are physically invalid (`xd`, `xb`, `xf`
/// not in `(0, 1)`, `xb < xf < xd` violated, `alpha <= 1`, `q` non-finite) or
/// when the `θ` solve fails.
pub fn underwood_rmin(xd: f64, xb: f64, xf: f64, q: f64, alpha: f64) -> Option<f64> {
    if !in_open_unit(xd)
        || !in_open_unit(xb)
        || !in_open_unit(xf)
        || !(xb < xf && xf < xd)
        || alpha <= 1.0
        || !q.is_finite()
    {
        return None;
    }
    let theta = underwood_theta(xf, q, alpha)?;
    if !theta.is_finite() {
        return None;
    }
    let rmin = alpha * xd / (alpha - theta) - 1.0;
    if rmin <= 0.0 || !rmin.is_finite() {
        return None;
    }
    Some(rmin)
}

/// Actual theoretical stages from minimum stages and reflux ratios
/// (Gilliland correlation, Eduljee form):
/// `X = (R − R_min) / (R + 1)`, `Y = 1 − exp[(1 + 54.4·X) / 11.0 · (X − 1)
/// / X^0.982]`, `N = 1 + (N_min − 1) / Y`.
///
/// This bridges [`fenske_min_stages`] and [`underwood_rmin`] into a single
/// design-stage estimate.
///
/// # Errors
///
/// Returns `None` when `n_min` or `r_min` is non-positive/non-finite, `r <= 0`
/// is non-finite, when `r <= r_min` (no feasible operating point), or when
/// `r_min + 1` is zero.
pub fn gilliland_stages(n_min: f64, r_min: f64, r: f64) -> Option<f64> {
    if n_min <= 0.0
        || r_min <= 0.0
        || r <= 0.0
        || !n_min.is_finite()
        || !r_min.is_finite()
        || !r.is_finite()
        || r <= r_min
        || r_min + 1.0 == 0.0
    {
        return None;
    }
    let x = (r - r_min) / (r + 1.0);
    let y = 1.0 - ((1.0 + 54.4 * x) / 11.0 * (x - 1.0) / x.powf(0.982)).exp();
    if y <= 0.0 || !y.is_finite() {
        return None;
    }
    Some(1.0 + (n_min - 1.0) / y)
}

/// Geometry and result of a McCabe–Thiele stage count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McCabeThiele {
    /// Total equilibrium-contact count (including the reboiler, excluding the
    /// condenser).
    pub stages: f64,
    /// 1-based stage at which the feed enters (`None` only if the feed sits
    /// below the last stage).
    pub feed_stage: Option<usize>,
}

/// McCabe–Thiele theoretical-stage count with constant relative volatility
/// `alpha`, stepping between the operating lines and the equilibrium curve
/// `y = α·x / (1 + (α − 1)·x)`.
///
/// * `xd` — overhead (distillate) mole fraction of the light key.
/// * `xb` — bottoms mole fraction of the light key.
/// * `xf` — feed mole fraction of the light key.
/// * `q`  — feed thermal condition (`0` saturated vapour, `1` saturated
///   liquid, `>1` subcooled, `<0` superheated vapour).
/// * `r`  — actual reflux ratio `L/D`.
///
/// The pinch intersection of the q-line and rectifying operating line is
/// stepped off exactly; the result includes the reboiler as the final stage
/// and reports the feed stage. The interpolation onto the final (partial)
/// stage is done in log-separation-ratio space so the `R → ∞` limit recovers
/// [`fenske_min_stages`].
///
/// # Errors
///
/// Returns `None` when the inputs are invalid (`xd`, `xb`, `xf` not in
/// `(0, 1)`, `xb < xf < xd` violated, `alpha <= 1`, `r <= 0` or non-finite),
/// when `r` is below the minimum reflux ([`underwood_rmin`]), or when the
/// stepping fails to converge (non-finite progress).
pub fn mccabe_thiele_stages(
    xd: f64,
    xb: f64,
    xf: f64,
    q: f64,
    alpha: f64,
    r: f64,
) -> Option<McCabeThiele> {
    if !in_open_unit(xd)
        || !in_open_unit(xb)
        || !in_open_unit(xf)
        || !(xb < xf && xf < xd)
        || alpha <= 1.0
        || r <= 0.0
        || !r.is_finite()
        || !q.is_finite()
    {
        return None;
    }
    // Rectifying line: y = (R/(R+1))·x + xd/(R+1), through (xd, xd).
    let rs = r / (r + 1.0);
    let rb = xd / (r + 1.0);
    // q-line / rectifying-line intersection.
    let x_i = (xf - rb) / (rs - q + 1.0);
    let y_i = rs * x_i + rb;
    if !(x_i > xb && x_i < xd) || !x_i.is_finite() {
        return None;
    }
    // Stripping line through (x_i, y_i) and (xb, xb).
    let ss = (y_i - xb) / (x_i - xb);
    let sb = xb - ss * xb;
    let rmin = underwood_rmin(xd, xb, xf, q, alpha)?;
    if r <= rmin {
        return None;
    }

    // Equilibrium relationship `y = α·x / (1 + (α − 1)·x)` and its inverse
    // `x = y / (α − (α − 1)·y)`. McCabe–Thiele steps from the top (xd, xd)
    // downward: equilibrium vapour `y = y_eq(x)`, then the operating line places
    // the liquid below it at `x_from_y(y_op)`.
    let x_from_y = |y: f64| y / (alpha - (alpha - 1.0) * y);

    let mut xl = xd;
    let mut stages = 1.0;
    let mut feed_stage: Option<usize> = None;
    for _ in 0..10_000 {
        if !xl.is_finite() {
            return None;
        }
        if feed_stage.is_none() && xl <= x_i {
            feed_stage = Some(stages as usize);
        }
        let y_next = if xl > x_i { rs * xl + rb } else { ss * xl + sb };
        let x_next = x_from_y(y_next);
        if !x_next.is_finite() {
            return None;
        }
        if x_next <= xb {
            // Fractional final (reboiler) stage via log-separation-ratio
            // interpolation, so the R → ∞ limit recovers `fenske_min_stages`.
            let log_s = |x: f64| ((xd / (1.0 - xd)) / (x / (1.0 - x))).ln();
            let f = (log_s(xb) - log_s(xl)) / (log_s(x_next) - log_s(xl));
            stages += f.clamp(0.0, 1.0);
            return Some(McCabeThiele { stages, feed_stage });
        }
        if x_next >= xl {
            return None;
        }
        xl = x_next;
        stages += 1.0;
    }
    None
}

// --- Heat exchangers --------------------------------------------------------

/// Log-mean temperature difference (K):
/// `LMTD = (ΔT_hot − ΔT_cold) / ln(ΔT_hot / ΔT_cold)`, where `dt_hot` is the
/// terminal temperature difference at the hot end and `dt_cold` at the cold
/// end of a counterflow exchanger. Both must be strictly positive.
///
/// When the two differences are (near) equal the ratio in the logarithm
/// approaches 1 and the expression is singular; in that limit `LMTD` is the
/// arithmetic mean `ΔT`, which this function returns (with a `1e-9` relative
/// tolerance on the difference).
///
/// # Errors
///
/// Returns `None` when `dt_hot` or `dt_cold` is non-positive or non-finite, or
/// when their ratio is non-finite.
pub fn lmtd(dt_hot: f64, dt_cold: f64) -> Option<f64> {
    if dt_hot <= 0.0 || dt_cold <= 0.0 || !dt_hot.is_finite() || !dt_cold.is_finite() {
        return None;
    }
    if (dt_hot - dt_cold).abs() <= 1e-9 * dt_hot.max(dt_cold) {
        return Some(0.5 * (dt_hot + dt_cold));
    }
    if !(dt_hot / dt_cold).is_finite() {
        return None;
    }
    Some((dt_hot - dt_cold) / (dt_hot / dt_cold).ln())
}

/// Number of transfer units for a heat exchanger:
/// `NTU = U·A / C_min`, where `u` is the overall coefficient (W/(m²·K)), `a`
/// the area (m²) and `c_min` the smaller heat-capacity rate (W/K).
///
/// # Errors
///
/// Returns `None` when `u`, `a` or `c_min` is non-positive or non-finite.
pub fn ntu(u: f64, a: f64, c_min: f64) -> Option<f64> {
    if u <= 0.0
        || a <= 0.0
        || c_min <= 0.0
        || !u.is_finite()
        || !a.is_finite()
        || !c_min.is_finite()
    {
        return None;
    }
    Some(u * a / c_min)
}

/// Capacity ratio `C_r = C_min / C_max` from the two heat-capacity rates
/// (W/K). `0` for a condensing/boiling fluid (`c_max → ∞`).
///
/// # Errors
///
/// Returns `None` when `c_min` is negative or non-finite, `c_max` is
/// non-positive or non-finite, or `c_min > c_max`.
pub fn capacity_ratio(c_min: f64, c_max: f64) -> Option<f64> {
    if c_min < 0.0 || c_max <= 0.0 || !c_min.is_finite() || !c_max.is_finite() || c_min > c_max {
        return None;
    }
    Some(c_min / c_max)
}

/// Heat-transfer rate (W) from effectiveness: `Q = ε·C_min·ΔT_in`, where `eps`
/// is the exchanger effectiveness, `c_min` the minimum heat-capacity rate
/// (W/K) and `dt_in` the inlet temperature difference (K, hot − cold).
///
/// # Errors
///
/// Returns `None` when `eps` is outside `[0, 1]`, `c_min` is negative or
/// non-finite, or `dt_in` is non-finite.
pub fn effectiveness_to_q(eps: f64, c_min: f64, dt_in: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&eps) || c_min < 0.0 || !c_min.is_finite() || !dt_in.is_finite() {
        return None;
    }
    Some(eps * c_min * dt_in)
}

/// Effectiveness of a counterflow exchanger (ε-NTU):
/// `ε = (1 − exp(−NTU·(1 − C_r))) / (1 − C_r·exp(−NTU·(1 − C_r)))` for
/// `C_r < 1`; in the `C_r → 1` limit this reduces to `NTU / (1 + NTU)`.
///
/// # Errors
///
/// Returns `None` when `ntu` is negative or non-finite, `cr` is outside
/// `[0, 1]`, or the expression is non-finite.
pub fn epsilon_ntu_counterflow(ntu: f64, cr: f64) -> Option<f64> {
    if ntu < 0.0 || !ntu.is_finite() || !(0.0..=1.0).contains(&cr) {
        return None;
    }
    if (1.0 - cr).abs() < 1e-9 {
        let e = ntu / (1.0 + ntu);
        return e.is_finite().then_some(e);
    }
    let a = -ntu * (1.0 - cr);
    let expa = a.exp();
    let e = (1.0 - expa) / (1.0 - cr * expa);
    e.is_finite().then_some(e)
}

/// Effectiveness of a parallel-flow (co-current) exchanger (ε-NTU):
/// `ε = (1 − exp(−NTU·(1 + C_r))) / (1 + C_r)`.
///
/// # Errors
///
/// Returns `None` when `ntu` is negative or non-finite, `cr` is outside
/// `[0, 1]`, or the expression is non-finite.
pub fn epsilon_ntu_parallel(ntu: f64, cr: f64) -> Option<f64> {
    if ntu < 0.0 || !ntu.is_finite() || !(0.0..=1.0).contains(&cr) {
        return None;
    }
    let e = (1.0 - (-ntu * (1.0 + cr)).exp()) / (1.0 + cr);
    e.is_finite().then_some(e)
}

/// Effectiveness when one stream is isothermal (`C_r = 0`, condensing or
/// evaporating): `ε = 1 − exp(−NTU)`. Identical for counter- and parallel-flow.
///
/// # Errors
///
/// Returns `None` when `ntu` is negative or non-finite.
pub fn epsilon_ntu_phase_change(ntu: f64) -> Option<f64> {
    if ntu < 0.0 || !ntu.is_finite() {
        return None;
    }
    Some(1.0 - (-ntu).exp())
}

/// Inverse of [`epsilon_ntu_counterflow`]: required NTU for a target
/// counterflow effectiveness `eps` at capacity ratio `cr`.
///
/// `cr < 1`: `NTU = − ln((1 − ε)/(1 − ε·C_r)) / (1 − C_r)`;
/// `cr → 1`: `NTU = ε / (1 − ε)`.
///
/// # Errors
///
/// Returns `None` when `eps` is outside `[0, 1)`, `cr` is outside `[0, 1]`, or
/// the required NTU is non-finite / negative.
pub fn ntu_from_epsilon_counterflow(eps: f64, cr: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&eps) || !(0.0..=1.0).contains(&cr) {
        return None;
    }
    let n = if (1.0 - cr).abs() < 1e-9 {
        eps / (1.0 - eps)
    } else {
        // NTU = -ln[(1 − ε) / (1 − ε·C_r)] / (1 − C_r)
        //     = [ln(1 − ε) − ln(1 − ε·C_r)] / (C_r − 1)
        ((1.0 - eps).ln() - (1.0 - cr * eps).ln()) / (cr - 1.0)
    };
    (n.is_finite() && n >= 0.0).then_some(n)
}

/// Overall heat-transfer coefficient (W/(m²·K)) from a **series** of thermal
/// resistances (K/W per unit area) such as tube wall, fouling and two films:
/// `1/U = Σ R_i`.
///
/// # Errors
///
/// Returns `None` when `resistances` is empty, any element is non-positive or
/// non-finite, or the series sum is zero.
pub fn overall_u(resistances: &[f64]) -> Option<f64> {
    if resistances.is_empty() {
        return None;
    }
    let total: f64 = resistances.iter().copied().fold(0.0, |a, b| a + b);
    if total <= 0.0 || !total.is_finite() || resistances.iter().any(|r| *r <= 0.0 || !r.is_finite())
    {
        return None;
    }
    Some(1.0 / total)
}

/// Overall heat-transfer coefficient (W/(m²·K)) from resistances in
/// **parallel** — e.g. two independent conduction paths (fin + exposed base)
/// between the same two temperatures. Uses
/// [`tpt_eng_heat_transfer::parallel_resistances`].
///
/// # Errors
///
/// Returns `None` when `resistances` is empty or any element is non-positive
/// or non-finite (the parallel combination via `tpt_eng_heat_transfer` is then
/// undefined).
pub fn overall_u_parallel(resistances: &[f64]) -> Option<f64> {
    if resistances.is_empty() || resistances.iter().any(|r| *r <= 0.0 || !r.is_finite()) {
        return None;
    }
    let r_eq = parallel_resistances(resistances)?;
    if !r_eq.is_finite() || r_eq <= 0.0 {
        return None;
    }
    Some(1.0 / r_eq)
}

// --- Pumps & compressors ----------------------------------------------------

/// Shaft power (W) of a pump: `P = ρ·g·Q·H / η`.
///
/// * `flow_m3s` — volumetric flow rate `Q` (m³/s).
/// * `head_m`    — developed head `H` (m of fluid).
/// * `rho`       — fluid density (kg/m³).
/// * `efficiency` — pump efficiency `η` in `(0, 1]`.
///
/// # Errors
///
/// Returns `None` when `flow_m3s` or `head_m` is negative or non-finite,
/// `rho` is non-positive or non-finite, or `efficiency` is outside `(0, 1]`.
pub fn pump_power(flow_m3s: f64, head_m: f64, rho: f64, efficiency: f64) -> Option<f64> {
    if flow_m3s < 0.0
        || head_m < 0.0
        || rho <= 0.0
        || !(efficiency > 0.0 && efficiency <= 1.0)
        || !flow_m3s.is_finite()
        || !head_m.is_finite()
        || !rho.is_finite()
        || !efficiency.is_finite()
    {
        return None;
    }
    Some(rho * G * flow_m3s * head_m / efficiency)
}

/// A quadratic pump (or fan) head curve `H = H0 − k·Q²`, where `Q` is the
/// volumetric flow rate (m³/s) and `H` the developed head (m).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PumpCurve {
    /// Shut-off head `H0` (m) at zero flow.
    pub h0: f64,
    /// Quadratic shape coefficient `k` (m/(m³/s)²).
    pub k: f64,
}

impl PumpCurve {
    /// Developed head `H` (m) at flow `q` (m³/s): `H = H0 − k·Q²`.
    pub fn head(&self, q: f64) -> f64 {
        self.h0 - self.k * q * q
    }

    /// Flow rate (m³/s) at the intersection with the system curve
    /// `H = R·Q²`: `Q = √(H0 / (k + R))`.
    ///
    /// # Errors
    ///
    /// Returns `None` when `self.h0 <= 0` or `k + R <= 0` (no positive-flow
    /// intersection), or when any input is non-finite.
    pub fn operating_point(&self, system_resistance: f64) -> Option<f64> {
        let denom = self.k + system_resistance;
        if self.h0 <= 0.0 || denom <= 0.0 || !self.h0.is_finite() || !denom.is_finite() {
            return None;
        }
        Some((self.h0 / denom).sqrt())
    }
}

/// Net positive suction head available `NPSH_a` (m):
/// `NPSH_a = (P_suction − P_vapour) / (ρ·g) + z_static − h_friction`.
///
/// * `p_suction` — absolute pressure at the pump suction (Pa).
/// * `p_vapour`  — vapour pressure of the liquid at suction temperature (Pa).
/// * `rho`       — liquid density (kg/m³).
/// * `z_static`  — static head of the suction source above the pump (m; use a
///   negative value for a flooded/booster arrangement).
/// * `h_friction` — friction + entrance losses on the suction line (m).
///
/// # Errors
///
/// Returns `None` when `rho` is non-positive or non-finite, or when `p_suction
/// − p_vapour` or `h_friction` is non-finite.
pub fn npsh_available(
    p_suction: f64,
    p_vapour: f64,
    rho: f64,
    z_static: f64,
    h_friction: f64,
) -> Option<f64> {
    if rho <= 0.0
        || !rho.is_finite()
        || !p_suction.is_finite()
        || !p_vapour.is_finite()
        || !z_static.is_finite()
        || !h_friction.is_finite()
    {
        return None;
    }
    Some((p_suction - p_vapour) / (rho * G) + z_static - h_friction)
}

/// Isentropic shaft power (W) of a compressor:
/// `P = ṁ·cp·T_in·((P_out/P_in)^((γ−1)/γ) − 1) / η`, using the ideal-gas
/// isentropic relation.
///
/// * `mass_flow` — mass flow rate `ṁ` (kg/s).
/// * `cp`        — specific heat at constant pressure (J/(kg·K)).
/// * `t_in`      — inlet temperature `T_in` (K).
/// * `p_ratio`   — pressure ratio `P_out / P_in` (> 1).
/// * `gamma`     — ratio of specific heats `γ` (dimensionless).
/// * `efficiency` — isentropic efficiency `η` in `(0, 1]`.
///
/// # Errors
///
/// Returns `None` when `mass_flow`, `cp` or `t_in` is non-positive / non-finite,
/// `p_ratio <= 1`, `gamma <= 1`, `efficiency` outside `(0, 1]`, or any input is
/// non-finite.
pub fn compressor_isentropic_power(
    mass_flow: f64,
    cp: f64,
    t_in: f64,
    p_ratio: f64,
    gamma: f64,
    efficiency: f64,
) -> Option<f64> {
    if mass_flow <= 0.0
        || cp <= 0.0
        || t_in <= 0.0
        || p_ratio <= 1.0
        || gamma <= 1.0
        || !(efficiency > 0.0 && efficiency <= 1.0)
        || !mass_flow.is_finite()
        || !cp.is_finite()
        || !t_in.is_finite()
        || !p_ratio.is_finite()
        || !gamma.is_finite()
        || !efficiency.is_finite()
    {
        return None;
    }
    let exponent = (gamma - 1.0) / gamma;
    let p = mass_flow * cp * t_in * (p_ratio.powf(exponent) - 1.0) / efficiency;
    p.is_finite().then_some(p)
}

/// Discharge temperature (K) of an isentropic compressor:
/// `T_out = T_in·(P_out/P_in)^((γ−1)/γ)`.
///
/// # Errors
///
/// Returns `None` when `t_in` is non-positive / non-finite, `p_ratio <= 1`,
/// `gamma <= 1`, or any input is non-finite.
pub fn compressor_discharge_temperature(t_in: f64, p_ratio: f64, gamma: f64) -> Option<f64> {
    if t_in <= 0.0
        || p_ratio <= 1.0
        || gamma <= 1.0
        || !t_in.is_finite()
        || !p_ratio.is_finite()
        || !gamma.is_finite()
    {
        return None;
    }
    Some(t_in * p_ratio.powf((gamma - 1.0) / gamma))
}

// --- Thin wrappers over the dependency crates -------------------------------

/// Inside-tube (or internal-flow) convective coefficient `h` (W/(m²·K)) from
/// the Dittus–Boelter correlation, via [`tpt_eng_heat_transfer`].
///
/// * `re`, `pr` — Reynolds and Prandtl numbers of the internal flow.
/// * `conductivity` — fluid thermal conductivity `k` (W/(m·K)).
/// * `diameter` — tube inner diameter `D` (m).
/// * `heating` — `true` selects the `Pr^0.4` (heating) exponent, `false` the
///   `Pr^0.3` (cooling) exponent.
///
/// # Panics
///
/// Does not panic; returns a non-finite `h` for non-positive `diameter`,
/// `conductivity` or for non-finite inputs.
pub fn tube_film_coefficient(
    re: f64,
    pr: f64,
    conductivity: f64,
    diameter: f64,
    heating: bool,
) -> f64 {
    let nu = nusselt_internal_pipe(re, pr, heating);
    convection_coefficient(nu, conductivity, diameter)
}

/// Radiant heat loss (W) from a surface of area `area` (m²) at temperature
/// `t_surface_k` (K) to an enclosure at `t_ambient_k` (K), with emissivity
/// `emissivity` and view/exchange factor `f` (1.0 for a large enclosure).
/// Computed with the Stefan–Boltzmann constant (see `SIGMA_FAC`).
///
/// # Panics
///
/// Does not panic; returns a non-finite value for non-positive `area` or for
/// non-finite temperature / factor inputs.
pub fn radiant_loss(area: f64, emissivity: f64, f: f64, t_surface_k: f64, t_ambient_k: f64) -> f64 {
    area * f * emissivity * SIGMA_FAC * (t_surface_k.powi(4) - t_ambient_k.powi(4))
}

/// Convenience alias for [`tpt_eng_heat_transfer::SIGMA`]; the Stefan–Boltzmann
/// constant (W/(m²·K⁴)).
const SIGMA_FAC: f64 = 5.670_374_419e-8;

/// Two infinite parallel grey plates exchange `q'' = σ·(T₁⁴ − T₂⁴) / (1/ε₁ +
/// 1/ε₂ − 1)` (W/m²). Re-exported thin wrapper over
/// [`tpt_eng_heat_transfer::parallel_grey_plates_flux`].
///
/// # Panics
///
/// Does not panic; returns a non-finite flux for non-finite temperatures or
/// emissivities.
pub fn grey_plate_flux(
    emissivity_1: f64,
    emissivity_2: f64,
    temperature_1_k: f64,
    temperature_2_k: f64,
) -> f64 {
    parallel_grey_plates_flux(emissivity_1, emissivity_2, temperature_1_k, temperature_2_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenske_basics() {
        // Fenske: N_min = ln[(x_d/(1−x_d))/(x_b/(1−x_b))] / ln α
        //        = ln(361) / ln(2.5) ≈ 6.43 (excludes reboiler + condenser).
        let n = fenske_min_stages(0.95, 0.05, 2.5).unwrap();
        assert!((n - 6.43).abs() < 0.05);
        // More stages with sharper split.
        let sharper = fenske_min_stages(0.99, 0.01, 2.5).unwrap();
        assert!(sharper > n);
        // More stages with higher relative volatility.
        let higher_a = fenske_min_stages(0.95, 0.05, 5.0).unwrap();
        assert!(higher_a < n);
        assert!(fenske_min_stages(0.5, 0.5, 2.0).is_none());
        assert!(fenske_min_stages(0.95, 0.05, 1.0).is_none());
    }

    #[test]
    fn diag() {
        let ethane = Component::from_name("ethane").unwrap();
        let propane = Component::from_name("propane").unwrap();
        let pe = pr_saturation_pressure(260.0, ethane);
        let pp = pr_saturation_pressure(260.0, propane);
        eprintln!("DIAG pe={pe} pp={pp} ratio={}", pp / pe);
        let a = relative_volatility(260.0, propane, ethane).unwrap();
        eprintln!("DIAG relvol={a}");
        let theta = underwood_theta(0.5, 0.0, 2.4).unwrap();
        let rmin = underwood_rmin(0.95, 0.05, 0.5, 0.0, 2.4).unwrap();
        eprintln!("DIAG theta={theta} rmin={rmin}");
        let n_min = fenske_min_stages(0.95, 0.05, 2.4).unwrap();
        let mt = mccabe_thiele_stages(0.95, 0.05, 0.5, 0.0, 2.4, 100.0).unwrap();
        eprintln!(
            "DIAG n_min={n_min} mt.stages={} feed={:?}",
            mt.stages, mt.feed_stage
        );
        let eps = epsilon_ntu_counterflow(3.0, 0.4).unwrap();
        let n = ntu_from_epsilon_counterflow(eps, 0.4).unwrap();
        eprintln!("DIAG eps={eps} n={n}");
    }

    #[test]
    fn lmtd_equal_returns_arithmetic_mean() {
        let v = lmtd(20.0, 20.0).unwrap();
        assert!((v - 20.0).abs() < 1e-9);
        let v2 = lmtd(30.0, 10.0).unwrap();
        let ratio: f64 = 30.0 / 10.0;
        let expected: f64 = (30.0 - 10.0) / ratio.ln();
        assert!((v2 - expected).abs() < 1e-9);
        assert!(lmtd(-1.0, 10.0).is_none());
    }

    #[test]
    fn epsilon_ntu_counterflow_limits() {
        // cr=1 → NTU/(1+NTU).
        let e1 = epsilon_ntu_counterflow(2.0, 1.0).unwrap();
        assert!((e1 - 2.0 / 3.0).abs() < 1e-9);
        // cr<1 → 1 as NTU → ∞.
        let hi = epsilon_ntu_counterflow(1e6, 0.5).unwrap();
        assert!((hi - 1.0).abs() < 1e-3);
        // cr=0 (phase change) → 1 − exp(−NTU).
        let e0 = epsilon_ntu_counterflow(2.0, 0.0).unwrap();
        assert!((e0 - (1.0 - (-2.0_f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn epsilon_ntu_round_trip() {
        let eps = epsilon_ntu_counterflow(3.0, 0.4).unwrap();
        let n = ntu_from_epsilon_counterflow(eps, 0.4).unwrap();
        eprintln!("DEBUG eps={eps} n={n}");
        assert!((n - 3.0).abs() < 1e-6);
    }

    #[test]
    fn overall_u_series_and_parallel() {
        let u = overall_u(&[0.001, 0.002, 0.0005]).unwrap();
        assert!((u - 1.0 / 0.0035).abs() < 1e-9);
        let up = overall_u_parallel(&[0.1, 0.1]).unwrap();
        assert!((up - 20.0).abs() < 1e-9);
    }

    #[test]
    fn pump_power_formula() {
        // Q=0.05 m³/s, H=30 m, ρ=1000, η=0.7 → ρ·g·Q·H/η.
        let p = pump_power(0.05, 30.0, 1000.0, 0.7).unwrap();
        assert!((p - 1000.0 * G * 0.05 * 30.0 / 0.7).abs() < 1e-6);
        // Invalid efficiency (η=0) → None.
        assert!(pump_power(0.05, 30.0, 1000.0, 0.0).is_none());
    }

    #[test]
    fn pump_operating_point_solves_quadratic() {
        let curve = PumpCurve { h0: 50.0, k: 2.0 };
        // system R=2 → Q = sqrt(50/(2+2)) = sqrt(12.5).
        let q = curve.operating_point(2.0).unwrap();
        assert!((q - 12.5_f64.sqrt()).abs() < 1e-9);
        let dp = curve.head(q);
        let dp_sys = 2.0 * q * q;
        assert!((dp - dp_sys).abs() < 1e-9);
        assert!(PumpCurve { h0: 0.0, k: 2.0 }.operating_point(2.0).is_none());
    }

    #[test]
    fn compressor_power_and_discharge() {
        // Air: cp=1005, γ=1.4, ṁ=1, T_in=300, PR=2, η=0.8.
        let pow = compressor_isentropic_power(1.0, 1005.0, 300.0, 2.0, 1.4, 0.8).unwrap();
        let exp = 1.0 * 1005.0 * 300.0 * (2.0_f64.powf(0.4 / 1.4) - 1.0) / 0.8;
        assert!((pow - exp).abs() < 1e-6);
        let t = compressor_discharge_temperature(300.0, 2.0, 1.4).unwrap();
        assert!((t - 300.0 * 2.0_f64.powf(0.4 / 1.4)).abs() < 1e-6);
        assert!(compressor_isentropic_power(1.0, 1005.0, 300.0, 2.0, 1.4, 0.0).is_none());
    }

    #[test]
    fn mccabe_thiele_exceeds_fenske_and_feeds() {
        // At a high (but finite) reflux the stage count must exceed the Fenske
        // minimum and be finite, with a feed stage located between the ends.
        let n_min = fenske_min_stages(0.95, 0.05, 2.4).unwrap();
        let mt = mccabe_thiele_stages(0.95, 0.05, 0.5, 0.0, 2.4, 100.0).unwrap();
        assert!(mt.stages.is_finite() && mt.stages > n_min);
        assert!(mt.feed_stage.is_some_and(|f| f >= 1));
        // Below the minimum reflux the design is infeasible.
        let rmin = underwood_rmin(0.95, 0.05, 0.5, 0.0, 2.4).unwrap();
        assert!(mccabe_thiele_stages(0.95, 0.05, 0.5, 0.0, 2.4, rmin - 1e-6).is_none());
    }

    #[test]
    fn relative_volatility_finite_and_above_one() {
        // Ethane is the light (more volatile) key, propane the heavy key.
        let ethane = Component::from_name("ethane").unwrap();
        let propane = Component::from_name("propane").unwrap();
        let a = relative_volatility(260.0, ethane, propane).unwrap();
        assert!(a.is_finite() && a > 1.0);
    }
}
