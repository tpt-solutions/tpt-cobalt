//! # tpt-eng-renewables
//!
//! Engineering primitives for renewable-energy systems:
//!
//! * Solar photovoltaic (PV) single-diode I–V model ([`PvCell`]).
//! * Wind-turbine power curves with a realistic cut-in / rated / cut-out
//!   envelope and the Betz-limit ceiling ([`wind_power`], [`betz_limit_power`]).
//! * Lithium-ion battery capacity-fade end-of-life modelling built on the
//!   Weibull life distribution from `tpt-eng-reliability`
//!   ([`cycles_to_threshold`]).
//!
//! All scalar quantities are plain `f64` in SI units (volts, amperes, watts,
//! metres, seconds, kelvin, °C) unless noted. No unit-typed (uom) arithmetic is
//! used so the crate stays dependency-light and numerically transparent.

use tpt_eng_reliability::{ReliabilityError, weibull_mean};

/// Boltzmann constant expressed in electron-volts per kelvin
/// (`k_B / q ≈ 8.617333·10⁻⁵ eV/K`), which makes the thermal voltage
/// `V_t(T) = k_B·T/q = KB_EV·T` come out directly in volts.
const KB_EV: f64 = 8.617_333_262e-5;

/// Reference plane-of-array irradiance (W/m²) at which [`PvCell`] reference
/// parameters are quoted (standard test conditions).
pub const G_REF: f64 = 1000.0;

/// Absolute zero in degrees Celsius.
const CELSIUS_OFFSET: f64 = 273.15;

/// A single photovoltaic cell (or an equivalent one-diode lump) described by the
/// five-parameter single-diode model.
///
/// Reference parameters (`isc_ref`, `voc_ref`) are quoted at the reference
/// irradiance [`G_REF`] and reference temperature `t_ref_k`. The model is
/// documented under [`PvCell::current_at`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PvCell {
    /// Short-circuit current at reference conditions (A).
    pub isc_ref: f64,
    /// Open-circuit voltage at reference conditions (V).
    pub voc_ref: f64,
    /// Reference cell temperature (K).
    pub t_ref_k: f64,
    /// Diode ideality factor (dimensionless, typically 1–2).
    pub n: f64,
    /// Series resistance (Ω).
    pub rs: f64,
    /// Shunt (parallel) resistance (Ω).
    pub rsh: f64,
    /// Effective band-gap energy of the cell material (eV), e.g. ≈ 1.12 for
    /// crystalline silicon.
    pub eg: f64,
}

impl PvCell {
    /// Construct a PV cell from its single-diode parameters.
    ///
    /// # Panics
    /// None. All parameters are accepted as-is; non-physical values simply
    /// produce non-physical [`PvCell::current_at`] outputs.
    #[must_use]
    pub fn new(
        isc_ref: f64,
        voc_ref: f64,
        t_ref_k: f64,
        n: f64,
        rs: f64,
        rsh: f64,
        eg: f64,
    ) -> Self {
        Self {
            isc_ref,
            voc_ref,
            t_ref_k,
            n,
            rs,
            rsh,
            eg,
        }
    }

    /// A typical crystalline-silicon cell at standard test
    /// conditions (25 °C, 1000 W/m²). Useful as a starting point for tests and
    /// examples. `rsh` is taken as a healthy 100 Ω so the shunt leakage at
    /// open circuit is small.
    #[must_use]
    pub fn silicon_reference() -> Self {
        Self::new(8.0, 0.6, 298.15, 1.3, 0.001, 100.0, 1.12)
    }

    /// Thermal voltage `V_t = k_B·T/q` (V) at a given absolute temperature (K).
    #[must_use]
    pub fn thermal_voltage(t_k: f64) -> f64 {
        KB_EV * t_k
    }

    /// Photogenerated current (A). Scales linearly with irradiance `g`
    /// (W/m²) relative to the reference irradiance [`G_REF`].
    ///
    /// Temperature dependence of `I_ph` (via a small positive temperature
    /// coefficient) is intentionally omitted to keep the model transparent;
    /// the dominant temperature effect on the I–V curve is captured by the
    /// saturation-current term in [`PvCell::current_at`].
    #[must_use]
    pub fn photocurrent(&self, irradiance: f64) -> f64 {
        self.isc_ref * (irradiance / G_REF)
    }

    /// Diode saturation (dark) current `I_0` (A) at the given cell temperature
    /// `temp_c` (°C).
    ///
    /// `I_0` is anchored at reference conditions from `I_sc` and `V_oc`
    /// (`I_0_ref = I_sc / (exp(V_oc/(n·V_t_ref)) − 1)`) and then scaled with
    /// temperature through the band-gap term
    /// `exp(−E_g/(n·k_B)·(1/T − 1/T_ref))`, so leakage grows as the cell
    /// warms.
    #[must_use]
    pub fn saturation_current(&self, temp_c: f64) -> f64 {
        let t_k = temp_c + CELSIUS_OFFSET;
        let vt_ref = Self::thermal_voltage(self.t_ref_k);
        let i0_ref = self.isc_ref / ((self.voc_ref / (self.n * vt_ref)).exp() - 1.0);
        let temp_term = -(self.eg / (self.n * KB_EV)) * (1.0 / t_k - 1.0 / self.t_ref_k);
        i0_ref * temp_term.exp()
    }

    /// Solve the single-diode equation for the terminal current (A) at a given
    /// terminal voltage `voltage` (V), plane-of-array irradiance `irradiance`
    /// (W/m²) and cell temperature `temp_c` (°C).
    ///
    /// The underlying implicit equation is
    ///
    /// ```text
    /// I = I_ph − I_0·(exp((V + I·R_s)/(n·V_t)) − 1) − (V + I·R_s)/R_sh
    /// ```
    ///
    /// where `I_ph` scales with irradiance and `I_0` scales with temperature
    /// (see [`PvCell::photocurrent`] / [`PvCell::saturation_current`]). It is
    /// solved by a fixed number of Picard (fixed-point) iterations starting
    /// from the photocurrent guess; for the well-conditioned parameter ranges
    /// of real cells this converges to well within engineering tolerance.
    ///
    /// Sanity limits: at `V = 0` the current approaches `I_ph` (short
    /// circuit), and at `V = V_oc` (reference conditions) the current
    /// approaches 0 (open circuit).
    #[must_use]
    pub fn current_at(&self, voltage: f64, irradiance: f64, temp_c: f64) -> f64 {
        let t_k = temp_c + CELSIUS_OFFSET;
        let vt = Self::thermal_voltage(t_k);
        let i_ph = self.photocurrent(irradiance);
        let i0 = self.saturation_current(temp_c);

        let mut i = i_ph; // initial guess: short-circuit current.
        for _ in 0..60 {
            let v_d = voltage + i * self.rs; // diode (internal) voltage.
            let i_new = i_ph - i0 * ((v_d / (self.n * vt)).exp() - 1.0) - v_d / self.rsh;
            i = i_new;
            if !i.is_finite() {
                break;
            }
        }
        i
    }
}

/// Wind-turbine cut-in speed (m/s) below which the turbine produces no power.
pub const WIND_CUT_IN: f64 = 3.0;
/// Wind-turbine rated wind speed (m/s) above which output is clamped to the
/// rated power.
pub const WIND_RATED: f64 = 12.0;
/// Wind-turbine cut-out speed (m/s) above which the turbine feathers and
/// produces no power.
pub const WIND_CUT_OUT: f64 = 25.0;

/// Available kinetic power in the air swept by a rotor of area `rotor_area`
/// (m²) at air density `rho` (kg/m³) and wind speed `v_wind` (m/s):
/// `0.5·ρ·A·v³` (W). This is the power available to *any* wind-energy
/// extractor before the power coefficient `C_p` is applied.
#[must_use]
pub fn wind_kinetic_power(rho: f64, rotor_area: f64, v_wind: f64) -> f64 {
    0.5 * rho * rotor_area * v_wind * v_wind * v_wind
}

/// Wind-turbine electrical output power (W) with a realistic operating
/// envelope.
///
/// The ideal aerodynamic power `0.5·ρ·A·C_p·v³` is returned only inside the
/// operating band `[WIND_CUT_IN, WIND_RATED]`; above the rated speed the
/// output is clamped to the rated power (`0.5·ρ·A·C_p·WIND_RATED³`, achieved by
/// pitching the blades) and outside the `[WIND_CUT_IN, WIND_CUT_OUT]` window
/// the turbine is parked and produces zero.
///
/// * `v_wind` — wind speed (m/s)
/// * `rho` — air density (kg/m³)
/// * `rotor_area` — swept rotor area (m²)
/// * `cp` — rotor power coefficient (dimensionless, must satisfy `C_p ≤ 16/27`)
///
/// # Panics
/// None.
#[must_use]
pub fn wind_power(v_wind: f64, rho: f64, rotor_area: f64, cp: f64) -> f64 {
    if !(WIND_CUT_IN..=WIND_CUT_OUT).contains(&v_wind) {
        return 0.0;
    }
    let p = wind_kinetic_power(rho, rotor_area, v_wind) * cp;
    let rated_p = wind_kinetic_power(rho, rotor_area, WIND_RATED) * cp;
    p.min(rated_p)
}

/// Maximum extractable wind power (W) under the Betz limit `C_p ≤ 16/27 ≈
/// `0.5926``.
///
/// Returns `16/27 · 0.5·ρ·A·v³`, i.e. the theoretical ceiling on what any
/// turbine (no matter how ideal) can remove from the wind stream. It is always
/// `<= 0.593 · 0.5·ρ·A·v³` and is an upper bound on [`wind_power`] for any
/// physically valid `C_p`.
///
/// * `rho` — air density (kg/m³)
/// * `area` — swept rotor area (m²)
/// * `v` — wind speed (m/s)
#[must_use]
pub fn betz_limit_power(rho: f64, area: f64, v: f64) -> f64 {
    const CP_BETZ: f64 = 16.0 / 27.0;
    wind_kinetic_power(rho, area, v) * CP_BETZ
}

/// Expected number of equivalent-full cycles until a lithium-ion cell reaches
/// its end-of-life capacity-loss `threshold`, modelled with a Weibull life
/// distribution.
///
/// The linear-fade nominal life is `threshold / capacity_fade_rate` cycles
/// (the number of cycles that would deplete `threshold` of initial capacity at
/// a constant per-cycle fade). This nominal life is used as the Weibull
/// *characteristic* life, scaled by the calibration multiplier `scale`, and the
/// **mean** (expected) end-of-life cycle count is then read from
/// `tpt-eng-reliability`'s Weibull mean `η·Γ(1 + 1/β)`.
///
/// * `capacity_fade_rate` — fractional capacity lost per equivalent-full cycle
///   (e.g. `2e-4` = 0.02 %/cycle). Must be positive.
/// * `threshold` — end-of-life capacity-loss fraction (e.g. `0.2` = 20 % fade).
///   Must be positive.
/// * `shape` — Weibull shape (β) parameter. Must be positive.
/// * `scale` — dimensionless calibration multiplier applied to the nominal
///   linear-fade life. Must be positive.
///
/// # Errors
/// Returns [`ReliabilityError::InvalidParameter`] if any of `capacity_fade_rate`,
/// `threshold`, `shape` or `scale` is non-positive (the underlying Weibull
/// mean requires positive parameters).
pub fn cycles_to_threshold(
    capacity_fade_rate: f64,
    threshold: f64,
    shape: f64,
    scale: f64,
) -> Result<f64, ReliabilityError> {
    if capacity_fade_rate <= 0.0 || threshold <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return Err(ReliabilityError::InvalidParameter(
            "capacity_fade_rate, threshold, shape and scale must be positive".into(),
        ));
    }
    let nominal = threshold / capacity_fade_rate; // linear-fade nominal life.
    let eta = nominal * scale;
    weibull_mean(eta, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pv_short_circuit_equals_photocurrent() {
        let cell = PvCell::silicon_reference();
        // At V = 0, I ≈ I_ph (= isc_ref at reference irradiance).
        let i_sc = cell.current_at(0.0, G_REF, 25.0);
        let i_ph = cell.photocurrent(G_REF);
        assert!((i_sc - i_ph).abs() < 1e-2, "i_sc={i_sc}, i_ph={i_ph}");
    }

    #[test]
    fn pv_open_circuit_current_is_zero() {
        let cell = PvCell::silicon_reference();
        // At V = Voc_ref (reference conditions) the current → 0.
        let i_oc = cell.current_at(cell.voc_ref, G_REF, 25.0);
        assert!(i_oc.abs() < 1e-2, "i_oc={i_oc}");
    }

    #[test]
    fn pv_current_decreases_with_voltage() {
        let cell = PvCell::silicon_reference();
        let i_low = cell.current_at(0.1, G_REF, 25.0);
        let i_high = cell.current_at(0.5, G_REF, 25.0);
        assert!(i_low > i_high, "i_low={i_low}, i_high={i_high}");
    }

    #[test]
    fn pv_scales_with_irradiance() {
        let cell = PvCell::silicon_reference();
        let half = cell.current_at(0.0, G_REF / 2.0, 25.0);
        let full = cell.current_at(0.0, G_REF, 25.0);
        assert!((half - full / 2.0).abs() < 1e-2);
    }

    #[test]
    fn wind_power_in_operating_band() {
        let rho = 1.225;
        let a = 10_000.0;
        let cp = 0.4;
        let v = 8.0; // inside [cut-in, rated].
        let p = wind_power(v, rho, a, cp);
        let expected = 0.5 * rho * a * cp * v * v * v;
        assert!((p - expected).abs() < 1e-6, "p={p}, expected={expected}");
    }

    #[test]
    fn wind_power_clamped_at_rated() {
        let rho = 1.225;
        let a = 10_000.0;
        let cp = 0.4;
        let v = 20.0; // between rated and cut-out → clamped to rated power.
        let p = wind_power(v, rho, a, cp);
        let rated_p = 0.5 * rho * a * cp * WIND_RATED * WIND_RATED * WIND_RATED;
        assert!((p - rated_p).abs() < 1e-6, "p={p}, rated_p={rated_p}");
        // The clamp must be strictly below the unconstrained v³ power.
        let unclamped = 0.5 * rho * a * cp * v * v * v;
        assert!(p < unclamped);
    }

    #[test]
    fn wind_power_zero_outside_band() {
        assert_eq!(wind_power(2.0, 1.225, 10_000.0, 0.4), 0.0); // below cut-in.
        assert_eq!(wind_power(26.0, 1.225, 10_000.0, 0.4), 0.0); // above cut-out.
    }

    #[test]
    fn betz_power_never_exceeds_fraction() {
        let rho = 1.225;
        let a = 10_000.0;
        for v in [1.0, 5.0, 12.0, 24.0] {
            let p = betz_limit_power(rho, a, v);
            let ceiling = 0.593 * 0.5 * rho * a * v * v * v;
            assert!(p <= ceiling, "p={p}, ceiling={ceiling}, v={v}");
            // Ratio should equal the Betz fraction 16/27.
            let kinetic = 0.5 * rho * a * v * v * v;
            assert!((p / kinetic - 16.0 / 27.0).abs() < 1e-12);
        }
    }

    #[test]
    fn battery_fade_rate_monotonic() {
        // Slower fade → more cycles to end-of-life.
        let slow = cycles_to_threshold(1e-4, 0.2, 2.0, 1.0).unwrap();
        let fast = cycles_to_threshold(4e-4, 0.2, 2.0, 1.0).unwrap();
        assert!(slow > fast, "slow={slow}, fast={fast}");
    }

    #[test]
    fn battery_threshold_monotonic() {
        // A larger allowed fade threshold → more cycles.
        let tight = cycles_to_threshold(2e-4, 0.1, 2.0, 1.0).unwrap();
        let loose = cycles_to_threshold(2e-4, 0.3, 2.0, 1.0).unwrap();
        assert!(loose > tight, "loose={loose}, tight={tight}");
    }

    #[test]
    fn battery_expected_life_reasonable() {
        // 0.02 %/cycle to 20 % fade → nominal 1000 cycles; Weibull(β=2) mean
        // is nominal·Γ(1.5) ≈ nominal·0.886.
        let mean = cycles_to_threshold(2e-4, 0.2, 2.0, 1.0).unwrap();
        assert!(mean > 800.0 && mean < 1000.0, "mean={mean}");
    }
}
