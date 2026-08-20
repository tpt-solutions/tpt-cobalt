//! # tpt-eng-building-sys
//!
//! Building-systems engineering primitives for HVAC load calculation,
//! plumbing fixture-unit demand (Hunter's curve), and electrical panel
//! scheduling.
//!
//! All scalar quantities are plain `f64` in SI units (watts `W`, metres `m`,
//! kelvin `K`, etc.) unless a function documents a specific unit. The crate is
//! intentionally **not** unit-typed (it does not use `uom`): the values are
//! documented inline and the math stays dependency-light.
//!
//! The crate leans on sibling crates for the underlying physics:
//! * [`tpt_eng_heat_transfer`] — thermal-resistance networks
//!   ([`tpt_eng_heat_transfer::series_resistances`],
//!   [`tpt_eng_heat_transfer::plane_wall_resistance`],
//!   [`tpt_eng_heat_transfer::heat_rate`]) and Nusselt/conviction
//!   correlations used for convective film coefficients.
//! * [`tpt_eng_electrical`] — balanced three-phase power
//!   ([`tpt_eng_electrical::three_phase_power`]) for relating line current to
//!   connected demand.
//! * [`tpt_eng_props_air`] — psychrometrics, surfaced here through
//!   [`tpt_eng_heat_transfer::water_saturation_pressure`] for `f64`-typed
//!   saturation-pressure lookups.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_building_sys::{envelope_ua, heating_load, fixture_unit_demand_gpm, FixtureType};
//!
//! let ua = envelope_ua(&[(100.0, 0.3), (40.0, 0.5)]);
//! let q = heating_load(ua, 20.0, 800.0, 500.0);
//! assert!(q > 0.0);
//!
//! // 300 WSFU on a flush-tank system → a few hundred gpm of expected demand.
//! let d = fixture_unit_demand_gpm(300.0, FixtureType::FlushTank).unwrap();
//! assert!(d > 0.0);
//! ```

use std::fmt;

use tpt_eng_electrical as electrical;
use tpt_eng_heat_transfer as ht;

/// Standard interior (still-air) film resistance for a vertical surface,
/// K·m²/W. Representative of ASHRAE winter design (≈ 0.12 K·m²/W).
pub const INTERIOR_FILM_RESISTANCE: f64 = 0.12;

/// Standard exterior (15.4 km/h / 9.3 mph wind) film resistance for a
/// vertical surface, K·m²/W (≈ 0.04 K·m²/W).
pub const EXTERIOR_FILM_RESISTANCE: f64 = 0.04;

/// Stefan–Boltzmann constant, W/(m²·K⁴), re-exported for radiative film
/// calculations.
pub const SIGMA: f64 = ht::SIGMA;

// -------------------------------------------------------------------------
// Errors
// -------------------------------------------------------------------------

/// Errors returned by building-systems calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingError {
    /// A fixture-unit demand was requested for a non-finite or negative
    /// fixture-unit count.
    InvalidFixtureUnits,
    /// A panel rating was supplied that was non-positive or non-finite.
    InvalidPanelRating,
    /// A convective film coefficient could not be formed because the
    /// characteristic length was non-positive.
    InvalidGeometry,
}

impl fmt::Display for BuildingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildingError::InvalidFixtureUnits => {
                write!(f, "fixture units must be a finite, non-negative number")
            }
            BuildingError::InvalidPanelRating => {
                write!(f, "panel rating must be a finite, positive number")
            }
            BuildingError::InvalidGeometry => {
                write!(f, "convective geometry must have a positive length")
            }
        }
    }
}

impl std::error::Error for BuildingError {}

// -------------------------------------------------------------------------
// HVAC — envelope and loads
// -------------------------------------------------------------------------

/// Total envelope conductance `UA` (W/K).
///
/// Sums `area · u_value` over every wall/roof/floor assembly. Each entry is
/// `(area_m2, u_value_W_per_m2K)`: `area` in square metres, `u_value` in
/// W/(m²·K). The product is the assembly conductance; summing them gives the
/// whole-envelope `UA`.
///
/// This mirrors the single-assembly resistance
/// [`tpt_eng_heat_transfer::plane_wall_resistance`] in the limit of one
/// combined resistance `R = 1/UA`.
pub fn envelope_ua(walls: &[(f64, f64)]) -> f64 {
    walls.iter().map(|&(a, u)| a * u).sum()
}

/// A single opaque assembly layer: material conductivity `k` (W/(m·K)) and
/// thickness `l` (m). Its conductive resistance is `l / k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssemblyLayer {
    /// Thermal conductivity, W/(m·K).
    pub conductivity: f64,
    /// Thickness, m.
    pub thickness: f64,
}

impl AssemblyLayer {
    /// Construct an assembly layer.
    pub fn new(conductivity: f64, thickness: f64) -> Self {
        Self {
            conductivity,
            thickness,
        }
    }

    /// Conductive resistance of this layer in K·m²/W (`l / k`).
    pub fn resistance(&self) -> f64 {
        self.thickness / self.conductivity
    }
}

/// U-value (W/(m²·K)) of a multi-layer opaque assembly including interior and
/// exterior surface film resistances.
///
/// `R_total = r_interior + Σ(l_i/k_i) + r_exterior`; the U-value is
/// `1 / R_total`. Uses [`tpt_eng_heat_transfer::series_resistances`].
pub fn assembly_u_value(layers: &[AssemblyLayer], interior_film: f64, exterior_film: f64) -> f64 {
    let mut r = vec![interior_film, exterior_film];
    r.extend(layers.iter().map(AssemblyLayer::resistance));
    let total = ht::series_resistances(&r);
    1.0 / total
}

/// Heating load (W): `envelope_UA · ΔT + infiltration_W + internal_gains_W`.
///
/// * `envelope_ua` — whole-building conductance, W/K.
/// * `design_dt` — design temperature difference (indoor−outdoor), K.
/// * `infiltration_w` — sensible loss due to infiltration/ventilation, W.
/// * `internal_gains_w` — internal gains, W. Pass a **negative** value to
///   credit heat that offsets the heating demand (the formula is the literal
///   sum `UA·ΔT + infiltration + gains` as commonly tabulated).
pub fn heating_load(
    envelope_ua: f64,
    design_dt: f64,
    infiltration_w: f64,
    internal_gains_w: f64,
) -> f64 {
    envelope_ua * design_dt + infiltration_w + internal_gains_w
}

/// Cooling load (W): `envelope_UA·ΔT + infiltration + solar_gains`.
///
/// Same structure as [`heating_load`] but with `solar_gains_w` as the gained
/// heat that *adds* to the cooling demand (typically positive).
pub fn cooling_load(
    envelope_ua: f64,
    design_dt: f64,
    infiltration_w: f64,
    solar_gains_w: f64,
) -> f64 {
    envelope_ua * design_dt + infiltration_w + solar_gains_w
}

/// Sensible heat loss/gain (W) through one assembly given its `UA` (W/K) and
/// the temperature difference across it. Thin wrapper over
/// [`tpt_eng_heat_transfer::heat_rate`] with a single resistance `1/UA`.
pub fn transmission_heat_rate(ua: f64, delta_t: f64) -> f64 {
    if ua <= 0.0 {
        return 0.0;
    }
    ht::heat_rate(delta_t, &[1.0 / ua])
}

/// Estimated annual heating energy (kWh) for a constant design load.
///
/// `E = heating_load_W · annual_degree_hours / 1000`, where
/// `annual_degree_hours` is the sum over the season of `ΔT · hours` (K·h).
/// The result is converted from W·h to kWh by the `/1000` factor.
pub fn annual_heating_energy_kwh(heating_load_w: f64, annual_degree_hours: f64) -> f64 {
    heating_load_w * annual_degree_hours / 1000.0
}

// -------------------------------------------------------------------------
// HVAC — infiltration
// -------------------------------------------------------------------------

/// Sensible infiltration/ventilation loss (W) via mass flow:
/// `q = ρ · V̇ · c_p · ΔT`.
///
/// * `air_density` — ρ, kg/m³ (≈ 1.2 at 20 °C, 101.325 kPa).
/// * `volumetric_flow` — V̇, m³/s.
/// * `cp_air` — specific heat, J/(kg·K) (≈ 1006 for dry air).
/// * `delta_t` — temperature difference, K.
pub fn infiltration_loss_w(
    air_density: f64,
    volumetric_flow: f64,
    cp_air: f64,
    delta_t: f64,
) -> f64 {
    air_density * volumetric_flow * cp_air * delta_t
}

/// Air-change-rate infiltration loss (W): `q = n·V·ρ·c_p·ΔT / 3600`.
///
/// `ach` is air changes per hour, `volume_m3` the conditioned volume, and the
/// `/3600` converts h⁻¹ to s⁻¹.
pub fn infiltration_loss_ach_w(
    ach: f64,
    volume_m3: f64,
    air_density: f64,
    cp_air: f64,
    delta_t: f64,
) -> f64 {
    ach * volume_m3 * air_density * cp_air * delta_t / 3600.0
}

// -------------------------------------------------------------------------
// HVAC — films (radiative / convective)
// -------------------------------------------------------------------------

/// Radiative film coefficient (W/(m²·K)) for a surface at `t_s` exchanging
/// with large surroundings at `t_surr` via a linearised Stefan–Boltzmann flux.
///
/// Uses `h_r = ε·σ·(T_s² + T_surr²)·(T_s + T_surr)` (exact linearisation of
/// `σ·(T_s⁴ − T_surr⁴)/(T_s − T_surr)`). `emissivity` is dimensionless in
/// `[0, 1]`; temperatures are in kelvin.
pub fn radiative_film_coefficient(
    emissivity: f64,
    surface_temp_k: f64,
    surround_temp_k: f64,
) -> f64 {
    emissivity
        * SIGMA
        * (surface_temp_k * surface_temp_k + surround_temp_k * surround_temp_k)
        * (surface_temp_k + surround_temp_k)
}

/// Exterior convective film coefficient (W/(m²·K)) for a vertical surface in
/// wind using the [`tpt_eng_heat_transfer`] flat-plate correlation.
///
/// `h = Nu·k / L` with `Nu = 0.664·Re^½·Pr^(1/3)` (laminar). `wind_speed` is in
/// m/s, `length` the characteristic dimension (m), `air_conductivity` W/(m·K),
/// `air_kinematic_viscosity` m²/s and `air_prandtl` dimensionless.
///
/// # Errors
///
/// Returns [`BuildingError::InvalidGeometry`] if `length` is not positive.
pub fn convective_film_coefficient(
    wind_speed: f64,
    length: f64,
    air_conductivity: f64,
    air_kinematic_viscosity: f64,
    air_prandtl: f64,
) -> Result<f64, BuildingError> {
    if length <= 0.0 || !length.is_finite() {
        return Err(BuildingError::InvalidGeometry);
    }
    let re = wind_speed * length / air_kinematic_viscosity;
    let nu = ht::nusselt_flat_plate(re, air_prandtl, ht::FlowRegime::Laminar);
    Ok(ht::convection_coefficient(nu, air_conductivity, length))
}

/// Combined surface film resistance (K·m²/W) from convective and radiative
/// film coefficients: `R_film = 1/(h_c + h_r)`.
pub fn combined_film_resistance(convective_h: f64, radiative_h: f64) -> f64 {
    1.0 / (convective_h + radiative_h)
}

// -------------------------------------------------------------------------
// Plumbing — Hunter's curve
// -------------------------------------------------------------------------

/// Fixture-group type for the two Hunter's-curve regimes.
///
/// * `FlushTank` — tank-type fixtures (e.g. residential WCs), lower demand
///   per fixture unit.
/// * `FlushValve` — valve-type fixtures (e.g. commercial WCs), higher peak
///   demand per fixture unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureType {
    /// Tank-type fixture group (IPC Table E103.3(3), "Private" column).
    FlushTank,
    /// Valve-type fixture group (IPC Table E103.3(3), "Public" column).
    FlushValve,
}

/// Approximate measured demand (gpm) at reference water-supply fixture units
/// (WSFU) from the IPC Appendix E Table E103.3(3). Values are reproduced from
/// the published Hunter's-curve table (gpm).
const HUNTER_TANK_GPM: &[(f64, f64)] = &[
    (1.0, 3.0),
    (2.0, 5.0),
    (3.0, 6.5),
    (4.0, 8.0),
    (5.0, 9.4),
    (6.0, 10.7),
    (7.0, 11.8),
    (8.0, 12.8),
    (9.0, 13.7),
    (10.0, 14.6),
    (15.0, 18.3),
    (20.0, 21.7),
    (25.0, 24.7),
    (30.0, 27.6),
    (40.0, 32.7),
    (50.0, 37.5),
    (60.0, 41.8),
    (70.0, 45.9),
    (80.0, 49.8),
    (90.0, 53.6),
    (100.0, 57.3),
    (120.0, 64.3),
    (140.0, 71.0),
    (160.0, 77.4),
    (180.0, 83.5),
    (200.0, 89.5),
    (225.0, 96.9),
    (250.0, 104.0),
    (275.0, 111.0),
    (300.0, 117.8),
    (350.0, 130.7),
    (400.0, 143.1),
    (450.0, 155.2),
    (500.0, 167.0),
    (600.0, 189.8),
    (700.0, 211.5),
    (800.0, 232.4),
    (900.0, 252.7),
];

/// Approximate measured demand (gpm) at reference water-supply fixture units
/// (WSFU) for flush-valve (public) fixtures. Beyond the last tabulated point
/// (900 WSFU ≈ 252.7 gpm) the curve is extrapolated by power-law fit.
const HUNTER_VALVE_GPM: &[(f64, f64)] = &[
    (1.0, 3.0),
    (2.0, 4.5),
    (3.0, 6.0),
    (4.0, 7.5),
    (5.0, 9.0),
    (6.0, 10.4),
    (7.0, 11.8),
    (8.0, 13.1),
    (9.0, 14.4),
    (10.0, 15.7),
    (15.0, 22.0),
    (20.0, 28.1),
    (25.0, 33.9),
    (30.0, 39.6),
    (40.0, 50.7),
    (50.0, 61.5),
    (60.0, 72.4),
    (70.0, 82.9),
    (80.0, 93.1),
    (90.0, 103.2),
    (100.0, 113.2),
    (120.0, 132.8),
    (140.0, 152.0),
    (160.0, 170.8),
    (180.0, 189.3),
    (200.0, 207.5),
    (225.0, 229.8),
    (250.0, 251.6),
    (275.0, 273.2),
    (300.0, 294.5),
    (350.0, 336.6),
    (400.0, 378.0),
    (450.0, 418.8),
    (500.0, 459.2),
    (600.0, 538.7),
    (700.0, 616.8),
    (800.0, 693.7),
    (900.0, 770.0),
];

/// Interpolate (piecewise-linear in log–log space) a demand from a Hunter
/// table for a fixture-unit count.
fn interpolate_hunter(table: &[(f64, f64)], fixture_units: f64) -> f64 {
    if fixture_units <= table[0].0 {
        return table[0].1;
    }
    let last = table.len() - 1;
    if fixture_units >= table[last].0 {
        // Power-law extrapolation: fit y = a·x^b through the final two points.
        let (x0, y0) = table[last - 1];
        let (x1, y1) = table[last];
        let b = (y1.ln() - y0.ln()) / (x1.ln() - x0.ln());
        let a = y1 / x1.powf(b);
        return a * fixture_units.powf(b);
    }
    for i in 0..last {
        let (x0, y0) = table[i];
        let (x1, y1) = table[i + 1];
        if fixture_units <= x1 {
            let t = (fixture_units.ln() - x0.ln()) / (x1.ln() - x0.ln());
            return y0 * (y1 / y0).powf(t);
        }
    }
    table[last].1
}

/// Expected water demand (US gallons per minute, gpm) for a total
/// water-supply fixture-unit (WSFU) count, per Hunter's curve.
///
/// Uses the IPC Appendix E Table E103.3(3) measured-demand values, interpolated
/// piecewise-linear in log–log space and extrapolated by power law beyond the
/// tabulated range. `fixture_type` selects the flush-tank (private) or
/// flush-valve (public) column.
///
/// # Errors
///
/// Returns [`BuildingError::InvalidFixtureUnits`] if `fixture_units` is
/// negative or non-finite.
///
/// # Panics
///
/// Panics if the compiled-in reference tables are shorter than two entries
/// (they are not, by construction).
pub fn fixture_unit_demand_gpm(
    fixture_units: f64,
    fixture_type: FixtureType,
) -> Result<f64, BuildingError> {
    if !fixture_units.is_finite() || fixture_units < 0.0 {
        return Err(BuildingError::InvalidFixtureUnits);
    }
    let table = match fixture_type {
        FixtureType::FlushTank => HUNTER_TANK_GPM,
        FixtureType::FlushValve => HUNTER_VALVE_GPM,
    };
    Ok(interpolate_hunter(table, fixture_units))
}

/// Convert a flow rate in US gallons per minute (gpm) to litres per second
/// (L/s). `1 gpm ≈ 0.06309 L/s`.
pub fn gpm_to_lps(gpm: f64) -> f64 {
    gpm * 0.063_090_196_4
}

/// Expected water demand in litres per second (L/s).
///
/// Wrapper over [`fixture_unit_demand_gpm`] with unit conversion via
/// [`gpm_to_lps`].
///
/// # Errors
///
/// Propagates [`BuildingError::InvalidFixtureUnits`] from the gpm routine.
pub fn fixture_unit_demand_lps(
    fixture_units: f64,
    fixture_type: FixtureType,
) -> Result<f64, BuildingError> {
    fixture_unit_demand_gpm(fixture_units, fixture_type).map(gpm_to_lps)
}

/// Standard water-supply fixture-unit (WSFU) assignments for common fixtures
/// (cold + hot combined), per IPC Table E103.3(2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fixture {
    /// Water closet, tank type.
    WaterClosetTank,
    /// Water closet, flush-valve type.
    WaterClosetValve,
    /// Lavatory (private).
    Lavatory,
    /// Kitchen sink.
    KitchenSink,
    /// Bathtub/shower (per fixture).
    BathtubShower,
    /// Clothes washer.
    WashingMachine,
    /// Dishwasher.
    Dishwasher,
    /// Hose bibb (outdoor faucet).
    HoseBibb,
    /// Urinal, flush-valve.
    UrinalValve,
}

impl Fixture {
    /// Water-supply fixture units (WSFU) for this fixture per IPC Table
    /// E103.3(2).
    pub fn fixture_units(self) -> f64 {
        match self {
            Fixture::WaterClosetTank => 3.0,
            Fixture::WaterClosetValve => 6.0,
            Fixture::Lavatory => 1.0,
            Fixture::KitchenSink => 1.5,
            Fixture::BathtubShower => 2.0,
            Fixture::WashingMachine => 1.5,
            Fixture::Dishwasher => 1.5,
            Fixture::HoseBibb => 2.5,
            Fixture::UrinalValve => 4.0,
        }
    }
}

/// Sum water-supply fixture units (WSFU) for a list of fixtures (each entry is
/// `(fixture, count)`).
pub fn sum_fixture_units(items: &[(Fixture, usize)]) -> f64 {
    items
        .iter()
        .map(|&(f, n)| f.fixture_units() * n as f64)
        .sum()
}

/// Minimum service-pipe diameter (m) carrying `flow_lps` (litres per second)
/// at velocity `v` (m/s): `d = √(4·Q / (1000·π·v))` (the /1000 converts L/s to
/// m³/s). Accepts L/s to pair directly with [`fixture_unit_demand_lps`].
pub fn required_pipe_diameter_m(flow_lps: f64, velocity_mps: f64) -> f64 {
    if velocity_mps <= 0.0 {
        return 0.0;
    }
    (4.0 * flow_lps / (1000.0 * std::f64::consts::PI * velocity_mps)).sqrt()
}

// -------------------------------------------------------------------------
// Electrical — panel scheduling
// -------------------------------------------------------------------------

/// Total connected (demand) load (W) of a panel — the plain sum of branch
/// loads.
///
/// `panel_rating_w` is the panel's nameplate rating in watts; it is validated
/// (must be finite and positive) but the connected total is independent of it.
/// See [`panel_utilization`] for the ratio against the rating.
///
/// # Panics
///
/// Panics in debug builds if `panel_rating_w` is non-positive or non-finite,
/// since a panel must have a positive rating.
pub fn panel_load(branch_loads_w: &[f64], panel_rating_w: f64) -> f64 {
    debug_assert!(
        panel_rating_w.is_finite() && panel_rating_w > 0.0,
        "panel_rating_w must be a finite, positive number"
    );
    branch_loads_w.iter().copied().sum()
}

/// Panel utilization fraction `total_load / rating` (dimensionless, ≥ 0).
///
/// A value ≤ 1 is a valid (non-overloaded) panel. Returns `None` if
/// `panel_rating_w` is non-positive or non-finite.
pub fn panel_utilization(total_load_w: f64, panel_rating_w: f64) -> Option<f64> {
    if !panel_rating_w.is_finite() || panel_rating_w <= 0.0 {
        return None;
    }
    Some(total_load_w / panel_rating_w)
}

/// Convert a three-phase line current (A) to connected power (W).
///
/// Thin wrapper over [`tpt_eng_electrical::three_phase_power`] returning the
/// real-power component for a given line-to-line voltage `v_ll` (V), line
/// current `i_line` (A) and lagging power factor `pf` ∈ `[0, 1]`.
pub fn three_phase_current_to_power_w(v_ll: f64, i_line: f64, pf: f64) -> f64 {
    electrical::three_phase_power(v_ll, i_line, pf).0
}

/// A scheduled branch circuit on a panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Branch {
    /// Circuit description (free-form; not used in calculation).
    pub name: &'static str,
    /// Connected load of the branch, W.
    pub load_w: f64,
}

impl Branch {
    /// Construct a branch.
    pub fn new(name: &'static str, load_w: f64) -> Self {
        Self { name, load_w }
    }

    /// Connected load, W.
    pub fn load(&self) -> f64 {
        self.load_w
    }
}

/// Sum the connected loads (W) of a list of [`Branch`] entries.
pub fn schedule_panel(branches: &[Branch], panel_rating_w: f64) -> f64 {
    let loads: Vec<f64> = branches.iter().map(Branch::load).collect();
    panel_load(&loads, panel_rating_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_ua_sums_area_times_u() {
        // Two walls: 100 m² @ 0.3 and 40 m² @ 0.5 → 30 + 20 = 50 W/K.
        let ua = envelope_ua(&[(100.0, 0.3), (40.0, 0.5)]);
        assert!((ua - 50.0).abs() < 1e-9);
        assert_eq!(envelope_ua(&[]), 0.0);
    }

    #[test]
    fn heating_load_formula() {
        // UA=50 W/K, ΔT=20 K, infil=800 W, gains=500 W.
        // 50·20 + 800 + 500 = 2300 W.
        let q = heating_load(50.0, 20.0, 800.0, 500.0);
        assert!((q - 2300.0).abs() < 1e-9);
        // Gains that offset demand (negative) reduce the total.
        let q2 = heating_load(50.0, 20.0, 800.0, -500.0);
        assert!((q2 - 1300.0).abs() < 1e-9);
    }

    #[test]
    fn cooling_load_formula() {
        // 50·20 + 800 + 1200 = 3000 W.
        let q = cooling_load(50.0, 20.0, 800.0, 1200.0);
        assert!((q - 3000.0).abs() < 1e-9);
    }

    #[test]
    fn assembly_u_value_ok() {
        // Two 0.1 m layers of k=0.04 (R=2.5 each) + films 0.12 + 0.04 = 5.16.
        let layers = [AssemblyLayer::new(0.04, 0.1), AssemblyLayer::new(0.04, 0.1)];
        let u = assembly_u_value(&layers, INTERIOR_FILM_RESISTANCE, EXTERIOR_FILM_RESISTANCE);
        let expected = 1.0 / (2.5 + 2.5 + 0.12 + 0.04);
        assert!((u - expected).abs() < 1e-12);
    }

    #[test]
    fn transmission_heat_rate_matches_ua() {
        // UA=50, ΔT=20 → 1000 W.
        let q = transmission_heat_rate(50.0, 20.0);
        assert!((q - 1000.0).abs() < 1e-9);
        // Zero UA → no transmission.
        assert_eq!(transmission_heat_rate(0.0, 20.0), 0.0);
    }

    #[test]
    fn annual_heating_energy() {
        // 2300 W over 4000 K·h = 9,200,000 W·h = 9200 kWh.
        let e = annual_heating_energy_kwh(2300.0, 4000.0);
        assert!((e - 9200.0).abs() < 1e-6);
    }

    #[test]
    fn infiltration_mass_and_ach_agree() {
        // 0.5 ACH in a 200 m³ volume, ρ=1.2, cp=1006, ΔT=20.
        let q_ach = infiltration_loss_ach_w(0.5, 200.0, 1.2, 1006.0, 20.0);
        // Equivalent volumetric flow = 0.5·200/3600 = 0.02777... m³/s.
        let q_mass = infiltration_loss_w(1.2, 0.5 * 200.0 / 3600.0, 1006.0, 20.0);
        assert!((q_ach - q_mass).abs() < 1e-9);
        assert!(q_ach > 600.0 && q_ach < 700.0);
    }

    #[test]
    fn radiative_film_coefficient_positive() {
        // Between two near-equal temps, h_r ≈ 4·ε·σ·T³.
        let t = 293.15;
        let hr = radiative_film_coefficient(0.9, t, t);
        let approx = 4.0 * 0.9 * SIGMA * t.powi(3);
        assert!((hr - approx).abs() < 1e-3);
        // Zero emissivity → zero radiative film.
        assert_eq!(radiative_film_coefficient(0.0, t, t), 0.0);
    }

    #[test]
    fn convective_film_uses_heat_transfer() {
        // 5 m/s wind, 3 m wall, air-ish k=0.026, ν=1.5e-5, Pr=0.7.
        let h = convective_film_coefficient(5.0, 3.0, 0.026, 1.5e-5, 0.7).unwrap();
        // Laminar Nu = 0.664·(5·3/1.5e-5)^0.5·0.7^(1/3) ≈ 78.2.
        let re = 5.0 * 3.0 / 1.5e-5;
        let nu = ht::nusselt_flat_plate(re, 0.7, ht::FlowRegime::Laminar);
        let expected = ht::convection_coefficient(nu, 0.026, 3.0);
        assert!((h - expected).abs() < 1e-12);
        // Non-positive length is rejected.
        assert_eq!(
            convective_film_coefficient(5.0, 0.0, 0.026, 1.5e-5, 0.7),
            Err(BuildingError::InvalidGeometry)
        );
    }

    #[test]
    fn combined_film_resistance_ok() {
        // h_c=8, h_r=4 → R = 1/12.
        assert!((combined_film_resistance(8.0, 4.0) - 1.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn fixture_unit_demand_monotonic_and_positive() {
        let tank = |fu: f64| fixture_unit_demand_gpm(fu, FixtureType::FlushTank).unwrap();
        assert!(tank(10.0) > 0.0);
        // Demand must increase with fixture units (within a column).
        assert!(tank(50.0) > tank(10.0));
        assert!(tank(500.0) > tank(100.0));
        // Flush-valve systems draw more than flush-tank at equal WSFU.
        assert!(fixture_unit_demand_gpm(100.0, FixtureType::FlushValve).unwrap() > tank(100.0));
    }

    #[test]
    fn fixture_unit_demand_matches_table_points() {
        // 10 WSFU (tank) ≈ 14.6 gpm; 300 WSFU (tank) ≈ 117.8 gpm.
        let d10 = fixture_unit_demand_gpm(10.0, FixtureType::FlushTank).unwrap();
        assert!((d10 - 14.6).abs() < 1e-6);
        let d300 = fixture_unit_demand_gpm(300.0, FixtureType::FlushTank).unwrap();
        assert!((d300 - 117.8).abs() < 1e-6);
    }

    #[test]
    fn fixture_unit_demand_invalid() {
        assert_eq!(
            fixture_unit_demand_gpm(-5.0, FixtureType::FlushTank),
            Err(BuildingError::InvalidFixtureUnits)
        );
        assert_eq!(
            fixture_unit_demand_gpm(f64::NAN, FixtureType::FlushValve),
            Err(BuildingError::InvalidFixtureUnits)
        );
    }

    #[test]
    fn gpm_to_lps_conversion() {
        let lps = fixture_unit_demand_lps(10.0, FixtureType::FlushTank).unwrap();
        let gpm = fixture_unit_demand_gpm(10.0, FixtureType::FlushTank).unwrap();
        assert!((lps - gpm_to_lps(gpm)).abs() < 1e-12);
        assert!((lps - gpm * 0.063_090_196_4).abs() < 1e-12);
    }

    #[test]
    fn sum_fixture_units_and_pipe() {
        let fu = sum_fixture_units(&[
            (Fixture::WaterClosetTank, 2),
            (Fixture::Lavatory, 3),
            (Fixture::KitchenSink, 1),
        ]);
        // 2·3 + 3·1 + 1·1.5 = 10.5 WSFU.
        assert!((fu - 10.5).abs() < 1e-9);
        // 10.5 WSFU tank → ~14.78 gpm ≈ 0.932 L/s at 1.5 m/s ≈ 0.0281 m.
        let d = required_pipe_diameter_m(gpm_to_lps(14.78), 1.5);
        assert!(d > 0.027 && d < 0.029);
    }

    #[test]
    fn panel_load_and_utilization() {
        let loads = [1200.0, 800.0, 2500.0, 500.0];
        let rating = 6000.0;
        let total = panel_load(&loads, rating);
        assert!((total - 5000.0).abs() < 1e-9);
        let util = panel_utilization(total, rating).unwrap();
        assert!((util - 5000.0 / 6000.0).abs() < 1e-12);
        // Valid panel → utilization ≤ 1.
        assert!(util <= 1.0);
    }

    #[test]
    fn panel_utilization_invalid_rating() {
        assert_eq!(panel_utilization(1000.0, 0.0), None);
        assert_eq!(panel_utilization(1000.0, f64::NAN), None);
    }

    #[test]
    fn three_phase_power_matches_electrical() {
        // 400 V, 10 A, pf 0.9 → S = √3·400·10·0.9.
        let p = three_phase_current_to_power_w(400.0, 10.0, 0.9);
        let expected = electrical::three_phase_power(400.0, 10.0, 0.9).0;
        assert!((p - expected).abs() < 1e-9);
    }

    #[test]
    fn schedule_panel_aggregates_branches() {
        let branches = [
            Branch::new("lights", 1200.0),
            Branch::new("receptacles", 2400.0),
            Branch::new("hvac", 3000.0),
        ];
        let total = schedule_panel(&branches, 10000.0);
        assert!((total - 6600.0).abs() < 1e-9);
        let util = panel_utilization(total, 10000.0).unwrap();
        assert!((util - 0.66).abs() < 1e-12);
    }

    #[test]
    fn heat_transfer_saturation_bridge() {
        // Bridge to psychrometrics (props-air, surfaced via heat-transfer);
        // 100 °C ≈ 101.3 kPa.
        let p = ht::water_saturation_pressure(373.15);
        assert!((p - 101_325.0).abs() < 2000.0);
    }
}
