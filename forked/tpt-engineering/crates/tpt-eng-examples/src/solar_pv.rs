//! Solar PV array sizing + three-phase grid connection.
//!
//! A cross-domain cookbook scenario that composes [`tpt_eng_renewables`]
//! (single-diode PV cell I–V model, maximum-power-point sweep) with
//! [`tpt_eng_electrical`] (balanced three-phase power) to size a rooftop array
//! and its grid-tie inverter connection. This sits alongside
//! [`crate::thermal_loop`] and [`crate::mechanical_design`] as a worked example
//! of "how do I use these crates together?".

use tpt_eng_electrical::three_phase_power;
use tpt_eng_renewables::PvCell;

/// Result of the solar-PV sizing scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPvReport {
    /// Cells per module (series string).
    pub cells_per_module: usize,
    /// Total modules in the array.
    pub modules: usize,
    /// Module maximum-power-point voltage (V).
    pub module_v_mp: f64,
    /// Module maximum-power-point current (A).
    pub module_i_mp: f64,
    /// Array DC power (W).
    pub array_dc_w: f64,
    /// Grid line-line voltage used for the tie (V).
    pub grid_v_ll: f64,
    /// Required three-phase line current (A).
    pub line_current_a: f64,
    /// Power factor assumed for the tie.
    pub power_factor: f64,
}

/// Size a rooftop PV array and its three-phase grid connection.
///
/// Returns a [`SolarPvReport`] summarising the module MPP, array DC power, and
/// the line current the grid-tie inverter must carry.
///
/// # Panics
///
/// Panics if the three-phase power cross-check (`three_phase_power` fed the
/// computed line current) does not recover the array's DC power, which would
/// indicate an internal inconsistency in the sizing maths.
pub fn run_solar_pv_sizing() -> SolarPvReport {
    let cell = PvCell::silicon_reference();
    let n_series = 60usize; // typical 60-cell module
    let irradiance = 1000.0; // W/m² (standard test conditions)
    let temp_c = 25.0;

    // Maximum-power-point sweep over the cell voltage.
    let voc = cell.voc_ref;
    let steps = 400usize;
    let mut best = (0.0f64, 0.0f64, 0.0f64); // (v, i, p)
    for k in 1..steps {
        let v = voc * (k as f64) / (steps as f64);
        let i = cell.current_at(v, irradiance, temp_c);
        let p = v * i;
        if p > best.2 {
            best = (v, i, p);
        }
    }
    let (v_cell, i_cell, _p_cell) = best;
    let module_v_mp = v_cell * n_series as f64;
    let module_i_mp = i_cell; // series current is unchanged
    let _p_module = module_v_mp * module_i_mp;

    // Array: 2 parallel strings of 10 series modules.
    let series_mods = 10usize;
    let parallel_strings = 2usize;
    let modules = series_mods * parallel_strings;
    let array_v = module_v_mp * series_mods as f64;
    let array_i = module_i_mp * parallel_strings as f64;
    let array_dc_w = array_v * array_i;

    // Three-phase grid tie at 400 V line-line, power factor 0.98.
    let grid_v_ll = 400.0;
    let pf = 0.98;
    let line_current_a = array_dc_w / (3.0f64.sqrt() * grid_v_ll * pf);

    // Cross-check with the electrical crate: feeding `line_current_a` at the
    // grid voltage and pf must recover the array's real power.
    let (p_check, _q) = three_phase_power(grid_v_ll, line_current_a, pf);
    assert!(
        (p_check - array_dc_w).abs() < 1e-6,
        "three-phase power cross-check failed"
    );

    SolarPvReport {
        cells_per_module: n_series,
        modules,
        module_v_mp,
        module_i_mp,
        array_dc_w,
        grid_v_ll,
        line_current_a,
        power_factor: pf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizing_is_consistent() {
        let r = run_solar_pv_sizing();
        // A 60-cell Si module sits around 30–40 V at MPP.
        assert!(r.module_v_mp > 20.0 && r.module_v_mp < 60.0);
        // 20 modules at ~250 W each => a few kW array.
        assert!(r.array_dc_w > 1000.0);
        let expected_i = r.array_dc_w / (3.0f64.sqrt() * r.grid_v_ll * r.power_factor);
        assert!((r.line_current_a - expected_i).abs() < 1e-9);
    }
}
