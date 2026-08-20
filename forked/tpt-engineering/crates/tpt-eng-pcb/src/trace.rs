//! Trace models: microstrip characteristic impedance and IPC-2221 current
//! capacity, plus cross-section / DC-resistance helpers.

use tpt_eng_electrical::dc_resistance;

use crate::MIL_TO_M;

/// Cross-sectional area of a rectangular trace (mils²) from its width and
/// thickness (both in metres).
///
/// ```text
/// A[mil²] = (w[m] / 2.54e-5) · (t[m] / 2.54e-5)
/// ```
///
/// IPC-2221 current-capacity relations are expressed in mils², so this helper
/// bridges SI trace geometry to those formulas.
pub fn trace_area_mil2(width_m: f64, thickness_m: f64) -> f64 {
    let w_mil = width_m / MIL_TO_M;
    let t_mil = thickness_m / MIL_TO_M;
    w_mil * t_mil
}

/// Characteristic impedance (Ω) of a surface microstrip.
///
/// Uses the widely cited closed-form approximations (Wheeler / IPC for narrow
/// strips, Edwards for wide strips), all in SI units:
///
/// * `w/h ≤ 1` (narrow):
///   ```text
///   Z0 = (87 / √(εᵣ + 1.41)) · ln( 5.98·h / (0.8·w + t) )
///   ```
/// * `w/h > 1` (wide, Edwards):
///   ```text
///   Z0 = (60 / √(εᵣ + 1.41)) · ln( 8·h/w + 0.25·w/h )
///   ```
///
/// where `w` is the trace width (m), `h` the substrate height beneath the trace
/// (m), `t` the trace thickness (m) and `εᵣ` the substrate relative
/// permittivity.
///
/// # Arguments
///
/// * `width_m` — trace width (m).
/// * `height_m` — substrate height between the trace and the reference plane (m).
/// * `relative_permittivity` — substrate relative permittivity `εᵣ` (dimensionless).
/// * `trace_thickness_m` — conductor thickness (m).
///
/// # Panics
///
/// Panics if any geometric argument is non-positive (logarithm / division by
/// zero).
pub fn microstrip_impedance(
    width_m: f64,
    height_m: f64,
    relative_permittivity: f64,
    trace_thickness_m: f64,
) -> f64 {
    let er = relative_permittivity + 1.41;
    let root = er.sqrt();
    let w_over_h = width_m / height_m;

    if w_over_h <= 1.0 {
        let ratio = 5.98 * height_m / (0.8 * width_m + trace_thickness_m);
        (87.0 / root) * ratio.ln()
    } else {
        let ratio = 8.0 * height_m / width_m + 0.25 * width_m / height_m;
        (60.0 / root) * ratio.ln()
    }
}

/// Maximum permissible current (A) of a trace per the IPC-2221 external/internal
/// temperature-rise nomograph.
///
/// ```text
/// I = k · ΔT^0.44 · A^0.725
/// ```
///
/// with `A` the cross-sectional area in **mils²** and `ΔT` the allowed
/// temperature rise (°C). The coefficient is `k = 0.048` for outer (external)
/// layers and `k = 0.024` for inner (internal) layers.
///
/// # Arguments
///
/// * `area_mil2` — cross-sectional area (mils²). Use [`trace_area_mil2`] to
///   convert SI geometry, or supply a value already in mils².
/// * `temp_rise_c` — allowed temperature rise above ambient (°C).
/// * `external` — `true` for an outer-layer trace, `false` for an inner-layer
///   trace.
///
/// # Panics
///
/// Panics if `area_mil2` or `temp_rise_c` is negative (non-physical inputs would
/// produce `NaN`).
pub fn ipc_2221_current_capacity(area_mil2: f64, temp_rise_c: f64, external: bool) -> f64 {
    assert!(
        area_mil2 >= 0.0,
        "cross-sectional area must be non-negative"
    );
    assert!(temp_rise_c >= 0.0, "temperature rise must be non-negative");
    let k = if external { 0.048 } else { 0.024 };
    k * temp_rise_c.powf(0.44) * area_mil2.powf(0.725)
}

/// DC resistance (Ω) of a straight rectangular trace of given length.
///
/// Combines the trace cross-section (`width_m · thickness_m`) with
/// [`tpt_eng_electrical::dc_resistance`] (`R = ρ·L / A`), so the copper
/// resistivity can be taken straight from `tpt-eng-electrical`'s material
/// tables.
///
/// # Arguments
///
/// * `length_m` — trace length (m).
/// * `width_m` — trace width (m).
/// * `thickness_m` — conductor thickness (m).
/// * `resistivity_ohm_m` — conductor resistivity (Ω·m).
///
/// # Panics
///
/// Panics if the cross-sectional area (`width_m · thickness_m`) is zero.
pub fn trace_dc_resistance(
    length_m: f64,
    width_m: f64,
    thickness_m: f64,
    resistivity_ohm_m: f64,
) -> f64 {
    let area = width_m * thickness_m;
    dc_resistance(length_m, area, resistivity_ohm_m)
}
