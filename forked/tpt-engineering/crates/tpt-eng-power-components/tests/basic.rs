//! Integration tests for `tpt-eng-power-components`.
//!
//! These exercise the public API through analytic reference cases:
//! ideal-transformer behaviour, the surge-impedance definition, the
//! degenerate and lossless `ABCD` two-port, and the generator / induction
//! machine relations.

use tpt_eng_electrical::{Complex, PerUnitSystem, admittance};
use tpt_eng_power_components::{
    Transformer, TransmissionLine, complex_scale, generator_electrical_power,
    generator_mechanical_power, generator_shaft_power_for_output, induction_mechanical_power,
    induction_rotor_copper_loss, induction_rotor_frequency, induction_slip, surge_impedance,
    synchronous_speed_rpm,
};

const TOL: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (tol {tol})"
    );
}

fn assert_complex_close(actual: Complex, expected: Complex, tol: f64) {
    assert!(
        (actual - expected).magnitude() <= tol,
        "expected {expected:?}, got {actual:?} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

#[test]
fn ideal_unity_ratio_transformer_copies_voltage() {
    let xfmr = Transformer::ideal(1.0);
    assert_eq!(xfmr.leakage_impedance, Complex::new(0.0, 0.0));
    assert_eq!(xfmr.magnetizing_admittance, Complex::new(0.0, 0.0));

    // With no leakage the load impedance divider is exactly unity, so the
    // secondary voltage equals the primary voltage for *any* load.
    for load in [
        Complex::new(1.0, 0.0),
        Complex::new(0.8, 0.6),
        Complex::new(12.5, -4.0),
        Complex::new(1.0e6, 0.0),
    ] {
        let v_sec = xfmr.secondary_voltage(230.0, load);
        assert_complex_close(v_sec, Complex::new(230.0, 0.0), 1e-9);
        // ...and the regulation is therefore zero.
        assert_close(xfmr.voltage_regulation(230.0, load), 0.0, 1e-12);
    }
}

#[test]
fn ideal_transformer_scales_by_turns_ratio() {
    let step_down = Transformer::ideal(10.0);
    let v = step_down.secondary_voltage(11.0e3, Complex::new(5.0, 2.0));
    assert_complex_close(v, Complex::new(1.1e3, 0.0), 1e-9);

    let step_up = Transformer::ideal(0.05); // 400 V -> 8 kV
    let v = step_up.secondary_voltage(400.0, Complex::new(100.0, 0.0));
    assert_complex_close(v, Complex::new(8.0e3, 0.0), 1e-9);

    // An ideal unit is a pure impedance scaler: Z_in = n²·Z_load.
    let z_in = step_down.input_impedance(Complex::new(5.0, 2.0));
    assert_complex_close(z_in, Complex::new(500.0, 200.0), 1e-9);
}

#[test]
fn leakage_impedance_produces_voltage_drop_and_regulation() {
    // Per-unit unit-ratio transformer, Z_leak = j0.1 pu, 1.0 pu resistive load.
    let xfmr = Transformer::new(1.0, Complex::new(0.0, 0.1), Complex::new(0.0, 0.0));
    let load = Complex::new(1.0, 0.0);
    let v_sec = xfmr.secondary_voltage(1.0, load);

    // V_sec = 1/(1 + j0.1) = (1 − j0.1)/1.01.
    assert_close(v_sec.re, 1.0 / 1.01, 1e-12);
    assert_close(v_sec.im, -0.1 / 1.01, 1e-12);

    // |V_sec| = 1/|1 + j0.1|, and the load current lags the primary voltage.
    let mag = 1.0 / (1.0_f64 + 0.01).sqrt();
    assert_close(v_sec.magnitude(), mag, 1e-12);
    assert!(v_sec.phase() < 0.0);

    // Regulation = (1 − |V_sec|)/|V_sec| ≈ 0.4988 %.
    assert_close(xfmr.voltage_regulation(1.0, load), (1.0 - mag) / mag, 1e-12);
    assert!(xfmr.voltage_regulation(1.0, load) > 0.0);
}

#[test]
fn leakage_referral_follows_square_of_turns_ratio() {
    // A 20:1 transformer with 4 + j40 Ω primary-referred leakage looks like
    // 0.01 + j0.1 Ω from the secondary.
    let xfmr = Transformer::new(20.0, Complex::new(4.0, 40.0), Complex::new(0.0, 0.0));
    assert_complex_close(
        xfmr.secondary_referred_leakage(),
        Complex::new(0.01, 0.1),
        1e-12,
    );

    // Explicit divider evaluation must match `secondary_voltage`.
    let load = Complex::new(2.0, 1.0);
    let expected = Complex::new(400.0 / 20.0, 0.0) * load
        / (load + Complex::new(4.0, 40.0) / Complex::new(400.0, 0.0));
    assert_complex_close(xfmr.secondary_voltage(400.0, load), expected, 1e-12);
}

#[test]
fn magnetizing_admittance_matches_impedance_constructor() {
    let z_m = Complex::new(500.0, -5000.0);
    let a = Transformer::with_magnetizing_impedance(1.0, Complex::new(0.01, 0.1), z_m);
    let b = Transformer::new(1.0, Complex::new(0.01, 0.1), admittance(z_m));
    assert_eq!(a, b);

    // The shunt branch draws exciting current but leaves the load voltage of
    // this L-model untouched.
    let load = Complex::new(1.0, 0.0);
    let without = Transformer::new(1.0, Complex::new(0.01, 0.1), Complex::new(0.0, 0.0));
    assert_complex_close(
        a.secondary_voltage(1.0, load),
        without.secondary_voltage(1.0, load),
        1e-15,
    );
    assert!(a.exciting_current(1.0).magnitude() > 0.0);
    assert!(a.no_load_loss(1.0) > 0.0);
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

#[test]
fn generator_electrical_power_is_mechanical_times_efficiency() {
    assert_close(generator_electrical_power(1.0e6, 0.95), 950.0e3, 1e-9);
    assert_close(generator_electrical_power(0.0, 0.95), 0.0, TOL);
    // A lossless machine converts everything; a stalled one converts nothing.
    assert_close(generator_electrical_power(2.5e6, 1.0), 2.5e6, TOL);
    assert_close(generator_electrical_power(2.5e6, 0.0), 0.0, TOL);
    // Inverse relation.
    assert_close(generator_mechanical_power(950.0e3, 0.95), 1.0e6, 1e-6);
}

#[test]
fn shaft_power_for_three_phase_output() {
    // 400 V, 100 A, pf 0.9 → P = √3·400·100·0.9 ≈ 62.35 kW at the terminals.
    let p_terminal = 3.0_f64.sqrt() * 400.0 * 100.0 * 0.9;
    let shaft = generator_shaft_power_for_output(400.0, 100.0, 0.9, 0.92);
    assert_close(shaft, p_terminal / 0.92, 1e-6);
    // Losses mean the shaft always carries more than the terminals deliver.
    assert!(shaft > p_terminal);
}

#[test]
fn induction_slip_is_zero_at_synchronous_speed() {
    let n_s = synchronous_speed_rpm(50.0, 4);
    assert_close(n_s, 1500.0, TOL);
    assert_eq!(induction_slip(n_s, n_s), 0.0);
    assert_eq!(induction_slip(1500.0, 1500.0), 0.0);
    // At synchronous speed there is no rotor emf and no rotor loss.
    assert_close(
        induction_rotor_frequency(50.0, induction_slip(n_s, n_s)),
        0.0,
        TOL,
    );
    assert_close(induction_rotor_copper_loss(10.0e3, 0.0), 0.0, TOL);
    assert_close(induction_mechanical_power(10.0e3, 0.0), 10.0e3, 1e-9);
}

#[test]
fn induction_machine_motoring_and_generating() {
    let n_s = synchronous_speed_rpm(60.0, 6); // 1200 rpm
    assert_close(n_s, 1200.0, TOL);

    // Motoring at 1164 rpm → s = 3 %.
    let s_motor = induction_slip(1164.0, n_s);
    assert_close(s_motor, 0.03, 1e-12);
    assert_close(induction_rotor_frequency(60.0, s_motor), 1.8, 1e-12);

    // Air-gap power splits into rotor copper loss and shaft power.
    let p_ag = 50.0e3;
    let cu = induction_rotor_copper_loss(p_ag, s_motor);
    let mech = induction_mechanical_power(p_ag, s_motor);
    assert_close(cu, 1.5e3, 1e-9);
    assert_close(cu + mech, p_ag, 1e-9);

    // Driven above synchronous speed the machine generates: slip is negative.
    let s_gen = induction_slip(1236.0, n_s);
    assert_close(s_gen, -0.03, 1e-12);
    assert!(induction_rotor_frequency(60.0, s_gen) < 0.0);
}

// ---------------------------------------------------------------------------
// Transmission line
// ---------------------------------------------------------------------------

#[test]
fn surge_impedance_is_sqrt_of_magnitude_ratio() {
    let z = Complex::new(0.1, 0.5);
    let y = Complex::new(0.0, 3.0e-6);
    let expected = (z.magnitude() / y.magnitude()).sqrt();
    assert_close(surge_impedance(z, y), expected, 1e-12);
    assert_close(
        TransmissionLine::new(z, y).surge_impedance(),
        expected,
        1e-12,
    );

    // Lossless line: |Z_c| = √(L/C), independent of frequency.
    let l = 1.0e-6_f64; // H/m
    let c = 1.1e-11_f64; // F/m
    let sqrt_lc = (l / c).sqrt();
    for f in [50.0, 60.0, 400.0] {
        let line = TransmissionLine::from_line_parameters(0.0, l, c, f);
        assert_close(line.surge_impedance(), sqrt_lc, 1e-9);
        // For a lossless line the characteristic impedance is purely real.
        let z_c = line.characteristic_impedance();
        assert_close(z_c.re, sqrt_lc, 1e-9);
        assert!(z_c.im.abs() < 1e-9);
    }
}

#[test]
fn abc_parameters_of_zero_length_are_the_identity() {
    let line = TransmissionLine::new(Complex::new(1.0e-4, 4.0e-4), Complex::new(0.0, 3.0e-9));
    let (a, b, c, d) = line.abc_parameters(0.0);
    assert_complex_close(a, Complex::new(1.0, 0.0), TOL);
    assert_complex_close(d, Complex::new(1.0, 0.0), TOL);
    assert_complex_close(b, Complex::new(0.0, 0.0), TOL);
    assert_complex_close(c, Complex::new(0.0, 0.0), TOL);

    // An identity two-port passes phasors through unchanged.
    let v_r = Complex::new(132.0e3, 0.0);
    let i_r = Complex::new(300.0, -120.0);
    let (v_s, i_s) = line.sending_end(v_r, i_r, 0.0);
    assert_complex_close(v_s, v_r, 1e-9);
    assert_complex_close(i_s, i_r, 1e-9);
}

#[test]
fn abc_parameters_are_symmetric_and_reciprocal() {
    let line = TransmissionLine::from_line_parameters(5.0e-5, 1.0e-6, 1.1e-11, 50.0);
    let length = 300.0e3; // 300 km long line
    let (a, b, c, d) = line.abc_parameters(length);

    // Symmetric line: A = D. Reciprocal network: A·D − B·C = 1.
    assert_complex_close(a, d, 1e-12);
    let det = a * d - b * c;
    assert_complex_close(det, Complex::new(1.0, 0.0), 1e-9);

    // B is an impedance, C an admittance: their product is dimensionless and
    // both must be non-trivial for a real line.
    assert!(b.magnitude() > 0.0 && c.magnitude() > 0.0);
}

#[test]
fn lossless_line_abc_reduces_to_trigonometric_form() {
    // γ = jβ with β = ω√(LC), Z_c = √(L/C) (real).
    let l = 1.0e-6;
    let c = 1.1e-11;
    let f = 50.0;
    let line = TransmissionLine::from_line_parameters(0.0, l, c, f);
    let length = 400.0e3;

    let omega = 2.0 * std::f64::consts::PI * f;
    let beta = omega * (l * c).sqrt();
    let z_c = (l / c).sqrt();

    let gamma = line.propagation_constant();
    assert!(gamma.re.abs() < 1e-15, "lossless line has zero attenuation");
    assert_close(gamma.im, beta, 1e-15);

    let (a, b, cc, d) = line.abc_parameters(length);
    let bl = beta * length;
    assert_complex_close(a, Complex::new(bl.cos(), 0.0), 1e-9);
    assert_complex_close(b, Complex::new(0.0, z_c * bl.sin()), 1e-6);
    assert_complex_close(cc, Complex::new(0.0, bl.sin() / z_c), 1e-12);
    assert_complex_close(d, a, 1e-12);
}

#[test]
fn cascading_half_sections_reproduces_full_line() {
    let line = TransmissionLine::from_line_parameters(6.0e-5, 1.3e-6, 8.5e-12, 60.0);
    let length = 500.0e3;

    let (a, b, c, d) = line.abc_parameters(length);
    let (a1, b1, c1, d1) = line.abc_parameters(length / 2.0);

    // Two identical half sections in cascade multiply their ABCD matrices.
    let a2 = a1 * a1 + b1 * c1;
    let b2 = a1 * b1 + b1 * d1;
    let c2 = c1 * a1 + d1 * c1;
    let d2 = c1 * b1 + d1 * d1;

    assert_complex_close(a2, a, 1e-9);
    assert_complex_close(b2, b, 1e-6);
    assert_complex_close(c2, c, 1e-12);
    assert_complex_close(d2, d, 1e-9);
}

#[test]
fn short_line_tends_to_the_lumped_series_model() {
    let line = TransmissionLine::from_line_parameters(5.0e-5, 1.0e-6, 1.1e-11, 50.0);
    let length = 20.0e3; // 20 km: shunt charging is negligible

    let (a, b, c, _d) = line.abc_parameters(length);
    let z_total = line.total_series_impedance(length);
    let y_total = line.total_shunt_admittance(length);

    // A ≈ 1 + ZY/2, B ≈ Z, C ≈ Y (errors are O((ZY)²) and O(ZY) respectively).
    assert_complex_close(
        a,
        Complex::new(1.0, 0.0) + complex_scale(z_total * y_total, 0.5),
        1e-7,
    );
    assert_complex_close(b, z_total, 1e-3);
    assert_complex_close(c, y_total, 1e-7);
    assert!((a.magnitude() - 1.0).abs() < 1e-3);
}

#[test]
fn open_circuited_receiving_end_scales_by_a() {
    let line = TransmissionLine::from_line_parameters(5.0e-5, 1.0e-6, 1.1e-11, 50.0);
    let length = 200.0e3;
    let (a, _b, c, _d) = line.abc_parameters(length);

    let v_r = Complex::new(132.0e3 / 3.0_f64.sqrt(), 0.0);
    let (v_s, i_s) = line.sending_end(v_r, Complex::new(0.0, 0.0), length);
    assert_complex_close(v_s, a * v_r, 1e-9);
    assert_complex_close(i_s, c * v_r, 1e-12);

    // Ferranti effect: an open-ended line has |V_r| > |V_s|, i.e. |A| < 1.
    assert!(a.magnitude() < 1.0);
    assert!(v_s.magnitude() < v_r.magnitude());
    // The open-end current is purely charging current, leading the voltage.
    assert!(i_s.im > 0.0);
}

#[test]
fn per_unit_line_parameters_use_the_declared_base() {
    // 0.05 Ω/km + j0.4 Ω/km, 3 nS/km shunt, 150 km, 100 MVA / 230 kV base.
    let line = TransmissionLine::new(Complex::new(5.0e-5, 4.0e-4), Complex::new(0.0, 3.0e-12));
    let base = PerUnitSystem::new(100.0e6, 230.0e3);
    let z_base = base.base_impedance();
    assert_close(z_base, 529.0, 1e-9);

    let length = 150.0e3;
    let z_pu = line.series_impedance_pu(length, base);
    assert_close(z_pu.re, 7.5 / z_base, 1e-12);
    assert_close(z_pu.im, 60.0 / z_base, 1e-12);

    let y_pu = line.shunt_admittance_pu(length, base);
    assert_close(y_pu.im, 4.5e-7 * z_base, 1e-15);

    // Surge impedance is a physical ohmic quantity, unaffected by the base.
    assert!(line.surge_impedance() > 0.0);
}
