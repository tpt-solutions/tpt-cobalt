//! Integration tests for `tpt-eng-thermal-mgmt`.

use tpt_eng_thermal_mgmt::{
    FanCurve, HeatSink, ThermalPath, fin_efficiency, fin_parameter, junction_temperature,
    junction_to_ambient,
};

#[test]
fn fin_efficiency_unity_at_zero_argument() {
    // As m·L → 0 the fin is isothermal with the base, so η → 1.
    assert!((fin_efficiency(0.0, 1.0) - 1.0).abs() < 1e-12);
    assert!((fin_efficiency(1.0, 0.0) - 1.0).abs() < 1e-12);
    assert!((fin_efficiency(1e-9, 1e-9) - 1.0).abs() < 1e-6);
}

#[test]
fn fin_efficiency_below_unity() {
    // For any finite m·L > 0 the efficiency is strictly less than 1.
    let cases = [(0.5, 0.05), (1.0, 0.03), (2.0, 0.1), (5.0, 0.02)];
    for (m, l) in cases {
        let eta = fin_efficiency(m, l);
        assert!(eta > 0.0 && eta < 1.0, "η={eta} for m={m}, L={l}");
    }
    // Monotonic: a taller / more poorly conducting fin is less efficient.
    let short = fin_efficiency(2.0, 0.01);
    let tall = fin_efficiency(2.0, 0.1);
    assert!(tall < short, "taller fin should be less efficient");
}

#[test]
fn fin_parameter_formula() {
    // m = sqrt(2·h / (k·t)).
    let m = fin_parameter(50.0, 200.0, 0.003);
    let expected = (2.0 * 50.0 / (200.0 * 0.003f64)).sqrt();
    assert!((m - expected).abs() < 1e-12);
}

#[test]
fn sink_resistance_positive_and_decreases_with_fins() {
    let sink = HeatSink {
        base_area: 0.01,
        fin_count: 10,
        fin_length: 0.04,
        fin_thickness: 0.002,
        fin_height: 0.03,
    };
    let h = 25.0;
    let k = 200.0;

    let r = sink.thermal_resistance(h, k);
    assert!(
        r.is_finite() && r > 0.0,
        "sink resistance must be positive, got {r}"
    );

    // Doubling the fin count (with the same base) must lower the resistance.
    let mut more_fins = sink;
    more_fins.fin_count = 20;
    let r_more = more_fins.thermal_resistance(h, k);
    assert!(r_more < r, "more fins should reduce θ_sa: {r_more} < {r}");
}

#[test]
fn sink_resistance_decreases_with_better_convection() {
    let sink = HeatSink {
        base_area: 0.01,
        fin_count: 12,
        fin_length: 0.04,
        fin_thickness: 0.002,
        fin_height: 0.03,
    };
    let r_low = sink.thermal_resistance(10.0, 200.0);
    let r_high = sink.thermal_resistance(50.0, 200.0);
    assert!(r_high < r_low, "stronger convection lowers θ_sa");
}

#[test]
fn fan_operating_point_solves_quadratic() {
    // a=100, b=1, R=1  →  q = sqrt(100 / 2) = sqrt(50).
    let curve = FanCurve { a: 100.0, b: 1.0 };
    let q = curve.operating_point(1.0).expect("should have a solution");
    assert!((q - 50.0_f64.sqrt()).abs() < 1e-9);

    // Verify the intersection exactly balances fan and system pressure.
    let dp_fan = curve.pressure(q);
    let dp_sys = 1.0 * q * q;
    assert!((dp_fan - dp_sys).abs() < 1e-9);
}

#[test]
fn fan_operating_point_no_solution() {
    // No shut-off pressure → no positive-flow intersection.
    let curve = FanCurve { a: 0.0, b: 1.0 };
    assert!(curve.operating_point(1.0).is_none());

    // Fan curve always above system curve → never meets.
    let curve = FanCurve { a: 100.0, b: -1.0 };
    assert!(curve.operating_point(1.0).is_none());
}

#[test]
fn thermal_path_series_and_parallel() {
    // Series of 0.2 and 0.3 K/W → 0.5 K/W.
    let series = ThermalPath::Series(vec![
        ThermalPath::Resistance(0.2),
        ThermalPath::Resistance(0.3),
    ]);
    assert!((series.total_resistance().unwrap() - 0.5).abs() < 1e-12);

    // Parallel of 0.2 and 0.3 K/W → 0.12 K/W.
    let parallel = ThermalPath::Parallel(vec![
        ThermalPath::Resistance(0.2),
        ThermalPath::Resistance(0.3),
    ]);
    assert!((parallel.total_resistance().unwrap() - 0.12).abs() < 1e-12);

    // Empty parallel is undefined.
    assert!(ThermalPath::Parallel(vec![]).total_resistance().is_none());

    // Nested: parallel branch of two series branches.
    let nested = ThermalPath::Parallel(vec![
        ThermalPath::Series(vec![
            ThermalPath::Resistance(0.2),
            ThermalPath::Resistance(0.3),
        ]),
        ThermalPath::Resistance(0.12),
    ]);
    // 0.5 || 0.12 = 1/(2 + 8.333...) = 0.0969...
    let r = nested.total_resistance().unwrap();
    assert!((r - 0.096774).abs() < 1e-4);
}

#[test]
fn junction_to_ambient_and_temperature() {
    // θ_ja is the series sum of the three stage resistances.
    let path = junction_to_ambient(0.5, 0.3, 1.2);
    assert!((path.total_resistance().unwrap() - 2.0).abs() < 1e-12);

    // Junction temperature rises linearly with power.
    let t_amb = 25.0;
    let t_p10 = junction_temperature(10.0, 2.0, t_amb);
    let t_p20 = junction_temperature(20.0, 2.0, t_amb);
    assert!((t_p10 - 45.0).abs() < 1e-12);
    assert!((t_p20 - 65.0).abs() < 1e-12);
    assert!(t_p20 > t_p10, "higher power → higher junction temperature");
}
