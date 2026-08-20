//! Integration tests for `tpt-eng-pcb`.

use tpt_eng_electrical::material_property;
use tpt_eng_pcb::{
    MIL_TO_M,
    footprint::Pad,
    ipc_2221_current_capacity, microstrip_impedance,
    stackup::{Layer, Stackup},
    trace_area_mil2, trace_dc_resistance,
    via::Via,
};

#[test]
fn stackup_total_thickness_is_sum_of_layers() {
    let s = Stackup::new(vec![
        Layer::new("L1 cu", 35e-6, 1.0, 5.8e7),
        Layer::new("core", 1.6e-3, 4.4, 0.0),
        Layer::new("L2 cu", 35e-6, 1.0, 5.8e7),
    ]);
    let expected = 35e-6 + 1.6e-3 + 35e-6;
    assert!((s.total_thickness() - expected).abs() < 1e-15);
}

#[test]
fn stackup_effective_dielectric_constant_is_weighted_harmonic_mean() {
    // Two equal-thickness layers, er = 2.0 and er = 8.0.
    let s = Stackup::new(vec![
        Layer::new("a", 0.5e-3, 2.0, 0.0),
        Layer::new("b", 0.5e-3, 8.0, 0.0),
    ]);
    // (t1+t2) / (t1/er1 + t2/er2) = 1e-3 / (0.5e-3/2 + 0.5e-3/8)
    //   = 1e-3 / (0.25e-3 + 0.0625e-3) = 1 / 0.3125 = 3.2
    assert!((s.effective_dielectric_constant() - 3.2).abs() < 1e-12);

    // A single layer returns its own er.
    let one = Stackup::new(vec![Layer::new("c", 1.0e-3, 4.4, 0.0)]);
    assert!(one.effective_dielectric_constant() == 4.4);
}

#[test]
fn microstrip_fifty_ohm_band() {
    // A ~50 Ω line on 1.6 mm FR-4 (er = 4.4) needs w ≈ 1.8 mm with this
    // closed-form approximation.
    let z0 = microstrip_impedance(1.8e-3, 1.6e-3, 4.4, 35e-6);
    assert!((z0 - 50.0).abs() < 6.0, "z0 = {z0}");
}

#[test]
fn microstrip_narrow_strip_is_higher_impedance() {
    // The task's literal example (w = 0.3 mm on h = 1.6 mm) is a *narrow* strip,
    // which is physically a high-impedance line, not 50 Ω. We assert the
    // correct order of magnitude for that geometry instead.
    let z0 = microstrip_impedance(0.3e-3, 1.6e-3, 4.4, 35e-6);
    assert!(z0 > 100.0 && z0 < 160.0, "z0 = {z0}");

    // Widening the trace must lower the impedance monotonically toward 50 Ω.
    let z_wide = microstrip_impedance(4.0e-3, 1.6e-3, 4.4, 35e-6);
    assert!(z_wide < z0);
}

#[test]
fn ipc_current_capacity_increases_with_area() {
    // Same temperature rise, twice the area -> more current.
    let a1 = trace_area_mil2(0.25e-3, 35e-6); // 0.25 mm × 35 µm
    let a2 = trace_area_mil2(0.50e-3, 35e-6); // 0.50 mm × 35 µm
    assert!(a2 > a1);

    let i1 = ipc_2221_current_capacity(a1, 10.0, true);
    let i2 = ipc_2221_current_capacity(a2, 10.0, true);
    assert!(i2 > i1, "i1={i1}, i2={i2}");
}

#[test]
fn ipc_current_capacity_reference_value() {
    // 200 mil² external trace, 10 °C rise -> a few amps of capacity.
    let i = ipc_2221_current_capacity(200.0, 10.0, true);
    // 0.048 · 10^0.44 · 200^0.725 ≈ 6.16 A.
    assert!((i - 6.16).abs() < 0.2, "i = {i}");

    // Internal layers carry about half the current for the same geometry.
    let i_int = ipc_2221_current_capacity(200.0, 10.0, false);
    assert!(i_int < i && i_int > 0.0);
}

#[test]
fn via_aspect_ratio_is_height_over_drill() {
    let v = Via::new(0.3e-3, 0.6e-3, 1.6e-3);
    assert!((v.aspect_ratio() - (1.6e-3 / 0.3e-3)).abs() < 1e-12);
    assert!((v.annular_ring_m() - 0.15e-3).abs() < 1e-15);
}

#[test]
fn pad_footprint_geometry() {
    let p1 = Pad::new(0.0, 0.0, 1.0e-3, 0.5e-3, 0.0);
    let p2 = Pad::new(0.0, 1.27e-3, 1.0e-3, 0.5e-3, 0.0);
    assert!(p1.is_surface_mount());
    // 1.27 mm pitch between two pads.
    assert!((p1.pitch_to(&p2) - 1.27e-3).abs() < 1e-15);
}

#[test]
fn trace_dc_resistance_uses_electrical_table() {
    // 10 mm of 0.25 mm × 35 µm copper: A = 8.75e-9 m², ρ = 1.68e-8 Ω·m.
    let cu = material_property("copper").unwrap();
    let r = trace_dc_resistance(10e-3, 0.25e-3, 35e-6, cu.resistivity_ohm_m);
    let expected = 1.68e-8 * 10e-3 / (0.25e-3 * 35e-6);
    assert!((r - expected).abs() < 1e-12, "r={r}, expected={expected}");

    // Resistance must scale with length.
    let r2 = trace_dc_resistance(20e-3, 0.25e-3, 35e-6, cu.resistivity_ohm_m);
    assert!(r2 > r);
}

#[test]
fn mil_conversion_roundtrip() {
    // 10 mil = 254 µm.
    assert!((10.0 * MIL_TO_M - 254e-6).abs() < 1e-12);
}
