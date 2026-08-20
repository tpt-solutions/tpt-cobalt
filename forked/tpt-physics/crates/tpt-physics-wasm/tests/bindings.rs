//! Integration-style usage examples for the WebAssembly bindings, exercised
//! from Rust (the browser-facing `Float32Array` accessors are exercised by the
//! `www/` playground; these tests drive the same JSON constructors and steppers
//! on the host to prove the binding surface works end to end).
//!
//! These mirror what the WebGL frontend does: build a scene from JSON, step it,
//! and read scalar/health state back out.

use tpt_physics_wasm::{CfdSimulation, DemSimulation, ElectroThermalSimulation, SphSimulation};

#[test]
fn dem_scene_builds_steps_and_stays_finite() {
    let json = r#"{
        "dt": 1e-4,
        "gravity": [0, -9.81, 0],
        "particles": [
            {"position": [0.2, 1.0, 0.3], "velocity": [0, 0, 0], "radius": 0.1, "density": 1000},
            {"position": [0.8, 1.5, 0.9], "velocity": [0, 0, 0], "radius": 0.1, "density": 1000},
            {"position": [0.5, 2.0, 0.6], "velocity": [0, 0, 0], "radius": 0.1, "density": 1000}
        ],
        "obstacles": []
    }"#;
    let mut sim = DemSimulation::new(json).expect("valid DEM scene");
    assert_eq!(sim.count(), 3);
    for _ in 0..200 {
        sim.step();
    }
    assert!(sim.kinetic_energy().is_finite(), "DEM diverged");
    assert!(
        sim.kinetic_energy() > 0.0,
        "particles should have accelerated"
    );
    println!(
        "DEM: {} particles, KE = {:.4e} J",
        sim.count(),
        sim.kinetic_energy()
    );
}

#[test]
fn cfd_scene_builds_and_steps() {
    let json = r#"{
        "nx": 48, "ny": 24, "tau": 0.6,
        "x_boundary": "periodic", "walls": "none",
        "force": [1e-5, 0], "rho0": 1.0, "u0": [0, 0]
    }"#;
    let mut sim = CfdSimulation::new(json).expect("valid CFD scene");
    assert_eq!(sim.nx(), 48);
    assert_eq!(sim.ny(), 24);
    for _ in 0..200 {
        sim.step();
    }
    println!("CFD: {} x {} lattice stepped", sim.nx(), sim.ny());
}

#[test]
fn sph_scene_collapses_and_stays_bounded() {
    let json = r#"{
        "h": 0.04, "rho0": 1000.0, "c": 20.0, "gamma": 1.0, "mu": 0.5,
        "gravity": [0, -9.81], "domain": [1.0, 1.0], "dt": 1e-4,
        "block": {"nx": 15, "ny": 30, "spacing": 0.03, "origin": [0.02, 0.02]}
    }"#;
    let mut sim = SphSimulation::new(json).expect("valid SPH scene");
    assert_eq!(sim.count(), 450);
    for _ in 0..1000 {
        sim.step();
    }
    assert!(sim.kinetic_energy().is_finite(), "SPH diverged");
    println!(
        "SPH: {} particles, h = {:.3}, KE = {:.4e} J",
        sim.count(),
        sim.smoothing_length(),
        sim.kinetic_energy()
    );
}

#[test]
fn electro_thermal_scene_heats_up() {
    let json = r#"{
        "n": 21, "t_init": 300.0, "dx": 0.01, "voltage": 10.0, "convection": 50.0
    }"#;
    let mut sim = ElectroThermalSimulation::new(json).expect("valid ET scene");
    let t0 = sim.max_temperature();
    for _ in 0..2000 {
        sim.step(1e-4);
    }
    let t1 = sim.max_temperature();
    assert!(t1 > t0, "rod should heat under voltage: {t0} -> {t1}");
    println!("Electro-thermal: hotspot {t0:.1} K -> {t1:.1} K");
}
