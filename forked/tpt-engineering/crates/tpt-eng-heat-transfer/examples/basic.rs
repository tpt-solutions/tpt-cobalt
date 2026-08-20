//! Basic runnable example: heat-transfer correlations and thermal networks.

use tpt_eng_heat_transfer::*;

fn main() {
    // Forced convection over a flat plate (laminar boundary layer).
    let nu = nusselt_flat_plate(1e5, 0.7, FlowRegime::Laminar);
    let h = convection_coefficient(nu, 0.026, 1.0);
    println!(
        "Flat plate (Re=1e5, Pr=0.7): Nu = {:.1}, h = {:.2} W/m²K",
        nu, h
    );

    // External cylinder (Churchill–Bernstein).
    let nu_c = nusselt_cylinder(1e4, 0.7);
    println!("Cylinder (Re=1e4, Pr=0.7): Nu = {:.1}", nu_c);

    // Internal pipe (Dittus–Boelter, heating).
    let nu_p = nusselt_internal_pipe(1e4, 5.0, true);
    println!("Pipe (Re=1e4, Pr=5, heating): Nu = {:.1}", nu_p);

    // 1-D conduction through a plane wall.
    let q = plane_wall_heat_rate(0.5, 1.0, 20.0, 0.1);
    let r_wall = plane_wall_resistance(0.5, 1.0, 0.1);
    println!(
        "\nPlane wall (k=0.5, A=1, ΔT=20, L=0.1): q = {:.0} W, R = {:.3} K/W",
        q, r_wall
    );

    // Cylindrical shell and critical insulation radius.
    let r_shell = cylindrical_shell_resistance(0.04, 1.0, 0.01, 0.02);
    let r_crit = critical_insulation_radius(0.04, 10.0);
    println!("Cylindrical shell R = {:.3} K/W", r_shell);
    println!("Critical insulation radius = {:.1e} m", r_crit);

    // Radiation between parallel grey plates.
    let q_rad = parallel_grey_plates_flux(0.9, 0.9, 373.0, 293.0);
    println!("Parallel grey plates q'' = {:.0} W/m²", q_rad);

    // Series + parallel resistance networks.
    let r_total = series_resistances(&[r_wall, r_shell]);
    let r_par = parallel_resistances(&[0.2, 0.3]).unwrap();
    println!(
        "Series R = {:.3} K/W, parallel R = {:.3} K/W",
        r_total, r_par
    );
    println!(
        "Heat rate across ΔT=10 K, R=0.5 K/W = {:.0} W",
        heat_rate(10.0, &[0.2, 0.3])
    );

    // Integration with the fluid-property crates.
    let p_sat = water_saturation_pressure(373.15);
    let (rho, _h) = water_film_properties(300.0, 3.0e6);
    println!(
        "\nWater p_sat(100 °C) = {:.0} Pa, ρ(300 K, 3 MPa) = {:.1} kg/m³",
        p_sat, rho
    );

    assert!(h > 0.0 && q > 0.0 && q_rad > 0.0);
    println!("heat-transfer basic example passed");
}
