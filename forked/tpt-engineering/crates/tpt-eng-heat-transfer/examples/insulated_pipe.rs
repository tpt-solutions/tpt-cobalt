//! Runnable example: heat loss from an insulated hot-water pipe (combined
//! convection, radial conduction and radiation in series).

use std::f64::consts::PI;
use tpt_eng_heat_transfer::{
    critical_insulation_radius, cylindrical_shell_resistance, heat_rate, nusselt_cylinder,
    parallel_grey_plates_flux, series_resistances,
};

fn main() {
    let l = 1.0; // per metre of pipe
    let r1 = 0.025; // pipe inner radius 25 mm
    let r2 = 0.028; // pipe outer (steel)
    let r3 = 0.048; // insulation outer
    let k_pipe = 45.0; // W/mK steel
    let k_ins = 0.04; // W/mK mineral wool
    let t_fluid = 363.0; // 90 °C
    let t_amb = 293.0; // 20 °C

    // External convection over the cylinder (Churchill–Bernstein, air).
    let nu = nusselt_cylinder(4000.0, 0.7);
    let h_o = nu * 0.026 / (2.0 * r3);
    let r_conv = 1.0 / (h_o * 2.0 * PI * r3 * l);

    // Radial conduction through pipe wall and insulation.
    let r_pipe = cylindrical_shell_resistance(k_pipe, l, r1, r2);
    let r_ins = cylindrical_shell_resistance(k_ins, l, r2, r3);

    // Radiation to ambient (linearised grey-body resistance).
    let q_rad = parallel_grey_plates_flux(0.8, 0.9, t_fluid, t_amb); // W/m²
    let area = 2.0 * PI * r3 * l;
    let r_rad = (t_fluid - t_amb) / (q_rad * area); // K/W

    let r_total = series_resistances(&[r_pipe, r_ins, r_conv, r_rad]);
    let q = heat_rate(t_fluid - t_amb, &[r_pipe, r_ins, r_conv, r_rad]);

    println!("Insulated pipe, per metre, fluid 90 °C / ambient 20 °C");
    println!("  outer convection h      = {:.2} W/m²K", h_o);
    println!("  R pipe wall            = {:.3e} K/W", r_pipe);
    println!("  R insulation           = {:.3e} K/W", r_ins);
    println!("  R convection (outside) = {:.3e} K/W", r_conv);
    println!("  R radiation            = {:.3e} K/W", r_rad);
    println!("  total R                = {:.3e} K/W", r_total);
    println!("  heat loss q'           = {:.1} W/m", q);
    println!(
        "  critical insul. radius = {:.1e} m (r3 = {:.3} m)",
        critical_insulation_radius(k_ins, h_o),
        r3
    );

    assert!(q > 0.0 && r_total > 0.0);
    println!("heat-transfer insulated-pipe example passed");
}
