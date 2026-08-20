// Basic runnable example for `tpt-eng-building-sys`.
//
// Demonstrates the core public API: envelope UA, heating/cooling loads,
// infiltration, plumbing fixture-unit demand (Hunter's curve), and electrical
// panel scheduling.

use tpt_eng_building_sys::{
    Branch, EXTERIOR_FILM_RESISTANCE, Fixture, FixtureType, INTERIOR_FILM_RESISTANCE,
    assembly_u_value, cooling_load, envelope_ua, fixture_unit_demand_gpm, heating_load,
    infiltration_loss_ach_w, panel_utilization, schedule_panel, sum_fixture_units,
};

fn main() {
    println!("=== tpt-eng-building-sys: core API ===");

    // --- Envelope & HVAC loads ---------------------------------------------
    println!("\nEnvelope & HVAC:");
    let ua = envelope_ua(&[(100.0, 0.3), (40.0, 0.5)]);
    println!("  Envelope UA (W/K)           : {ua:.3}");
    let q_heat = heating_load(ua, 20.0, 800.0, 500.0);
    println!("  Heating load (ΔT=20 K)      : {q_heat:.3} W");
    let q_cool = cooling_load(ua, 20.0, 800.0, 1200.0);
    println!("  Cooling load (solar 1200 W) : {q_cool:.3} W");

    let layers = [
        tpt_eng_building_sys::AssemblyLayer::new(0.04, 0.1),
        tpt_eng_building_sys::AssemblyLayer::new(0.04, 0.1),
    ];
    let u = assembly_u_value(&layers, INTERIOR_FILM_RESISTANCE, EXTERIOR_FILM_RESISTANCE);
    println!("  Assembly U-value (wall)     : {u:.3} W/(m²·K)");

    let q_infil = infiltration_loss_ach_w(0.5, 200.0, 1.2, 1006.0, 20.0);
    println!("  Infiltration loss (0.5 ACH) : {q_infil:.3} W");

    // --- Plumbing (Hunter's curve) -----------------------------------------
    println!("\nPlumbing (Hunter's curve):");
    let fu = sum_fixture_units(&[
        (Fixture::WaterClosetTank, 2),
        (Fixture::Lavatory, 3),
        (Fixture::KitchenSink, 1),
    ]);
    println!("  Fixture units (WC×2, lav×3, kit) : {fu:.3} WSFU");
    let d_tank = fixture_unit_demand_gpm(fu, FixtureType::FlushTank).unwrap();
    let d_valve = fixture_unit_demand_gpm(fu, FixtureType::FlushValve).unwrap();
    println!("  Demand (tank)   : {d_tank:.3} gpm");
    println!("  Demand (valve)  : {d_valve:.3} gpm");

    // --- Electrical panel scheduling ---------------------------------------
    println!("\nElectrical panel:");
    let branches = [
        Branch::new("lights", 1200.0),
        Branch::new("receptacles", 2400.0),
        Branch::new("hvac", 3000.0),
    ];
    let total = schedule_panel(&branches, 10000.0);
    let util = panel_utilization(total, 10000.0).unwrap();
    println!("  Connected load   : {total:.3} W");
    println!("  Panel utilization: {util:.3} ({:.1}%)", util * 100.0);
}
