// Richer example: whole-building HVAC load and service-water sizing.
//
// We assemble a multi-surface envelope, compute design heating and cooling
// loads including infiltration and internal/solar gains, estimate annual
// heating energy, then size the domestic-water service from a fixture schedule
// and the required pipe diameter.

use tpt_eng_building_sys::{
    AssemblyLayer, Branch, EXTERIOR_FILM_RESISTANCE, Fixture, FixtureType,
    INTERIOR_FILM_RESISTANCE, annual_heating_energy_kwh, cooling_load, envelope_ua,
    fixture_unit_demand_lps, gpm_to_lps, heating_load, infiltration_loss_w,
    required_pipe_diameter_m, schedule_panel, sum_fixture_units, transmission_heat_rate,
};

fn main() {
    println!("=== Whole-building HVAC & domestic-water sizing ===");

    // --- Envelope from individual surfaces ---------------------------------
    let walls = [
        (120.0, 0.28), // facade: 120 m² @ 0.28 W/m²K
        (60.0, 0.22),  // roof:    60 m² @ 0.22 W/m²K
        (40.0, 0.35),  // floor:   40 m² @ 0.35 W/m²K
        (15.0, 2.80),  // glazing: 15 m² @ 2.80 W/m²K
    ];
    let ua = envelope_ua(&walls);
    println!("\nEnvelope conductance UA = {ua:.3} W/K");

    // Detailed U-value of one opaque wall (brick + insulation + block).
    let wall_layers = [
        AssemblyLayer::new(0.84, 0.10),  // brick
        AssemblyLayer::new(0.040, 0.08), // insulation
        AssemblyLayer::new(0.20, 0.15),  // concrete block
    ];
    let u_wall = tpt_eng_building_sys::assembly_u_value(
        &wall_layers,
        INTERIOR_FILM_RESISTANCE,
        EXTERIOR_FILM_RESISTANCE,
    );
    println!("  Detailed wall U-value = {u_wall:.3} W/(m²·K)");

    // --- Design loads ------------------------------------------------------
    let design_dt = 22.0; // indoor 21 °C − outdoor −1 °C
    let infil_rate = 0.5; // ACH
    let volume = 480.0; // m³
    let infil = infiltration_loss_w(1.2, infil_rate * volume / 3600.0, 1006.0, design_dt);
    let internal = 1500.0; // W internal gains (lights/people)
    let q_heat = heating_load(ua, design_dt, infil, internal);
    let solar = 2500.0; // W solar gain through glazing
    let q_cool = cooling_load(ua, design_dt, infil, solar);
    println!("\nDesign heating load = {q_heat:.3} W");
    println!("Design cooling load = {q_cool:.3} W");
    println!(
        "Heat through facade alone = {:.3} W",
        transmission_heat_rate(120.0 * 0.28, design_dt)
    );

    let annual_e = annual_heating_energy_kwh(q_heat, 4000.0); // 4000 K·h season
    println!("Estimated annual heating energy = {annual_e:.3} kWh");

    // --- Domestic water service (Hunter's curve) ---------------------------
    let fixtures = [
        (Fixture::WaterClosetTank, 4),
        (Fixture::WaterClosetValve, 2),
        (Fixture::Lavatory, 8),
        (Fixture::KitchenSink, 2),
        (Fixture::BathtubShower, 4),
        (Fixture::WashingMachine, 2),
        (Fixture::Dishwasher, 1),
        (Fixture::HoseBibb, 1),
    ];
    let wsfu = sum_fixture_units(&fixtures);
    println!("\nTotal water-supply fixture units = {wsfu:.3} WSFU");
    let lps = fixture_unit_demand_lps(wsfu, FixtureType::FlushValve).unwrap();
    println!("Expected demand (valve) = {lps:.3} L/s");
    let diam = required_pipe_diameter_m(lps, 1.5);
    println!("Min service-pipe diameter @ 1.5 m/s = {diam:.3} m");

    // --- Panel scheduling for the same building ----------------------------
    let branches = [
        Branch::new("lighting", 1500.0),
        Branch::new("receptacles", 3500.0),
        Branch::new("hvac", q_heat),
        Branch::new("water-heater", 4500.0),
    ];
    let total = schedule_panel(&branches, 20000.0);
    let util = tpt_eng_building_sys::panel_utilization(total, 20000.0).unwrap();
    println!(
        "\nPanel connected load = {total:.3} W, utilization = {util:.3} ({:.1}%)",
        util * 100.0
    );

    let _ = gpm_to_lps; // keep helper referenced for completeness
}
