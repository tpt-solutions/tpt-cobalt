//! Basic runnable example: IAPWS-IF97 water/steam states and saturation.

use tpt_eng_props_water::{Region, saturation_pressure, saturation_temperature, state};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    mass_density::kilogram_per_cubic_meter,
    pressure::{kilopascal, megapascal},
    specific_heat_capacity::kilojoule_per_kilogram_kelvin,
    specific_volume::cubic_meter_per_kilogram,
    thermodynamic_temperature::kelvin,
};

fn main() {
    // --- Saturation relations (Region 4). ---
    let t = ThermodynamicTemperature::new::<kelvin>(373.15); // 100 °C
    let p_sat = saturation_pressure(t);
    println!("p_sat(100 °C) = {:.3} kPa", p_sat.get::<kilopascal>());

    // Round-trip: recover the temperature from the saturation pressure.
    let t_back = saturation_temperature(p_sat);
    println!(
        "T_sat(p_sat)  = {:.2} °C (round-trip)",
        t_back.get::<kelvin>() - 273.15
    );

    // --- Compressed-liquid state (Region 1). ---
    let s_liq = state(
        ThermodynamicTemperature::new::<kelvin>(300.0),
        Pressure::new::<megapascal>(3.0),
    )
    .expect("liquid state in Region 1");
    println!("\nLiquid water @ 300 K, 3 MPa (Region {:?}):", s_liq.region);
    println!(
        "  density         = {:.1} kg/m³",
        s_liq.density.get::<kilogram_per_cubic_meter>()
    );
    println!(
        "  specific volume = {:.6} m³/kg",
        s_liq.specific_volume.get::<cubic_meter_per_kilogram>()
    );
    println!("  enthalpy        = {:.3} kJ/kg", s_liq.enthalpy);
    println!(
        "  entropy         = {:.5} kJ/(kg·K)",
        s_liq.entropy.get::<kilojoule_per_kilogram_kelvin>()
    );
    println!(
        "  cp              = {:.4} kJ/(kg·K)",
        s_liq
            .isobaric_heat_capacity
            .get::<kilojoule_per_kilogram_kelvin>()
    );

    // --- Superheated-vapour state (Region 2). ---
    let s_vap = state(
        ThermodynamicTemperature::new::<kelvin>(300.0),
        Pressure::new::<megapascal>(0.0035),
    )
    .expect("vapour state in Region 2");
    assert_eq!(s_vap.region, Region::Two);
    println!(
        "\nSuperheated steam @ 300 K, 3.5 kPa (Region {:?}):",
        s_vap.region
    );
    println!(
        "  density         = {:.4} kg/m³",
        s_vap.density.get::<kilogram_per_cubic_meter>()
    );
    println!("  enthalpy        = {:.2} kJ/kg", s_vap.enthalpy);

    assert!(s_liq.density.get::<kilogram_per_cubic_meter>() > 900.0);
    assert!(s_vap.density.get::<kilogram_per_cubic_meter>() < 0.05);
    println!("\nprops-water basic example passed");
}
