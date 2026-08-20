//! Runnable example: fluid-property lookups across the `tpt-eng-props`
//! umbrella (water/steam, moist air, combustion fuels).

use tpt_eng_props::{air, fuels, water};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::bar, thermodynamic_temperature::kelvin};

fn main() {
    // IAPWS-IF97 saturation pressure of water at 100 °C (~1.013 bar).
    let t = ThermodynamicTemperature::new::<kelvin>(373.15);
    let psat = water::saturation_pressure(t);
    println!(
        "water saturation pressure @100 °C = {:.3} bar",
        psat.get::<bar>()
    );

    // ASHRAE humidity ratio for 2 kPa water-vapour partial pressure in
    // atmospheric air.
    let pw = Pressure::new::<bar>(0.02);
    let p_atm = Pressure::new::<bar>(1.01325);
    let w = air::humidity_ratio(pw, p_atm).expect("valid partial pressure");
    println!("humidity ratio = {:.4} kg water / kg dry air", w);

    // Diesel lower heating value.
    let lhv = fuels::Fuel::Diesel.lhv_mj_kg();
    println!("diesel LHV = {:.1} MJ/kg", lhv);

    assert!(psat.get::<bar>() > 0.9 && psat.get::<bar>() < 1.2);
    assert!(w > 0.0);
    assert!(lhv > 40.0);
    println!("property lookup passed");
}
