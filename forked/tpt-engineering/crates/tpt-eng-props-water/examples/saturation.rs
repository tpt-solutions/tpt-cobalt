//! Runnable example: IAPWS-IF97 water/steam saturation relations.

use tpt_eng_props_water::{saturation_pressure, saturation_temperature};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::bar, thermodynamic_temperature::kelvin};

fn main() {
    let t = ThermodynamicTemperature::new::<kelvin>(373.15); // 100 °C
    let psat = saturation_pressure(t);
    println!("saturation pressure @100 °C = {:.3} bar", psat.get::<bar>());

    // Round-trip: temperature from the saturation pressure we just computed.
    let t_back = saturation_temperature(psat);
    println!(
        "saturation temperature @1.013 bar = {:.2} °C",
        t_back.get::<kelvin>() - 273.15
    );

    assert!((psat.get::<bar>() - 1.01325).abs() < 0.05);
    println!("saturation example passed");
}
