//! Basic runnable example: the `tpt-eng-props` umbrella re-exports.

use tpt_eng_props::{air, fuels, mixture, water};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::bar, thermodynamic_temperature::kelvin};

fn main() {
    // Water/steam (IAPWS-IF97).
    let t = ThermodynamicTemperature::new::<kelvin>(373.15);
    let psat = water::saturation_pressure(t);
    println!("water p_sat(100 °C)        = {:.3} bar", psat.get::<bar>());

    // Moist air (ASHRAE).
    let pw = Pressure::new::<bar>(0.02);
    let p_atm = Pressure::new::<bar>(1.01325);
    let w = air::humidity_ratio(pw, p_atm).expect("valid partial pressure");
    println!("air humidity ratio        = {:.4} kg/kg dry air", w);

    // Combustion fuels.
    println!(
        "diesel LHV                = {:.1} MJ/kg",
        fuels::Fuel::Diesel.lhv_mj_kg()
    );

    // Process-fluid mixture (Peng–Robinson).
    let ch4 = mixture::Component::from_name("methane").unwrap();
    let mix = mixture::Mixture::pure(ch4);
    let z = mixture::peng_robinson_z(300.0, 5e6, &mix);
    println!("methane Z (300 K, 5 MPa)  = {:.4}", z.vapour().unwrap());

    assert!(psat.get::<bar>() > 0.9 && psat.get::<bar>() < 1.2);
    assert!(w > 0.0 && z.vapour().unwrap() > 0.0);
    println!("props umbrella basic example passed");
}
