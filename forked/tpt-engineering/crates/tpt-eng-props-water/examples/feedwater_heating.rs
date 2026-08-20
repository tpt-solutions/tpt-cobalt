//! Runnable example: constant-pressure feedwater heating duty from IAPWS-IF97
//! state enthalpies (a small boiler / HRSG sub-model).

use tpt_eng_props_water::state;
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::megapascal, thermodynamic_temperature::kelvin};

fn main() {
    // Heat 10 kg/s of water from 25 °C to 150 °C at 1 MPa (feed-pump discharge).
    let p = Pressure::new::<megapascal>(1.0);
    let t1 = ThermodynamicTemperature::new::<kelvin>(298.15);
    let t2 = ThermodynamicTemperature::new::<kelvin>(423.15);

    let s1 = state(t1, p).expect("feedwater inlet state");
    let s2 = state(t2, p).expect("feedwater outlet state");

    let m_dot = 10.0; // kg/s
    let dh = s2.enthalpy - s1.enthalpy; // kJ/kg
    let q = m_dot * dh; // kW (h is kJ/kg, m_dot in kg/s)

    println!("Feedwater heating @ 1 MPa, m_dot = {:.0} kg/s", m_dot);
    println!("  h_in            = {:.2} kJ/kg", s1.enthalpy);
    println!("  h_out           = {:.2} kJ/kg", s2.enthalpy);
    println!("  Δh              = {:.2} kJ/kg", dh);
    println!("  heating duty Q  = {:.1} kW ({:.2} MW)", q, q / 1000.0);

    assert!(q > 0.0);
    println!("props-water feedwater heating example passed");
}
