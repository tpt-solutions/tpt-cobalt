//! Runnable example: boiler fuel requirement to raise feedwater enthalpy,
//! combining the water/steam and fuels umbrellas.

use tpt_eng_props::{fuels, water};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::megapascal, thermodynamic_temperature::kelvin};

fn main() {
    let p = Pressure::new::<megapascal>(2.0);
    let t_in = ThermodynamicTemperature::new::<kelvin>(320.0);
    let t_out = ThermodynamicTemperature::new::<kelvin>(520.0);

    let h_in = water::state(t_in, p).expect("feedwater inlet").enthalpy;
    let h_out = water::state(t_out, p).expect("feedwater outlet").enthalpy;

    let m_dot = 20.0; // kg/s
    let q_steam = m_dot * (h_out - h_in); // kW
    let eta = 0.88; // boiler efficiency
    let fuel = fuels::Fuel::NaturalGas;
    let m_fuel = (q_steam * 1.0e3 / eta) / (fuel.lhv_mj_kg() * 1.0e6); // kg/s

    println!("Boiler: {:.0} kg/s water @ 2 MPa, 320→520 K", m_dot);
    println!("  heat to steam Q  = {:.1} MW", q_steam / 1000.0);
    println!(
        "  natural-gas fuel = {:.2} kg/s ({:.0} kg/h)",
        m_fuel,
        m_fuel * 3600.0
    );

    assert!(q_steam > 0.0 && m_fuel > 0.0);
    println!("props umbrella cogeneration example passed");
}
