//! Runnable example: cooling-coil load and condensate for dehumidifying
//! ventilation air (ASHRAE moist-air relations).

use tpt_eng_props_air::{humidity_ratio, moist_air_enthalpy, saturation_pressure_water};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::kilopascal, thermodynamic_temperature::degree_celsius};

fn main() {
    let p_atm = Pressure::new::<kilopascal>(101.325);

    // Inlet: hot, humid outdoor air at 35 °C, 50 % RH.
    let t1 = ThermodynamicTemperature::new::<degree_celsius>(35.0);
    let rh1 = 0.50;
    let psat1 = saturation_pressure_water(t1);
    let pw1 = Pressure::new::<kilopascal>(rh1 * psat1.get::<kilopascal>());
    let w1 = humidity_ratio(pw1, p_atm).expect("inlet partial pressure");
    let h1 = moist_air_enthalpy(t1, w1);

    // Outlet: coil leaves air saturated at 12 °C (dehumidifying cooling).
    let t2 = ThermodynamicTemperature::new::<degree_celsius>(12.0);
    let psat2 = saturation_pressure_water(t2);
    let w2 = humidity_ratio(psat2, p_atm).expect("outlet saturation pressure");
    let h2 = moist_air_enthalpy(t2, w2);

    let m_da = 1.0; // kg dry air / s
    let q_cool = m_da * (h1 - h2); // kW (h in kJ/kg)
    let condensate = m_da * (w1 - w2); // kg water / s

    println!("Cooling coil, m_da = {:.1} kg dry air/s", m_da);
    println!(
        "  inlet  : T = 35 °C, RH = 50 % -> w = {:.4}, h = {:.1} kJ/kg",
        w1, h1
    );
    println!(
        "  outlet : T = 12 °C, sat.      -> w = {:.4}, h = {:.1} kJ/kg",
        w2, h2
    );
    println!("  cooling load       = {:.1} kW", q_cool);
    println!(
        "  condensate removal = {:.4} kg/s ({:.2} L/min)",
        condensate,
        condensate * 60.0
    );

    assert!(q_cool > 0.0 && condensate > 0.0);
    println!("props-air cooling-coil example passed");
}
