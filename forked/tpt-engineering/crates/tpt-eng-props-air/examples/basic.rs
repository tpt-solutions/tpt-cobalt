//! Basic runnable example: ASHRAE moist-air psychrometric state.

use tpt_eng_props_air::{
    dew_point, humidity_ratio, moist_air_enthalpy, relative_humidity, saturation_pressure_water,
    vapour_pressure_from_ratio,
};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    pressure::{kilopascal, pascal},
    thermodynamic_temperature::kelvin,
};

fn main() {
    let t = ThermodynamicTemperature::new::<kelvin>(298.15); // 25 °C
    let p = Pressure::new::<kilopascal>(101.325);
    let pw = Pressure::new::<kilopascal>(2.0); // water-vapour partial pressure

    let psat = saturation_pressure_water(t);
    let w = humidity_ratio(pw, p).expect("valid partial pressure");
    let rh = relative_humidity(pw, t).expect("valid state");
    let dp = dew_point(pw).expect("valid partial pressure");
    let h = moist_air_enthalpy(t, w);

    println!("Moist air @ 25 °C, 101.325 kPa, p_w = 2.0 kPa:");
    println!(
        "  saturation pressure = {:.3} kPa",
        psat.get::<kilopascal>()
    );
    println!("  humidity ratio      = {:.4} kg water / kg dry air", w);
    println!("  relative humidity   = {:.1} %", rh * 100.0);
    println!(
        "  dew point           = {:.2} °C",
        dp.get::<kelvin>() - 273.15
    );
    println!("  moist-air enthalpy  = {:.1} kJ/kg dry air", h);

    // Round-trip: recover the partial pressure from the humidity ratio.
    let pw2 = vapour_pressure_from_ratio(w, p).expect("valid ratio");
    println!(
        "  p_w recovered       = {:.4} kPa (round-trip)",
        pw2.get::<kilopascal>()
    );

    assert!((pw2.get::<pascal>() - pw.get::<pascal>()).abs() < 1e-6);
    assert!(rh > 0.0 && rh <= 1.0);
    println!("props-air basic example passed");
}
