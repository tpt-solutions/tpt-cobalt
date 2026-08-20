//! Runnable example: ASHRAE moist-air psychrometrics.

use tpt_eng_props_air::{dew_point, humidity_ratio, relative_humidity};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::kilopascal, thermodynamic_temperature::kelvin};

fn main() {
    let t = ThermodynamicTemperature::new::<kelvin>(298.15); // 25 °C
    let p = Pressure::new::<kilopascal>(101.325);
    let pw = Pressure::new::<kilopascal>(2.0); // water-vapour partial pressure

    let w = humidity_ratio(pw, p).expect("valid partial pressure");
    let rh = relative_humidity(pw, t).expect("valid state");
    let dp = dew_point(pw).expect("valid partial pressure");

    println!("humidity ratio       = {:.4} kg water / kg dry air", w);
    println!("relative humidity     = {:.1} %", rh * 100.0);
    println!(
        "dew point             = {:.1} °C",
        dp.get::<kelvin>() - 273.15
    );

    assert!(w > 0.0 && rh > 0.0 && rh <= 1.0);
    println!("psychrometrics example passed");
}
