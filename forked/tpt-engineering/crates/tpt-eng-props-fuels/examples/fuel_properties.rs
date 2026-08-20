//! Runnable example: combustion-fuel heating values, density, and stoichiometry.

use tpt_eng_props_fuels::Fuel;
use tpt_math_units::uom::si::mass_density::kilogram_per_cubic_meter;

fn main() {
    let fuels = [
        Fuel::Methane,
        Fuel::Hydrogen,
        Fuel::NaturalGas,
        Fuel::Diesel,
    ];
    for f in fuels {
        println!(
            "LHV = {:.1} MJ/kg, density = {:.1} kg/m³, stoichiometric AFR = {:.1}",
            f.lhv_mj_kg(),
            f.density().get::<kilogram_per_cubic_meter>(),
            f.stoichiometric_air_fuel_ratio(),
        );
    }
    // Hydrogen has the highest heating value per kg; diesel the highest density.
    assert!(Fuel::Hydrogen.lhv_mj_kg() > Fuel::Methane.lhv_mj_kg());
    assert!(
        Fuel::Diesel.density().get::<kilogram_per_cubic_meter>()
            > Fuel::Hydrogen.density().get::<kilogram_per_cubic_meter>()
    );
    println!("fuel properties example passed");
}
