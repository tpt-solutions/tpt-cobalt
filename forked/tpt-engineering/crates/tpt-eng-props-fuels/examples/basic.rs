//! Basic runnable example: combustion-fuel heating values, density,
//! stoichiometry and CO₂ factors.

use tpt_eng_props_fuels::{BlendedFuel, Fuel};
use tpt_math_units::uom::si::mass_density::kilogram_per_cubic_meter;

fn main() {
    for f in [
        Fuel::Methane,
        Fuel::Hydrogen,
        Fuel::NaturalGas,
        Fuel::Diesel,
    ] {
        println!(
            "{f:<10?} LHV = {:>6.1} MJ/kg  HHV = {:>6.1} MJ/kg  ρ = {:>6.1} kg/m³  AFR = {:>5.1}  CO₂ = {:>5.3} kg/MJ",
            f.lhv_mj_kg(),
            f.hhv_mj_kg(),
            f.density().get::<kilogram_per_cubic_meter>(),
            f.stoichiometric_air_fuel_ratio(),
            f.co2_kg_per_mj(),
        );
    }

    // Hydrogen-blended natural gas: 30 % H₂ by volume/mole.
    let blend = BlendedFuel::new(0.30);
    println!(
        "\n30% H₂ blend: LHV = {:.1} MJ/kg, AFR = {:.1}, CO₂ = {:.3} kg/MJ",
        blend.lhv,
        blend.stoichiometric_air_fuel_ratio(),
        blend.co2_kg_per_mj(),
    );

    assert!(Fuel::Hydrogen.lhv_mj_kg() > Fuel::Methane.lhv_mj_kg());
    assert!(blend.co2_kg_per_mj() < Fuel::Methane.co2_kg_per_mj());
    println!("props-fuels basic example passed");
}
