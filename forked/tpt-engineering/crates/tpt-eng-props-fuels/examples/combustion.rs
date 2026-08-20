//! Runnable example: fuel and combustion-air requirement for a furnace duty,
//! with a CO₂ emission estimate.

use tpt_eng_props_fuels::Fuel;

fn main() {
    let heat_demand = 5.0e6; // W = 5 MW thermal
    let fuel = Fuel::NaturalGas;
    let eta = 0.85; // burner thermal efficiency

    let fuel_power = heat_demand / eta; // W of fuel input
    let lhv_j_kg = fuel.lhv_mj_kg() * 1.0e6; // J/kg
    let m_fuel = fuel_power / lhv_j_kg; // kg/s
    let afr = fuel.stoichiometric_air_fuel_ratio();
    let m_air = m_fuel * afr; // kg/s

    // co2_kg_per_mj is per MJ of LHV; fuel_power is W -> MJ/s = W/1e6.
    let co2_rate = (fuel_power / 1.0e6) * fuel.co2_kg_per_mj(); // kg/s

    println!(
        "Furnace duty = {:.1} MW, fuel = {fuel:?}, η = {} %",
        heat_demand / 1e6,
        eta * 100.0
    );
    println!("  fuel input        = {:.1} MW", fuel_power / 1e6);
    println!(
        "  fuel flow         = {:.3} kg/s ({:.0} kg/h)",
        m_fuel,
        m_fuel * 3600.0
    );
    println!(
        "  stoichiometric air = {:.1} kg/s ({:.0} kg/h)",
        m_air,
        m_air * 3600.0
    );
    println!("  CO₂ emission      = {:.1} kg/h", co2_rate * 3600.0);

    assert!(m_fuel > 0.0 && m_air > m_fuel);
    println!("props-fuels combustion example passed");
}
