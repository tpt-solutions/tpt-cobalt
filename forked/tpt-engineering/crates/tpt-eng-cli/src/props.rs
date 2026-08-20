//! Fluid/fuel property lookups: water/steam (IAPWS-IF97), moist air
//! (ASHRAE psychrometrics), and fuels.

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use tpt_eng_props_air::{self, Error as AirError};
use tpt_eng_props_fuels::Fuel;
use tpt_eng_props_water::{self, Error as WaterError};
use tpt_math_units::uom::si::f64::{Pressure, ThermodynamicTemperature};
use tpt_math_units::uom::si::{
    mass_density::kilogram_per_cubic_meter,
    pressure::pascal,
    thermodynamic_temperature::{degree_celsius, kelvin},
};

#[derive(Args)]
pub struct PropsArgs {
    #[command(subcommand)]
    pub cmd: PropsCmd,
}

#[derive(Subcommand)]
pub enum PropsCmd {
    /// Water/steam (IAPWS-IF97) property lookup at (T, P).
    Water {
        /// Temperature.
        temp: f64,
        /// Pressure.
        pressure: f64,
        /// Temperature unit: k | c.
        #[arg(long, default_value = "k")]
        temp_unit: String,
        /// Pressure unit: pa | kpa | mpa | bar | psi.
        #[arg(long, default_value = "mpa")]
        pressure_unit: String,
    },
    /// Moist-air (ASHRAE) lookup from dry-bulb T, relative humidity and total P.
    Air {
        /// Dry-bulb temperature.
        temp: f64,
        /// Relative humidity (percent, 0-100).
        rh: f64,
        /// Total pressure.
        #[arg(long, default_value = "101.325")]
        pressure: f64,
        /// Temperature unit: k | c.
        #[arg(long, default_value = "c")]
        temp_unit: String,
        /// Pressure unit: pa | kpa | mpa | bar | psi.
        #[arg(long, default_value = "kpa")]
        pressure_unit: String,
    },
    /// Fuel property lookup (heating values, density, AFR, CO2).
    Fuel {
        /// Fuel name: methane | hydrogen | natural-gas | diesel | blend.
        name: String,
        /// Hydrogen mole/volume fraction (0-1) for `blend`.
        #[arg(long)]
        h2: Option<f64>,
    },
}

fn to_kelvin(value: f64, unit: &str) -> f64 {
    match unit.trim().to_ascii_lowercase().as_str() {
        "c" | "degc" | "celsius" => value + 273.15,
        "k" | "kelvin" => value,
        _ => value,
    }
}

fn parse_pressure(value: f64, unit: &str) -> Result<f64> {
    let factor = match unit.trim().to_ascii_lowercase().as_str() {
        "pa" => 1.0,
        "kpa" => 1e3,
        "mpa" => 1e6,
        "bar" => 1e5,
        "psi" => 6894.757293168,
        other => bail!("unknown pressure unit: {other}"),
    };
    Ok(value * factor)
}

pub fn run(args: PropsArgs) -> Result<()> {
    match args.cmd {
        PropsCmd::Water {
            temp,
            pressure,
            temp_unit,
            pressure_unit,
        } => run_water(temp, pressure, &temp_unit, &pressure_unit),
        PropsCmd::Air {
            temp,
            rh,
            pressure,
            temp_unit,
            pressure_unit,
        } => run_air(temp, rh, pressure, &temp_unit, &pressure_unit),
        PropsCmd::Fuel { name, h2 } => run_fuel(&name, h2),
    }
}

fn run_water(temp: f64, pressure: f64, temp_unit: &str, pressure_unit: &str) -> Result<()> {
    let t = ThermodynamicTemperature::new::<kelvin>(to_kelvin(temp, temp_unit));
    let p = Pressure::new::<pascal>(parse_pressure(pressure, pressure_unit)?);
    let s = tpt_eng_props_water::state(t, p)
        .map_err(|e: WaterError| anyhow::anyhow!("water: {e:?}"))?;

    println!("Water/steam state (IAPWS-IF97, region {:?})", s.region);
    println!(
        "  temperature      = {:.4} K",
        s.temperature.get::<kelvin>()
    );
    println!(
        "  pressure         = {:.4} kPa",
        s.pressure.get::<pascal>() / 1e3
    );
    println!(
        "  specific volume = {:.9} m^3/kg",
        s.specific_volume
            .get::<tpt_math_units::uom::si::specific_volume::cubic_meter_per_kilogram>()
    );
    println!(
        "  density          = {:.4} kg/m^3",
        s.density.get::<kilogram_per_cubic_meter>()
    );
    println!("  enthalpy         = {:.4} kJ/kg", s.enthalpy);
    println!(
        "  entropy          = {:.6} kJ/(kg·K)",
        s.entropy
            .get::<tpt_math_units::uom::si::specific_heat_capacity::kilojoule_per_kilogram_kelvin>(
            )
    );
    println!("  internal energy  = {:.4} kJ/kg", s.internal_energy);
    println!(
        "  cp               = {:.6} kJ/(kg·K)",
        s.isobaric_heat_capacity.get::<tpt_math_units::uom::si::specific_heat_capacity::kilojoule_per_kilogram_kelvin>()
    );
    println!(
        "  cv               = {:.6} kJ/(kg·K)",
        s.isochoric_heat_capacity.get::<tpt_math_units::uom::si::specific_heat_capacity::kilojoule_per_kilogram_kelvin>()
    );
    Ok(())
}

fn run_air(temp: f64, rh: f64, pressure: f64, temp_unit: &str, pressure_unit: &str) -> Result<()> {
    if !(0.0..=100.0).contains(&rh) {
        bail!("relative humidity must be within 0-100 percent (got {rh})");
    }
    let t = ThermodynamicTemperature::new::<degree_celsius>(to_kelvin(temp, temp_unit) - 273.15);
    let p = Pressure::new::<pascal>(parse_pressure(pressure, pressure_unit)?);
    let psat = tpt_eng_props_air::saturation_pressure_water(t);
    let psat_pa = psat.get::<pascal>();
    let pw = psat_pa * (rh / 100.0);
    if pw >= p.get::<pascal>() {
        bail!("saturation pressure exceeds total pressure at this condition");
    }
    let p_w = Pressure::new::<pascal>(pw);

    let w = tpt_eng_props_air::humidity_ratio(p_w, p)
        .map_err(|e: AirError| anyhow::anyhow!("air: {e:?}"))?;
    let dp =
        tpt_eng_props_air::dew_point(p_w).map_err(|e: AirError| anyhow::anyhow!("air: {e:?}"))?;
    let h = tpt_eng_props_air::moist_air_enthalpy(t, w);

    println!("Moist-air state (ASHRAE)");
    println!("  dry-bulb         = {:.3} °C", t.get::<degree_celsius>());
    println!("  relative humidity = {:.2} %", rh);
    println!("  saturation p      = {:.3} Pa", psat_pa);
    println!("  vapour p          = {:.3} Pa", pw);
    println!("  humidity ratio W  = {:.6} kg/kg", w);
    println!("  dew point         = {:.3} °C", dp.get::<degree_celsius>());
    println!("  enthalpy          = {:.3} kJ/kg dry air", h);
    Ok(())
}

fn run_fuel(name: &str, h2: Option<f64>) -> Result<()> {
    let name = name.trim().to_ascii_lowercase();
    match name.as_str() {
        "blend" => {
            let f = h2.ok_or_else(|| anyhow::anyhow!("blend requires --h2 <0..1>"))?;
            let b = tpt_eng_props_fuels::BlendedFuel::new(f);
            println!("Blended fuel (natural gas + H2, h2 = {:.3})", b.h2_fraction);
            println!("  molar mass        = {:.3} kg/kmol", b.molar_mass);
            println!("  LHV               = {:.3} MJ/kg", b.lhv);
            println!("  HHV               = {:.3} MJ/kg", b.hhv);
            println!(
                "  density           = {:.4} kg/m^3",
                b.density.get::<kilogram_per_cubic_meter>()
            );
            println!(
                "  stoichiometric AFR = {:.3} kg air/kg fuel",
                b.stoichiometric_air_fuel_ratio()
            );
            println!("  CO2               = {:.4} kg/MJ LHV", b.co2_kg_per_mj());
        }
        _ => {
            let fuel = match name.as_str() {
                "methane" => Fuel::Methane,
                "hydrogen" | "h2" => Fuel::Hydrogen,
                "natural-gas" | "naturalgas" | "ng" => Fuel::NaturalGas,
                "diesel" => Fuel::Diesel,
                other => {
                    bail!("unknown fuel: {other} (try methane/hydrogen/natural-gas/diesel/blend)")
                }
            };
            println!("Fuel: {name}");
            println!("  LHV               = {:.3} MJ/kg", fuel.lhv_mj_kg());
            println!("  HHV               = {:.3} MJ/kg", fuel.hhv_mj_kg());
            println!(
                "  density           = {:.4} kg/m^3",
                fuel.density().get::<kilogram_per_cubic_meter>()
            );
            println!(
                "  stoichiometric AFR = {:.3} kg air/kg fuel",
                fuel.stoichiometric_air_fuel_ratio()
            );
            println!(
                "  CO2               = {:.4} kg/MJ LHV",
                fuel.co2_kg_per_mj()
            );
        }
    }
    Ok(())
}
