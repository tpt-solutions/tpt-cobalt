//! Minimal unit-conversion support for the CLI (length, mass, force, pressure, temperature).

use anyhow::{Result, bail};

/// Unit categories supported by [`convert`].
pub const CATEGORIES: &[&str] = &["length", "mass", "force", "pressure", "temperature"];

struct LinearUnit {
    aliases: &'static [&'static str],
    category: &'static str,
    factor: f64,
}

const LINEAR_UNITS: &[LinearUnit] = &[
    // length (base: metre)
    LinearUnit {
        aliases: &["m", "metre", "meter"],
        category: "length",
        factor: 1.0,
    },
    LinearUnit {
        aliases: &["cm"],
        category: "length",
        factor: 0.01,
    },
    LinearUnit {
        aliases: &["mm"],
        category: "length",
        factor: 0.001,
    },
    LinearUnit {
        aliases: &["um", "µm"],
        category: "length",
        factor: 1e-6,
    },
    LinearUnit {
        aliases: &["km"],
        category: "length",
        factor: 1000.0,
    },
    LinearUnit {
        aliases: &["in", "inch"],
        category: "length",
        factor: 0.0254,
    },
    LinearUnit {
        aliases: &["ft", "foot", "feet"],
        category: "length",
        factor: 0.3048,
    },
    LinearUnit {
        aliases: &["yd", "yard"],
        category: "length",
        factor: 0.9144,
    },
    LinearUnit {
        aliases: &["mi", "mile"],
        category: "length",
        factor: 1609.344,
    },
    // mass (base: kilogram)
    LinearUnit {
        aliases: &["kg"],
        category: "mass",
        factor: 1.0,
    },
    LinearUnit {
        aliases: &["g"],
        category: "mass",
        factor: 0.001,
    },
    LinearUnit {
        aliases: &["mg"],
        category: "mass",
        factor: 1e-6,
    },
    LinearUnit {
        aliases: &["t", "tonne"],
        category: "mass",
        factor: 1000.0,
    },
    LinearUnit {
        aliases: &["lb", "lbm"],
        category: "mass",
        factor: 0.45359237,
    },
    LinearUnit {
        aliases: &["oz"],
        category: "mass",
        factor: 0.028349523125,
    },
    // force (base: newton)
    LinearUnit {
        aliases: &["N"],
        category: "force",
        factor: 1.0,
    },
    LinearUnit {
        aliases: &["kN"],
        category: "force",
        factor: 1000.0,
    },
    LinearUnit {
        aliases: &["lbf"],
        category: "force",
        factor: 4.4482216152605,
    },
    LinearUnit {
        aliases: &["kip"],
        category: "force",
        factor: 4448.2216152605,
    },
    // pressure (base: pascal)
    LinearUnit {
        aliases: &["Pa"],
        category: "pressure",
        factor: 1.0,
    },
    LinearUnit {
        aliases: &["kPa"],
        category: "pressure",
        factor: 1000.0,
    },
    LinearUnit {
        aliases: &["MPa"],
        category: "pressure",
        factor: 1e6,
    },
    LinearUnit {
        aliases: &["bar"],
        category: "pressure",
        factor: 1e5,
    },
    LinearUnit {
        aliases: &["psi"],
        category: "pressure",
        factor: 6894.757293168,
    },
];

fn lookup_linear(unit: &str) -> Option<(&'static str, f64)> {
    let u = unit.trim();
    LINEAR_UNITS
        .iter()
        .find(|entry| entry.aliases.iter().any(|a| a.eq_ignore_ascii_case(u)))
        .map(|entry| (entry.category, entry.factor))
}

fn to_celsius(value: f64, from: &str) -> f64 {
    match from.trim().to_ascii_lowercase().as_str() {
        "c" | "°c" | "celsius" => value,
        "f" | "°f" | "fahrenheit" => (value - 32.0) * 5.0 / 9.0,
        "k" | "kelvin" => value - 273.15,
        _ => value,
    }
}

fn from_celsius(value: f64, to: &str) -> f64 {
    match to.trim().to_ascii_lowercase().as_str() {
        "c" | "°c" | "celsius" => value,
        "f" | "°f" | "fahrenheit" => value * 9.0 / 5.0 + 32.0,
        "k" | "kelvin" => value + 273.15,
        _ => value,
    }
}

/// Convert `value` from unit `from` to unit `to`.
///
/// Returns an error if either unit is unknown or the units belong to different categories.
pub fn convert(value: f64, from: &str, to: &str) -> Result<f64> {
    let from_l = from.trim().to_ascii_lowercase();
    let to_l = to.trim().to_ascii_lowercase();

    let is_temp = matches!(
        from_l.as_str(),
        "c" | "°c" | "celsius" | "f" | "°f" | "fahrenheit" | "k" | "kelvin"
    ) && matches!(
        to_l.as_str(),
        "c" | "°c" | "celsius" | "f" | "°f" | "fahrenheit" | "k" | "kelvin"
    );

    if is_temp {
        return Ok(from_celsius(to_celsius(value, &from_l), &to_l));
    }

    let (cat_from, f_from) =
        lookup_linear(from).ok_or_else(|| anyhow::anyhow!("unknown unit: {from}"))?;
    let (cat_to, f_to) = lookup_linear(to).ok_or_else(|| anyhow::anyhow!("unknown unit: {to}"))?;

    if cat_from != cat_to {
        bail!("incompatible units: {from} is {cat_from}, {to} is {cat_to}");
    }

    Ok(value * f_from / f_to)
}

/// Return all known unit aliases grouped by category (for help output).
pub fn list_units() -> String {
    let mut out = String::new();
    for category in CATEGORIES {
        out.push_str(category);
        out.push_str(": ");
        let mut first = true;
        for entry in LINEAR_UNITS.iter().filter(|e| e.category == *category) {
            if !first {
                out.push_str(", ");
            }
            out.push_str(entry.aliases[0]);
            first = false;
        }
        out.push('\n');
    }
    out.push_str("temperature: C, F, K\n");
    out
}
