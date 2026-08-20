//! Material property values: scalar, temperature-dependent, and anisotropic.
//!
//! A [`Property`] is the value model behind every named material property (e.g.
//! `"youngs-modulus"`, `"thermal-expansion"`). Three forms are supported:
//!
//! * [`Property::Scalar`] — a single value with a unit string.
//! * [`Property::TemperatureDependent`] — a set of `(temperature, value)` points
//!   evaluated by linear interpolation via [`Property::value_at`].
//! * [`Property::Anisotropic`] — per-direction scalars (a hook for directional
//!   or tensor-valued properties), accessed via
//!   [`Property::value_in_direction`].
//!
//! Value retrieval never returns `f64::NAN`: [`Property::value_at`] and the
//! representative-value helpers [`Property::average`] and
//! [`Property::interpolate`] return `Option<f64>`, yielding `None` when a
//! value cannot be produced (e.g. an empty temperature-dependent sample set).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single `(temperature, value)` sample for a temperature-dependent property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TempPoint {
    /// Temperature at which `value` was measured/specified (in the property's
    /// temperature unit, conventionally Kelvin or degrees Celsius — recorded by
    /// the caller).
    pub temp: f64,
    /// The property value at `temp`.
    pub value: f64,
}

/// A material property value.
///
/// Serialized as an internally-tagged enum on the `type` field
/// (`"scalar"`, `"temperature_dependent"`, `"anisotropic"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Property {
    /// A single value valid at all temperatures.
    Scalar {
        /// The value.
        value: f64,
        /// Unit string (e.g. `"GPa"`, `"W/(m·K)"`).
        unit: String,
    },
    /// A value defined at discrete temperatures, interpolated between them.
    TemperatureDependent {
        /// Unit string.
        unit: String,
        /// The `(temperature, value)` samples (any order; sorted on evaluation).
        points: Vec<TempPoint>,
    },
    /// A direction-dependent value (anisotropic hook). Keys are free-form
    /// direction labels such as `"x"`, `"y"`, `"z"`, `"11"`, `"22"`, `"33"`.
    Anisotropic {
        /// Unit string.
        unit: String,
        /// Per-direction scalar values.
        values: BTreeMap<String, f64>,
    },
}

impl Property {
    /// The unit string of this property.
    pub fn unit(&self) -> &str {
        match self {
            Property::Scalar { unit, .. } => unit,
            Property::TemperatureDependent { unit, .. } => unit,
            Property::Anisotropic { unit, .. } => unit,
        }
    }

    /// Whether this property varies with temperature.
    pub fn is_temperature_dependent(&self) -> bool {
        matches!(self, Property::TemperatureDependent { .. })
    }

    /// Whether this property is direction-dependent.
    pub fn is_anisotropic(&self) -> bool {
        matches!(self, Property::Anisotropic { .. })
    }

    /// Evaluate the property at temperature `temp`.
    ///
    /// Returns `None` when no value can be produced — in particular an
    /// [`Property::Anisotropic`] with no recorded directions, which has no
    /// representative scalar.
    ///
    /// * [`Property::Scalar`] returns its constant value.
    /// * [`Property::TemperatureDependent`] linearly interpolates between the
    ///   bracketing samples and clamps to the end samples outside the range.
    /// * [`Property::Anisotropic`] returns the arithmetic mean of its
    ///   directional values (a representative isotropic estimate; use
    ///   [`Property::value_in_direction`] for a specific direction).
    pub fn value_at(&self, temp: f64) -> Option<f64> {
        match self {
            Property::Scalar { value, .. } => Some(*value),
            Property::TemperatureDependent { points, .. } => interpolate(points, temp),
            Property::Anisotropic { values, .. } => Self::average_anisotropic(values),
        }
    }

    /// The representative value of the property as a single number.
    ///
    /// * [`Property::Scalar`] returns `Some(value)`.
    /// * [`Property::TemperatureDependent`] returns the arithmetic mean of its
    ///   sampled values, or `None` if it has no samples.
    /// * [`Property::Anisotropic`] returns the arithmetic mean of its directional
    ///   values, or `None` if it has no directions.
    pub fn average(&self) -> Option<f64> {
        match self {
            Property::Scalar { value, .. } => Some(*value),
            Property::TemperatureDependent { points, .. } => {
                if points.is_empty() {
                    return None;
                }
                Some(points.iter().map(|p| p.value).sum::<f64>() / points.len() as f64)
            }
            Property::Anisotropic { values, .. } => Self::average_anisotropic(values),
        }
    }

    /// Interpolate the property value at `temp`.
    ///
    /// Returns `None` for [`Property::Anisotropic`] (direction-specific values
    /// are not temperature-sampled) and for an empty
    /// [`Property::TemperatureDependent`] sample set; otherwise the same value
    /// [`Property::value_at`] would return.
    pub fn interpolate(&self, temp: f64) -> Option<f64> {
        match self {
            Property::Scalar { value, .. } => Some(*value),
            Property::TemperatureDependent { points, .. } => interpolate(points, temp),
            Property::Anisotropic { .. } => None,
        }
    }

    /// Arithmetic mean of anisotropic directional values, or `None` if empty.
    fn average_anisotropic(values: &BTreeMap<String, f64>) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        Some(values.values().sum::<f64>() / values.len() as f64)
    }

    /// Value in a specific direction, for anisotropic properties. Returns `None`
    /// for scalar/temperature-dependent properties (which are direction-free).
    pub fn value_in_direction(&self, direction: &str) -> Option<f64> {
        match self {
            Property::Anisotropic { values, .. } => values.get(direction).copied(),
            _ => None,
        }
    }

    /// The set of direction labels, for anisotropic properties.
    pub fn directions(&self) -> Vec<String> {
        match self {
            Property::Anisotropic { values, .. } => values.keys().cloned().collect(),
            _ => Vec::new(),
        }
    }
}

/// Linearly interpolate `(temperature, value)` samples at `temp`, clamping
/// outside the sampled range. Returns `None` for an empty sample set.
fn interpolate(points: &[TempPoint], temp: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.temp
            .partial_cmp(&b.temp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if temp <= pts[0].temp {
        return Some(pts[0].value);
    }
    if temp >= pts[pts.len() - 1].temp {
        return Some(pts[pts.len() - 1].value);
    }
    for w in pts.windows(2) {
        let (lo, hi) = (&w[0], &w[1]);
        if temp >= lo.temp && temp <= hi.temp {
            let span = hi.temp - lo.temp;
            if span == 0.0 {
                return Some(lo.value);
            }
            let t = (temp - lo.temp) / span;
            return Some(lo.value + t * (hi.value - lo.value));
        }
    }
    Some(pts[pts.len() - 1].value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_value() {
        let p = Property::Scalar {
            value: 200.0,
            unit: "GPa".into(),
        };
        assert!((p.value_at(0.0).unwrap() - 200.0).abs() < 1e-15);
        assert_eq!(p.unit(), "GPa");
        assert!(!p.is_temperature_dependent());
    }

    #[test]
    fn temperature_interpolation() {
        let p = Property::TemperatureDependent {
            unit: "GPa".into(),
            points: vec![
                TempPoint {
                    temp: 0.0,
                    value: 210.0,
                },
                TempPoint {
                    temp: 100.0,
                    value: 190.0,
                },
            ],
        };
        assert!(p.is_temperature_dependent());
        assert!((p.value_at(50.0).unwrap() - 200.0).abs() < 1e-15);
        // clamps below/above the range
        assert!((p.value_at(-50.0).unwrap() - 210.0).abs() < 1e-15);
        assert!((p.value_at(500.0).unwrap() - 190.0).abs() < 1e-15);
    }

    #[test]
    fn anisotropic_value() {
        let p = Property::Anisotropic {
            unit: "GPa".into(),
            values: {
                let mut m = BTreeMap::new();
                m.insert("x".to_string(), 200.0);
                m.insert("y".to_string(), 10.0);
                m
            },
        };
        assert!(p.is_anisotropic());
        assert!((p.value_in_direction("x").unwrap() - 200.0).abs() < 1e-15);
        assert!((p.value_at(0.0).unwrap() - 105.0).abs() < 1e-15); // mean
        assert_eq!(p.directions(), vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn empty_inputs_return_none() {
        let scalar = Property::Scalar {
            value: 1.0,
            unit: "x".into(),
        };
        let td = Property::TemperatureDependent {
            unit: "x".into(),
            points: vec![],
        };
        let aniso = Property::Anisotropic {
            unit: "x".into(),
            values: BTreeMap::new(),
        };
        // value_at still resolves for scalar and anisotropic-with-data; None only
        // when there is genuinely no value to report.
        assert_eq!(scalar.value_at(0.0), Some(1.0));
        assert_eq!(td.value_at(0.0), None);
        assert_eq!(aniso.value_at(0.0), None);
        assert_eq!(td.average(), None);
        assert_eq!(aniso.average(), None);
        assert_eq!(td.interpolate(0.0), None);
        assert_eq!(aniso.interpolate(0.0), None);
    }
}
