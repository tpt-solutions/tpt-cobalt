//! A collection of [`Material`]s with loading, querying, and licensing checks.

use std::collections::BTreeSet;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::category::MaterialCategory;
use crate::error::{MaterialError, Result};
use crate::material::Material;

/// Licenses accepted for bundled material data. Anything not in this list is
/// treated as potentially proprietary and fails
/// [`MaterialLibrary::validate`].
pub const ALLOWED_LICENSES: &[&str] = &[
    "original",
    "user-provided",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "zlib",
    "cc0-1.0",
    "public-domain",
    "open-source",
];

/// A library (collection) of materials.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialLibrary {
    /// The materials in the library.
    pub materials: Vec<Material>,
}

impl MaterialLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        MaterialLibrary::default()
    }

    /// Number of materials.
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// True if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Add a material.
    pub fn add(&mut self, material: Material) {
        self.materials.push(material);
    }

    /// Borrow a material by id.
    pub fn get_by_id(&self, id: &str) -> Option<&Material> {
        self.materials.iter().find(|m| m.id == id)
    }

    /// Borrow a material by name (first match).
    pub fn get_by_name(&self, name: &str) -> Option<&Material> {
        self.materials.iter().find(|m| m.name == name)
    }

    /// Materials in a given category.
    pub fn filter_by_category(&self, category: MaterialCategory) -> Vec<&Material> {
        self.materials
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Deserialize a library from a JSON string.
    ///
    /// # Errors
    /// Returns an error if `json` is malformed or does not match the
    /// `MaterialLibrary` schema.
    pub fn from_json(json: &str) -> Result<Self> {
        let lib: MaterialLibrary = serde_json::from_str(json)?;
        Ok(lib)
    }

    /// Serialize the library to a pretty JSON string.
    ///
    /// # Errors
    /// Returns an error if serialization of the library fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// # Errors
    /// Returns an error if the CSV is malformed, missing a required column
    /// (`id`, `name`, or `category`), or a category or scalar-property value
    /// fails to parse.
    ///
    /// Read a library from CSV. The first columns must be `id`, `name`,
    /// `category`; every remaining column is treated as a scalar property name
    /// (parsed as `f64`). A `description` or `source` column, if present, is
    /// honored; all other extra columns become scalar properties.
    pub fn from_csv(reader: impl Read) -> Result<Self> {
        let mut rdr = csv::Reader::from_reader(reader);
        let headers = rdr.headers()?.clone();
        let id_i = column(&headers, "id")?;
        let name_i = column(&headers, "name")?;
        let cat_i = column(&headers, "category")?;
        let desc_i = optional_column(&headers, "description");
        let source_i = optional_column(&headers, "source");

        let mut prop_cols: Vec<(usize, String)> = Vec::new();
        for (i, h) in headers.iter().enumerate() {
            match h.trim().to_ascii_lowercase().as_str() {
                "id" | "name" | "category" | "description" | "source" => {}
                other => prop_cols.push((i, other.to_string())),
            }
        }

        let mut materials = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let id = rec.get(id_i).unwrap_or("").to_string();
            let name = rec.get(name_i).unwrap_or("").to_string();
            let cat_str = rec.get(cat_i).unwrap_or("").to_string();
            let category = cat_str
                .parse::<MaterialCategory>()
                .map_err(|e| MaterialError::Parse { what: e })?;
            let mut m = Material::new(id.clone(), name, category);
            if let Some(i) = desc_i {
                if let Some(d) = rec.get(i) {
                    if !d.is_empty() {
                        m = m.with_description(d.to_string());
                    }
                }
            }
            if let Some(i) = source_i {
                if let Some(s) = rec.get(i) {
                    if !s.is_empty() {
                        m = m.with_source(crate::provenance::DataSource::file(s.to_string()));
                    }
                }
            }
            for (i, pname) in &prop_cols {
                if let Some(v) = rec.get(*i) {
                    let v = v.trim();
                    if v.is_empty() {
                        continue;
                    }
                    let value: f64 = v.parse().map_err(|_| MaterialError::Parse {
                        what: format!("property `{pname}` value `{v}` is not a number"),
                    })?;
                    m = m.with_property(
                        pname.clone(),
                        crate::property::Property::Scalar {
                            value,
                            unit: String::new(),
                        },
                    );
                }
            }
            materials.push(m);
        }
        Ok(MaterialLibrary { materials })
    }

    /// Write the library to CSV, using the union of all scalar property names
    /// across the library as columns. Temperature/anisotropic properties are
    /// exported as their representative value at the reference temperature
    /// `ref_temp` (NaN becomes an empty cell).
    ///
    /// # Errors
    /// Returns an error if writing a record or flushing the underlying writer
    /// fails.
    pub fn to_csv(&self, writer: impl Write, ref_temp: f64) -> Result<()> {
        let mut wtr = csv::Writer::from_writer(writer);
        let mut prop_keys: BTreeSet<String> = BTreeSet::new();
        for m in &self.materials {
            for k in m.properties.keys() {
                prop_keys.insert(k.clone());
            }
        }
        let mut headers = vec!["id", "name", "category", "description", "source"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        headers.extend(prop_keys.iter().cloned());
        wtr.write_record(&headers)?;

        for m in &self.materials {
            let mut row: Vec<String> = vec![
                m.id.clone(),
                m.name.clone(),
                m.category.to_string(),
                m.description.clone(),
                m.source.label.clone(),
            ];
            for k in &prop_keys {
                match m.value(k, ref_temp) {
                    Some(v) if v.is_finite() => row.push(format!("{v}")),
                    _ => row.push(String::new()),
                }
            }
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
        Ok(())
    }

    /// Validate the library against the data-policy rules:
    ///
    /// * every material must carry a source with a non-empty label
    ///   (traceability), and
    /// * if a material records a `license` attribute, it must be in
    ///   [`ALLOWED_LICENSES`] (no proprietary data).
    ///
    /// Returns `Ok(())` when the library complies, or
    /// [`MaterialError::Validation`] listing the offending materials.
    ///
    /// # Errors
    /// Returns [`MaterialError::Validation`] listing the offending materials
    /// when a material lacks a source or records a disallowed license.
    pub fn validate(&self) -> Result<()> {
        let mut problems = Vec::new();
        for m in &self.materials {
            if m.source.label.trim().is_empty() {
                problems.push(format!("`{}` has no recorded source", m.id));
            }
            if let Some(lic) = m.metadata.attributes.get("license") {
                let allowed = ALLOWED_LICENSES.iter().any(|a| a.eq_ignore_ascii_case(lic));
                if !allowed {
                    problems.push(format!("`{}` records disallowed license `{lic}`", m.id));
                }
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(MaterialError::Validation {
                what: problems.join("; "),
            })
        }
    }

    /// True if the library passes [`MaterialLibrary::validate`].
    pub fn is_compliant(&self) -> bool {
        self.validate().is_ok()
    }
}

fn column(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h.trim().eq_ignore_ascii_case(name))
        .ok_or_else(|| MaterialError::Parse {
            what: format!("missing required CSV column `{name}`"),
        })
}

fn optional_column(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|h| h.trim().eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Property;

    fn sample() -> MaterialLibrary {
        let mut lib = MaterialLibrary::new();
        lib.add(
            Material::new("steel-s355", "S355", MaterialCategory::Metal)
                .with_source(crate::provenance::DataSource::standard("EN 10025"))
                .with_property(
                    "youngs-modulus",
                    Property::Scalar {
                        value: 210.0,
                        unit: "GPa".into(),
                    },
                ),
        );
        lib.add(
            Material::new("aluminium-6061", "AA6061", MaterialCategory::Metal)
                .with_source(crate::provenance::DataSource::standard("AMS 4027"))
                .with_property(
                    "youngs-modulus",
                    Property::Scalar {
                        value: 69.0,
                        unit: "GPa".into(),
                    },
                ),
        );
        lib
    }

    #[test]
    fn json_round_trip() {
        let lib = sample();
        let json = lib.to_json().unwrap();
        let back = MaterialLibrary::from_json(&json).unwrap();
        assert_eq!(lib, back);
    }

    #[test]
    fn query_helpers() {
        let lib = sample();
        assert_eq!(lib.len(), 2);
        assert!(lib.get_by_id("steel-s355").is_some());
        assert_eq!(lib.filter_by_category(MaterialCategory::Metal).len(), 2);
        let e = lib.get_by_id("steel-s355").unwrap();
        assert!((e.value("youngs-modulus", 0.0).unwrap() - 210.0).abs() < 1e-15);
    }

    #[test]
    fn csv_round_trip() {
        let lib = sample();
        let mut buf = Vec::new();
        lib.to_csv(&mut buf, 0.0).unwrap();
        let parsed = MaterialLibrary::from_csv(buf.as_slice()).unwrap();
        assert_eq!(parsed.len(), 2);
        // CSV export stores scalar properties at ref_temp.
        let e = parsed.get_by_id("steel-s355").unwrap();
        assert!((e.value("youngs-modulus", 0.0).unwrap() - 210.0).abs() < 1e-15);
    }

    #[test]
    fn validate_flags_missing_source_and_bad_license() {
        let mut lib = MaterialLibrary::new();
        lib.add(Material::new("x", "X", MaterialCategory::Other)); // no source
        let mut bad = Material::new("y", "Y", MaterialCategory::Other)
            .with_source(crate::provenance::DataSource::file("vendor.pdf"));
        bad.metadata
            .attributes
            .insert("license".into(), "proprietary".into());
        lib.add(bad);
        let err = lib.validate().unwrap_err();
        assert!(err.to_string().contains("x"));
        assert!(err.to_string().contains("proprietary"));
        assert!(!lib.is_compliant());
    }

    #[test]
    fn validate_passes_compliant_library() {
        assert!(sample().validate().is_ok());
    }
}
