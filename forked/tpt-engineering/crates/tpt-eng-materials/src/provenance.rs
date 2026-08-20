//! Minimal source-tracking and metadata types, local to this crate.
//!
//! `tpt-math` has no equivalent of these plain data-bag types (they carry no
//! numeric logic), so they're defined here directly rather than pulled from
//! elsewhere.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Free-form metadata describing a material.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    /// Short human-readable name.
    pub name: String,
    /// Free-form custom attributes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl Metadata {
    /// Create metadata with a required name and no attributes.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: BTreeMap::new(),
        }
    }
}

/// The kind of source a datum came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    /// Read from an external file or database.
    File,
    /// Produced by a calculation.
    #[default]
    Calculated,
    /// Defined by a standard, specification, or allowable.
    Standard,
}

/// Tracks the origin of a material's data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct DataSource {
    /// The kind of source.
    pub kind: SourceKind,
    /// Free-text origin label (file path, standard reference, ...).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

impl DataSource {
    /// A file source with the given path/name.
    pub fn file(label: impl Into<String>) -> Self {
        Self {
            kind: SourceKind::File,
            label: label.into(),
        }
    }

    /// A standard/specification source with the given reference.
    pub fn standard(label: impl Into<String>) -> Self {
        Self {
            kind: SourceKind::Standard,
            label: label.into(),
        }
    }
}

impl Default for DataSource {
    fn default() -> Self {
        Self {
            kind: SourceKind::Calculated,
            label: String::new(),
        }
    }
}
