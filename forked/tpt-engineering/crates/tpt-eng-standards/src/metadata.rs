//! Minimal descriptive-metadata type, local to this crate.
//!
//! `tpt-math` has no equivalent (it carries no numeric logic), so it's
//! defined here directly rather than pulled from elsewhere.

use serde::{Deserialize, Serialize};

/// Free-form metadata describing a load case or combination.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    /// Short human-readable name.
    pub name: String,
}

impl Metadata {
    /// Create metadata with a required name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}
