//! Shape-specialized kernel dispatch.
//!
//! Manages multiple kernel variants and selects the best one based on
//! the dispatch table (`tuning/dispatch_table.json`).

use crate::ir::{ElemType, KernelVariant, KERNEL_TEMPLATES};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dispatch table entry: maps a kernel variant key to the best-performing
/// kernel variant for that configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchEntry {
    /// The dispatch key: `{name}_{elem}_{shape}`
    pub key: String,
    /// The selected variant name
    pub selected_variant: String,
    /// Performance metadata
    pub execution_time_ms: f64,
    /// GPU model this entry was tuned for
    pub gpu_model: String,
}

/// Dispatch table: collection of dispatch entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchTable {
    pub entries: HashMap<String, DispatchEntry>,
    pub version: String,
}

impl Default for DispatchTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchTable {
    /// Create a new empty dispatch table.
    pub fn new() -> Self {
        DispatchTable {
            entries: HashMap::new(),
            version: "1.0".to_string(),
        }
    }

    /// Load a dispatch table from a JSON file.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize the dispatch table to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Get the dispatch entry for a kernel variant.
    pub fn get_entry(&self, variant: &KernelVariant) -> Option<&DispatchEntry> {
        let key = variant.dispatch_key();
        self.entries.get(&key)
    }

    /// Insert or update a dispatch entry.
    pub fn insert_entry(&mut self, entry: DispatchEntry) {
        self.entries.insert(entry.key.clone(), entry);
    }

    /// Generate dispatch entries for all supported kernel templates.
    pub fn generate_default_entries(&mut self, gpu_model: &str) {
        for &name in KERNEL_TEMPLATES {
            for elem in [ElemType::F32, ElemType::F16, ElemType::BF16] {
                let variant = KernelVariant {
                    name: name.to_string(),
                    elem,
                    shape_params: vec![1024], // default shape
                };
                let key = variant.dispatch_key();
                self.entries.insert(
                    key.clone(),
                    DispatchEntry {
                        key,
                        selected_variant: format!("{}_{}", name, elem.name()),
                        execution_time_ms: 1.0,
                        gpu_model: gpu_model.to_string(),
                    },
                );
            }
        }
    }
}

/// Dispatch a kernel call to the best variant.
pub fn dispatch_kernel(table: &DispatchTable, variant: &KernelVariant) -> Option<String> {
    table.get_entry(variant).map(|e| e.selected_variant.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_table_new() {
        let table = DispatchTable::new();
        assert!(table.entries.is_empty());
    }

    #[test]
    fn test_generate_default_entries() {
        let mut table = DispatchTable::new();
        table.generate_default_entries("RTX_3090");
        assert!(!table.entries.is_empty());
        // Should have entries for 9 templates * 3 element types = 27
        assert_eq!(table.entries.len(), 27);
    }

    #[test]
    fn test_dispatch_kernel() {
        let mut table = DispatchTable::new();
        table.generate_default_entries("RTX_3090");
        let variant = KernelVariant {
            name: "vector_add".to_string(),
            elem: ElemType::F32,
            shape_params: vec![1024],
        };
        let result = dispatch_kernel(&table, &variant);
        assert!(result.is_some());
    }
}
