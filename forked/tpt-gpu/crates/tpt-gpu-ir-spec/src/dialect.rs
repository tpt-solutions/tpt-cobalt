use std::collections::HashMap;

/// Semver string for the TPTIR text-format dialect. Consumers should check
/// this when reading TPTIR files to guard against format changes.
pub const TPTIR_DIALECT_VERSION: &str = "0.1.0";

/// A named dialect groups a set of ops and type extensions.
#[derive(Debug, Clone)]
pub struct Dialect {
    pub name: String,
    pub version: String,
}

impl Dialect {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }

    pub fn tptir_core() -> Self {
        Self::new("tptir", TPTIR_DIALECT_VERSION)
    }
}

/// Registry of all dialects active in a compilation unit.
#[derive(Debug, Default)]
pub struct DialectRegistry {
    dialects: HashMap<String, Dialect>,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, dialect: Dialect) {
        self.dialects.insert(dialect.name.clone(), dialect);
    }

    pub fn get(&self, name: &str) -> Option<&Dialect> {
        self.dialects.get(name)
    }

    pub fn with_tptir_core() -> Self {
        let mut reg = Self::new();
        reg.register(Dialect::tptir_core());
        reg
    }
}
