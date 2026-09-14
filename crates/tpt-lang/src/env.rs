//! Lexically scoped environments for the TPT script runtime.
//!
//! An [`Environment`] maps variable names to [`Value`]s and chains to a parent
//! scope via `Arc<Mutex<…>>`, giving shared-mutable closure semantics
//! (the documented divergence from the spec's tracing-GC sketch).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::value::Value;

/// A single lexical scope.
#[derive(Debug, Default)]
pub struct Environment {
    values: Mutex<HashMap<String, Value>>,
    parent: Option<Arc<Environment>>,
}

impl Environment {
    /// Create a fresh root (global) environment.
    pub fn new() -> Self {
        Environment::default()
    }

    /// Create a child scope whose lookups fall through to `self`.
    #[must_use]
    pub fn child(parent: &Arc<Environment>) -> Environment {
        Environment { values: Mutex::new(HashMap::new()), parent: Some(Arc::clone(parent)) }
    }

    /// Define (or overwrite) a variable in *this* scope only.
    pub fn define(&self, name: impl Into<String>, value: Value) {
        self.values.lock().unwrap().insert(name.into(), value);
    }

    /// Assign to an existing binding found in this scope or any ancestor.
    /// Returns `Err(())` when no binding with that name exists anywhere on the
    /// scope chain (assignment never creates bindings).
    pub fn assign(&self, name: &str, value: Value) -> Result<(), ()> {
        if self.values.lock().unwrap().contains_key(name) {
            self.values.lock().unwrap().insert(name.to_string(), value);
            return Ok(());
        }
        match &self.parent {
            Some(p) => p.assign(name, value),
            None => Err(()),
        }
    }

    /// Look up a name in this scope or any ancestor.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(v) = self.values.lock().unwrap().get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.get(name))
    }

    /// Whether a binding exists on this scope or any ancestor.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// All names visible from this scope (nearest scope last).
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = Some(self);
        while let Some(scope) = cur {
            for k in scope.values.lock().unwrap().keys() {
                if !out.contains(k) {
                    out.push(k.clone());
                }
            }
            cur = scope.parent.as_deref();
        }
        out
    }

    /// A snapshot of *this scope's own* bindings (no ancestors) — used by the
    /// `module` statement to collect a block's namespace.
    #[must_use]
    pub fn own_bindings(&self) -> HashMap<String, Value> {
        self.values.lock().unwrap().clone()
    }
}