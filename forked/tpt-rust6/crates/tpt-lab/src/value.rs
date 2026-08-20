//! Type-erased cell values plus the runtime type registry primitives.

use std::any::{type_name, Any};
use std::fmt;
use std::sync::Arc;

/// A type-erased notebook value that remembers the static type it was built
/// from. This is the unit stored by every [`crate::Notebook`] cell and is what
/// backs the runtime type registry (`Notebook::type_of`).
#[derive(Clone)]
pub struct Value {
    any: Arc<dyn Any + Send + Sync>,
    type_name: &'static str,
}

impl Value {
    /// Erase `v`, recording `std::any::type_name::<T>()`.
    pub fn new<T: Any + Send + Sync>(v: T) -> Self {
        Self {
            any: Arc::new(v),
            type_name: type_name::<T>(),
        }
    }

    /// The recorded type name, e.g. `"i64"` or `"tpt_omni::table::Table"`.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Borrow the value as `T`, or `None` on a type mismatch.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.any.downcast_ref::<T>()
    }

    /// Borrow the erased value (used by the rendering dispatch).
    pub fn as_any(&self) -> &(dyn Any + Send + Sync) {
        &*self.any
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value<{}>", self.type_name)
    }
}

/// Best-effort Rust literal for a value, used by `Notebook::export_rust`.
///
/// Returns `None` for types with no obvious literal form (tables, tensors,
/// user structs); the exporter emits a comment for those instead.
pub(crate) fn literal_repr(v: &Value) -> Option<String> {
    if let Some(x) = v.downcast_ref::<i64>() {
        return Some(format!("{x}i64"));
    }
    if let Some(x) = v.downcast_ref::<i32>() {
        return Some(format!("{x}i32"));
    }
    if let Some(x) = v.downcast_ref::<usize>() {
        return Some(format!("{x}usize"));
    }
    if let Some(x) = v.downcast_ref::<f64>() {
        return Some(format!("{x:?}f64"));
    }
    if let Some(x) = v.downcast_ref::<f32>() {
        return Some(format!("{x:?}f32"));
    }
    if let Some(x) = v.downcast_ref::<bool>() {
        return Some(format!("{x}"));
    }
    if let Some(x) = v.downcast_ref::<String>() {
        return Some(format!("{x:?}.to_string()"));
    }
    if let Some(x) = v.downcast_ref::<&'static str>() {
        return Some(format!("{x:?}"));
    }
    if let Some(x) = v.downcast_ref::<Vec<f64>>() {
        let items: Vec<String> = x.iter().map(|v| format!("{v:?}")).collect();
        return Some(format!("vec![{}]", items.join(", ")));
    }
    if let Some(x) = v.downcast_ref::<Vec<i64>>() {
        let items: Vec<String> = x.iter().map(|v| format!("{v}i64")).collect();
        return Some(format!("vec![{}]", items.join(", ")));
    }
    None
}
