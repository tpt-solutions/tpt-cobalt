use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single attribute value attached to an op or function.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<AttrValue>),
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AttrValue::Int(v) => write!(f, "{}", v),
            AttrValue::Float(v) => write!(f, "{:.6}", v),
            AttrValue::Bool(v) => write!(f, "{}", v),
            AttrValue::String(s) => write!(f, "\"{}\"", s),
            AttrValue::Array(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Named attribute dictionary — mirrors MLIR's attribute dictionary syntax.
///
/// In text format: `{alpha = 1.0, transpose_a = false}`
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Attr(pub HashMap<String, AttrValue>);

impl Attr {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
    pub fn set(&mut self, key: impl Into<String>, value: AttrValue) {
        self.0.insert(key.into(), value);
    }
    pub fn get(&self, key: &str) -> Option<&AttrValue> {
        self.0.get(key)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }
        let mut pairs: Vec<_> = self.0.iter().collect();
        pairs.sort_by_key(|(k, _)| k.as_str());
        write!(f, "{{")?;
        for (i, (k, v)) in pairs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} = {}", k, v)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attr_display_sorted() {
        let mut a = Attr::new();
        a.set("beta", AttrValue::Float(0.0));
        a.set("alpha", AttrValue::Float(1.0));
        let s = a.to_string();
        assert!(s.starts_with('{'));
        let alpha_pos = s.find("alpha").unwrap();
        let beta_pos = s.find("beta").unwrap();
        assert!(alpha_pos < beta_pos, "keys should be sorted");
    }
}
