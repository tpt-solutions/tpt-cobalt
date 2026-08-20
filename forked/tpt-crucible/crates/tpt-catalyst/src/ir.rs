use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TptIrError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),
    #[error("ingestion error: {0}")]
    Ingestion(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TptIr {
    pub version: String,
    pub metadata: ModelMetadata,
    pub graph: ComputationalGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    pub name: String,
    pub source_format: String,
    pub parameter_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComputationalGraph {
    pub nodes: Vec<OpNode>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpNode {
    pub id: usize,
    pub op_type: String,
    pub name: String,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub tensor_name: String,
}

impl TptIr {
    pub fn new(name: String, source_format: String) -> Self {
        Self {
            version: "1.0.0".into(),
            metadata: ModelMetadata {
                name,
                source_format,
                parameter_count: 0,
            },
            graph: ComputationalGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
        }
    }

    pub fn to_json(&self) -> Result<String, TptIrError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn to_binary(&self) -> Result<Vec<u8>, TptIrError> {
        Ok(serde_json::to_vec(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self, TptIrError> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_file(path: &Path) -> Result<Self, TptIrError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    pub fn save(&self, path: &Path) -> Result<(), TptIrError> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_roundtrip() {
        let ir = TptIr::new("test_model".into(), "pytorch".into());
        let json = ir.to_json().unwrap();
        let restored = TptIr::from_json(&json).unwrap();
        assert_eq!(ir, restored);
    }
}
