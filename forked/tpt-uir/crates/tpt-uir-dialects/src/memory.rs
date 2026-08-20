extern crate alloc;

use alloc::vec;

use tpt_uir_core::attr::Attribute;
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::{Operation, ValueId};

/// Builder helpers for the memory dialect (`tpt_memory.*`).
pub struct MemOp;

impl MemOp {
    /// `tpt_memory.scope_begin { lifetime }`
    pub fn scope_begin(lifetime: &str) -> Operation {
        Operation {
            id: 0,
            op_name: OpName::new("tpt_memory", "scope_begin"),
            operands: vec![],
            results: vec![],
            regions: vec![],
            attributes: vec![Attribute::string("lifetime", lifetime)],
        }
    }

    /// `tpt_memory.alloc { size_bytes, scope }`
    ///
    /// `size_bytes` is the [`ValueId`] of the allocated buffer (or its size
    /// source) produced by the surrounding alloc-bearing operation.
    pub fn mem_alloc(size_bytes: ValueId, scope: &str) -> Operation {
        Operation {
            id: 0,
            op_name: OpName::new("tpt_memory", "alloc"),
            operands: vec![size_bytes],
            results: vec![],
            regions: vec![],
            attributes: vec![Attribute::string("scope", scope)],
        }
    }

    /// `tpt_memory.scope_end { lifetime }`
    pub fn scope_end(lifetime: &str) -> Operation {
        Operation {
            id: 0,
            op_name: OpName::new("tpt_memory", "scope_end"),
            operands: vec![],
            results: vec![],
            regions: vec![],
            attributes: vec![Attribute::string("lifetime", lifetime)],
        }
    }
}
