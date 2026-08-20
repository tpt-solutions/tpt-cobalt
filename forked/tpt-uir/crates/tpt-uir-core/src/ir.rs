use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::attr::Attribute;
use crate::op_name::OpName;
use crate::types::Type;

/// Unique identifier for a value (an SSA register or graph edge).
pub type ValueId = u32;

/// Unique identifier for an operation (an SSA op or graph node).
pub type OpId = u32;

/// The fundamental unit of computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Operation {
    pub id: OpId,
    /// Namespaced operation name (e.g. `"tpt_gpu.add_f16"`).
    pub op_name: OpName,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    /// Nested control flow or scoping (e.g. loops, memory scopes).
    pub regions: Vec<Region>,
    pub attributes: Vec<Attribute>,
}

/// A sequence of operations with block arguments (SSA phi nodes / graph inputs).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Block {
    pub arguments: Vec<(ValueId, Type)>,
    pub operations: Vec<Operation>,
}

/// A collection of blocks (a function, loop body, or memory scope).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Region {
    pub blocks: Vec<Block>,
}

/// Validate SSA well-formedness of a region.
///
/// Every [`ValueId`] used as an operand of an operation must be defined either
/// by a block argument of the enclosing block or by a `result` of a preceding
/// operation within that block. Nested regions are validated independently.
/// All violations are collected and returned (not just the first).
pub fn validate_region(region: &Region) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    validate_region_inner(region, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_region_inner(region: &Region, errors: &mut Vec<String>) {
    for (bi, block) in region.blocks.iter().enumerate() {
        let mut defined: BTreeSet<ValueId> = BTreeSet::new();
        for (vid, _ty) in &block.arguments {
            defined.insert(*vid);
        }
        for (oi, op) in block.operations.iter().enumerate() {
            for operand in &op.operands {
                if !defined.contains(operand) {
                    errors.push(format!(
                        "block {}: operation {} (id {}) references undefined value {:?}",
                        bi, oi, op.id, operand
                    ));
                }
            }
            for result in &op.results {
                defined.insert(*result);
            }
            for nested in &op.regions {
                validate_region_inner(nested, errors);
            }
        }
    }
}
