extern crate alloc;

use alloc::vec::Vec;

use crate::attr::Attribute;
use crate::ir::{Block, Operation, Region};
use crate::op_name::OpName;
use crate::types::Type;

use crate::ir::{OpId, ValueId};

/// Fluent builder for a single [`Operation`].
///
/// `op_name` must be supplied (via [`OpBuilder::name`]) before
/// [`OpBuilder::build`] is called, or `build` panics.
pub struct OpBuilder {
    id: OpId,
    op_name: Option<OpName>,
    operands: Vec<ValueId>,
    results: Vec<ValueId>,
    regions: Vec<Region>,
    attributes: Vec<Attribute>,
}

impl OpBuilder {
    pub fn new(id: OpId) -> Self {
        OpBuilder {
            id,
            op_name: None,
            operands: Vec::new(),
            results: Vec::new(),
            regions: Vec::new(),
            attributes: Vec::new(),
        }
    }

    pub fn name(mut self, op_name: OpName) -> Self {
        self.op_name = Some(op_name);
        self
    }

    pub fn operand(mut self, value: ValueId) -> Self {
        self.operands.push(value);
        self
    }

    pub fn result(mut self, value: ValueId) -> Self {
        self.results.push(value);
        self
    }

    pub fn region(mut self, region: Region) -> Self {
        self.regions.push(region);
        self
    }

    pub fn attribute(mut self, attribute: Attribute) -> Self {
        self.attributes.push(attribute);
        self
    }

    pub fn build(self) -> Operation {
        let op_name = self
            .op_name
            .unwrap_or_else(|| panic!("OpBuilder::build requires an op_name (id {})", self.id));
        Operation {
            id: self.id,
            op_name,
            operands: self.operands,
            results: self.results,
            regions: self.regions,
            attributes: self.attributes,
        }
    }
}

/// Fluent builder for a [`Block`].
pub struct BlockBuilder {
    arguments: Vec<(ValueId, Type)>,
    operations: Vec<Operation>,
}

impl BlockBuilder {
    pub fn new() -> Self {
        BlockBuilder {
            arguments: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn arg(mut self, value: ValueId, ty: Type) -> Self {
        self.arguments.push((value, ty));
        self
    }

    pub fn op(mut self, operation: Operation) -> Self {
        self.operations.push(operation);
        self
    }

    pub fn build(self) -> Block {
        Block {
            arguments: self.arguments,
            operations: self.operations,
        }
    }
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for a [`Region`].
pub struct RegionBuilder {
    blocks: Vec<Block>,
}

impl RegionBuilder {
    pub fn new() -> Self {
        RegionBuilder { blocks: Vec::new() }
    }

    pub fn block(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }

    pub fn build(self) -> Region {
        Region {
            blocks: self.blocks,
        }
    }
}

impl Default for RegionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::attr::Attribute;
    use crate::op_name::OpName;
    use crate::types::Type;
    use alloc::vec;

    #[test]
    fn builds_region_block_op() {
        let region = RegionBuilder::new()
            .block(
                BlockBuilder::new()
                    .arg(0, Type::Index)
                    .op(OpBuilder::new(1)
                        .name(OpName::parse("core.add").unwrap())
                        .operand(0)
                        .result(1)
                        .attribute(Attribute::i64("axis", 0))
                        .build())
                    .build(),
            )
            .build();

        assert_eq!(region.blocks.len(), 1);
        let block = &region.blocks[0];
        assert_eq!(block.arguments.len(), 1);
        assert_eq!(block.operations.len(), 1);
        let op = &block.operations[0];
        assert_eq!(op.id, 1);
        assert_eq!(op.operands, vec![0]);
        assert_eq!(op.results, vec![1]);
        assert_eq!(op.attributes.len(), 1);
    }

    #[test]
    #[should_panic(expected = "op_name")]
    fn op_builder_requires_name() {
        let _ = OpBuilder::new(7).build();
    }
}
