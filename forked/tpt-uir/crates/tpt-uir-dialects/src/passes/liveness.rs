extern crate alloc;

use alloc::vec::Vec;

use tpt_uir_core::op_name as core_op_name;
use tpt_uir_core::{Operation, Region, ValueId};

use crate::memory::MemOp;

fn is_alloc_op(op_name: &tpt_uir_core::op_name::OpName) -> bool {
    let s = op_name.to_string();
    s == core_op_name::TPT_GPU_ALLOC || s == core_op_name::TPT_MEMORY_ALLOC
}

fn max_op_id(region: &Region) -> u32 {
    let mut max = 0u32;
    for block in &region.blocks {
        for op in &block.operations {
            if op.id > max {
                max = op.id;
            }
        }
    }
    max
}

/// Wrap alloc-bearing operations in an explicit memory scope.
///
/// For every block that contains at least one alloc operation (currently
/// `tpt_gpu.alloc` or `tpt_memory.alloc`), this inserts a `mem.scope_begin`
/// before the first operation, a `mem.alloc` immediately after each alloc
/// operation (referencing that operation's first result value), and a
/// `mem.scope_end` after the last operation. All injected ops are assigned
/// fresh, unique [`tpt_uir_core::OpId`]s.
pub fn inject_memory_ops(region: &mut Region, scope_name: &str) {
    let mut counter = max_op_id(region).saturating_add(1);
    let scope = scope_name.to_string();

    for block in &mut region.blocks {
        let has_alloc = block.operations.iter().any(|op| is_alloc_op(&op.op_name));
        if !has_alloc {
            continue;
        }

        let mut new_ops: Vec<Operation> = Vec::new();

        let mut begin = MemOp::scope_begin(&scope);
        begin.id = counter;
        counter = counter.saturating_add(1);
        new_ops.push(begin);

        for op in block.operations.iter() {
            new_ops.push(op.clone());
            if is_alloc_op(&op.op_name) {
                let size_vid: ValueId = op.results.first().copied().unwrap_or(0);
                let mut alloc = MemOp::mem_alloc(size_vid, &scope);
                alloc.id = counter;
                counter = counter.saturating_add(1);
                new_ops.push(alloc);
            }
        }

        let mut end = MemOp::scope_end(&scope);
        end.id = counter;
        counter = counter.saturating_add(1);
        new_ops.push(end);

        block.operations = new_ops;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_uir_core::{Block, OpName};

    #[test]
    fn injects_memory_ops_around_alloc() {
        let alloc_op = Operation {
            id: 1,
            op_name: OpName::parse("tpt_gpu.alloc").unwrap(),
            operands: vec![],
            results: vec![10],
            regions: vec![],
            attributes: vec![],
        };
        let other = Operation {
            id: 2,
            op_name: OpName::parse("core.add").unwrap(),
            operands: vec![10],
            results: vec![11],
            regions: vec![],
            attributes: vec![],
        };
        let mut region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![alloc_op, other],
            }],
        };

        inject_memory_ops(&mut region, "layer_1");

        let ops = &region.blocks[0].operations;
        assert_eq!(ops.len(), 5);
        assert!(ops[0].op_name.to_string() == core_op_name::TPT_MEMORY_SCOPE_BEGIN);
        assert_eq!(ops[1].id, 1);
        assert!(ops[2].op_name.to_string() == core_op_name::TPT_MEMORY_ALLOC);
        assert_eq!(ops[2].operands, vec![10]);
        assert_eq!(ops[3].id, 2);
        assert!(ops[4].op_name.to_string() == core_op_name::TPT_MEMORY_SCOPE_END);

        // All ops have unique ids; the original ops (1, 2) are preserved and the
        // three injected ops (scope_begin, alloc, scope_end) got fresh ids.
        let mut ids: Vec<u32> = ops.iter().map(|o| o.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(ops[0].id, 3);
        assert_eq!(ops[2].id, 4);
        assert_eq!(ops[4].id, 5);
    }

    #[test]
    fn no_alloc_no_injection() {
        let other = Operation {
            id: 2,
            op_name: OpName::parse("core.add").unwrap(),
            operands: vec![],
            results: vec![],
            regions: vec![],
            attributes: vec![],
        };
        let mut region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![other.clone()],
            }],
        };
        inject_memory_ops(&mut region, "layer_1");
        assert_eq!(region.blocks[0].operations.len(), 1);
    }
}
