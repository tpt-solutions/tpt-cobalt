extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::types::{Dimension, ShapeSpec, Type};
use tpt_uir_core::{OpId, Operation, Region, ValueId};

pub use tpt_uir_core::op_name::{
    TPT_GPU_ALLOC, TPT_GPU_BARRIER, TPT_GPU_DEALLOC, TPT_GPU_LAUNCH, TPT_GPU_THREAD_ID,
};

/// Marker type for the GPU dialect.
pub struct GpuDialect;

/// Builder helpers for GPU-dialect operations.
///
/// The GPU dialect requires that tensor shapes use `Dimension::Bounded` or are
/// absent (`None`). It never uses `Fixed` or `Symbolic` dimensions.
pub struct GpuOp;

impl GpuOp {
    /// Construct a GPU operation, validating that any shape-bearing attribute
    /// uses `Dimension::Bounded` (or is absent).
    pub fn build(
        id: OpId,
        op_name: OpName,
        operands: Vec<ValueId>,
        results: Vec<ValueId>,
        attributes: Vec<Attribute>,
        regions: Vec<Region>,
    ) -> Result<Operation, String> {
        for attr in &attributes {
            if let AttributeValue::Shape(shape) = &attr.value {
                if !shape_all_bounded(shape) {
                    return Err(format!(
                        "GPU dialect requires shapes to be `None` or `Dimension::Bounded`; \
                         invalid dimension found in attribute `{}`",
                        attr.key
                    ));
                }
            }
        }
        Ok(Operation {
            id,
            op_name,
            operands,
            results,
            regions,
            attributes,
        })
    }
}

fn shape_all_bounded(shape: &ShapeSpec) -> bool {
    shape
        .dimensions
        .iter()
        .all(|d| matches!(d, Dimension::Bounded { .. }))
}

/// Validate that a region only uses `None` or `Bounded` dimensions (GPU invariant).
pub fn validate_gpu_region(region: &Region) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    for block in &region.blocks {
        for (vid, ty) in &block.arguments {
            if let Type::Tensor(tt) = ty {
                if let Some(ref shape) = tt.shape {
                    if !shape_all_bounded(shape) {
                        errors.push(format!(
                            "block argument {:?} has a non-Bounded tensor shape",
                            vid
                        ));
                    }
                }
            }
        }
        for op in &block.operations {
            for attr in &op.attributes {
                if let AttributeValue::Shape(ref shape) = attr.value {
                    if !shape_all_bounded(shape) {
                        errors.push(format!(
                            "operation {} has a non-Bounded shape attribute `{}`",
                            op.id, attr.key
                        ));
                    }
                }
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_uir_core::types::{ScalarType, ShapeSpec, TensorType};
    use tpt_uir_core::{Block, OpName, Region, Type};

    #[test]
    fn gpu_rejects_fixed_shape() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![(
                    0u32,
                    Type::Tensor(TensorType {
                        dtype: ScalarType::F32,
                        shape: Some(ShapeSpec {
                            dimensions: vec![Dimension::Fixed(4096)],
                        }),
                    }),
                )],
                operations: vec![],
            }],
        };
        assert!(validate_gpu_region(&region).is_err());
    }

    #[test]
    fn gpu_accepts_bounded_shape() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![(
                    0u32,
                    Type::Tensor(TensorType {
                        dtype: ScalarType::F32,
                        shape: Some(ShapeSpec {
                            dimensions: vec![Dimension::Bounded {
                                symbol: "n".into(),
                                max_value: 4096,
                            }],
                        }),
                    }),
                )],
                operations: vec![],
            }],
        };
        assert!(validate_gpu_region(&region).is_ok());
    }

    #[test]
    fn gpu_build_rejects_bad_shape_attr() {
        let bad = GpuOp::build(
            1,
            OpName::parse("tpt_gpu.launch").unwrap(),
            vec![],
            vec![],
            vec![Attribute::shape(
                "shape",
                ShapeSpec {
                    dimensions: vec![Dimension::Fixed(4)],
                },
            )],
            vec![],
        );
        assert!(bad.is_err());
    }
}
