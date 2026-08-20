extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::types::{Dimension, Type};
use tpt_uir_core::{OpId, Operation, Region, ValueId};

pub use tpt_uir_core::op_name::{
    TPT_CRUCIBLE_ANALOG_CONV, TPT_CRUCIBLE_MAP_FLASH, TPT_CRUCIBLE_ROUTE_FPGA,
};

/// Marker type for the Crucible dialect.
pub struct CrucibleDialect;

/// Builder helpers for Crucible-dialect operations.
pub struct CrucibleOp;

impl CrucibleOp {
    pub fn build(
        id: OpId,
        op_name: OpName,
        operands: Vec<ValueId>,
        results: Vec<ValueId>,
        attributes: Vec<Attribute>,
        regions: Vec<Region>,
    ) -> Operation {
        Operation {
            id,
            op_name,
            operands,
            results,
            regions,
            attributes,
        }
    }
}

/// Validate that a Crucible region uses only `Dimension::Fixed` dimensions.
///
/// Crucible requires strictly static shapes, so any `Symbolic` or `Bounded`
/// dimension is rejected.
pub fn validate_crucible_region(region: &Region) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    for block in &region.blocks {
        for (vid, ty) in &block.arguments {
            if let Type::Tensor(tt) = ty {
                if let Some(ref shape) = tt.shape {
                    if let Some(bad) = shape
                        .dimensions
                        .iter()
                        .find(|d| !matches!(d, Dimension::Fixed(_)))
                    {
                        errors.push(format!(
                            "block argument {:?} has non-Fixed dimension {:?} \
                             (Crucible requires Fixed)",
                            vid, bad
                        ));
                    }
                }
            }
        }
        for op in &block.operations {
            for attr in &op.attributes {
                if let AttributeValue::Shape(shape) = &attr.value {
                    if let Some(bad) = shape
                        .dimensions
                        .iter()
                        .find(|d| !matches!(d, Dimension::Fixed(_)))
                    {
                        errors.push(format!(
                            "operation {} has non-Fixed dimension {:?} in shape attribute `{}`",
                            op.id, bad, attr.key
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

/// Hardware routing attribute helper.
pub fn target_ip_block(block_name: &str) -> Attribute {
    Attribute::string("target_ip_block", block_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_uir_core::attr::AttributeValue;
    use tpt_uir_core::types::{ScalarType, ShapeSpec, TensorType};
    use tpt_uir_core::{Block, OpName, Region, Type};

    #[test]
    fn crucible_rejects_symbolic_and_bounded() {
        for dim in [
            Dimension::Symbolic("b".into()),
            Dimension::Bounded {
                symbol: "n".into(),
                max_value: 8,
            },
        ] {
            let region = Region {
                blocks: vec![Block {
                    arguments: vec![(
                        0u32,
                        Type::Tensor(TensorType {
                            dtype: ScalarType::F32,
                            shape: Some(ShapeSpec {
                                dimensions: vec![dim.clone()],
                            }),
                        }),
                    )],
                    operations: vec![],
                }],
            };
            assert!(validate_crucible_region(&region).is_err());
        }
    }

    #[test]
    fn crucible_accepts_fixed() {
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
        assert!(validate_crucible_region(&region).is_ok());
    }

    #[test]
    fn target_ip_block_attr() {
        let a = target_ip_block("dsp_slice_4");
        assert!(a.key == "target_ip_block");
        assert_eq!(a.value, AttributeValue::String("dsp_slice_4".to_string()));
    }

    #[test]
    fn op_name_constants_roundtrip() {
        assert_eq!(
            OpName::parse(TPT_CRUCIBLE_MAP_FLASH).unwrap().to_string(),
            TPT_CRUCIBLE_MAP_FLASH
        );
    }
}
