extern crate alloc;

use alloc::vec::Vec;

use tpt_uir_core::Region;

use crate::crucible::validate_crucible_region;
use crate::gpu::validate_gpu_region;

/// Trait implemented by dialect marker types to validate a [`Region`] against
/// the invariants of that dialect.
///
/// In addition to the dialect-specific [`ValidateDialect::validate`] routine,
/// the trait provides default helpers for verifying dialect name-prefix and
/// version attributes. See [`ValidateDialect::validate_all`] to run the full
/// suite of checks at once.
pub trait ValidateDialect {
    /// Dialect name prefix (e.g. `"tpt_gpu"`). Every op in the region must
    /// belong to this dialect.
    fn dialect_prefix() -> &'static str;

    /// Dialect-specific structural validation (shapes, invariants, etc.).
    fn validate(region: &Region) -> Result<(), Vec<String>>;

    /// The current dialect version emitted by this crate. Consumers attaching
    /// `dialect_version` / `op_dialect_version` attributes should pin to this.
    fn current_version() -> i64 {
        1
    }

    /// Verify that every operation's dialect matches [`ValidateDialect::dialect_prefix`]
    /// and that any version attributes are non-negative.
    fn validate_versions(region: &Region) -> Result<(), Vec<String>> {
        let prefix = Self::dialect_prefix();
        let mut errors: Vec<String> = Vec::new();
        for block in &region.blocks {
            for op in &block.operations {
                if !op.op_name.dialect.starts_with(prefix) {
                    errors.push(format!(
                        "operation {} has dialect '{}' which does not match expected prefix '{}'",
                        op.id, op.op_name.dialect, prefix
                    ));
                }
                for attr in &op.attributes {
                    if attr.key == "dialect_version" || attr.key == "op_dialect_version" {
                        if let tpt_uir_core::attr::AttributeValue::I64(v) = attr.value {
                            if v < 0 {
                                errors.push(format!(
                                    "operation {} has negative {} attribute",
                                    op.id, attr.key
                                ));
                            }
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

    /// Run both [`ValidateDialect::validate`] and
    /// [`ValidateDialect::validate_versions`], accumulating every violation.
    fn validate_all(region: &Region) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();
        if let Err(e) = Self::validate(region) {
            errors.extend(e);
        }
        if let Err(e) = Self::validate_versions(region) {
            errors.extend(e);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl ValidateDialect for crate::gpu::GpuDialect {
    fn dialect_prefix() -> &'static str {
        "tpt_gpu"
    }

    fn validate(region: &Region) -> Result<(), Vec<String>> {
        validate_gpu_region(region)
    }
}

impl ValidateDialect for crate::crucible::CrucibleDialect {
    fn dialect_prefix() -> &'static str {
        "tpt_crucible"
    }

    fn validate(region: &Region) -> Result<(), Vec<String>> {
        validate_crucible_region(region)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crucible::CrucibleDialect;
    use crate::gpu::{GpuDialect, TPT_GPU_LAUNCH};
    use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType};
    use tpt_uir_core::{Block, Region, Type};

    #[test]
    fn gpu_dialect_trait_rejects_fixed() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![(
                    0u32,
                    Type::Tensor(TensorType {
                        dtype: ScalarType::F32,
                        shape: Some(ShapeSpec {
                            dimensions: vec![Dimension::Fixed(4)],
                        }),
                    }),
                )],
                operations: vec![],
            }],
        };
        assert!(GpuDialect::validate(&region).is_err());
    }

    #[test]
    fn crucible_dialect_trait_accepts_fixed() {
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
        assert!(CrucibleDialect::validate(&region).is_ok());
    }

    #[test]
    fn validate_versions_rejects_foreign_dialect() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![tpt_uir_core::Operation {
                    id: 1,
                    op_name: tpt_uir_core::OpName::parse("core.add").unwrap(),
                    operands: vec![],
                    results: vec![],
                    regions: vec![],
                    attributes: vec![],
                }],
            }],
        };
        assert!(GpuDialect::validate_versions(&region).is_err());
        assert!(GpuDialect::validate_all(&region).is_err());
    }

    #[test]
    fn validate_versions_accepts_matching_dialect() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![tpt_uir_core::Operation {
                    id: 1,
                    op_name: tpt_uir_core::OpName::parse(TPT_GPU_LAUNCH).unwrap(),
                    operands: vec![],
                    results: vec![],
                    regions: vec![],
                    attributes: vec![tpt_uir_core::attr::Attribute::dialect_version(1)],
                }],
            }],
        };
        assert!(GpuDialect::validate_versions(&region).is_ok());
        assert!(GpuDialect::validate_all(&region).is_ok());
    }

    #[test]
    fn validate_versions_rejects_negative_version() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![tpt_uir_core::Operation {
                    id: 1,
                    op_name: tpt_uir_core::OpName::parse(TPT_GPU_LAUNCH).unwrap(),
                    operands: vec![],
                    results: vec![],
                    regions: vec![],
                    attributes: vec![tpt_uir_core::attr::Attribute::op_dialect_version(-1)],
                }],
            }],
        };
        assert!(GpuDialect::validate_versions(&region).is_err());
    }
}
