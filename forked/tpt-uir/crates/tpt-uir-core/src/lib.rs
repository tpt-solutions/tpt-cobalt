#![no_std]
#![doc = include_str!("../README.md")]
extern crate alloc;

pub mod attr;
pub mod builder;
pub mod ir;
pub mod op_name;
pub mod quant;
pub mod types;

pub use attr::*;
pub use builder::*;
pub use ir::*;
pub use op_name::*;
pub use quant::*;
pub use types::*;

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn scalar_type_sizes() {
        use ScalarType::*;
        assert_eq!(I8.size_bytes(), 1);
        assert_eq!(I16.size_bytes(), 2);
        assert_eq!(I32.size_bytes(), 4);
        assert_eq!(I64.size_bytes(), 8);
        assert_eq!(U8.size_bytes(), 1);
        assert_eq!(U16.size_bytes(), 2);
        assert_eq!(U32.size_bytes(), 4);
        assert_eq!(U64.size_bytes(), 8);
        assert_eq!(F16.size_bytes(), 2);
        assert_eq!(F32.size_bytes(), 4);
        assert_eq!(F64.size_bytes(), 8);
        assert_eq!(BF16.size_bytes(), 2);
        assert_eq!(Bool.size_bytes(), 1);
        assert_eq!(Q8_0.size_bytes(), 1);
        // Packed 4-bit formats have no whole-byte per-element size.
        assert_eq!(Q4_0.size_bytes(), 0);
        assert_eq!(Q4_1.size_bytes(), 0);
    }

    #[test]
    fn opname_parse_roundtrip() {
        let n = OpName::parse("tpt_gpu.launch").unwrap();
        assert!(n.dialect == "tpt_gpu");
        assert!(n.op == "launch");
        assert_eq!(n.to_string(), "tpt_gpu.launch");

        assert!(OpName::parse("invalid").is_err());
        assert!(OpName::parse(".op").is_err());
        assert!(OpName::parse("d.").is_err());
        assert!(OpName::parse("").is_err());
    }

    #[test]
    fn ssa_validation_ok() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![(0u32, Type::Index)],
                operations: vec![Operation {
                    id: 1,
                    op_name: OpName::parse("core.add").unwrap(),
                    operands: vec![0],
                    results: vec![1],
                    regions: vec![],
                    attributes: vec![],
                }],
            }],
        };
        assert!(validate_region(&region).is_ok());
    }

    #[test]
    fn ssa_validation_error() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![Operation {
                    id: 1,
                    op_name: OpName::parse("core.add").unwrap(),
                    operands: vec![99],
                    results: vec![],
                    regions: vec![],
                    attributes: vec![],
                }],
            }],
        };
        let errs = validate_region(&region);
        assert!(errs.is_err());
        assert!(!errs.unwrap_err().is_empty());
    }
}
