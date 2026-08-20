#![no_std]
#![doc = include_str!("../README.md")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;

use postcard::{from_bytes, to_allocvec};
use tpt_uir_core::{Operation, Region};

/// Serialize a single operation to a postcard byte vector.
pub fn serialize_op(op: &Operation) -> Result<Vec<u8>, postcard::Error> {
    to_allocvec(op)
}

/// Deserialize a single operation from postcard bytes.
pub fn deserialize_op(bytes: &[u8]) -> Result<Operation, postcard::Error> {
    from_bytes(bytes)
}

/// Serialize a region to a postcard byte vector.
pub fn serialize_region(region: &Region) -> Result<Vec<u8>, postcard::Error> {
    to_allocvec(region)
}

/// Deserialize a region from postcard bytes.
pub fn deserialize_region(bytes: &[u8]) -> Result<Region, postcard::Error> {
    from_bytes(bytes)
}

/// Write a region to a `.tptuir` file.
#[cfg(feature = "std")]
pub fn write_tptuir<P: AsRef<std::path::Path>>(path: P, region: &Region) -> std::io::Result<()> {
    let bytes = serialize_region(region)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, bytes)
}

/// Read a region from a `.tptuir` file.
#[cfg(feature = "std")]
pub fn read_tptuir<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Region, std::boxed::Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let region = deserialize_region(&bytes)?;
    Ok(region)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec;
    use tpt_uir_core::attr::Attribute;
    use tpt_uir_core::op_name::OpName;
    use tpt_uir_core::types::Type;
    use tpt_uir_core::{Block, Operation, Region};

    fn sample_op() -> Operation {
        Operation {
            id: 1,
            op_name: OpName::parse("core.add").unwrap(),
            operands: vec![0, 1],
            results: vec![2],
            regions: vec![],
            attributes: vec![Attribute::i64("axis", 0)],
        }
    }

    fn sample_region() -> Region {
        Region {
            blocks: vec![Block {
                arguments: vec![(0u32, Type::Index), (1u32, Type::Index)],
                operations: vec![sample_op()],
            }],
        }
    }

    #[test]
    fn roundtrip_op() {
        let op = sample_op();
        let bytes = serialize_op(&op).unwrap();
        let back = deserialize_op(&bytes).unwrap();
        assert_eq!(op, back);
    }

    #[test]
    fn roundtrip_region() {
        let region = sample_region();
        let bytes = serialize_region(&region).unwrap();
        let back = deserialize_region(&bytes).unwrap();
        assert_eq!(region, back);
    }

    #[test]
    fn roundtrip_region_with_nested_block() {
        let inner = Operation {
            id: 9,
            op_name: OpName::parse("core.mul").unwrap(),
            operands: vec![0],
            results: vec![3],
            regions: vec![],
            attributes: vec![],
        };
        let outer = Operation {
            id: 2,
            op_name: OpName::parse("core.store").unwrap(),
            operands: vec![2],
            results: vec![],
            regions: vec![Region {
                blocks: vec![Block {
                    arguments: vec![],
                    operations: vec![inner],
                }],
            }],
            attributes: vec![],
        };
        let region = Region {
            blocks: vec![Block {
                arguments: vec![(0u32, Type::Index), (1u32, Type::Index)],
                operations: vec![sample_op(), outer],
            }],
        };
        let bytes = serialize_region(&region).unwrap();
        let back = deserialize_region(&bytes).unwrap();
        assert_eq!(region, back);
    }

    #[cfg(feature = "std")]
    #[test]
    fn roundtrip_file() {
        let region = sample_region();
        let path = std::env::temp_dir().join("tpt_uir_test.tptuir");
        write_tptuir(&path, &region).unwrap();
        let back = read_tptuir(&path).unwrap();
        assert_eq!(region, back);
        let _ = std::fs::remove_file(&path);
    }

    // Wire-compatibility regression test.
    //
    // The `Quantization` variant was appended as the LAST `AttributeValue`
    // variant. Postcard encodes enum variants by numeric index, so the indices
    // of every pre-existing variant (I64=0 ... OpName=5) must stay fixed. This
    // fixture encodes `AttributeValue::OpName("tpt_gpu.launch")` using only
    // those pre-change indices (variant 5 = 0x05, then two length-prefixed
    // postcard strings). It must still decode on the current code.
    #[test]
    fn wire_compat_attr_value_opname() {
        // 0x05 = variant index 5 (OpName)
        // 0x07 + "tpt_gpu", 0x06 + "launch"
        let fixture: &[u8] = &[
            0x05, 0x07, b't', b'p', b't', b'_', b'g', b'p', b'u', 0x06, b'l', b'a', b'u', b'n',
            b'c', b'h',
        ];
        let decoded: tpt_uir_core::attr::AttributeValue = postcard::from_bytes(fixture).unwrap();
        assert_eq!(
            decoded,
            tpt_uir_core::attr::AttributeValue::OpName(
                tpt_uir_core::OpName::parse("tpt_gpu.launch").unwrap()
            )
        );

        // Round-trips to identical bytes (proves the index is stable).
        let reencoded = postcard::to_allocvec(&decoded).unwrap();
        assert_eq!(reencoded, fixture);
    }
}
