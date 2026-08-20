#![doc = include_str!("../README.md")]

pub mod lex;
pub mod parse;
pub mod print;

pub use parse::parse as parse_text_raw;

use tpt_uir_core::ir::Region;

/// Error produced while parsing the textual TPT-UIR format.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError(pub String);

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TPT-UIR parse error: {}", self.0)
    }
}

/// Serialize a [`Region`] to the textual TPT-UIR format.
pub fn to_text(region: &Region) -> String {
    print::to_text(region)
}

/// Parse a textual TPT-UIR string into a [`Region`].
pub fn parse_text(src: &str) -> Result<Region, ParseError> {
    parse::parse(src).map_err(ParseError)
}

/// Wrapper that pretty-prints a [`Region`] via its `Display` implementation.
pub struct Pretty<'a>(pub &'a Region);

impl<'a> core::fmt::Display for Pretty<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", print::to_text(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_uir_core::attr::Attribute;
    use tpt_uir_core::op_name::OpName;
    use tpt_uir_core::quant::QuantizationParams;
    use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType, Type};
    use tpt_uir_core::{Block, Operation};

    fn sample() -> Region {
        Region {
            blocks: vec![Block {
                arguments: vec![
                    (
                        0u32,
                        Type::Tensor(TensorType {
                            dtype: ScalarType::F32,
                            shape: Some(ShapeSpec {
                                dimensions: vec![
                                    Dimension::Fixed(1),
                                    Dimension::Symbolic("n".into()),
                                    Dimension::Bounded {
                                        symbol: "m".into(),
                                        max_value: 1024,
                                    },
                                ],
                            }),
                        }),
                    ),
                    (1u32, Type::Index),
                ],
                operations: vec![
                    Operation {
                        id: 1,
                        op_name: OpName::parse("core.add").unwrap(),
                        operands: vec![0, 1],
                        results: vec![2],
                        regions: vec![],
                        attributes: vec![
                            Attribute::i64("axis", -3),
                            Attribute::f64("scale", 1.5),
                            Attribute::string("name", "acc"),
                            Attribute::quantization(
                                "q",
                                QuantizationParams {
                                    block_size: 32,
                                    bytes_per_block: 18,
                                    num_blocks: 4,
                                    scales: Some(1),
                                },
                            ),
                        ],
                    },
                    Operation {
                        id: 2,
                        op_name: OpName::parse("tpt_gpu.launch").unwrap(),
                        operands: vec![2],
                        results: vec![],
                        regions: vec![Region {
                            blocks: vec![Block {
                                arguments: vec![],
                                operations: vec![Operation {
                                    id: 9,
                                    op_name: OpName::parse("core.mul").unwrap(),
                                    operands: vec![2],
                                    results: vec![3],
                                    regions: vec![],
                                    attributes: vec![],
                                }],
                            }],
                        }],
                        attributes: vec![],
                    },
                ],
            }],
        }
    }

    #[test]
    fn roundtrip_region() {
        let region = sample();
        let text = to_text(&region);
        let back = parse_text(&text).expect("parse should succeed");
        assert_eq!(region, back, "round-trip mismatch:\n{}", text);
    }

    #[test]
    fn empty_region() {
        let region = Region {
            blocks: vec![Block {
                arguments: vec![],
                operations: vec![],
            }],
        };
        let text = to_text(&region);
        assert_eq!(parse_text(&text).unwrap(), region);
    }

    #[test]
    fn pretty_display() {
        let region = sample();
        let s = format!("{}", Pretty(&region));
        assert!(s.contains("^bb0"));
        assert!(s.contains("core.add"));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_text("not a region @@@").is_err());
    }
}
