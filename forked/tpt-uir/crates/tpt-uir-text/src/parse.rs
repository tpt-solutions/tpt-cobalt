//! Recursive-descent parser for the TPT-UIR textual format.

use crate::lex::{lex, Tok};
use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::ir::{Block, Operation, Region};
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::quant::QuantizationParams;
use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType, Type};

use tpt_uir_core::OpId;
use tpt_uir_core::ValueId;

type PResult<T> = Result<T, String>;

pub struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Tok) -> PResult<()> {
        match self.next() {
            Some(ref t) if t == want => Ok(()),
            Some(t) => Err(format!("expected {:?}, found {:?}", want, t)),
            None => Err(format!("expected {:?}, found end of input", want)),
        }
    }

    fn expect_ident(&mut self) -> PResult<String> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            Some(t) => Err(format!("expected identifier, found {:?}", t)),
            None => Err("expected identifier, found end of input".to_string()),
        }
    }

    fn expect_int(&mut self) -> PResult<i64> {
        match self.next() {
            Some(Tok::Int(v)) => Ok(v),
            Some(t) => Err(format!("expected integer, found {:?}", t)),
            None => Err("expected integer, found end of input".to_string()),
        }
    }

    fn expect_str(&mut self) -> PResult<String> {
        match self.next() {
            Some(Tok::Str(s)) => Ok(s),
            Some(t) => Err(format!("expected string, found {:?}", t)),
            None => Err("expected string, found end of input".to_string()),
        }
    }

    // --- region / block / op ---

    fn parse_region(&mut self) -> PResult<Region> {
        let mut blocks = Vec::new();
        while let Some(Tok::Caret) = self.peek() {
            blocks.push(self.parse_block()?);
        }
        Ok(Region { blocks })
    }

    fn parse_block(&mut self) -> PResult<Block> {
        self.expect(&Tok::Caret)?;
        let _name = self.expect_ident()?;
        let mut arguments = Vec::new();
        if let Some(Tok::LParen) = self.peek() {
            self.next();
            if !matches!(self.peek(), Some(Tok::RParen)) {
                loop {
                    let vid = self.parse_value_ref()?;
                    self.expect(&Tok::Colon)?;
                    let ty = self.parse_type()?;
                    arguments.push((vid, ty));
                    match self.next() {
                        Some(Tok::Comma) => continue,
                        Some(Tok::RParen) => break,
                        other => {
                            return Err(format!(
                                "expected ',' or ')' in block args, found {:?}",
                                other
                            ))
                        }
                    }
                }
            } else {
                self.next(); // RParen
            }
        }
        self.expect(&Tok::Colon)?;
        let mut operations = Vec::new();
        while let Some(t) = self.peek() {
            if matches!(t, Tok::Caret) || matches!(t, Tok::RBracket) {
                break;
            }
            operations.push(self.parse_op()?);
        }
        Ok(Block {
            arguments,
            operations,
        })
    }

    fn parse_op(&mut self) -> PResult<Operation> {
        let mut id: OpId = 0;
        if let Some(Tok::Hash) = self.peek() {
            self.next();
            id = self.expect_int()? as OpId;
            self.expect(&Tok::Colon)?;
        }
        let mut results = Vec::new();
        if let Some(Tok::Percent) = self.peek() {
            results.push(self.parse_value_ref()?);
            // support `%a, %b =` multiple results (rare)
            while let Some(Tok::Comma) = self.peek() {
                self.next();
                results.push(self.parse_value_ref()?);
            }
            self.expect(&Tok::Equal)?;
        }
        let opname_str = self.expect_str()?;
        let op_name =
            OpName::parse(&opname_str).map_err(|_| format!("invalid op name '{}'", opname_str))?;
        self.expect(&Tok::LParen)?;
        let mut operands = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            loop {
                operands.push(self.parse_value_ref()?);
                match self.next() {
                    Some(Tok::Comma) => continue,
                    Some(Tok::RParen) => break,
                    other => {
                        return Err(format!(
                            "expected ',' or ')' in operands, found {:?}",
                            other
                        ))
                    }
                }
            }
        } else {
            self.next(); // RParen
        }
        let mut attributes = Vec::new();
        if let Some(Tok::LBrace) = self.peek() {
            self.next();
            if !matches!(self.peek(), Some(Tok::RBrace)) {
                loop {
                    attributes.push(self.parse_attribute()?);
                    match self.next() {
                        Some(Tok::Comma) => continue,
                        Some(Tok::RBrace) => break,
                        other => {
                            return Err(format!(
                                "expected ',' or '}}' in attributes, found {:?}",
                                other
                            ))
                        }
                    }
                }
            } else {
                self.next(); // RBrace
            }
        }
        let mut regions = Vec::new();
        while let Some(Tok::LBracket) = self.peek() {
            self.next();
            regions.push(self.parse_region()?);
            self.expect(&Tok::RBracket)?;
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

    fn parse_value_ref(&mut self) -> PResult<ValueId> {
        self.expect(&Tok::Percent)?;
        let v = self.expect_int()?;
        Ok(v as ValueId)
    }

    fn parse_attribute(&mut self) -> PResult<Attribute> {
        let key = self.expect_ident()?;
        self.expect(&Tok::Equal)?;
        let value = self.parse_attr_value()?;
        Ok(Attribute { key, value })
    }

    fn parse_attr_value(&mut self) -> PResult<AttributeValue> {
        match self.peek() {
            Some(Tok::Ident(s)) if s == "tensor" => {
                self.next();
                Ok(AttributeValue::Type(self.parse_tensor_body()?))
            }
            Some(Tok::Ident(s)) if s == "shape" => {
                self.next();
                Ok(AttributeValue::Shape(self.parse_shape()?))
            }
            Some(Tok::Ident(s)) if s == "quant" => {
                self.next();
                Ok(AttributeValue::Quantization(self.parse_quant_body()?))
            }
            Some(Tok::Ident(s)) if s == "opname" => {
                self.next();
                self.expect(&Tok::LParen)?;
                let s = self.expect_str()?;
                self.expect(&Tok::RParen)?;
                let op = OpName::parse(&s).map_err(|_| format!("invalid opname '{}'", s))?;
                Ok(AttributeValue::OpName(op))
            }
            Some(Tok::Str(_)) => Ok(AttributeValue::String(self.expect_str()?)),
            Some(Tok::Float(f)) => {
                let f = *f;
                self.next();
                Ok(AttributeValue::F64(f))
            }
            Some(Tok::Int(v)) => {
                let v = *v;
                self.next();
                Ok(AttributeValue::I64(v))
            }
            other => Err(format!("unexpected attribute value: {:?}", other)),
        }
    }

    // --- type / shape / quant ---

    fn parse_type(&mut self) -> PResult<Type> {
        match self.peek() {
            Some(Tok::Ident(s)) if s == "index" => {
                self.next();
                Ok(Type::Index)
            }
            Some(Tok::Ident(s)) if s == "tensor" => {
                self.next();
                self.parse_tensor_body()
            }
            Some(t) => Err(format!("expected type, found {:?}", t)),
            None => Err("expected type, found end of input".to_string()),
        }
    }

    fn parse_tensor_body(&mut self) -> PResult<Type> {
        self.expect(&Tok::Less)?;
        let dtype = self.parse_scalar()?;
        let mut shape = None;
        if let Some(Tok::Comma) = self.peek() {
            self.next();
            shape = Some(self.parse_shape()?);
        }
        self.expect(&Tok::Greater)?;
        Ok(Type::Tensor(TensorType { dtype, shape }))
    }

    fn parse_shape(&mut self) -> PResult<ShapeSpec> {
        let name = self.expect_ident()?;
        if name != "shape" {
            return Err(format!("expected 'shape', found '{}'", name));
        }
        self.parse_shape_body()
    }

    fn parse_shape_body(&mut self) -> PResult<ShapeSpec> {
        self.expect(&Tok::Less)?;
        let mut dimensions = Vec::new();
        if !matches!(self.peek(), Some(Tok::Greater)) {
            loop {
                dimensions.push(self.parse_dim()?);
                match self.next() {
                    Some(Tok::Comma) => continue,
                    Some(Tok::Greater) => break,
                    other => {
                        return Err(format!("expected ',' or '>' in shape, found {:?}", other))
                    }
                }
            }
        } else {
            self.next(); // Greater
        }
        Ok(ShapeSpec { dimensions })
    }

    fn parse_dim(&mut self) -> PResult<Dimension> {
        match self.peek() {
            Some(Tok::Int(v)) => {
                let v = *v;
                self.next();
                Ok(Dimension::Fixed(v as usize))
            }
            Some(Tok::Ident(name)) => {
                let name = name.clone();
                self.next();
                if let Some(Tok::Colon) = self.peek() {
                    self.next();
                    let max = self.expect_int()?;
                    Ok(Dimension::Bounded {
                        symbol: name,
                        max_value: max as usize,
                    })
                } else {
                    Ok(Dimension::Symbolic(name))
                }
            }
            other => Err(format!("expected dimension, found {:?}", other)),
        }
    }

    fn parse_quant_body(&mut self) -> PResult<QuantizationParams> {
        self.expect(&Tok::Less)?;
        let block_size = self.expect_int()? as u32;
        self.expect(&Tok::Comma)?;
        let bytes_per_block = self.expect_int()? as u32;
        let mut num_blocks = 0u32;
        let mut scales = None;
        if let Some(Tok::Comma) = self.peek() {
            self.next();
            // optional num_blocks, then optional scales
            if let Some(Tok::Int(_)) = self.peek() {
                num_blocks = self.expect_int()? as u32;
                if let Some(Tok::Comma) = self.peek() {
                    self.next();
                    scales = Some(self.expect_int()? as u32);
                }
            } else {
                scales = Some(self.expect_int()? as u32);
            }
        }
        self.expect(&Tok::Greater)?;
        Ok(QuantizationParams {
            block_size,
            bytes_per_block,
            num_blocks,
            scales,
        })
    }

    fn parse_scalar(&mut self) -> PResult<ScalarType> {
        let name = self.expect_ident()?;
        scalar_from_str(&name)
    }
}

fn scalar_from_str(s: &str) -> PResult<ScalarType> {
    use ScalarType::*;
    Ok(match s {
        "i8" => I8,
        "i16" => I16,
        "i32" => I32,
        "i64" => I64,
        "u8" => U8,
        "u16" => U16,
        "u32" => U32,
        "u64" => U64,
        "f16" => F16,
        "f32" => F32,
        "f64" => F64,
        "bf16" => BF16,
        "bool" => Bool,
        "q4_0" => Q4_0,
        "q4_1" => Q4_1,
        "q8_0" => Q8_0,
        _ => return Err(format!("unknown scalar type '{}'", s)),
    })
}

/// Parse a textual TPT-UIR region into a [`Region`].
pub fn parse(src: &str) -> Result<Region, String> {
    let toks = lex(src).map_err(|e| e.message)?;
    let mut p = Parser { toks, pos: 0 };
    let region = p.parse_region()?;
    if p.pos != p.toks.len() {
        return Err("trailing tokens after region".to_string());
    }
    Ok(region)
}
