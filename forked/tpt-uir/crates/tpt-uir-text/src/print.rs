//! Pretty-printing of the TPT-UIR IR into the textual format.

use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::ir::{Block, Operation, Region};
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::quant::QuantizationParams;
use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, Type};
use tpt_uir_core::ValueId;

fn scalar_name(s: ScalarType) -> &'static str {
    use ScalarType::*;
    match s {
        I8 => "i8",
        I16 => "i16",
        I32 => "i32",
        I64 => "i64",
        U8 => "u8",
        U16 => "u16",
        U32 => "u32",
        U64 => "u64",
        F16 => "f16",
        F32 => "f32",
        F64 => "f64",
        BF16 => "bf16",
        Bool => "bool",
        Q4_0 => "q4_0",
        Q4_1 => "q4_1",
        Q8_0 => "q8_0",
    }
}

fn print_dim(d: &Dimension) -> String {
    match d {
        Dimension::Fixed(n) => n.to_string(),
        Dimension::Symbolic(name) => name.clone(),
        Dimension::Bounded { symbol, max_value } => format!("{}:{}", symbol, max_value),
    }
}

fn print_shape(sh: &ShapeSpec) -> String {
    let inner = sh
        .dimensions
        .iter()
        .map(print_dim)
        .collect::<Vec<_>>()
        .join(", ");
    format!("shape<{}>", inner)
}

fn print_type(t: &Type) -> String {
    match t {
        Type::Index => "index".to_string(),
        Type::Scalar(s) => scalar_name(*s).to_string(),
        Type::Tensor(tt) => {
            let mut s = format!("tensor<{}", scalar_name(tt.dtype));
            if let Some(sh) = &tt.shape {
                s.push_str(", ");
                s.push_str(&print_shape(sh));
            }
            s.push('>');
            s
        }
    }
}

fn print_quant(q: &QuantizationParams) -> String {
    match q.scales {
        Some(s) => format!(
            "quant<{}, {}, {}, {}>",
            q.block_size, q.bytes_per_block, q.num_blocks, s
        ),
        None => format!(
            "quant<{}, {}, {}>",
            q.block_size, q.bytes_per_block, q.num_blocks
        ),
    }
}

fn print_attr_value(v: &AttributeValue) -> String {
    match v {
        AttributeValue::I64(x) => x.to_string(),
        AttributeValue::F64(x) => x.to_string(),
        AttributeValue::String(s) => format!("{:?}", s),
        AttributeValue::Type(t) => print_type(t),
        AttributeValue::Shape(sh) => print_shape(sh),
        AttributeValue::OpName(op) => format!("opname(\"{}\")", op),
        AttributeValue::Quantization(q) => print_quant(q),
    }
}

fn print_attribute(a: &Attribute) -> String {
    format!("{} = {}", a.key, print_attr_value(&a.value))
}

fn print_value(v: ValueId) -> String {
    format!("%{}", v)
}

fn print_op(op: &Operation, indent: &str) -> String {
    let mut s = String::new();
    s.push_str(indent);
    s.push_str(&format!("#{}: ", op.id));
    if !op.results.is_empty() {
        let res = op
            .results
            .iter()
            .map(|r| print_value(*r))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&res);
        s.push_str(" = ");
    }
    s.push_str(&format!("\"{}\"", OpName::to_string(&op.op_name)));
    s.push('(');
    let ops = op
        .operands
        .iter()
        .map(|o| print_value(*o))
        .collect::<Vec<_>>()
        .join(", ");
    s.push_str(&ops);
    s.push(')');
    if !op.attributes.is_empty() {
        let attrs = op
            .attributes
            .iter()
            .map(print_attribute)
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(" {{ {} }}", attrs));
    }
    // Nested regions.
    for region in &op.regions {
        s.push_str(" [\n");
        s.push_str(&print_region(region, &format!("{}  ", indent)));
        s.push_str(indent);
        s.push(']');
    }
    s
}

fn print_block(block: &Block, idx: usize, indent: &str) -> String {
    let mut s = String::new();
    s.push_str(indent);
    s.push_str(&format!("^bb{}", idx));
    if !block.arguments.is_empty() {
        let args = block
            .arguments
            .iter()
            .map(|(v, t)| format!("{}: {}", print_value(*v), print_type(t)))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!("({})", args));
    }
    s.push_str(":\n");
    for op in &block.operations {
        s.push_str(&print_op(op, &format!("{}  ", indent)));
        s.push('\n');
    }
    s
}

fn print_region(region: &Region, indent: &str) -> String {
    let mut s = String::new();
    for (i, block) in region.blocks.iter().enumerate() {
        s.push_str(&print_block(block, i, indent));
    }
    s
}

/// Render a [`Region`] as a textual TPT-UIR string.
pub fn to_text(region: &Region) -> String {
    print_region(region, "")
}

/// Wraps a [`Region`] for `Display` (pretty-printing).
pub struct Pretty<'a>(pub &'a Region);

impl<'a> core::fmt::Display for Pretty<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", to_text(self.0))
    }
}

/// Parse `src` into a [`Region`] (convenience re-export).
pub fn parse_text(src: &str) -> Result<Region, String> {
    crate::parse::parse(src)
}
