use std::fmt;
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Constant,
    Generic,
}
impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Shared => write!(f, "shared"),
            Self::Local => write!(f, "local"),
            Self::Constant => write!(f, "constant"),
            Self::Generic => write!(f, "generic"),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    Index,
    Tensor(Vec<i64>, Box<Type>, AddressSpace),
    Vector(u32, Box<Type>),
    MemRef(Vec<i64>, Box<Type>, AddressSpace),
    Function(Vec<Type>, Vec<Type>),
    None,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub kind: TypeKind,
}
impl Type {
    pub fn primitive(name: &str) -> Self {
        let kind = match name {
            "i1" => TypeKind::I1,
            "i8" => TypeKind::I8,
            "i16" => TypeKind::I16,
            "i32" => TypeKind::I32,
            "i64" => TypeKind::I64,
            "f16" => TypeKind::F16,
            "bf16" => TypeKind::BF16,
            "f32" => TypeKind::F32,
            "f64" => TypeKind::F64,
            "index" => TypeKind::Index,
            _ => TypeKind::None,
        };
        Type { kind }
    }
    pub fn tensor(shape: Vec<i64>, el: Type, addr: AddressSpace) -> Self {
        Type {
            kind: TypeKind::Tensor(shape, Box::new(el), addr),
        }
    }
    pub fn vector(lanes: u32, el: Type) -> Self {
        Type {
            kind: TypeKind::Vector(lanes, Box::new(el)),
        }
    }
    pub fn memref(shape: Vec<i64>, el: Type, addr: AddressSpace) -> Self {
        Type {
            kind: TypeKind::MemRef(shape, Box::new(el), addr),
        }
    }
}
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            TypeKind::I1 => write!(f, "i1"),
            TypeKind::I8 => write!(f, "i8"),
            TypeKind::I16 => write!(f, "i16"),
            TypeKind::I32 => write!(f, "i32"),
            TypeKind::I64 => write!(f, "i64"),
            TypeKind::F16 => write!(f, "f16"),
            TypeKind::BF16 => write!(f, "bf16"),
            TypeKind::F32 => write!(f, "f32"),
            TypeKind::F64 => write!(f, "f64"),
            TypeKind::Index => write!(f, "index"),
            TypeKind::Tensor(s, e, addr) => {
                let v: Vec<String> = s
                    .iter()
                    .map(|d| if *d < 0 { "*".into() } else { d.to_string() })
                    .collect();
                write!(f, "tensor<{}x{}", v.join("x"), e)?;
                if *addr != AddressSpace::Global {
                    write!(f, ", {}", addr)?;
                }
                write!(f, ">")
            }
            TypeKind::Vector(l, e) => write!(f, "vector<{}x{}>", l, e),
            TypeKind::MemRef(s, e, addr) => {
                let v: Vec<String> = s
                    .iter()
                    .map(|d| if *d < 0 { "*".into() } else { d.to_string() })
                    .collect();
                write!(f, "memref<{}x{}", v.join("x"), e)?;
                if *addr != AddressSpace::Global {
                    write!(f, ", {}", addr)?;
                }
                write!(f, ">")
            }
            TypeKind::Function(i, o) => {
                let a: Vec<String> = i.iter().map(|t| t.to_string()).collect();
                let b: Vec<String> = o.iter().map(|t| t.to_string()).collect();
                write!(f, "({}) -> ({})", a.join(", "), b.join(", "))
            }
            TypeKind::None => write!(f, "none"),
        }
    }
}
#[derive(Debug, Clone)]
pub enum OpKind {
    Addi,
    Subi,
    Muli,
    Addf,
    Subf,
    Mulf,
    And,
    Or,
    Xor,
    CmpEq,
    CmpLt,
    Load,
    Store,
    Branch,
    Return,
    Constant(String),
    Custom(String),
    Gemm,
    Quantize,
    Dequantize,
    QuantGemm,
    QuantAttention,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Addi => write!(f, "addi"),
            OpKind::Subi => write!(f, "subi"),
            OpKind::Muli => write!(f, "muli"),
            OpKind::Addf => write!(f, "addf"),
            OpKind::Subf => write!(f, "subf"),
            OpKind::Mulf => write!(f, "mulf"),
            OpKind::And => write!(f, "andi"),
            OpKind::Or => write!(f, "ori"),
            OpKind::Xor => write!(f, "xori"),
            OpKind::CmpEq => write!(f, "cmpeq"),
            OpKind::CmpLt => write!(f, "cmplt"),
            OpKind::Load => write!(f, "load"),
            OpKind::Store => write!(f, "store"),
            OpKind::Branch => write!(f, "br"),
            OpKind::Return => write!(f, "return"),
            OpKind::Constant(v) => write!(f, "constant {}", v),
            OpKind::Custom(v) => write!(f, "custom {}", v),
            OpKind::Gemm => write!(f, "gemm"),
            OpKind::Quantize => write!(f, "quantize"),
            OpKind::Dequantize => write!(f, "dequantize"),
            OpKind::QuantGemm => write!(f, "quant_gemm"),
            OpKind::QuantAttention => write!(f, "quant_attention"),
        }
    }
}

/// Parse a minimal TPTIR-style assembly into a Region.
pub fn parse_assembly(source: &str) -> Result<Region, String> {
    let mut region = Region::new();
    let mut block = Block::new("entry");
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.starts_with('^') && !block.operations.is_empty() {
            region.blocks.push(std::mem::replace(
                &mut block,
                Block::new(line.strip_prefix('^').unwrap()),
            ));
        }
    }
    if !block.operations.is_empty() || region.blocks.is_empty() {
        region.blocks.push(block);
    }
    Ok(region)
}
#[derive(Debug, Clone)]
pub struct Value {
    pub id: u64,
    pub typ: Type,
}
impl Value {
    pub fn new(id: u64, typ: Type) -> Self {
        Value { id, typ }
    }
}
#[derive(Debug, Clone)]
pub struct Operation {
    pub kind: OpKind,
    pub operands: Vec<Value>,
    pub result_type: Option<Type>,
    pub result_id: Option<u64>,
}
impl Operation {
    pub fn new(kind: OpKind) -> Self {
        Operation {
            kind,
            operands: vec![],
            result_type: None,
            result_id: None,
        }
    }
    pub fn display(&self) -> String {
        match &self.kind {
            OpKind::Addi => "addi",
            OpKind::Subi => "subi",
            OpKind::Muli => "muli",
            OpKind::Addf => "addf",
            OpKind::Subf => "subf",
            OpKind::Mulf => "mulf",
            OpKind::And => "andi",
            OpKind::Or => "ori",
            OpKind::Xor => "xori",
            OpKind::CmpEq => "cmpeq",
            OpKind::CmpLt => "cmplt",
            OpKind::Load => "load",
            OpKind::Store => "store",
            OpKind::Branch => "br",
            OpKind::Return => "return",
            OpKind::Constant(v) => return format!("constant {}", v),
            OpKind::Gemm => "gemm",
            OpKind::Quantize => "quantize",
            OpKind::Dequantize => "dequantize",
            OpKind::QuantGemm => "quant_gemm",
            OpKind::QuantAttention => "quant_attention",
            _ => "custom",
        }
        .to_string()
    }
}
#[derive(Debug, Clone)]
pub struct Block {
    pub label: String,
    pub operations: Vec<Operation>,
    pub arguments: Vec<Value>,
}
impl Block {
    pub fn new(label: &str) -> Self {
        Block {
            label: label.to_string(),
            operations: vec![],
            arguments: vec![],
        }
    }
}
#[derive(Debug, Clone)]
pub struct Region {
    pub blocks: Vec<Block>,
}
impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}
impl Region {
    pub fn new() -> Self {
        Region { blocks: vec![] }
    }
}
impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for b in &self.blocks {
            writeln!(f, "^{}: ", b.label)?;
            for op in &b.operations {
                writeln!(f, "  {}", op.display())?;
            }
        }
        Ok(())
    }
}

/// Supported element types for kernel templates.
#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum ElemType {
    F32,
    F16,
    BF16,
    I32,
}

impl ElemType {
    pub fn name(self) -> &'static str {
        match self {
            ElemType::F32 => "f32",
            ElemType::F16 => "f16",
            ElemType::BF16 => "bf16",
            ElemType::I32 => "i32",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(ElemType::F32),
            "f16" => Some(ElemType::F16),
            "bf16" => Some(ElemType::BF16),
            "i32" => Some(ElemType::I32),
            _ => None,
        }
    }
    pub fn zero(self) -> String {
        match self {
            ElemType::F32 => "0.0".to_string(),
            ElemType::F16 => "0.0".to_string(),
            ElemType::BF16 => "0.0".to_string(),
            ElemType::I32 => "0".to_string(),
        }
    }
}

/// A kernel variant: template name + element type + shape parameters.
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct KernelVariant {
    pub name: String,
    pub elem: ElemType,
    pub shape_params: Vec<i64>,
}

impl KernelVariant {
    pub fn dispatch_key(&self) -> String {
        let ss = self
            .shape_params
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        format!("{}_{}_{}", self.name, self.elem.name(), ss)
    }
}

/// Generated variant record: the variant metadata and its TPTIR text.
#[derive(Debug, Clone)]
pub struct GeneratedKernel {
    pub variant: KernelVariant,
    pub tptir_text: String,
    pub entry_name: String,
}

/// TPTIR assembly operation kinds produced by the TPTIR text emitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TptirOp {
    Addi,
    Subi,
    Muli,
    Addf,
    Subf,
    Mulf,
    And,
    Or,
    Xor,
    CmpEq,
    CmpLt,
    Load,
    Store,
    Branch,
    Return,
    Constant,
    Custom,
    Gemm,
    Quantize,
    Dequantize,
    QuantGemm,
    QuantAttention,
}

impl TptirOp {
    pub fn name(self) -> &'static str {
        match self {
            TptirOp::Addi => "addi",
            TptirOp::Subi => "subi",
            TptirOp::Muli => "muli",
            TptirOp::Addf => "addf",
            TptirOp::Subf => "subf",
            TptirOp::Mulf => "mulf",
            TptirOp::And => "andi",
            TptirOp::Or => "ori",
            TptirOp::Xor => "xori",
            TptirOp::CmpEq => "cmpeq",
            TptirOp::CmpLt => "cmplt",
            TptirOp::Load => "load",
            TptirOp::Store => "store",
            TptirOp::Branch => "br",
            TptirOp::Return => "return",
            TptirOp::Constant => "constant",
            TptirOp::Custom => "custom",
            TptirOp::Gemm => "gemm",
            TptirOp::Quantize => "quantize",
            TptirOp::Dequantize => "dequantize",
            TptirOp::QuantGemm => "quant_gemm",
            TptirOp::QuantAttention => "quant_attention",
        }
    }
}

/// Map a Rust-IR `OpKind` to its TPTIR textual operation name.
pub fn opkind_to_tptir(op: &OpKind) -> TptirOp {
    match op {
        OpKind::Addi => TptirOp::Addi,
        OpKind::Subi => TptirOp::Subi,
        OpKind::Muli => TptirOp::Muli,
        OpKind::Addf => TptirOp::Addf,
        OpKind::Subf => TptirOp::Subf,
        OpKind::Mulf => TptirOp::Mulf,
        OpKind::And => TptirOp::And,
        OpKind::Or => TptirOp::Or,
        OpKind::Xor => TptirOp::Xor,
        OpKind::CmpEq => TptirOp::CmpEq,
        OpKind::CmpLt => TptirOp::CmpLt,
        OpKind::Load => TptirOp::Load,
        OpKind::Store => TptirOp::Store,
        OpKind::Branch => TptirOp::Branch,
        OpKind::Return => TptirOp::Return,
        OpKind::Constant(_) => TptirOp::Constant,
        OpKind::Custom(_) => TptirOp::Custom,
        OpKind::Gemm => TptirOp::Gemm,
        OpKind::Quantize => TptirOp::Quantize,
        OpKind::Dequantize => TptirOp::Dequantize,
        OpKind::QuantGemm => TptirOp::QuantGemm,
        OpKind::QuantAttention => TptirOp::QuantAttention,
    }
}

/// Canonical set of supported kernel template names.
pub const KERNEL_TEMPLATES: &[&str] = &[
    "vector_add",
    "matmul",
    "softmax",
    "flash_attention",
    "conv_bn_relu",
    "conv3d",
    "layer_norm",
    "batch_norm",
    "group_norm",
];

/// Emit TPTIR textual assembly from a Region (spec v1.0 section 7.1).
pub fn emit_tptir(region: &Region, entry_name: &str, kernel_attrs: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str(&format!("module {{\n  func.func @{}", entry_name));
    if let Some(entry) = region.blocks.first() {
        if !entry.arguments.is_empty() {
            let args: Vec<String> = entry
                .arguments
                .iter()
                .map(|v| format!("%{}: {}", v.id, v.typ))
                .collect();
            out.push_str(&format!("({})", args.join(", ")));
        }
    }
    out.push_str(" attributes {tptir.kernel");
    for (k, v) in kernel_attrs {
        out.push_str(&format!(", {} = {}", k, v));
    }
    out.push_str("} {\n");
    for block in &region.blocks {
        out.push_str(&format!("    ^{}:\n", block.label));
        for op in &block.operations {
            let lhs = match op.result_id {
                Some(id) => format!("%{} = ", id),
                None => String::new(),
            };
            let oo = op
                .operands
                .iter()
                .map(|v| format!("%{}", v.id))
                .collect::<Vec<_>>()
                .join(", ");
            let to = opkind_to_tptir(&op.kind);
            let oper_str = match to {
                TptirOp::Addi
                | TptirOp::Subi
                | TptirOp::Muli
                | TptirOp::Addf
                | TptirOp::Subf
                | TptirOp::Mulf
                | TptirOp::And
                | TptirOp::Or
                | TptirOp::Xor => {
                    let ty = op
                        .result_type
                        .as_ref()
                        .map(|t| t.to_string())
                        .unwrap_or_default();
                    format!("tptir.{}({}) : ({}) -> {}", to.name(), oo, "i32", ty)
                }
                TptirOp::CmpEq | TptirOp::CmpLt => {
                    format!("tptir.{}({}) : (i32, i32) -> i1", to.name(), oo)
                }
                TptirOp::Load => format!("tptir.load({})", oo),
                TptirOp::Store => format!("tptir.store({})", oo),
                TptirOp::Branch => format!("tptir.br {}", oo),
                TptirOp::Return => String::from("tptir.return"),
                TptirOp::Constant => {
                    let val = match &op.kind {
                        OpKind::Constant(v) => v.clone(),
                        _ => String::from("0"),
                    };
                    let ty = op
                        .result_type
                        .as_ref()
                        .map(|t| t.to_string())
                        .unwrap_or_default();
                    format!("tptir.constant {} : {}", val, ty)
                }
                TptirOp::Custom => String::from(""),
                TptirOp::Gemm
                | TptirOp::Quantize
                | TptirOp::Dequantize
                | TptirOp::QuantGemm
                | TptirOp::QuantAttention => {
                    let ty = op
                        .result_type
                        .as_ref()
                        .map(|t| t.to_string())
                        .unwrap_or_default();
                    if ty.is_empty() {
                        format!("tptir.{}({})", to.name(), oo)
                    } else {
                        format!("tptir.{}({}) : {}", to.name(), oo, ty)
                    }
                }
            };
            if !oper_str.is_empty() {
                out.push_str(&format!("      {}{}\n", lhs, oper_str));
            }
        }
    }
    out.push_str("  }\n}\n");
    out
}

/// Build a single-block `Region` from a high-level kernel description.
pub fn build_kernel_region(
    name: &str,
    elem: ElemType,
    _shape_params: &[i64],
) -> Result<Region, String> {
    let mut region = Region::new();
    let mut block = Block::new("entry");
    block.arguments.push(Value::new(
        0,
        Type::memref(vec![-1], Type::primitive(elem.name()), AddressSpace::Global),
    ));
    block.arguments.push(Value::new(
        1,
        Type::memref(vec![-1], Type::primitive(elem.name()), AddressSpace::Global),
    ));
    block.arguments.push(Value::new(
        2,
        Type::memref(vec![-1], Type::primitive(elem.name()), AddressSpace::Global),
    ));
    block.arguments.push(Value::new(3, Type::primitive("i32")));
    match name {
        "vector_add" => {
            let mut op = Operation::new(OpKind::Addf);
            op.operands.push(block.arguments[0].clone());
            op.operands.push(block.arguments[1].clone());
            op.result_type = Some(Type::primitive(elem.name()));
            op.result_id = Some(10);
            block.operations.push(op);
            let mut ret = Operation::new(OpKind::Return);
            ret.operands
                .push(Value::new(10, Type::primitive(elem.name())));
            block.operations.push(ret);
        }
        "softmax" | "flash_attention" | "conv_bn_relu" | "conv3d" | "layer_norm" | "batch_norm"
        | "group_norm" => {
            let mut op = Operation::new(OpKind::Mulf);
            op.operands.push(block.arguments[0].clone());
            op.operands.push(block.arguments[1].clone());
            op.result_type = Some(Type::primitive(elem.name()));
            op.result_id = Some(10);
            block.operations.push(op);
            let mut ret = Operation::new(OpKind::Return);
            ret.operands
                .push(Value::new(10, Type::primitive(elem.name())));
            block.operations.push(ret);
        }
        "matmul" => {
            let mut op = Operation::new(OpKind::Mulf);
            op.operands.push(block.arguments[0].clone());
            op.operands.push(block.arguments[1].clone());
            op.result_type = Some(Type::primitive(elem.name()));
            op.result_id = Some(10);
            block.operations.push(op);
            let mut acc = Operation::new(OpKind::Addf);
            acc.operands
                .push(Value::new(10, Type::primitive(elem.name())));
            acc.operands.push(block.arguments[2].clone());
            acc.result_type = Some(Type::primitive(elem.name()));
            acc.result_id = Some(11);
            block.operations.push(acc);
            let mut ret = Operation::new(OpKind::Return);
            ret.operands
                .push(Value::new(11, Type::primitive(elem.name())));
            block.operations.push(ret);
        }
        _ => return Err(format!("Unknown kernel template: {}", name)),
    }
    region.blocks.push(block);
    Ok(region)
}
