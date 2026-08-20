"""Append new content to ir.rs."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

# Read existing content
existing = p.read_text(encoding="utf-8")

# New content to append
new_content = """
/// Supported element types for kernel templates.
#[derive(Debug,Clone,Copy,PartialEq,Hash,Eq)]
pub enum ElemType{ F32, F16, BF16, I32 }

impl ElemType{
 pub fn name(self)->&str{match self{ElemType::F32=>"f32",ElemType::F16=>"f16",ElemType::BF16=>"bf16",ElemType::I32=>"i32"}}
 pub fn parse(s:&str)->Option<Self>{match s{"f32"=>Some(ElemType::F32),"f16"=>Some(ElemType::F16),"bf16"=>Some(ElemType::BF16),"i32"=>Some(ElemType::I32),_=>None}}
 pub fn zero(self)->String{match self{ElemType::F32=>"0.0",ElemType::F16=>"0.0",ElemType::BF16=>"0.0",ElemType::I32=>"0"}}
}

/// A kernel variant: template name + element type + shape parameters.
#[derive(Debug,Clone,PartialEq,Hash,Eq)]
pub struct KernelVariant{
 pub name:String,
 pub elem:ElemType,
 pub shape_params:Vec<i64>,
}

impl KernelVariant{
 pub fn dispatch_key(&self)->String{
  let ss=self.shape_params.iter().map(|s|s.to_string()).collect::<Vec<_>>().join("x");
  format!("{}_{}_{}", self.name, self.elem.name(), ss)
 }
}

/// Generated variant record: the variant metadata and its TPTIR text.
#[derive(Debug,Clone)]
pub struct GeneratedKernel{
 pub variant:KernelVariant,
 pub tptir_text:String,
 pub entry_name:String,
}

/// TPTIR assembly operation kinds produced by the TPTIR text emitter.
#[derive(Debug,Clone,Copy,PartialEq)]
pub enum TptirOp{
 Addi,Subi,Muli,Addf,Subf,Mulf,And,Or,Xor,CmpEq,CmpLt,Load,Store,Branch,Return,Constant,Custom
}

impl TptirOp{
 pub fn name(self)->&str{match self{
  TptirOp::Addi=>"addi",TptirOp::Subi=>"subi",TptirOp::Muli=>"muli",
  TptirOp::Addf=>"addf",TptirOp::Subf=>"subf",TptirOp::Mulf=>"mulf",
  TptirOp::And=>"andi",TptirOp::Or=>"ori",TptirOp::Xor=>"xori",
  TptirOp::CmpEq=>"cmpeq",TptirOp::CmpLt=>"cmplt",
  TptirOp::Load=>"load",TptirOp::Store=>"store",
  TptirOp::Branch=>"br",TptirOp::Return=>"return",
  TptirOp::Constant=>"constant",TptirOp::Custom=>"custom",
 }}
}

/// Map a Rust-IR `OpKind` to its TPTIR textual operation name.
pub fn opkind_to_tptir(op:&OpKind)->TptirOp{
 match op{
  OpKind::Addi=>TptirOp::Addi,OpKind::Subi=>TptirOp::Subi,OpKind::Muli=>TptirOp::Muli,
  OpKind::Addf=>TptirOp::Addf,OpKind::Subf=>TptirOp::Subf,OpKind::Mulf=>TptirOp::Mulf,
  OpKind::And=>TptirOp::And,OpKind::Or=>TptirOp::Or,OpKind::Xor=>TptirOp::Xor,
  OpKind::CmpEq=>TptirOp::CmpEq,OpKind::CmpLt=>TptirOp::CmpLt,
  OpKind::Load=>TptirOp::Load,OpKind::Store=>TptirOp::Store,
  OpKind::Branch=>TptirOp::Branch,OpKind::Return=>TptirOp::Return,
  OpKind::Constant(_)=>TptirOp::Constant,OpKind::Custom(_)=>TptirOp::Custom,
 }
}

/// Canonical set of supported kernel template names.
pub const KERNEL_TEMPLATES:[&str]=["vector_add","matmul","softmax","flash_attention","conv_bn_relu"];
"""

# Write combined content
p.write_text(existing + new_content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
