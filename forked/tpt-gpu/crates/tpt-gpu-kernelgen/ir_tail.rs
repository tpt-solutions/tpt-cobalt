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
pub struct KernelVariant{ pub name:String, pub elem:ElemType, pub shape_params:Vec<i64> }
impl KernelVariant{
 pub fn dispatch_key(&self)->String{
  let ss=self.shape_params.iter().map(|s|s.to_string()).collect::<Vec<_>>().join("x");
  format!("{}_{}_{}", self.name, self.elem.name(), ss)
 }
}

/// Generated variant record: the variant metadata and its TPTIR text.
#[derive(Debug,Clone)]
pub struct GeneratedKernel{ pub variant:KernelVariant, pub tptir_text:String, pub entry_name:String }

/// TPTIR assembly operation kinds produced by the TPTIR text emitter.
#[derive(Debug,Clone,Copy,PartialEq)]
pub enum TptirOp{ Addi,Subi,Muli,Addf,Subf,Mulf,And,Or,Xor,CmpEq,CmpLt,Load,Store,Branch,Return,Constant,Custom }
impl TptirOp{
 pub fn name(self)->&str{match self{
  TptirOp::Addi=>"addi",TptirOp::Subi=>"subi",TptirOp::Muli=>"muli",
  TptirOp::Addf=>"addf",TptirOp::Subf=>"subf",TptirOp::Mulf=>"mulf",
  TptirOp::And=>"andi",TptirOp::Or=>"ori",TptirOp::Xor=>"xori",
  TptirOp::CmpEq=>"cmpeq",TptirOp::CmpLt=>"cmplt",
  TptirOp::Load=>"load",TptirOp::Store=>"store",
  TptirOp::Branch=>"br",TptirOp::Return=>"return",
  TptirOp::Constant=>"constant",TptirOp::Custom=>"custom" }}
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
  OpKind::Constant(_)=>TptirOp::Constant,OpKind::Custom(_)=>TptirOp::Custom }}
}

/// Canonical set of supported kernel template names.
pub const KERNEL_TEMPLATES:[&str]=["vector_add","matmul","softmax","flash_attention","conv_bn_relu"];


/// Emit TPTIR textual assembly from a Region (spec v1.0 section 7.1).
pub fn emit_tptir(region:&Region, entry_name:&str, kernel_attrs:&[(String,String)])->String{
 let mut out=String::new();
 out.push_str(&format!("module {{\n  func.func @{}", entry_name));
 if let Some(entry)=region.blocks.first(){
  if !entry.arguments.is_empty(){
   let args:Vec<String>=entry.arguments.iter().map(|v|format!("%{}: {}", v.id, v.typ)).collect();
   out.push_str(&format!("({})", args.join(", ")));
  } }
 out.push_str(" attributes {tptir.kernel");
 for (k,v) in kernel_attrs{ out.push_str(&format!(", {} = {}", k, v)); }
 out.push_str("} {\n");
 for block in &region.blocks{
  out.push_str(&format!("    ^{}:\n", block.label));
  for op in &block.operations{
   let lhs=match op.result_id{Some(id)=>format!("%{} = ", id),None=>String::new()};
   let oo=op.operands.iter().map(|v|format!("%{}",v.id)).collect::<Vec<_>>().join(", ");
   let to=opkind_to_tptir(&op.kind);
   let oper_str=match to{
    TptirOp::Addi|TptirOp::Subi|TptirOp::Muli|TptirOp::Addf|TptirOp::Subf|TptirOp::Mulf|TptirOp::And|TptirOp::Or|TptirOp::Xor=>{
     let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default();
     format!("tptir.{}({}) : ({}) -> {}", to.name(), oo, "i32", ty) }
    TptirOp::CmpEq|TptirOp::CmpLt=>format!("tptir.{}({}) : (i32, i32) -> i1", to.name(), oo),
    TptirOp::Load=>format!("tptir.load({})", oo),
    TptirOp::Store=>format!("tptir.store({})", oo),
    TptirOp::Branch=>format!("tptir.br {}", oo),
    TptirOp::Return=>String::from("tptir.return"),
    TptirOp::Constant=>{ let val=match&op.kind{OpKind::Constant(v)=>v.clone(),_=>String::from("0")}; let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default(); format!("tptir.constant {} : {}", val, ty) }
    TptirOp::Custom=>String::from(""), };
   if !oper_str.is_empty(){ out.push_str(&format!("      {}{}\n", lhs, oper_str)); }
  } }
 out.push_str("  }\n}\n"); out
}

/// Build a single-block `Region` from a high-level kernel description.
pub fn build_kernel_region(name:&str, elem:ElemType, _shape_params:&[i64])->Result<Region,String>{
 let mut region=Region::new();
 let mut block=Block::new("entry");
 block.arguments.push(Value::new(0,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(1,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(2,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(3,Type::primitive("i32")));
 match name{
  "vector_add"=>{
   let mut op=Operation::new(OpKind::Addf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(10,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  "softmax" | "flash_attention" | "conv_bn_relu" =>{
   let mut op=Operation::new(OpKind::Mulf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(10,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  "matmul"=>{
   let mut op=Operation::new(OpKind::Mulf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut acc=Operation::new(OpKind::Addf);
   acc.operands.push(Value::new(10,Type::primitive(elem.name())));
   acc.operands.push(block.arguments[2].clone());
   acc.result_type=Some(Type::primitive(elem.name()));
   acc.result_id=Some(11);
   block.operations.push(acc);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(11,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  _=>return Err(format!("Unknown kernel template: {}",name)),
 }
 region.blocks.push(block);
 Ok(region)
}


/// Emit TPTIR textual assembly from a Region (spec v1.0 section 7.1).
pub fn emit_tptir(region:&Region, entry_name:&str, kernel_attrs:&[(String,String)])->String{
 let mut out=String::new();
 out.push_str(&format!("module {{\n  func.func @{}", entry_name));
 if let Some(entry)=region.blocks.first(){
  if !entry.arguments.is_empty(){
   let args:Vec<String>=entry.arguments.iter().map(|v|format!("%{}: {}", v.id, v.typ)).collect();
   out.push_str(&format!("({})", args.join(", ")));
  } }
 out.push_str(" attributes {tptir.kernel");
 for (k,v) in kernel_attrs{ out.push_str(&format!(", {} = {}", k, v)); }
 out.push_str("} {\n");
 for block in &region.blocks{
  out.push_str(&format!("    ^{}:\n", block.label));
  for op in &block.operations{
   let lhs=match op.result_id{Some(id)=>format!("%{} = ", id),None=>String::new()};
   let oo=op.operands.iter().map(|v|format!("%{}",v.id)).collect::<Vec<_>>().join(", ");
   let to=opkind_to_tptir(&op.kind);
   let oper_str=match to{
    TptirOp::Addi|TptirOp::Subi|TptirOp::Muli|TptirOp::Addf|TptirOp::Subf|TptirOp::Mulf|TptirOp::And|TptirOp::Or|TptirOp::Xor=>{
     let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default();
     format!("tptir.{}({}) : ({}) -> {}", to.name(), oo, "i32", ty) }
    TptirOp::CmpEq|TptirOp::CmpLt=>format!("tptir.{}({}) : (i32, i32) -> i1", to.name(), oo),
    TptirOp::Load=>format!("tptir.load({})", oo),
    TptirOp::Store=>format!("tptir.store({})", oo),
    TptirOp::Branch=>format!("tptir.br {}", oo),
    TptirOp::Return=>String::from("tptir.return"),
    TptirOp::Constant=>{ let val=match&op.kind{OpKind::Constant(v)=>v.clone(),_=>String::from("0")}; let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default(); format!("tptir.constant {} : {}", val, ty) }
    TptirOp::Custom=>String::from(""), };
   if !oper_str.is_empty(){ out.push_str(&format!("      {}{}\n", lhs, oper_str)); }
  } }
 out.push_str("  }\n}\n"); out
}

/// Build a single-block `Region` from a high-level kernel description.
pub fn build_kernel_region(name:&str, elem:ElemType, _shape_params:&[i64])->Result<Region,String>{
 let mut region=Region::new();
 let mut block=Block::new("entry");
 block.arguments.push(Value::new(0,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(1,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(2,Type::memref(vec![-1],Type::primitive(elem.name()),AddressSpace::Global)));
 block.arguments.push(Value::new(3,Type::primitive("i32")));
 match name{
  "vector_add"=>{
   let mut op=Operation::new(OpKind::Addf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(10,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  "softmax" | "flash_attention" | "conv_bn_relu" =>{
   let mut op=Operation::new(OpKind::Mulf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(10,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  "matmul"=>{
   let mut op=Operation::new(OpKind::Mulf);
   op.operands.push(block.arguments[0].clone());
   op.operands.push(block.arguments[1].clone());
   op.result_type=Some(Type::primitive(elem.name()));
   op.result_id=Some(10);
   block.operations.push(op);
   let mut acc=Operation::new(OpKind::Addf);
   acc.operands.push(Value::new(10,Type::primitive(elem.name())));
   acc.operands.push(block.arguments[2].clone());
   acc.result_type=Some(Type::primitive(elem.name()));
   acc.result_id=Some(11);
   block.operations.push(acc);
   let mut ret=Operation::new(OpKind::Return);
   ret.operands.push(Value::new(11,Type::primitive(elem.name())));
   block.operations.push(ret);
  }
  _=>return Err(format!("Unknown kernel template: {}",name)),
 }
 region.blocks.push(block);
 Ok(region)
}
