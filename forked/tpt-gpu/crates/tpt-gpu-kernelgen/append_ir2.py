"""Append emit_tptir and build_kernel_region to ir.rs."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

existing = p.read_text(encoding="utf-8")

new_content = """
/// Emit TPTIR textual assembly from a Region (spec v1.0 section 7.1).
pub fn emit_tptir(region:&Region, entry_name:&str, kernel_attrs:&[(String,String)])->String{
 let mut out=String::new();
 out.push_str(&format!("module {{\\n  func.func @{}", entry_name));
 if let Some(entry)=region.blocks.first(){
  if !entry.arguments.is_empty(){
   let args:Vec<String>=entry.arguments.iter().map(|v|format!("%{}: {}", v.id, v.typ)).collect();
   out.push_str(&format!("({})", args.join(", ")));
  }
 }
 out.push_str(" attributes {tptir.kernel");
 for (k,v) in kernel_attrs{ out.push_str(&format!(", {} = {}", k, v)); }
 out.push_str("} {\\n");
 for block in &region.blocks{
  out.push_str(&format!("    ^{}:\\n", block.label));
  for op in &block.operations{
   let lhs=match op.result_id{Some(id)=>format!("%{} = ", id),None=>String::new()};
   let oo=op.operands.iter().map(|v|format!("%{}",v.id)).collect::<Vec<_>>().join(", ");
   let to=opkind_to_tptir(&op.kind);
   let oper_str=match to{
    TptirOp::Addi|TptirOp::Subi|TptirOp::Muli|
    TptirOp::Addf|TptirOp::Subf|TptirOp::Mulf|
    TptirOp::And|TptirOp::Or|TptirOp::Xor=>{
     let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default();
     format!("tptir.{}({}) : ({}) -> {}", to.name(), oo, "i32", ty)
    }
    TptirOp::CmpEq|TptirOp::CmpLt=>format!("tptir.{}({}) : (i32, i32) -> i1", to.name(), oo),
    TptirOp::Load=>format!("tptir.load({})", oo),
    TptirOp::Store=>format!("tptir.store({})", oo),
    TptirOp::Branch=>format!("tptir.br {}", oo),
    TptirOp::Return=>String::from("tptir.return"),
    TptirOp::Constant=>{
     let val=match&op.kind{OpKind::Constant(v)=>v.clone(),_=>String::from("0")};
     let ty=op.result_type.as_ref().map(|t|t.to_string()).unwrap_or_default();
     format!("tptir.constant {} : {}", val, ty)
    }
    TptirOp::Custom=>String::from(""),
   };
   if !oper_str.is_empty(){
    out.push_str(&format!("      {}{}\\n", lhs, oper_str));
   }
  }
 }
 out.push_str("  }\\n}\\n");
 out
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
"""

p.write_text(existing + new_content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
