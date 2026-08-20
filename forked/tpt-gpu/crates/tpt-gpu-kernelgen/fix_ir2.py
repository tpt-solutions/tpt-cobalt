"""Fix remaining ir.rs issues."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

content = p.read_text(encoding="utf-8")

# Fix 1: Missing lifetime specifier in ElemType::name
content = content.replace(
    'pub fn name(self)->&str{match self{ElemType::F32=>"f32",ElemType::F16=>"f16",ElemType::BF16=>"bf16",ElemType::I32=>"i32"}}',
    'pub fn name(self)->&\'static str{match self{ElemType::F32=>"f32",ElemType::F16=>"f16",ElemType::BF16=>"bf16",ElemType::I32=>"i32"}}'
)

# Fix 2: Missing lifetime specifier in TptirOp::name
content = content.replace(
    'pub fn name(self)->&str{match self{',
    'pub fn name(self)->&\'static str{match self{'
)

# Fix 3: KERNEL_TEMPLATES type issue - add &
content = content.replace(
    'pub const KERNEL_TEMPLATES:[&str]=',
    'pub const KERNEL_TEMPLATES:&[&str]='
)

# Fix 4: zero() method return type issue
content = content.replace(
    'pub fn zero(self)->String{match self{ElemType::F32=>"0.0",ElemType::F16=>"0.0",ElemType::BF16=>"0.0",ElemType::I32=>"0"}}',
    'pub fn zero(self)->String{match self{ElemType::F32=>"0.0".to_string(),ElemType::F16=>"0.0".to_string(),ElemType::BF16=>"0.0".to_string(),ElemType::I32=>"0".to_string()}}'
)

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
