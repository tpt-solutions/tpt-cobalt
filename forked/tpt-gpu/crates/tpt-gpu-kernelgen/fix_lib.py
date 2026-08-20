"""Fix lib.rs by removing duplicate module declarations."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\lib.rs")

content = """pub mod ffi;
pub mod ir;
pub mod passes;
pub mod validate;
pub mod fusion;
pub mod dispatch;
pub mod tuning;
pub mod bench;

pub const VERSION: &str = "0.1.0";

pub fn compile(source: &str, target: &str) -> Result<String, String> {
    #[cfg(feature = "ffi")] { ffi::compile_via_ffi(source, target) }
    #[cfg(not(feature = "ffi"))] { compile_native(source, target) }
}

pub fn compile_native(source: &str, target: &str) -> Result<String, String> {
    let region = ir::parse_assembly(source)?;
    let passes = passes::default_pipeline();
    let _changes = passes.run(&region);
    match target {
        "tptisa" | "text" => Ok(region.to_string()),
        "llvmir" => Ok(generate_llvm_ir(&region)),
        _ => Err(format!("Unknown target: {}", target)),
    }
}

fn generate_llvm_ir(region: &ir::Region) -> String {
    let mut out = String::from("; LLVM IR\\ndefine void @kernel() {\\n");
    for block in &region.blocks {
        out.push_str(&format!("  {}:\\n", block.label));
        for op in &block.operations {
            out.push_str(&format!("    {}\\n", op.display()));
        }
    }
    out.push_str("}\\n");
    out
}

pub fn version() -> String {
    format!("tptc-rs v{}", VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_version() {
        assert!(version().contains("0.1.0"));
    }
    #[test]
    fn test_ir_types() {
        let t = ir::Type::primitive("i32");
        assert_eq!(t.to_string(), "i32");
    }
    #[test]
    fn test_block() {
        let b = ir::Block::new("entry");
        assert_eq!(b.label, "entry");
    }
}
"""

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
