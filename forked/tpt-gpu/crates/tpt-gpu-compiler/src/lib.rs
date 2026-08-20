pub mod bench;
pub mod dispatch;
pub mod ffi;
pub mod fusion;
pub mod ir;
pub mod passes;
pub mod provenance;
pub mod tuning;
pub mod validate;

pub const VERSION: &str = "0.1.0";

/// Convenience helper: build a kernel region, run the standard pass
/// pipeline and emit TPTIR with a provenance header prepended. Mirrors
/// the flow used by the `tpt generate` CLI path in `crates/tpt-gpu-kernelgen`.
pub fn generate_with_provenance(
    kernel: &str,
    elem: crate::ir::ElemType,
    shape: &[i64],
    score_gflops: f64,
) -> Result<String, String> {
    let region = crate::ir::build_kernel_region(kernel, elem, shape)?;
    let _changes = crate::passes::default_pipeline().run(&region);
    let body = crate::ir::emit_tptir(&region, kernel, &[]);
    let prov = crate::provenance::Provenance::new(kernel, score_gflops);
    Ok(format!("{}{}", prov.to_mlir_comment(), body))
}

pub fn compile(source: &str, target: &str) -> Result<String, String> {
    #[cfg(feature = "ffi")]
    {
        ffi::compile_via_ffi(source, target)
    }
    #[cfg(not(feature = "ffi"))]
    {
        compile_native(source, target)
    }
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
    let mut out = String::from("; LLVM IR\ndefine void @kernel() {\n");
    for block in &region.blocks {
        out.push_str(&format!("  {}:\n", block.label));
        for op in &block.operations {
            out.push_str(&format!("    {}\n", op.display()));
        }
    }
    out.push_str("}\n");
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
    #[test]
    fn test_generate_with_provenance_header() {
        let out = generate_with_provenance("vector_add", ir::ElemType::F32, &[1024], 0.0).unwrap();
        assert!(out.starts_with("# TPTIR Provenance:"));
        assert!(out.contains("module"));
    }
}
