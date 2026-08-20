pub mod rust_emit;
pub mod tptir_emit;

use crate::ast::Program;

/// The two emitted artifacts from a single TPT Script program.
///
/// Functions annotated `@requires_gpu(true)` are lowered to TPTIR for GPU
/// kernel emission via the layer3 tptc compiler stack.  All other functions
/// are lowered to Rust source for host-side execution.
#[derive(Debug, Default, Clone)]
pub struct CodegenOutput {
    /// Rust source code for non-GPU (host-side) functions. Empty if none.
    pub rust_source: String,
    /// TPTIR text for GPU-kernel functions.  Empty if no GPU functions exist.
    /// Feed this into `layer3_tptc::compile_native(tptir_source, "tptisa")`
    /// or `"llvmir"` to obtain machine-level output.
    pub tptir_source: String,
    /// Non-fatal TPTIR emission errors (e.g. unsupported control flow).
    /// The emitted TPTIR contains `; ERROR:` comments at the affected sites.
    /// An empty vec means the TPTIR is fully lowered.
    pub tptir_errors: Vec<String>,
}

/// Emit code from a parsed `Program`.
///
/// The program is split into two streams:
/// - Host functions (no `@requires_gpu`) → `output.rust_source`
/// - GPU kernel functions (`@requires_gpu(true)`) → `output.tptir_source`
pub fn emit(program: &Program) -> CodegenOutput {
    let mut tptir_emitter = tptir_emit::TptIrEmitter::new();
    let tptir_source = tptir_emitter.emit_program(program);
    let tptir_errors = tptir_emitter.take_errors();
    CodegenOutput {
        rust_source: rust_emit::RustEmitter::new().emit_program(program),
        tptir_source,
        tptir_errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile_str, semantic::type_check};

    fn emit_from(src: &str) -> CodegenOutput {
        let prog = compile_str(src).expect("compile_str failed");
        let _tc = type_check(&prog);
        emit(&prog)
    }

    #[test]
    fn test_emit_splits_gpu_and_host() {
        let out = emit_from(
            r#"
@requires_gpu(true)
fn kernel(a: Tensor[f32, m, n]) -> Tensor[f32, m, n] {
    let r = tpt.relu(a)
    return r
}

fn host_fn(x: f32) -> f32 {
    return x
}
"#,
        );
        assert!(
            out.rust_source.contains("host_fn"),
            "rust_source: {}",
            out.rust_source
        );
        assert!(
            out.rust_source.contains("GPU kernel `kernel`"),
            "rust_source: {}",
            out.rust_source
        );
        assert!(
            out.tptir_source.contains("func @kernel"),
            "tptir_source: {}",
            out.tptir_source
        );
        assert!(
            !out.tptir_source.contains("host_fn"),
            "tptir should not contain host_fn"
        );
    }

    #[test]
    fn test_emit_no_gpu_functions() {
        let out = emit_from("fn add(a: f32, b: f32) -> f32 { return a + b }");
        assert!(out.rust_source.contains("pub fn add"));
        assert!(out.tptir_source.contains("no GPU kernel functions found"));
    }
}
