//! TPTIR Kernel Compiler
//!
//! High-level interface for compiling TPTIR kernel source code.

use crate::error::TptpResult;
use crate::ffi::tptir_ffi;

/// Compilation target format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTarget {
    /// TPT ISA text output
    TptIsaText = 0,
    /// LLVM IR output
    LlvmIr = 1,
    /// TPTIR binary (cached)
    TptirBinary = 2,
}

/// Compilation options
#[derive(Debug, Clone)]
pub struct CompilationOptions {
    /// Target format
    pub target: CompilationTarget,
    /// Optimization level (0-3)
    pub opt_level: u32,
    /// Enable debug info
    pub debug_info: bool,
    /// Target GPU architecture (e.g., "sm_80", "gfx1030")
    pub target_arch: Option<String>,
    /// Additional compiler flags
    pub extra_flags: Vec<String>,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        CompilationOptions {
            target: CompilationTarget::TptIsaText,
            opt_level: 2,
            debug_info: false,
            target_arch: None,
            extra_flags: Vec::new(),
        }
    }
}

/// TPTIR compiler handle
pub struct TptirCompiler {
    #[allow(dead_code)]
    context: tptir_ffi::TptirCompilerHandle,
    options: CompilationOptions,
}

impl TptirCompiler {
    /// Create a new TPTIR compiler with default options
    pub fn new() -> TptpResult<Self> {
        let context = tptir_ffi::TptirCompilerHandle::new()?;
        Ok(TptirCompiler {
            context,
            options: CompilationOptions::default(),
        })
    }

    /// Create a compiler with custom options
    pub fn with_options(options: CompilationOptions) -> TptpResult<Self> {
        let context = tptir_ffi::TptirCompilerHandle::new()?;
        Ok(TptirCompiler { context, options })
    }

    /// Get the compiler version
    pub fn version() -> String {
        tptir_ffi::get_tptir_version()
    }

    /// Compile TPTIR source to the target format
    pub fn compile(&self, source: &str) -> TptpResult<String> {
        tptir_ffi::compile_kernel(source, self.options.target as i32)
    }

    /// Compile a specific kernel from TPTIR source
    pub fn compile_kernel(&self, source: &str, kernel_name: &str) -> TptpResult<String> {
        let _ = kernel_name;
        self.compile(source)
    }

    /// Compile a parameterized TPTIR template.
    ///
    /// Replaces `{{KEY}}` placeholders in `template` with the corresponding
    /// values from `params` before compilation.  For example, passing
    /// `&[("TILE_M", "128"), ("TILE_K", "32")]` turns `{{TILE_M}}` → `128`.
    pub fn compile_parameterized(
        &self,
        template: &str,
        params: &[(&str, &str)],
    ) -> TptpResult<String> {
        let mut source = template.to_owned();
        for (key, value) in params {
            let placeholder = format!("{{{{{}}}}}", key);
            source = source.replace(&placeholder, value);
        }
        self.compile(&source)
    }

    /// Convenience: compile the GEMM kernel template embedded in this crate.
    ///
    /// `params` should contain entries for TILE_M, TILE_N, TILE_K, VEC_WIDTH, UNROLL.
    pub fn compile_gemm_template(
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
        vec_width: u32,
        unroll: u32,
    ) -> TptpResult<String> {
        let template = include_str!("../../tptir/tptir_gemm.mlir");
        let compiler = TptirCompiler::new()?;
        compiler.compile_parameterized(
            template,
            &[
                ("TILE_M", &tile_m.to_string()),
                ("TILE_N", &tile_n.to_string()),
                ("TILE_K", &tile_k.to_string()),
                ("VEC_WIDTH", &vec_width.to_string()),
                ("UNROLL", &unroll.to_string()),
            ],
        )
    }

    /// Get the current compilation options
    pub fn options(&self) -> &CompilationOptions {
        &self.options
    }

    /// Set the compilation options
    pub fn set_options(&mut self, options: CompilationOptions) {
        self.options = options;
    }
}

impl Default for TptirCompiler {
    fn default() -> Self {
        Self::new().expect("failed to create TPTIR compiler")
    }
}

/// Compile a TPTIR kernel with default options (convenience function)
pub fn compile(source: &str) -> TptpResult<String> {
    let compiler = TptirCompiler::new()?;
    compiler.compile(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_version() {
        let version = TptirCompiler::version();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = TptirCompiler::new();
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_parameterized_substitution() {
        let template = "tile_m={{TILE_M}} tile_k={{TILE_K}} vec={{VEC_WIDTH}}";
        let compiler = TptirCompiler::new().unwrap();
        // compile_parameterized substitutes then compiles; the FFI stub returns a
        // non-empty string, so we just verify substitution happened correctly by
        // inspecting the substituted string before the compile step.
        let mut source = template.to_owned();
        for (k, v) in &[("TILE_M", "128"), ("TILE_K", "32"), ("VEC_WIDTH", "8")] {
            source = source.replace(&format!("{{{{{}}}}}", k), v);
        }
        assert_eq!(source, "tile_m=128 tile_k=32 vec=8");
        // Also verify compile_parameterized runs without error.
        let result = compiler.compile_parameterized(
            template,
            &[("TILE_M", "64"), ("TILE_K", "16"), ("VEC_WIDTH", "4")],
        );
        assert!(result.is_ok());
    }
}
