//! TPTIR Kernel Compilation
//!
//! Provides compilation of TPTIR kernel source to executable code
//! via the tptc C API.

pub mod compile;

pub use compile::{CompilationOptions, CompilationTarget, TptirCompiler};
