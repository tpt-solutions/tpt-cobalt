//! FFI bindings to TPTIR C API
//!
//! Provides Rust wrappers around the tptir C API for kernel compilation.

pub mod tptir_ffi;

pub use tptir_ffi::{compile_kernel, TptirCompilerHandle};

#[cfg(feature = "tptir-backend")]
pub use tptir_ffi::TptirModule;
