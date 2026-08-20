//! Build script for tptc-rs.
//!
//! When the `ffi` feature is enabled, compiles the bundled minimal
//! `ffi_stubs.cpp` (a passthrough, not the full C++ TPTIR/MLIR/LLVM toolchain),
//! so the Rust crate can link without relying on an external LLVM/MLIR build.
//!
//! The default build (`ffi` off) prefers the native Rust compiler path
//! under [`crate::compile_native`]; the FFI path is kept available as a
//! fallback translation unit call from [`crate::ffi`].

fn main() {
    // The Rust `ffi` module declares `extern "C"` symbols
    // `tptir_init`, `tptir_shutdown`, `tptir_get_version`, `tptir_compile`,
    // `tptir_string_free`. Compile the bundled passthrough stubs so the
    // crate can link without depending on a full C++ TPTIR/MLIR/LLVM
    // toolchain. When a real native TPTIR C library is available you can
    // replace this cc::Build block with
    // `println!("cargo:rustc-link-lib=tptc");` etc.
    let mut cc = cc::Build::new();
    cc.cpp(true);
    cc.file("src/ffi_stubs.cpp");
    println!("cargo:rerun-if-changed=src/ffi_stubs.cpp");
    cc.compile("ffi_stubs");
}
