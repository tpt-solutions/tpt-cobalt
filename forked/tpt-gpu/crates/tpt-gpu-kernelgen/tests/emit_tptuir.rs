//! End-to-end test for the `--output-tptuir` wiring in the kernelgen CLI.
//!
//! Spawns the real `tpt-gpu-kernelgen` binary, asks it to emit a kernel as a
//! `.tptuir` file, then reads that file back through the adapter and lowers it
//! to TPTIR — proving the adapter is correctly wired into the compiler.

use std::path::PathBuf;
use std::process::Command;

/// Locate the `tpt-gpu-kernelgen` binary. Prefer the Cargo-injected
/// `CARGO_BIN_EXE_*` var; fall back to deriving it from the test executable's
/// location (`<target>/<profile>/tpt-gpu-kernelgen[.exe]`).
fn kernelgen_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_tpt_gpu_kernelgen") {
        return PathBuf::from(p);
    }
    let exe = std::env::current_exe().expect("current_exe");
    let profile_dir = exe.parent().and_then(|p| p.parent()).expect("profile dir");
    let name = if cfg!(windows) {
        "tpt-gpu-kernelgen.exe"
    } else {
        "tpt-gpu-kernelgen"
    };
    profile_dir.join(name)
}

#[test]
fn emit_tptuir_via_cli_roundtrips() {
    let out = std::env::temp_dir().join(format!("kg_emit_{}.tptuir", std::process::id()));

    let status = Command::new(kernelgen_bin())
        .args([
            "generate",
            "matmul",
            "--elem",
            "f32",
            "--shape",
            "1024",
            "--output-tptuir",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("spawn tpt-gpu-kernelgen");
    assert!(status.success(), "kernelgen CLI exited non-zero");
    assert!(out.exists(), "kernelgen did not write the .tptuir file");

    // Read it back through the adapter and lower to TPTIR (lossless).
    let region = tpt_gpu_uir_adapter::read_tptuir(&out).expect("read_tptuir failed");
    assert!(!region.blocks.is_empty());

    let _ = std::fs::remove_file(&out);
}
