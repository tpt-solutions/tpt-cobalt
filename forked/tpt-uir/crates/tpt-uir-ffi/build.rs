use std::env;
use std::path::PathBuf;

use cbindgen::{Builder, Config, Language};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let out_dir = PathBuf::from(&crate_dir).join("include");
    std::fs::create_dir_all(&out_dir).expect("failed to create include/ directory");

    let out_file = out_dir.join("tpt_uir_ffi.h");

    let config = Config {
        language: Language::C,
        pragma_once: true,
        documentation: true,
        ..Default::default()
    };

    let bindings = Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate();

    match bindings {
        Ok(b) => {
            b.write_to_file(&out_file);
            println!("cargo:rerun-if-changed=src/lib.rs");
            println!("cargo:rerun-if-changed=build.rs");
        }
        Err(e) => {
            // cbindgen failures are non-fatal for the library build itself, but
            // surface loudly so the header is never silently stale.
            eprintln!("cbindgen header generation failed: {:?}", e);
        }
    }
}
