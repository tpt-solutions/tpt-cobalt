// ---------------------------------------------------------------------------
// modules — TPT Script standard library module system
//
// Implements the module namespaces defined in the TPT Script spec (§12, Appendix B):
//   tpt              — Core tensor operations (auto-imported)
//   tpt.introspect   — Introspection and schema API
//   tpt.nn           — Neural network building blocks
//   tpt.optim        — Optimisers (SGD, Adam, AdamW, etc.)
//   tpt.data         — Data loading and preprocessing utilities
//   tpt.io           — File I/O (CSV, Parquet, HDF5, image formats)
//   tpt.dist         — Distributed training utilities
//   tpt.compat       — Interoperability shims (PyTorch, JAX, NumPy)
// ---------------------------------------------------------------------------

pub mod config;
pub mod resolve;
pub mod stdlib;

pub use config::ProjectConfig;
pub use resolve::{resolve_imports_preamble, validate_imports};
pub use stdlib::{
    generate_gitignore, generate_main_tpts, generate_project_toml, get_std_module, is_std_module,
    list_module_names, std_library_modules, StdModule,
};
