// ---------------------------------------------------------------------------
// resolve — TPT Script import resolution
//
// Resolves import declarations to standard library modules and generates
// the appropriate Rust preamble for emitted code.
// ---------------------------------------------------------------------------

use std::collections::HashSet;

use super::config::ProjectConfig;
use crate::ast::{Item, Program};

/// Resolve imports and generate a Rust preamble for the emitted code.
pub fn resolve_imports_preamble(program: &Program, config: &ProjectConfig) -> String {
    let mut modules_imported: HashSet<String> = HashSet::new();
    let mut lines: Vec<String> = Vec::new();

    for item in &program.items {
        if let Item::Import(import) = item {
            let path_str = import.path.join("::");

            if modules_imported.contains(&path_str) {
                continue;
            }
            modules_imported.insert(path_str.clone());

            match path_str.as_str() {
                "tpt" => {
                    lines.push("// Core tensor operations (auto-imported)".into());
                    lines.push("use tpt::prelude::*;".into());
                }
                "tpt::introspect" | "tpt.introspect" => {
                    lines.push("use tpt::introspect;".into());
                }
                "tpt::nn" | "tpt.nn" => {
                    lines.push("use tpt::nn;".into());
                    if config.is_gpu_enabled() {
                        lines.push("use tpt::nn::layers::*;".into());
                    }
                }
                "tpt::optim" | "tpt.optim" => {
                    lines.push("use tpt::optim;".into());
                }
                "tpt::data" | "tpt.data" => {
                    lines.push("use tpt::data;".into());
                }
                "tpt::io" | "tpt.io" => {
                    lines.push("use tpt::io;".into());
                }
                "tpt::dist" | "tpt.dist" => {
                    if config.is_distributed_enabled() {
                        lines.push("use tpt::dist;".into());
                    } else {
                        lines.push(
                            "// WARNING: tpt.dist imported but 'distributed' feature not enabled"
                                .into(),
                        );
                        lines.push("use tpt::dist;".into());
                    }
                }
                "tpt::compat" | "tpt.compat" => {
                    lines.push("use tpt::compat;".into());
                }
                _ => {
                    let import_line = if let Some(alias) = &import.alias {
                        format!("use {} as {};", path_str, alias)
                    } else {
                        format!("use {};", path_str)
                    };
                    lines.push(import_line);
                }
            }
        }
    }

    if lines.is_empty() {
        lines.push("// Auto-imported: core tensor operations".into());
        lines.push("use tpt::prelude::*;".into());
    }

    lines.join("\n") + "\n"
}

/// Validate that all imports in a program resolve to known modules.
pub fn validate_imports(program: &Program) -> Vec<String> {
    let mut unresolved = Vec::new();

    for item in &program.items {
        if let Item::Import(import) = item {
            let path_str = import.path.join("::");
            if path_str.starts_with("tpt") && !is_valid_tpt_module(&path_str) {
                unresolved.push(format!("Unknown standard library module: `{}`", path_str));
            }
        }
    }

    unresolved
}

/// Check if a `tpt.*` path is a valid standard library module.
fn is_valid_tpt_module(path: &str) -> bool {
    matches!(
        path,
        "tpt"
            | "tpt::introspect"
            | "tpt::nn"
            | "tpt::optim"
            | "tpt::data"
            | "tpt::io"
            | "tpt::dist"
            | "tpt::compat"
            | "tpt.introspect"
            | "tpt.nn"
            | "tpt.optim"
            | "tpt.data"
            | "tpt.io"
            | "tpt.dist"
            | "tpt.compat"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_imports_auto() {
        let src = "fn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let config = ProjectConfig::new("test");
        let preamble = resolve_imports_preamble(&prog, &config);
        assert!(preamble.contains("use tpt::prelude::*"));
    }

    #[test]
    fn test_resolve_imports_nn() {
        let src = "import tpt.nn\nfn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let config = ProjectConfig::new("test");
        let preamble = resolve_imports_preamble(&prog, &config);
        assert!(preamble.contains("use tpt::nn"));
    }

    #[test]
    fn test_resolve_imports_gpu_nn() {
        let src = "import tpt.nn\nfn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let mut config = ProjectConfig::new("test");
        config.features.insert("gpu".into());
        let preamble = resolve_imports_preamble(&prog, &config);
        assert!(preamble.contains("use tpt::nn"));
        assert!(preamble.contains("use tpt::nn::layers::*"));
    }

    #[test]
    fn test_resolve_imports_dist_warns() {
        let src = "import tpt.dist\nfn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let config = ProjectConfig::new("test");
        let preamble = resolve_imports_preamble(&prog, &config);
        assert!(preamble.contains("WARNING"));
    }

    #[test]
    fn test_validate_imports_valid() {
        let src = "import tpt\nimport tpt.nn\nfn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let unresolved = validate_imports(&prog);
        assert!(unresolved.is_empty(), "unresolved: {:?}", unresolved);
    }

    #[test]
    fn test_validate_imports_unknown() {
        let src = "import tpt.nonexistent\nfn main() { }";
        let prog = crate::compile_str(src).unwrap();
        let unresolved = validate_imports(&prog);
        assert!(!unresolved.is_empty());
    }
}
