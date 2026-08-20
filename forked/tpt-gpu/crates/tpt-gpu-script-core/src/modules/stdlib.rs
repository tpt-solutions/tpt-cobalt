// ---------------------------------------------------------------------------
// stdlib — TPT Script standard library module registry
//
// Provides the 8 standard library modules defined in the spec:
//   tpt, tpt.introspect, tpt.nn, tpt.optim, tpt.data, tpt.io, tpt.dist, tpt.compat
// ---------------------------------------------------------------------------

/// A standard library module in the TPT Script namespace.
#[derive(Debug, Clone)]
pub struct StdModule {
    /// Full module path, e.g. "tpt.nn"
    pub path: String,
    /// Short description
    pub description: String,
    /// Operations provided by this module (names only)
    pub operations: Vec<&'static str>,
}

/// Returns all standard library modules.
pub fn std_library_modules() -> Vec<StdModule> {
    vec![
        StdModule {
            path: "tpt".into(),
            description: "Core tensor operations (auto-imported)".into(),
            operations: vec![
                "matmul",
                "add",
                "sub",
                "mul",
                "div",
                "relu",
                "gelu",
                "sigmoid",
                "tanh",
                "softmax",
                "transpose",
                "reshape",
                "concat",
                "split",
                "zeros",
                "ones",
                "eye",
                "arange",
                "linspace",
                "random",
                "randn",
                "exp",
                "log",
                "sqrt",
                "abs",
                "neg",
                "pow",
                "sum",
                "mean",
                "max",
                "min",
                "conv2d",
                "pool",
                "batch_norm",
                "layer_norm",
                "attention",
                "cross_entropy",
                "mse",
            ],
        },
        StdModule {
            path: "tpt.introspect".into(),
            description: "Introspection and schema API".into(),
            operations: vec![
                "list_operations",
                "get_schema",
                "validate_code",
                "get_capabilities",
                "get_current_hardware",
                "check_compatibility",
                "generate_openapi_schema",
                "generate_docs",
            ],
        },
        StdModule {
            path: "tpt.nn".into(),
            description: "Neural network building blocks".into(),
            operations: vec![
                "Linear",
                "Conv2d",
                "Conv3d",
                "BatchNorm",
                "LayerNorm",
                "GroupNorm",
                "Dropout",
                "Attention",
                "Transformer",
                "TransformerEncoder",
                "TransformerDecoder",
                "Embedding",
                "LSTM",
                "GRU",
                "Sequential",
                "Module",
                "ReLU",
                "GELU",
                "SiLU",
                "Sigmoid",
                "Tanh",
                "Softmax",
                "MaxPool2d",
                "AvgPool2d",
                "AdaptiveAvgPool2d",
                "Flatten",
                "ResNetBlock",
                "ResNet",
                "ViT",
                "GPTBlock",
            ],
        },
        StdModule {
            path: "tpt.optim".into(),
            description: "Optimisers (SGD, Adam, AdamW, etc.)".into(),
            operations: vec![
                "SGD",
                "Adam",
                "AdamW",
                "RMSprop",
                "Adagrad",
                "Adadelta",
                "LBFGS",
                "Rprop",
                "StepLR",
                "MultiStepLR",
                "ExponentialLR",
                "CosineAnnealingLR",
                "ReduceLROnPlateau",
                "OneCycleLR",
                "clip_grad_norm",
                "clip_grad_value",
            ],
        },
        StdModule {
            path: "tpt.data".into(),
            description: "Data loading and preprocessing utilities".into(),
            operations: vec![
                "DataLoader",
                "Dataset",
                "TensorDataset",
                "ConcatDataset",
                "Subset",
                "RandomSampler",
                "SequentialSampler",
                "BatchSampler",
                "collate_fn",
                "transform",
                "Normalize",
                "Resize",
                "CenterCrop",
                "RandomHorizontalFlip",
                "RandomRotation",
                "ToTensor",
            ],
        },
        StdModule {
            path: "tpt.io".into(),
            description: "File I/O (CSV, Parquet, HDF5, image formats)".into(),
            operations: vec![
                "load_csv",
                "save_csv",
                "load_parquet",
                "save_parquet",
                "load_hdf5",
                "save_hdf5",
                "load_image",
                "save_image",
                "load_numpy",
                "save_numpy",
                "load_pickle",
                "save_pickle",
                "read_file",
                "write_file",
                "exists",
                "mkdir",
                "list_dir",
            ],
        },
        StdModule {
            path: "tpt.dist".into(),
            description: "Distributed training utilities".into(),
            operations: vec![
                "init_process_group",
                "destroy_process_group",
                "get_rank",
                "get_world_size",
                "all_reduce",
                "all_gather",
                "broadcast",
                "scatter",
                "barrier",
                "send",
                "recv",
                "DistributedDataParallel",
                "FullyShardedDataParallel",
                "TensorParallel",
                "PipelineParallel",
            ],
        },
        StdModule {
            path: "tpt.compat".into(),
            description: "Interoperability shims (PyTorch, JAX, NumPy)".into(),
            operations: vec![
                "from_torch",
                "to_torch",
                "from_jax",
                "to_jax",
                "from_numpy",
                "to_numpy",
                "from_tpts",
                "to_tpts",
            ],
        },
    ]
}

/// Check if a module path is a valid standard library module.
pub fn is_std_module(path: &[String]) -> bool {
    let path_str = path.join("::");
    let path_str_alt = path.join(".");
    std_library_modules()
        .iter()
        .any(|m| m.path == path_str || m.path == path_str_alt)
}

/// Get a standard library module by its path.
pub fn get_std_module(path: &[String]) -> Option<StdModule> {
    let path_str = path.join("::");
    let path_str_alt = path.join(".");
    std_library_modules()
        .into_iter()
        .find(|m| m.path == path_str || m.path == path_str_alt)
}

// ---------------------------------------------------------------------------
// Project scaffolding generators
// ---------------------------------------------------------------------------

/// Generate a default `tpt.toml` for a new project.
pub fn generate_project_toml(name: &str) -> String {
    format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nauthors = [\"Your Name <you@example.com>\"]\ndescription = \"A TPT Script project\"\nlicense = \"Apache-2.0\"\n\n[features]\ngpu = true\ndistributed = false\n\n[profile]\nopt-level = 2\ndebug-assertions = false\ntarget = \"rust\"\n\n[dependencies]\ntpt = \"1.0\"\n",
        name
    )
}

/// Generate a default `main.tpts` for a new project.
pub fn generate_main_tpts() -> String {
    String::from(
        "// TPT Script — Main entry point\n// Generated by `tpt new`\n\nimport tpt\nimport tpt.nn\nimport tpt.optim\n\n@doc(\"Main entry point\")\nfn main() {\n    // Your TPT Script program starts here\n    let x = tpt.zeros([4, 4], dtype=f32)\n    let y = tpt.relu(x)\n    tpt.print(y)\n}\n"
    )
}

/// Generate a `.gitignore` for a TPT Script project.
pub fn generate_gitignore() -> String {
    String::from(
        "# TPT Script build artifacts\n/target/\n*.tptir\n*.tptisa\n\n# IDE\n.vscode/\n.idea/\n*.swp\n\n# OS\n.DS_Store\nThumbs.db\n"
    )
}

/// List all available standard library module names.
pub fn list_module_names() -> Vec<&'static str> {
    vec![
        "tpt",
        "tpt.introspect",
        "tpt.nn",
        "tpt.optim",
        "tpt.data",
        "tpt.io",
        "tpt.dist",
        "tpt.compat",
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_library_modules_count() {
        let modules = std_library_modules();
        assert!(modules.len() >= 8);
    }

    #[test]
    fn test_is_std_module() {
        assert!(is_std_module(&["tpt".to_string(), "nn".to_string()]));
        assert!(is_std_module(&["tpt".to_string(), "optim".to_string()]));
        assert!(is_std_module(&["tpt".to_string()]));
        assert!(!is_std_module(&[
            "tpt".to_string(),
            "nonexistent".to_string()
        ]));
        assert!(!is_std_module(&["my_module".to_string()]));
    }

    #[test]
    fn test_generate_project_toml() {
        let toml = generate_project_toml("my-project");
        assert!(toml.contains("name = \"my-project\""));
        assert!(toml.contains("[package]"));
    }

    #[test]
    fn test_generate_main_tpts() {
        let src = generate_main_tpts();
        assert!(src.contains("import tpt"));
        assert!(src.contains("fn main()"));
    }
}
