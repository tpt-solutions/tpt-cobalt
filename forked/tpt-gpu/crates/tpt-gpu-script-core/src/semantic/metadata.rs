use crate::ast::{Annotation, AnnotationArg, AnnotationValue, FunctionDecl, TypeDecl};
use crate::semantic::constraints::{parse_constraint, ConstraintExpr, ConstraintParseError};

// ---------------------------------------------------------------------------
// Structured metadata types extracted from annotations
// ---------------------------------------------------------------------------

/// Documentation for one input parameter (from `@input`).
#[derive(Debug, Clone)]
pub struct InputMeta {
    /// The raw type string, e.g. `"Tensor[f32, m, k]"`.
    pub type_str: String,
    pub description: Option<String>,
}

/// Documentation for the return value (from `@output`).
#[derive(Debug, Clone)]
pub struct OutputMeta {
    pub type_str: String,
    pub description: Option<String>,
}

/// A single `@constraint` annotation.
#[derive(Debug, Clone)]
pub struct ConstraintMeta {
    /// Raw expression string, e.g. `"a.shape[1] == b.shape[0]"`.
    pub expr_str: String,
    /// Parsed constraint (if the expression was valid).
    pub expr: Result<ConstraintExpr, ConstraintParseError>,
    /// Error message to emit when the constraint is violated.
    pub error_msg: Option<String>,
}

/// Hardware capability requirements extracted from `@requires_*` annotations.
#[derive(Debug, Clone, Default)]
pub struct HardwareCaps {
    pub requires_gpu: bool,
    pub requires_tensor_cores: bool,
    pub min_vram_gb: u32,
    pub supports_distributed: bool,
    pub max_batch_size: Option<u32>,
    pub preferred_dtype: Option<String>,
    pub gpu_optimized: bool,
}

/// Distributed / deployment execution metadata.
#[derive(Debug, Clone, Default)]
pub struct ExecutionMeta {
    /// e.g. `"fsdp"`, `"ddp"`, `"tensor_parallel"`, `"pipeline_parallel"`
    pub distributed_strategy: Option<String>,
    pub distributed_devices: Option<u32>,
    /// e.g. `"edge"`, `"cloud"`, `"mobile"`
    pub deploy_target: Option<String>,
    pub deploy_optimize: bool,
    pub async_exec: bool,
}

/// All structured metadata for a single function.
#[derive(Debug, Clone, Default)]
pub struct FunctionMeta {
    pub doc: Option<String>,
    pub inputs: Vec<InputMeta>,
    pub output: Option<OutputMeta>,
    pub examples: Vec<String>,
    pub constraints: Vec<ConstraintMeta>,
    pub complexity: Option<String>,
    pub memory_complexity: Option<String>,
    pub flops: Option<String>,
    pub differentiable: Option<bool>,
    pub gradient_checkpoint: Option<bool>,
    pub hardware: HardwareCaps,
    pub execution: ExecutionMeta,
}

/// All structured metadata for a type alias.
#[derive(Debug, Clone, Default)]
pub struct TypeMeta {
    pub doc: Option<String>,
    pub examples: Vec<String>,
    pub constraints: Vec<ConstraintMeta>,
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract structured metadata from a function's annotations.
pub fn extract_function_metadata(decl: &FunctionDecl) -> FunctionMeta {
    let mut meta = FunctionMeta::default();
    for ann in &decl.annotations {
        apply_annotation_to_fn(&mut meta, ann);
    }
    meta
}

/// Extract structured metadata from a type alias's annotations.
pub fn extract_type_metadata(decl: &TypeDecl) -> TypeMeta {
    let mut meta = TypeMeta::default();
    for ann in &decl.annotations {
        apply_annotation_to_type(&mut meta, ann);
    }
    meta
}

fn apply_annotation_to_fn(meta: &mut FunctionMeta, ann: &Annotation) {
    match ann.name.as_str() {
        "doc" => {
            if let Some(s) = first_string(&ann.args) {
                meta.doc = Some(s);
            }
        }
        "input" => {
            let type_str = first_string(&ann.args).unwrap_or_default();
            let description = named_string(&ann.args, "description");
            meta.inputs.push(InputMeta {
                type_str,
                description,
            });
        }
        "output" => {
            let type_str = first_string(&ann.args).unwrap_or_default();
            let description = named_string(&ann.args, "description");
            meta.output = Some(OutputMeta {
                type_str,
                description,
            });
        }
        "example" => {
            if let Some(s) = first_string(&ann.args) {
                meta.examples.push(s);
            }
        }
        "constraint" => {
            let expr_str = first_string(&ann.args).unwrap_or_default();
            let error_msg = named_string(&ann.args, "error");
            let expr = parse_constraint(&expr_str);
            meta.constraints.push(ConstraintMeta {
                expr_str,
                expr,
                error_msg,
            });
        }
        "complexity" => {
            meta.complexity = first_string(&ann.args);
        }
        "memory" => {
            meta.memory_complexity = first_string(&ann.args);
        }
        "flops" => {
            meta.flops = first_string(&ann.args);
        }
        "differentiable" => {
            meta.differentiable = first_bool(&ann.args);
        }
        "gradient_checkpoint" => {
            meta.gradient_checkpoint =
                named_bool(&ann.args, "enabled").or_else(|| first_bool(&ann.args));
        }
        "requires_gpu" => {
            if let Some(b) = first_bool(&ann.args) {
                meta.hardware.requires_gpu = b;
            }
        }
        "requires_tensor_cores" => {
            if let Some(b) = first_bool(&ann.args) {
                meta.hardware.requires_tensor_cores = b;
            }
        }
        "min_vram_gb" => {
            if let Some(n) = first_int(&ann.args) {
                meta.hardware.min_vram_gb = n as u32;
            }
        }
        "supports_distributed" => {
            if let Some(b) = first_bool(&ann.args) {
                meta.hardware.supports_distributed = b;
            }
        }
        "max_batch_size" => {
            if let Some(n) = first_int(&ann.args) {
                meta.hardware.max_batch_size = Some(n as u32);
            }
        }
        "preferred_dtype" => {
            meta.hardware.preferred_dtype = first_string(&ann.args);
        }
        "gpu_optimized" => {
            if let Some(b) = first_bool(&ann.args) {
                meta.hardware.gpu_optimized = b;
            }
        }
        "distributed" => {
            meta.execution.distributed_strategy = named_string(&ann.args, "strategy");
            if let Some(n) = named_int(&ann.args, "devices") {
                meta.execution.distributed_devices = Some(n as u32);
            }
        }
        "deploy" => {
            meta.execution.deploy_target = named_string(&ann.args, "target");
            meta.execution.deploy_optimize = named_bool(&ann.args, "optimize").unwrap_or(false);
        }
        "async_exec" => {
            meta.execution.async_exec = true;
        }
        // Unknown annotations are silently skipped per spec §7.7.
        _ => {}
    }
}

fn apply_annotation_to_type(meta: &mut TypeMeta, ann: &Annotation) {
    match ann.name.as_str() {
        "doc" => {
            meta.doc = first_string(&ann.args);
        }
        "example" => {
            if let Some(s) = first_string(&ann.args) {
                meta.examples.push(s);
            }
        }
        "constraint" => {
            let expr_str = first_string(&ann.args).unwrap_or_default();
            let error_msg = named_string(&ann.args, "error");
            let expr = parse_constraint(&expr_str);
            meta.constraints.push(ConstraintMeta {
                expr_str,
                expr,
                error_msg,
            });
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Argument extraction helpers
// ---------------------------------------------------------------------------

fn first_string(args: &[AnnotationArg]) -> Option<String> {
    for arg in args {
        let val = match arg {
            AnnotationArg::Positional { value, .. } => value,
            AnnotationArg::Named { value, .. } => value,
        };
        if let AnnotationValue::String(s) = val {
            return Some(s.clone());
        }
    }
    None
}

fn first_bool(args: &[AnnotationArg]) -> Option<bool> {
    for arg in args {
        let val = match arg {
            AnnotationArg::Positional { value, .. } => value,
            AnnotationArg::Named { value, .. } => value,
        };
        if let AnnotationValue::Bool(b) = val {
            return Some(*b);
        }
    }
    None
}

fn first_int(args: &[AnnotationArg]) -> Option<i64> {
    for arg in args {
        let val = match arg {
            AnnotationArg::Positional { value, .. } => value,
            AnnotationArg::Named { value, .. } => value,
        };
        if let AnnotationValue::Int(n) = val {
            return Some(*n);
        }
    }
    None
}

fn named_string(args: &[AnnotationArg], key: &str) -> Option<String> {
    for arg in args {
        if let AnnotationArg::Named {
            key: k,
            value: AnnotationValue::String(s),
            ..
        } = arg
        {
            if k == key {
                return Some(s.clone());
            }
        }
    }
    None
}

fn named_bool(args: &[AnnotationArg], key: &str) -> Option<bool> {
    for arg in args {
        if let AnnotationArg::Named {
            key: k,
            value: AnnotationValue::Bool(b),
            ..
        } = arg
        {
            if k == key {
                return Some(*b);
            }
        }
    }
    None
}

fn named_int(args: &[AnnotationArg], key: &str) -> Option<i64> {
    for arg in args {
        if let AnnotationArg::Named {
            key: k,
            value: AnnotationValue::Int(n),
            ..
        } = arg
        {
            if k == key {
                return Some(*n);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;

    fn parse_fn(src: &str) -> FunctionDecl {
        let prog = parse(tokenize(src).unwrap()).unwrap();
        match prog.items.into_iter().next().unwrap() {
            crate::ast::Item::Function(f) => f,
            _ => panic!("expected function"),
        }
    }

    #[test]
    fn test_doc_extraction() {
        let f = parse_fn(r#"@doc("Matrix multiply") fn f() {}"#);
        let meta = extract_function_metadata(&f);
        assert_eq!(meta.doc.as_deref(), Some("Matrix multiply"));
    }

    #[test]
    fn test_hardware_caps() {
        let f = parse_fn(
            r#"@requires_gpu(true) @requires_tensor_cores(true) @min_vram_gb(16) fn f() {}"#,
        );
        let meta = extract_function_metadata(&f);
        assert!(meta.hardware.requires_gpu);
        assert!(meta.hardware.requires_tensor_cores);
        assert_eq!(meta.hardware.min_vram_gb, 16);
    }

    #[test]
    fn test_constraint_extraction() {
        let f = parse_fn(
            r#"@constraint("a.shape[1] == b.shape[0]", error="Inner dims must match") fn f() {}"#,
        );
        let meta = extract_function_metadata(&f);
        assert_eq!(meta.constraints.len(), 1);
        assert_eq!(meta.constraints[0].expr_str, "a.shape[1] == b.shape[0]");
        assert_eq!(
            meta.constraints[0].error_msg.as_deref(),
            Some("Inner dims must match")
        );
        assert!(meta.constraints[0].expr.is_ok());
    }

    #[test]
    fn test_distributed_extraction() {
        let f = parse_fn(r#"@distributed(strategy="fsdp", devices=8) fn f() {}"#);
        let meta = extract_function_metadata(&f);
        assert_eq!(meta.execution.distributed_strategy.as_deref(), Some("fsdp"));
        assert_eq!(meta.execution.distributed_devices, Some(8));
    }

    #[test]
    fn test_complexity_extraction() {
        let f = parse_fn(r#"@complexity("O(m * n * k)") @memory("O(m * n)") fn f() {}"#);
        let meta = extract_function_metadata(&f);
        assert_eq!(meta.complexity.as_deref(), Some("O(m * n * k)"));
        assert_eq!(meta.memory_complexity.as_deref(), Some("O(m * n)"));
    }

    #[test]
    fn test_differentiable() {
        let f = parse_fn(r#"@differentiable(true) fn f() {}"#);
        let meta = extract_function_metadata(&f);
        assert_eq!(meta.differentiable, Some(true));
    }

    #[test]
    fn test_unknown_annotation_skipped() {
        // Unknown annotations should produce no error per spec §7.7.
        let f = parse_fn(r#"@future_feature(42) fn f() {}"#);
        let meta = extract_function_metadata(&f);
        // Just shouldn't panic; metadata is empty.
        assert!(meta.doc.is_none());
    }
}
