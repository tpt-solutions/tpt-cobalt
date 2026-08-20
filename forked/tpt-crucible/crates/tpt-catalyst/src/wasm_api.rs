//! WASM API for compiling models in the browser.

use wasm_bindgen::prelude::*;

use crate::package::PackageManifest;

/// Compile result returned to JavaScript.
#[wasm_bindgen]
pub struct CompileResult {
    manifest_json: String,
    success: bool,
    error_message: String,
}

#[wasm_bindgen]
impl CompileResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }

    #[wasm_bindgen(getter)]
    pub fn manifest(&self) -> String {
        self.manifest_json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn error(&self) -> String {
        self.error_message.clone()
    }
}

/// Compile a TPT-IR JSON string into a package manifest.
///
/// This is the primary WASM entry point for browser-based compilation.
/// Takes TPT-IR JSON and target configuration, returns a PackageManifest.
#[wasm_bindgen]
pub fn compile(ir_json: &str, targets_json: &str) -> CompileResult {
    let ir_result = crate::package::PackageManifest::from_json(ir_json);

    match ir_result {
        Ok(manifest) => {
            let targets: Vec<String> = serde_json::from_str(targets_json).unwrap_or_default();
            let mut pkg = manifest;
            for target in &targets {
                pkg.targets.push(crate::package::TargetEntry {
                    name: target.clone(),
                    artifacts: Vec::new(),
                });
            }

            match pkg.to_json() {
                Ok(json) => CompileResult {
                    manifest_json: json,
                    success: true,
                    error_message: String::new(),
                },
                Err(e) => CompileResult {
                    manifest_json: String::new(),
                    success: false,
                    error_message: format!("Serialization error: {e}"),
                },
            }
        }
        Err(e) => CompileResult {
            manifest_json: String::new(),
            success: false,
            error_message: format!("IR parse error: {e}"),
        },
    }
}

/// Validate a TPT-IR JSON string and return validation results.
#[wasm_bindgen]
pub fn validate_ir(ir_json: &str) -> String {
    match crate::package::PackageManifest::from_json(ir_json) {
        Ok(m) => {
            let result = serde_json::json!({
                "valid": true,
                "model_name": m.model_name,
                "targets": m.targets.len(),
            });
            result.to_string()
        }
        Err(e) => {
            let result = serde_json::json!({
                "valid": false,
                "error": e.to_string(),
            });
            result.to_string()
        }
    }
}

/// Operators not supported on each hardware target.
const ALLOY_UNSUPPORTED: &[&str] = &["FlashAttention", "SwiGLU", "RMSNorm"];
const FUSION_UNSUPPORTED: &[&str] = &["SwiGLU"];
const ELEMENT_UNSUPPORTED: &[&str] = &["FlashAttention", "Embedding", "Softmax", "LayerNorm"];
const CIM_UNSUPPORTED: &[&str] = &["FlashAttention", "Softmax", "LayerNorm", "RMSNorm"];
const NEUROMORPHIC_UNSUPPORTED: &[&str] = &["FlashAttention", "LayerNorm", "RMSNorm"];

/// Maximum accepted IR JSON size (50 MB).
const MAX_IR_BYTES: usize = 50 * 1024 * 1024;

/// Check compatibility of operators against a hardware target.
///
/// Returns a JSON object with: target, score (0.0–1.0), status (pass/warn/fail),
/// total_ops, unsupported_count, warnings[], model_name.
#[wasm_bindgen]
pub fn check_compatibility(ir_json: &str, target: &str) -> String {
    if ir_json.len() > MAX_IR_BYTES {
        return serde_json::json!({
            "error": "IR JSON exceeds 50 MB limit",
            "target": target,
            "score": 0.0,
            "status": "fail",
        })
        .to_string();
    }

    let ir = match crate::ir::TptIr::from_json(ir_json) {
        Ok(ir) => ir,
        Err(e) => {
            return serde_json::json!({
                "error": format!("Invalid IR JSON: {e}"),
                "target": target,
                "score": 0.0,
                "status": "fail",
            })
            .to_string();
        }
    };

    let unsupported: &[&str] = match target {
        "alloy" => ALLOY_UNSUPPORTED,
        "fusion" => FUSION_UNSUPPORTED,
        "element" => ELEMENT_UNSUPPORTED,
        "cim" | "silicon" => CIM_UNSUPPORTED,
        "neuromorphic" | "pulse" => NEUROMORPHIC_UNSUPPORTED,
        _ => &[],
    };

    let total = ir.graph.nodes.len();
    let mut warnings: Vec<String> = Vec::new();
    let mut unsupported_count = 0usize;

    for node in &ir.graph.nodes {
        if unsupported.contains(&node.op_type.as_str()) {
            warnings.push(format!(
                "Op '{}' (node '{}') not supported on {}",
                node.op_type, node.name, target
            ));
            unsupported_count += 1;
        }
    }

    let score = if total == 0 {
        1.0_f64
    } else {
        (total - unsupported_count) as f64 / total as f64
    };

    let status = if unsupported_count == 0 {
        "pass"
    } else if score >= 0.8 {
        "warn"
    } else {
        "fail"
    };

    serde_json::json!({
        "target": target,
        "score": score,
        "status": status,
        "total_ops": total,
        "unsupported_count": unsupported_count,
        "warnings": warnings,
        "model_name": ir.metadata.name,
    })
    .to_string()
}
