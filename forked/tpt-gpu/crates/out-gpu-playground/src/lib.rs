use serde::Serialize;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CompileResult {
    rust_source: String,
    tptir_source: String,
    errors: Vec<ErrorInfo>,
    perf_estimate: PerfEstimate,
}

#[derive(Serialize)]
struct ErrorInfo {
    code: String,
    message: String,
    line: u32,
    col: u32,
    fix_code: Option<String>,
    suggestion: Option<String>,
}

#[derive(Serialize)]
struct PerfEstimate {
    ops: Vec<OpEstimate>,
    total_gflops: f64,
    sim_time_ms: f64,
    memory_gb: f64,
    note: String,
}

#[derive(Serialize)]
struct OpEstimate {
    op: String,
    count: usize,
    gflops_each: f64,
    description: String,
}

// ---------------------------------------------------------------------------
// WASM entry point
// ---------------------------------------------------------------------------

/// Compile TPT Script source and return a JSON string with:
/// - `rust_source`    — host-side Rust output
/// - `tptir_source`   — GPU kernel TPTIR output
/// - `errors`         — structured diagnostics with fix suggestions
/// - `perf_estimate`  — simulated performance breakdown
#[wasm_bindgen]
pub fn compile(source: &str) -> String {
    let result = match tpt_gpu_script_core::compile_full(source) {
        Ok((checker, output)) => {
            let errors: Vec<ErrorInfo> = checker
                .errors
                .iter()
                .map(|e| ErrorInfo {
                    code: e.code.to_string(),
                    message: e.message.clone(),
                    line: e.span.line,
                    col: e.span.col,
                    fix_code: e.fix_code.clone(),
                    suggestion: e.suggestion.clone(),
                })
                .collect();

            let perf = estimate_perf(&output.tptir_source);

            CompileResult {
                rust_source: output.rust_source,
                tptir_source: output.tptir_source,
                errors,
                perf_estimate: perf,
            }
        }
        Err(e) => CompileResult {
            rust_source: String::new(),
            tptir_source: String::new(),
            errors: vec![ErrorInfo {
                code: "COMPILE_ERROR".to_string(),
                message: e.to_string(),
                line: 0,
                col: 0,
                fix_code: None,
                suggestion: None,
            }],
            perf_estimate: PerfEstimate {
                ops: vec![],
                total_gflops: 0.0,
                sim_time_ms: 0.0,
                memory_gb: 0.0,
                note: "Compilation failed — fix errors to see a perf estimate.".to_string(),
            },
        },
    };

    serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
}

// ---------------------------------------------------------------------------
// Performance estimator
//
// Scans TPTIR text for known `tpt.*` operations and applies reference FLOP
// counts at symbolic dimensions (M=N=K=1024, seq=512, hidden=768).
// Reported against "TPT SimGPU": 20 TFLOPS f32 / 500 GB/s bandwidth.
// All estimates are illustrative — real numbers require hardware profiling.
// ---------------------------------------------------------------------------

struct OpDef {
    name: &'static str,
    gflops: f64,
    mem_gb: f64,
    description: &'static str,
}

const SIM_TFLOPS: f64 = 20.0;
const SIM_BW_GBS: f64 = 500.0;

// Reference symbolic dims used when tensor dims are `*` in TPTIR.
const M: f64 = 1024.0;
const N: f64 = 1024.0;
const K: f64 = 1024.0;
const SEQ: f64 = 512.0;
const HEADS: f64 = 8.0;
const BATCH: f64 = 1.0;
const ELEM: f64 = 1_048_576.0; // 1M elements for elementwise ops

fn op_table() -> Vec<OpDef> {
    vec![
        OpDef {
            name: "tpt.gemm",
            gflops: 2.0 * M * N * K / 1e9,
            mem_gb: (M * K + K * N + M * N) * 4.0 / 1e9,
            description: "GEMM (M=N=K=1024 symbolic)",
        },
        OpDef {
            name: "tpt.matmul",
            gflops: 2.0 * M * N * K / 1e9,
            mem_gb: (M * K + K * N + M * N) * 4.0 / 1e9,
            description: "MatMul (M=N=K=1024 symbolic)",
        },
        OpDef {
            name: "tpt.attention",
            gflops: 4.0 * BATCH * HEADS * SEQ * SEQ / 1e9,
            mem_gb: BATCH * HEADS * SEQ * SEQ * 4.0 / 1e9,
            description: "Attention (B=1, H=8, S=512 symbolic)",
        },
        OpDef {
            name: "tpt.conv2d",
            gflops: 2.0 * 64.0 * 64.0 * 3.0 * 3.0 * 224.0 * 224.0 / 1e9,
            mem_gb: (64.0 * 3.0 * 3.0 * 64.0 + 64.0 * 224.0 * 224.0) * 4.0 / 1e9,
            description: "Conv2D (64→64, 3×3, 224×224 symbolic)",
        },
        OpDef {
            name: "tpt.conv3d",
            gflops: 2.0 * 32.0 * 32.0 * 3.0 * 3.0 * 3.0 * 64.0 * 64.0 * 64.0 / 1e9,
            mem_gb: 32.0 * 32.0 * 3.0 * 3.0 * 3.0 * 4.0 / 1e9,
            description: "Conv3D (32→32, 3×3×3, 64³ symbolic)",
        },
        OpDef {
            name: "tpt.softmax",
            gflops: 5.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Softmax (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.layer_norm",
            gflops: 5.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "LayerNorm (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.group_norm",
            gflops: 5.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "GroupNorm (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.batch_norm",
            gflops: 3.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "BatchNorm (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.relu",
            gflops: ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "ReLU (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.gelu",
            gflops: 8.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "GELU (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.silu",
            gflops: 4.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "SiLU (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.sigmoid",
            gflops: 4.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Sigmoid (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.tanh",
            gflops: 6.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Tanh (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.leaky_relu",
            gflops: ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "LeakyReLU (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.elu",
            gflops: 3.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "ELU (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.exp",
            gflops: 2.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Exp (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.log",
            gflops: 2.0 * ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Log (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.sqrt",
            gflops: ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Sqrt (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.dropout",
            gflops: ELEM / 1e9,
            mem_gb: ELEM * 4.0 / 1e9,
            description: "Dropout (1M elements symbolic)",
        },
        OpDef {
            name: "tpt.embedding",
            gflops: SEQ * 768.0 / 1e9,
            mem_gb: SEQ * 768.0 * 4.0 / 1e9,
            description: "Embedding lookup (seq=512, dim=768 symbolic)",
        },
    ]
}

fn estimate_perf(tptir: &str) -> PerfEstimate {
    if tptir.contains("no GPU kernel functions found") || tptir.trim().is_empty() {
        return PerfEstimate {
            ops: vec![],
            total_gflops: 0.0,
            sim_time_ms: 0.0,
            memory_gb: 0.0,
            note: "No GPU kernels — all functions run on the host CPU.".to_string(),
        };
    }

    let table = op_table();
    let mut estimates: Vec<OpEstimate> = Vec::new();
    let mut total_gflops = 0.0_f64;
    let mut total_mem_gb = 0.0_f64;

    for def in &table {
        // Count occurrences of `def.name` in the TPTIR text.
        // We match the op name followed by a space, newline, or end-of-string
        // so `tpt.relu` doesn't match `tpt.relu_6` etc.
        let count = count_op(tptir, def.name);
        if count > 0 {
            let gflops = def.gflops * count as f64;
            let mem = def.mem_gb * count as f64;
            total_gflops += gflops;
            total_mem_gb += mem;
            estimates.push(OpEstimate {
                op: def.name.to_string(),
                count,
                gflops_each: def.gflops,
                description: def.description.to_string(),
            });
        }
    }

    // Roofline model: max(compute time, memory time)
    let compute_ms = if total_gflops > 0.0 {
        total_gflops / SIM_TFLOPS * 1000.0
    } else {
        0.0
    };
    let mem_ms = if total_mem_gb > 0.0 {
        total_mem_gb / SIM_BW_GBS * 1000.0
    } else {
        0.0
    };
    let sim_time_ms = compute_ms.max(mem_ms);

    let bottleneck = if estimates.is_empty() {
        "No recognized compute ops found."
    } else if compute_ms >= mem_ms {
        "Compute-bound"
    } else {
        "Memory-bandwidth-bound"
    };

    let note = format!(
        "Simulated on TPT SimGPU ({} TFLOPS f32 / {} GB/s) — symbolic dims (M=N=K=1024, seq=512). {}. Roofline: compute={:.3}ms, mem={:.3}ms.",
        SIM_TFLOPS as u32,
        SIM_BW_GBS as u32,
        bottleneck,
        compute_ms,
        mem_ms
    );

    PerfEstimate {
        ops: estimates,
        total_gflops,
        sim_time_ms,
        memory_gb: total_mem_gb,
        note,
    }
}

/// Count non-overlapping occurrences of `op` in `tptir` where `op` is
/// followed by a space, comma, newline, or end-of-string (to avoid substring hits).
fn count_op(tptir: &str, op: &str) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while let Some(idx) = tptir[pos..].find(op) {
        let abs = pos + idx;
        let after = abs + op.len();
        // Check the character immediately after the match.
        let next = tptir[after..].chars().next();
        match next {
            None | Some(' ') | Some('\n') | Some('\r') | Some('\t') | Some(',') | Some('}') => {
                count += 1;
            }
            _ => {}
        }
        pos = after;
    }
    count
}
