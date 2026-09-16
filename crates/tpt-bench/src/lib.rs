//! # tpt-bench — the Cobalt benchmark suite (Phase 7)
//!
//! Two halves:
//!
//! 1. **Criterion micro-benchmarks** (`benches/matmul.rs`, `benches/ml.rs`)
//!    over the core kernels — matmul/bmm, Linear forward/backward,
//!    transformer block, and the TPT-Script `train_step` path.
//! 2. **A report generator** ([`report_json`] / the `tpt-bench-report`
//!    binary) that measures the same kernels with wall-clock timings and
//!    emits JSON + a Markdown table, together with
//!    `benches/PYTORCH_PROTOCOL.md` — the fixed recipe for running the
//!    *identical* workload in PyTorch on the same machine so the numbers
//!    are comparable (same shapes, same dtype, warmup/measurement rules).
//!
//! The suite is the Phase 7 "PyTorch benchmark suite" deliverable's Cobalt
//! half; the PyTorch half is a ~40-line script per protocol, kept out of
//! this repo (no Python in the workspace).

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use tpt_ml::Module;
use tpt_tensor::Tensor;

/// One measured kernel.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Measurement {
    pub name: String,
    /// Element count of the dominant tensor (for FLOP reasoning).
    pub elements: usize,
    pub iters: usize,
    pub total: Duration,
    pub mean_us: f64,
}

fn measure<F: FnMut()>(name: &str, elements: usize, iters: usize, mut f: F) -> Measurement {
    // warmup (same rule the PyTorch protocol uses: 10 untimed warmup runs)
    for _ in 0..10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = start.elapsed();
    Measurement {
        name: name.to_string(),
        elements,
        iters,
        total,
        mean_us: total.as_secs_f64() * 1e6 / iters as f64,
    }
}

fn rand_mat(rows: usize, cols: usize) -> Tensor {
    // deterministic LCG fill: reproducible reports, no rng dependency
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f64 / 1000.0 - 1.0
        })
        .collect();
    Tensor::from_typed(data).reshape(&[rows, cols]).unwrap()
}

/// Run every kernel in the fixed suite. `iters` trades runtime for stability.
pub fn run_report(iters: usize) -> Vec<Measurement> {
    let mut out = Vec::new();

    for (n, it) in [(64usize, iters * 10), (128, iters * 4), (256, iters)] {
        let a = rand_mat(n, n);
        let b = rand_mat(n, n);
        out.push(measure(&format!("matmul_{n}x{n}_f64"), n * n, it, || {
            let _ = a.matmul(&b);
        }));
    }

    // linear forward + backward through the tape
    let x = rand_mat(128, 256);
    let w = rand_mat(256, 128).with_autograd();
    out.push(measure(
        "linear_128x256x128_fwd_bwd",
        128 * 256,
        iters,
        || {
            let w = w.clone();
            let x = x.clone().with_autograd();
            let y = tpt_autograd::matmul(&x, &w);
            tpt_autograd::backward(&y);
        },
    ));

    // transformer block forward (tpt-ml)
    let block = tpt_ml::attention::TransformerBlock::new(64, 4, 128);
    // [B, T, D] per the block's contract
    let input = rand_mat(2 * 8, 64).reshape(&[2, 8, 64]).unwrap();
    out.push(measure(
        "transformer_block_b8_t8_d64_fwd",
        2 * 8 * 64,
        iters,
        || {
            let _ = block.forward(&input);
        },
    ));

    // full TPT-Script train_step (the interpreter path, incl. AdamW)
    let mut interp = tpt_lang::Interpreter::new();
    interp
        .run("let net = mlp(8, 32, 4)\nlet xs = ones([16, 8])\nlet ys = ones([16, 4])\n")
        .unwrap();
    out.push(measure("tpt_script_train_step_b16", 16 * 8, iters, || {
        let _ = interp.run("train_step(net, xs, ys, 0.01)");
    }));

    out
}

/// Render the measurements as a Markdown table.
pub fn report_markdown(ms: &[Measurement]) -> String {
    let mut md = String::from("| kernel | elements | iters | mean (µs) |\n|---|---|---|---|\n");
    for m in ms {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {:.1} |",
            m.name, m.elements, m.iters, m.mean_us
        );
    }
    md
}

/// Render the measurements as JSON.
pub fn report_json(ms: &[Measurement]) -> String {
    serde_json::to_string_pretty(ms).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_runs_and_renders() {
        let ms = run_report(2);
        assert!(ms.len() >= 5, "expected the fixed kernel suite");
        assert!(ms.iter().all(|m| m.mean_us.is_finite() && m.mean_us >= 0.0));
        let md = report_markdown(&ms);
        assert!(md.contains("matmul_256x256_f64"));
        assert!(md.contains("tpt_script_train_step_b16"));
        let json = report_json(&ms);
        assert!(json.contains("\"name\""));
    }
}
