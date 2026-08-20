//! tpt-optimizer CLI
//!
//! Usage:
//!   tpt-optimizer grid      [--kernel <name>] [--elem <type>]
//!   tpt-optimizer climb     [--kernel <name>] [--tile-m N] ...
//!   tpt-optimizer ai        [--kernel <name>] [--iterations N]
//!   tpt-optimizer optimize  [--kernel <name>] [--elem <type>] [--ai]
//!   tpt-optimizer bench     [--target <pct>] [--ai] [--output <file>]
//!   tpt-optimizer beat-gemm [--ai] [--ai-iters N] [--output <file>]

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use tpt_gpu_kernel_optimizer::*;

#[derive(Parser)]
#[command(
    name = "tpt-optimizer",
    version,
    about = "TPT GPU kernel parameter optimizer"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Phase 1: exhaustive grid search over the parameter space
    Grid {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        #[arg(long, default_value = "10")]
        top: usize,
    },
    /// Phase 2: hill-climbing from a given starting point
    Climb {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "32")]
        tile_m: u32,
        #[arg(long, default_value = "32")]
        tile_n: u32,
        #[arg(long, default_value = "16")]
        tile_k: u32,
        #[arg(long, default_value = "4")]
        vec_width: u32,
        #[arg(long, default_value = "4")]
        unroll: u32,
        #[arg(long, default_value = "50")]
        max_iters: usize,
    },
    /// Phase 3: AI-guided search
    Ai {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "10")]
        iterations: usize,
    },
    /// Full pipeline: grid to hill-climb to optional AI refinement
    Optimize {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        #[arg(long)]
        ai: bool,
        #[arg(long, default_value = "10")]
        ai_iters: usize,
        #[arg(long)]
        output: Option<std::path::PathBuf>,
    },
    /// Run the GEMM >= 90% cuBLAS efficiency milestone optimizer loop
    Bench {
        #[arg(long, default_value = "90.0")]
        target: f64,
        #[arg(long)]
        ai: bool,
        #[arg(long, default_value = "10")]
        ai_iters: usize,
        #[arg(long, short)]
        output: Option<std::path::PathBuf>,
    },
    /// Run the Attention >= 90% FlashAttention v2 efficiency milestone optimizer loop
    BenchAttention {
        #[arg(long, default_value = "90.0")]
        target: f64,
        #[arg(long)]
        ai: bool,
        #[arg(long, default_value = "10")]
        ai_iters: usize,
        #[arg(long, short)]
        output: Option<std::path::PathBuf>,
    },
    /// Beat cuBLAS on at least one problem size using fused GEMM + AI-guided tuning
    BeatGemm {
        /// Enable AI-guided phase (requires AI provider env var)
        #[arg(long)]
        ai: bool,
        #[arg(long, default_value = "10")]
        ai_iters: usize,
        /// Output path (.md or .json)
        #[arg(long, short)]
        output: Option<std::path::PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Grid { kernel, elem, top } => handle_grid(kernel, elem, top),
        Commands::Climb {
            kernel,
            tile_m,
            tile_n,
            tile_k,
            vec_width,
            unroll,
            max_iters,
        } => handle_climb(kernel, tile_m, tile_n, tile_k, vec_width, unroll, max_iters),
        Commands::Ai { kernel, iterations } => handle_ai(kernel, iterations),
        Commands::Optimize {
            kernel,
            elem,
            ai,
            ai_iters,
            output,
        } => handle_optimize(kernel, elem, ai, ai_iters, output),
        Commands::Bench {
            target,
            ai,
            ai_iters,
            output,
        } => handle_bench(target, ai, ai_iters, output),
        Commands::BenchAttention {
            target,
            ai,
            ai_iters,
            output,
        } => handle_bench_attention(target, ai, ai_iters, output),
        Commands::BeatGemm {
            ai,
            ai_iters,
            output,
        } => handle_beat_gemm(ai, ai_iters, output),
    }
}

fn handle_grid(kernel: String, elem: String, top: usize) -> anyhow::Result<()> {
    let space = ParamSpace::for_kernel(&kernel).map_err(|e| anyhow::anyhow!(e))?;
    println!(
        "Grid search: {} (elem={}) -- {} configs",
        kernel,
        elem,
        space.total_configs()
    );
    let eval = SimulatedEvaluator::new(&kernel);
    let results = grid_search(&space, &eval);
    println!("\nTop {} results:", top.min(results.len()));
    for r in results.iter().take(top) {
        println!("  {:>7.2} GFLOPS  {}", r.score, r.params.display());
    }
    Ok(())
}

fn handle_climb(
    kernel: String,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    vec_width: u32,
    unroll: u32,
    max_iters: usize,
) -> anyhow::Result<()> {
    let space = ParamSpace::for_kernel(&kernel).map_err(|e| anyhow::anyhow!(e))?;
    let start = TuningParams(HashMap::from([
        ("tile_m".into(), tile_m),
        ("tile_n".into(), tile_n),
        ("tile_k".into(), tile_k),
        ("vec_width".into(), vec_width),
        ("unroll".into(), unroll),
    ]));
    println!("Hill-climb: {} -- start: {}", kernel, start.display());
    let eval = SimulatedEvaluator::new(&kernel);
    let result = hill_climb(&space, &start, &eval, max_iters);
    println!(
        "\nBest: {:.2} GFLOPS  {}  ({} evals)",
        result.score,
        result.params.display(),
        result.eval_count
    );
    Ok(())
}

fn handle_ai(kernel: String, iterations: usize) -> anyhow::Result<()> {
    let space = ParamSpace::for_kernel(&kernel).map_err(|e| anyhow::anyhow!(e))?;
    let provider = tpt_gpu_shared::provider_from_env();
    println!(
        "AI-guided search: {} -- provider: {} -- {} iterations",
        kernel,
        provider.name(),
        iterations
    );
    let start = TuningParams(HashMap::from([
        ("tile_m".into(), 32u32),
        ("tile_n".into(), 32),
        ("tile_k".into(), 16),
        ("vec_width".into(), 4),
        ("unroll".into(), 4),
    ]));
    let eval = SimulatedEvaluator::new(&kernel);
    let result = ai_guided_search(
        &space,
        &start,
        &eval,
        provider.as_ref(),
        &kernel,
        iterations,
    );
    println!(
        "\nBest: {:.2} GFLOPS  {}  ({} evals)",
        result.score,
        result.params.display(),
        result.eval_count
    );
    Ok(())
}

fn handle_optimize(
    kernel: String,
    elem: String,
    ai: bool,
    ai_iters: usize,
    output: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let space = ParamSpace::for_kernel(&kernel).map_err(|e| anyhow::anyhow!(e))?;
    let eval = SimulatedEvaluator::new(&kernel);
    println!(
        "[1/{}] Grid search ({} configs)...",
        if ai { 3 } else { 2 },
        space.total_configs()
    );
    let grid_results = grid_search(&space, &eval);
    let best_grid = &grid_results[0];
    println!(
        "  best: {:.2} GFLOPS  {}",
        best_grid.score,
        best_grid.params.display()
    );
    println!(
        "[2/{}] Hill-climbing from best grid point...",
        if ai { 3 } else { 2 }
    );
    let hc_result = hill_climb(&space, &best_grid.params, &eval, 100);
    println!(
        "  best: {:.2} GFLOPS  {}  ({} evals)",
        hc_result.score,
        hc_result.params.display(),
        hc_result.eval_count
    );
    let final_result = if ai {
        println!("[3/3] AI-guided refinement ({} iterations)...", ai_iters);
        let provider = tpt_gpu_shared::provider_from_env();
        println!("  provider: {}", provider.name());
        let r = ai_guided_search(
            &space,
            &hc_result.params,
            &eval,
            provider.as_ref(),
            &kernel,
            ai_iters,
        );
        println!(
            "  best: {:.2} GFLOPS  {}  ({} evals)",
            r.score,
            r.params.display(),
            r.eval_count
        );
        r
    } else {
        hc_result
    };
    println!(
        "\nFinal: {:.2} GFLOPS  {}",
        final_result.score,
        final_result.params.display()
    );
    let json_out = serde_json::json!({
        "kernel": kernel, "elem": elem,
        "params": final_result.params.0,
        "score_gflops": final_result.score,
        "total_evals": final_result.eval_count,
    });
    if let Some(path) = output {
        std::fs::write(&path, serde_json::to_string_pretty(&json_out)?)?;
        println!("Wrote results to {}", path.display());
    } else {
        println!("\n{}", serde_json::to_string_pretty(&json_out)?);
    }
    Ok(())
}

fn handle_bench(
    target: f64,
    ai: bool,
    ai_iters: usize,
    output: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    println!("GEMM >= 90% cuBLAS Efficiency Milestone");
    println!("========================================");
    println!("Target efficiency: {:.1}%", target);
    println!(
        "AI-guided phase: {}",
        if ai { "enabled" } else { "disabled" }
    );
    println!();
    let results = optimize_all_gemm_problems(target, ai, ai_iters);
    let report_md = generate_milestone_report(&results, target);
    let report_json = generate_milestone_json(&results, target);
    println!("\n{}", report_md);
    if let Some(path) = output {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("md");
        let content = match ext {
            "json" => serde_json::to_string_pretty(&report_json).expect("failed to serialize"),
            _ => report_md,
        };
        std::fs::write(&path, content).expect("failed to write output file");
        println!("Report written to: {}", path.display());
    }
    let all_pass = results.iter().all(|r| r.meets_target);
    if !all_pass {
        eprintln!(
            "\nMilestone NOT achieved: some problem sizes below {:.1}%",
            target
        );
        std::process::exit(1);
    } else {
        eprintln!("\nMilestone ACHIEVED: all problem sizes >= {:.1}%", target);
    }
    Ok(())
}

fn handle_beat_gemm(
    ai: bool,
    ai_iters: usize,
    output: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    println!("GEMM > cuBLAS Milestone — AI-Guided Fusion");
    println!("==========================================");
    println!("Strategy: fused GEMM+bias+activation vs cuBLAS+bias+activation pipeline");
    println!(
        "AI-guided phase: {}",
        if ai { "enabled" } else { "disabled" }
    );
    println!();

    let results = run_beat_cublas_campaign(ai, ai_iters);
    let report_md = generate_beat_cublas_report(&results);
    let report_json = generate_beat_cublas_json(&results);

    println!("\n{}", report_md);

    if let Some(path) = output {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("md");
        let content = match ext {
            "json" => serde_json::to_string_pretty(&report_json).expect("serialize failed"),
            _ => report_md.clone(),
        };
        std::fs::write(&path, content)?;
        println!("Report written to: {}", path.display());
    }

    let beats = results.iter().any(|r| r.beats_cublas);
    if beats {
        eprintln!("\nMilestone ACHIEVED: at least one problem size beats cuBLAS pipeline");
        Ok(())
    } else {
        eprintln!("\nMilestone NOT YET: no problem size exceeds 100% fused efficiency");
        std::process::exit(1);
    }
}

fn handle_bench_attention(
    target: f64,
    ai: bool,
    ai_iters: usize,
    output: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    println!("Attention >= 90% FlashAttention v2 Efficiency Milestone");
    println!("=======================================================");
    println!("Target efficiency: {:.1}%", target);
    println!(
        "AI-guided phase: {}",
        if ai { "enabled" } else { "disabled" }
    );
    println!();
    let results = optimize_all_attention_problems(target, ai, ai_iters);
    let report_md = generate_attention_milestone_report(&results, target);
    let report_json = generate_attention_milestone_json(&results, target);
    println!("\n{}", report_md);
    if let Some(path) = output {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("md");
        let content = match ext {
            "json" => serde_json::to_string_pretty(&report_json).expect("failed to serialize"),
            _ => report_md,
        };
        std::fs::write(&path, content).expect("failed to write output file");
        println!("Report written to: {}", path.display());
    }
    let all_pass = results.iter().all(|r| r.meets_target);
    if !all_pass {
        eprintln!(
            "\nMilestone NOT achieved: some problem sizes below {:.1}%",
            target
        );
        std::process::exit(1);
    } else {
        eprintln!("\nMilestone ACHIEVED: all problem sizes >= {:.1}%", target);
    }
    Ok(())
}
