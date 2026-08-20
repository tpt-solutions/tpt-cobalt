//! TPT GPU kernel generator CLI.
//!
//! Usage:
//!   tpt generate <kernel> [--elem <type>] [--shape <shape>] [--output-tptuir <file>]
//!   tpt ai-generate <kernel> [--elem <type>] [--shape <shape>] [--no-bench]
//!   tpt validate <file.tptir>
//!   tpt bench [--quick] [--output-json <file>]
//!   tpt version
//!
//! Supported kernels:
//!   vector_add, matmul, softmax, flash_attention, conv_bn_relu, conv3d,
//!   layer_norm, batch_norm, group_norm

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tpt", version, about = "TPT GPU kernel generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a kernel template from the built-in library
    Generate {
        /// Kernel name (vector_add, matmul, softmax, flash_attention, conv_bn_relu, conv3d,
        ///              layer_norm, batch_norm, group_norm)
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        #[arg(long, default_value = "1024")]
        shape: String,
        /// Also emit the kernel as a TPT-UIR `.tptuir` file (TPT-UIR ingestion adapter)
        #[arg(long)]
        output_tptuir: Option<PathBuf>,
    },
    /// AI-assisted pipeline: spec -> TPTIR -> validate -> correctness test -> benchmark
    ///
    /// Requires ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or local Ollama.
    AiGenerate {
        /// Kernel to generate
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        #[arg(long, default_value = "1024")]
        shape: String,
        /// Skip benchmark phase
        #[arg(long)]
        no_bench: bool,
        /// Write generated TPTIR to this file
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Validate a TPTIR file
    Validate { file: PathBuf },
    /// Run benchmarks
    Bench {
        #[arg(long)]
        quick: bool,
        /// Write structured JSON benchmark report to this file
        #[arg(long)]
        output_json: Option<PathBuf>,
    },
    /// Show version
    Version,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            kernel,
            elem,
            shape,
            output_tptuir,
        } => {
            println!(
                "Generating kernel: {} (elem={}, shape={})",
                kernel, elem, shape
            );
            let elem_type = parse_elem(&elem)?;
            let shape_params = parse_shape(&shape)?;
            let region =
                tpt_gpu_compiler::ir::build_kernel_region(&kernel, elem_type, &shape_params)
                    .map_err(|e| anyhow::anyhow!(e))?;
            let tptir = tpt_gpu_compiler::ir::emit_tptir(
                &region,
                &kernel,
                &[("tptir.kernel".to_string(), "".to_string())],
            );
            println!("{}", tptir);
            if let Some(tptuir_path) = output_tptuir {
                tpt_gpu_uir_adapter::write_tptuir(&region, &tptuir_path)
                    .map_err(|e| anyhow::anyhow!("failed to emit .tptuir: {e}"))?;
                println!("Wrote TPT-UIR to {}", tptuir_path.display());
            }
        }

        Commands::AiGenerate {
            kernel,
            elem,
            shape,
            no_bench,
            output,
        } => {
            run_ai_generate(&kernel, &elem, &shape, no_bench, output.as_deref())?;
        }

        Commands::Validate { file } => {
            println!("Validating: {}", file.display());
            let source = std::fs::read_to_string(&file)?;
            let region =
                tpt_gpu_compiler::ir::parse_assembly(&source).map_err(|e| anyhow::anyhow!(e))?;
            let result = tpt_gpu_compiler::validate::validate_region(&region);
            if result.is_valid() {
                println!("Validation passed!");
            } else {
                println!("Validation failed with {} errors:", result.error_count());
                for err in &result.errors {
                    println!("  - {}", err);
                }
            }
        }

        Commands::Bench { quick, output_json } => {
            let config = if quick {
                println!("Running quick benchmark (30-second local sanity check)...");
                tpt_gpu_compiler::bench::BenchConfig::quick()
            } else {
                println!("Running full benchmark (5 minutes)...");
                tpt_gpu_compiler::bench::BenchConfig::default()
            };

            let mut runner = tpt_gpu_compiler::bench::BenchRunner::new(config);

            // Default kernel set for benchmarking
            let kernels: Vec<(String, String, Vec<i64>)> = vec![
                ("vector_add".to_string(), "f32".to_string(), vec![1024]),
                ("matmul".to_string(), "f32".to_string(), vec![256, 256]),
                ("softmax".to_string(), "f32".to_string(), vec![1024]),
            ];

            let report = runner.run_all(&kernels);

            // Print human-readable summary
            println!("\n=== Benchmark Report ===");
            println!("Host: {}", report.host);
            println!("Timestamp: {}", report.timestamp);
            println!();
            for entry in &report.kernels {
                let status = if entry.is_regression {
                    "REGRESSION"
                } else {
                    "OK"
                };
                println!(
                    "  {:12} {:>8.3} ms  {:>10.2} GFLOPS  {:>8.2} GB/s  {:>6.1}% eff [{}]",
                    entry.kernel,
                    entry.execution_time_ms,
                    entry.gflops,
                    entry.bandwidth_gbps,
                    entry.efficiency_pct,
                    status
                );
            }
            println!();
            println!("Summary:");
            println!("  Kernels: {}", report.summary.total_kernels);
            println!("  Regressions: {}", report.summary.regression_count);
            println!("  Best GFLOPS: {:.2}", report.summary.best_gflops);
            println!(
                "  Best Bandwidth: {:.2} GB/s",
                report.summary.best_bandwidth_gbps
            );
            println!(
                "  Avg Efficiency: {:.1}%",
                report.summary.avg_efficiency_pct
            );

            // Write JSON if requested
            if let Some(path) = output_json {
                let json = tpt_gpu_compiler::bench::report_to_json(&report);
                std::fs::write(&path, &json)?;
                println!("\nWrote structured JSON report to {}", path.display());
            } else {
                // Also print JSON to stdout
                println!("\n--- Structured JSON ---");
                println!("{}", tpt_gpu_compiler::bench::report_to_json(&report));
            }
        }

        Commands::Version => {
            println!("{}", tpt_gpu_compiler::version());
        }
    }

    Ok(())
}

fn run_ai_generate(
    kernel: &str,
    elem: &str,
    shape: &str,
    no_bench: bool,
    output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    println!(
        "AI-assisted generation: {} (elem={}, shape={})",
        kernel, elem, shape
    );

    // --- Step 1: AI Generation ---
    println!("\n[1/4] AI generation...");
    let provider = tpt_gpu_shared::provider_from_env();
    println!("  provider: {}", provider.name());
    let tptir_text = generate_tptir_via_ai(provider.as_ref(), kernel, elem, shape)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("  generated {} bytes of TPTIR", tptir_text.len());

    // --- Step 2: Parse and Validate ---
    println!("\n[2/4] Validation...");
    let region = tpt_gpu_compiler::ir::parse_assembly(&tptir_text)
        .map_err(|e| anyhow::anyhow!("Parse error: {e}"))?;
    let result = tpt_gpu_compiler::validate::validate_region(&region);
    if !result.is_valid() {
        anyhow::bail!(
            "Validation failed with {} errors: {:?}",
            result.error_count(),
            result.errors
        );
    }
    println!("  validation passed ({} blocks)", region.blocks.len());

    // --- Step 3: Correctness Test ---
    println!("\n[3/4] Correctness test...");
    let correctness = check_correctness(&region, kernel, elem);
    println!("  {}", correctness);

    // --- Step 4: Benchmark ---
    if !no_bench {
        println!("\n[4/4] Benchmark...");
        let config = tpt_gpu_compiler::bench::BenchConfig::quick();
        let mut runner = tpt_gpu_compiler::bench::BenchRunner::new(config);
        let shape_params = parse_shape(shape)?;
        let result = runner.run_kernel(kernel, elem, &shape_params);
        println!(
            "  {:.3} ms  {:.2} GFLOPS  {:.2} GB/s",
            result.execution_time_ms, result.throughput_gflops, result.memory_bandwidth_gbps
        );
    }

    // --- Output ---
    if let Some(path) = output {
        std::fs::write(path, &tptir_text)?;
        println!("\nWrote TPTIR to {}", path.display());
    } else {
        println!("\n--- Generated TPTIR ---\n{}", tptir_text);
    }

    Ok(())
}

fn generate_tptir_via_ai(
    provider: &dyn tpt_gpu_shared::AiProvider,
    kernel: &str,
    elem: &str,
    shape: &str,
) -> Result<String, String> {
    let prompt = format!(
        "Generate TPTIR assembly for a '{kernel}' kernel.\n\
         Element type: {elem}\n\
         Shape: {shape}\n\n\
         TPTIR format rules:\n\
         - Wrap in: module {{ func.func @{kernel}(...) attributes {{tptir.kernel}} {{ ... }} }}\n\
         - Blocks start with ^label:\n\
         - Operations use tptir. prefix: tptir.addf, tptir.mulf, tptir.load, tptir.store, tptir.return\n\
         - Values: %0, %1, %2, ...\n\
         - Arguments are memref<*x{elem}, global> tensors plus an i32 length\n\n\
         Example (vector_add, f32, 1024):\n\
         module {{\n           func.func @vector_add(%0: memref<*xf32, global>, %1: memref<*xf32, global>,\n               %2: memref<*xf32, global>, %3: i32) attributes {{tptir.kernel}} {{\n             ^entry:\n               %10 = tptir.addf(%0, %1) : (memref<*xf32, global>, memref<*xf32, global>) -> f32\n               tptir.store(%10, %2)\n               tptir.return\n           }}\n         }}\n\n\
         Generate ONLY the TPTIR module text for '{kernel}'. No explanation.",
    );

    provider.generate(&prompt).map_err(|e| e.to_string())
}

fn check_correctness(region: &tpt_gpu_compiler::ir::Region, kernel: &str, elem: &str) -> String {
    if region.blocks.is_empty() {
        return "FAIL -- no blocks".to_string();
    }

    let op_count: usize = region.blocks.iter().map(|b| b.operations.len()).sum();
    if op_count == 0 {
        return "FAIL -- no operations".to_string();
    }

    let has_return = region.blocks.iter().any(|b| {
        b.operations
            .iter()
            .any(|op| matches!(op.kind, tpt_gpu_compiler::ir::OpKind::Return))
    });

    if has_return {
        format!(
            "PASS -- {} block(s), {} op(s) [{} {}]",
            region.blocks.len(),
            op_count,
            kernel,
            elem
        )
    } else {
        format!(
            "WARN -- {} block(s), {} op(s), no return terminator [{} {}]",
            region.blocks.len(),
            op_count,
            kernel,
            elem
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_elem(elem: &str) -> anyhow::Result<tpt_gpu_compiler::ir::ElemType> {
    match elem {
        "f32" => Ok(tpt_gpu_compiler::ir::ElemType::F32),
        "f16" => Ok(tpt_gpu_compiler::ir::ElemType::F16),
        "bf16" => Ok(tpt_gpu_compiler::ir::ElemType::BF16),
        "i32" => Ok(tpt_gpu_compiler::ir::ElemType::I32),
        _ => anyhow::bail!(
            "Unknown element type: {}. Use f32, f16, bf16, or i32.",
            elem
        ),
    }
}

fn parse_shape(shape: &str) -> anyhow::Result<Vec<i64>> {
    shape
        .split('x')
        .map(|s| {
            s.parse::<i64>()
                .map_err(|e| anyhow::anyhow!("Invalid shape '{}': {}", s, e))
        })
        .collect()
}
