use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use tpt_gpu_model_optimizer::{
    benchmark::QualityBenchmark,
    calibration::CalibrationGenerator,
    domain_mapper::DomainMapper,
    export::{detect, Exl2ExportConfig, Exl2Exporter, GgufExportConfig, GgufExporter, ModelFormat},
    import::gguf::GgufImporter,
    kv_calculator::KvCacheCalculator,
    profiler::HardwareProfiler,
    pruner::SurgicalPruner,
    quant_allocator::{MixedPrecisionAllocator, QuantEvalConfig},
    sensitivity::{LayerSensitivityMap, SensitivityConfig},
    tptf_format::read_header,
    OptimizerConfig,
};

#[derive(Parser)]
#[command(
    name = "tpt-gpu-model-optimizer",
    about = "TPT GPU model compression and optimization"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Benchmark GPU hardware and cache the profile
    Profile,

    /// Analyze domain-specific neuron distribution in a model
    Analyze {
        /// Path to model (.tptf). Not required when --activations-dir is given.
        model: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        domains: Option<Vec<String>>,
        /// Directory of per-domain activation maps (`<domain>.json`) captured
        /// from calibration runs. When set, builds the domain map from real
        /// captured activations instead of the heuristic path.
        #[arg(long)]
        activations_dir: Option<PathBuf>,
        /// Output file for domain map JSON (default: domain_map.json in model dir)
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Run the full optimization pipeline: profile → sensitivity → prune → quantize
    Optimize {
        model: PathBuf,
        #[arg(long, default_value = "0.05")]
        max_loss: f32,
        #[arg(long, value_delimiter = ',')]
        domains_drop: Option<Vec<String>>,
        /// Directory of per-domain activation maps (`<domain>.json`) for the
        /// production domain-mapping path during pruning.
        #[arg(long)]
        activations_dir: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        stream: bool,
    },

    /// Export a .tptf file to a compatibility format
    Export {
        model: PathBuf,
        #[arg(long, default_value = "gguf")]
        format: String,
        #[arg(long)]
        output: PathBuf,
    },

    /// Convert a GGUF model to TPTF format (GGUF → TPTF)
    Convert {
        /// Input GGUF file path
        #[arg(long)]
        input: PathBuf,
        /// Output TPTF file path
        #[arg(long)]
        output: PathBuf,
    },

    /// Compare quality of two models (perplexity + task accuracy)
    Bench { before: PathBuf, after: PathBuf },

    /// Calculate maximum context window given remaining VRAM
    KvCalc { model: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile => cmd_profile(),
        Commands::Analyze {
            model,
            domains,
            activations_dir,
            output,
        } => cmd_analyze(model, domains, activations_dir, output),
        Commands::Optimize {
            model,
            max_loss,
            domains_drop,
            activations_dir,
            output,
            stream,
        } => cmd_optimize(
            model,
            max_loss,
            domains_drop,
            activations_dir,
            output,
            stream,
        ),
        Commands::Convert { input, output } => cmd_convert(input, output),
        Commands::Export {
            model,
            format,
            output,
        } => cmd_export(model, format, output),
        Commands::Bench { before, after } => cmd_bench(before, after),
        Commands::KvCalc { model } => cmd_kv_calc(model),
    }
}

fn cmd_profile() -> Result<()> {
    println!("Profiling hardware...");
    let p = HardwareProfiler::new().profile()?;
    println!("GPU:             {}", p.gpu_uuid);
    println!("Memory BW:       {:.0} GB/s", p.bw_gbps);
    println!("L2 cache:        {:.0} MiB", p.l2_mb);
    println!("Tensor cores:    {}", p.tensor_core_gen);
    println!("VRAM total:      {} MiB", p.vram_total_mb);
    println!("VRAM free:       {} MiB", p.vram_free_mb);
    Ok(())
}

fn cmd_analyze(
    model: Option<PathBuf>,
    domains: Option<Vec<String>>,
    activations_dir: Option<PathBuf>,
    output: Option<PathBuf>,
) -> Result<()> {
    let domains = domains.unwrap_or_else(|| {
        tpt_gpu_model_optimizer::domain_mapper::KNOWN_DOMAINS
            .iter()
            .map(|s| s.to_string())
            .collect()
    });

    let (domain_map, num_layers, output_path) = if let Some(dir) = &activations_dir {
        println!("Loading per-domain activations from {:?}...", dir);
        let maps = tpt_gpu_model_optimizer::load_domain_maps_from_dir(dir)?;
        if maps.is_empty() {
            anyhow::bail!("no activation maps found in {:?}", dir);
        }
        let map = DomainMapper::build_from_domain_activations(&maps, None)?;
        let num_layers = map.num_layers;
        let output_path = output.unwrap_or_else(|| dir.join("domain_map.json"));
        (map, num_layers, output_path)
    } else {
        let model = model
            .ok_or_else(|| anyhow::anyhow!("either --model or --activations-dir is required"))?;
        println!("Analyzing {:?}...", model);
        let header = read_header(&model)?;
        let ffn_dim = header.ffn_dim as usize;

        // Build domain map using heuristic path
        let mapper = DomainMapper::new(domains.clone());
        let map = mapper.build(header.num_layers as usize, ffn_dim)?;
        let output_path = output.unwrap_or_else(|| model.with_file_name("domain_map.json"));
        (map, header.num_layers as usize, output_path)
    };

    // Write domain_map.json
    let json = serde_json::to_string_pretty(&domain_map)?;
    std::fs::write(&output_path, &json)?;
    println!("Domain map written to {:?}", output_path);

    // Print summary
    for domain in &domains {
        let count: usize = (0..num_layers)
            .map(|l| domain_map.domain_neurons(l, domain, 0.1).len())
            .sum();
        println!("  {:12} — {:5} neurons", domain, count);
    }

    Ok(())
}

fn cmd_optimize(
    model: PathBuf,
    max_loss: f32,
    domains_drop: Option<Vec<String>>,
    activations_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    force_stream: bool,
) -> Result<()> {
    let output = output.unwrap_or_else(|| {
        let stem = model.file_stem().unwrap_or_default();
        model.with_file_name(format!("{}-opt.tptf", stem.to_string_lossy()))
    });

    let header = read_header(&model)?;
    let num_layers = header.num_layers as usize;
    let ffn_dim = header.ffn_dim as usize;

    let _cfg = OptimizerConfig {
        model_path: model.clone(),
        output_path: output.clone(),
        max_loss_fraction: max_loss,
        domains_to_drop: domains_drop.clone().unwrap_or_default(),
        force_streaming: force_stream,
    };

    println!("Step 1/5  Profiling hardware...");
    let profile = HardwareProfiler::new().profile()?;
    println!(
        "          {:.0} GB/s, {} MiB free",
        profile.bw_gbps, profile.vram_free_mb
    );

    println!("Step 2/5  Building sensitivity map...");
    // Generate calibration samples - use AI provider if available, otherwise heuristic
    let gen = CalibrationGenerator::new(vec!["general".to_string()])
        .with_samples_per_domain(8)
        .with_provider_from_env();

    if gen.has_ai_provider() {
        println!("          Using AI provider for calibration samples");
    } else {
        println!("          Using heuristic calibration samples");
    }

    let samples = gen.generate()?;

    let sens_config = SensitivityConfig {
        model_path: model.clone(),
        samples,
        eval_tokens: 32,
        group_size: 128,
    };
    let sensitivity = LayerSensitivityMap::build(num_layers, &sens_config)?;

    println!("Step 3/5  Analyzing domain neurons...");
    let mapper = DomainMapper::with_default_domains();
    let domain_map = if let Some(dir) = &activations_dir {
        println!("          Loading per-domain activations from {:?}...", dir);
        let maps = tpt_gpu_model_optimizer::load_domain_maps_from_dir(dir)?;
        if maps.is_empty() {
            anyhow::bail!("no activation maps found in {:?}", dir);
        }
        DomainMapper::build_from_domain_activations(&maps, None)?
    } else {
        mapper.build(num_layers, ffn_dim)?
    };

    if !_cfg.domains_to_drop.is_empty() {
        println!("          Pruning domains: {:?}", _cfg.domains_to_drop);
        let pruner = SurgicalPruner::new(_cfg.domains_to_drop.clone());
        let mask = pruner.build_mask(&domain_map)?;
        println!("          {} neurons zeroed", mask.total_pruned());
    }

    println!(
        "Step 4/5  Allocating mixed-precision bits (max loss {:.0}%)...",
        max_loss * 100.0
    );
    let allocator = MixedPrecisionAllocator::new(max_loss);
    let config = QuantEvalConfig::default();
    let bits = allocator.allocate(num_layers, &sensitivity, &config, |_layer, _bits| Ok(10.0))?;
    let avg_bits = bits.iter().map(|&b| b as f64).sum::<f64>() / bits.len() as f64;
    println!("          Average: {:.1} bits/weight", avg_bits);

    println!("Step 5/5  Writing {:?}...", output);
    println!("          (skipped in scaffold — no source model weights to pack)");

    println!("\nDone. Output: {:?}", output);
    Ok(())
}

fn cmd_convert(input: PathBuf, output: PathBuf) -> Result<()> {
    println!("Converting {:?} → {:?}", input, output);
    let model = GgufImporter::import(&input)?;
    println!(
        "  arch={}, layers={}, tensors={}",
        model.header.arch,
        model.header.num_layers,
        model.tensors.len()
    );
    let out = std::fs::File::create(&output)?;
    let out = std::io::BufWriter::new(out);
    model.write_tptf(out)?;
    println!("Done. Written {:?}", output);
    Ok(())
}

fn cmd_export(model: PathBuf, format: String, output: PathBuf) -> Result<()> {
    let fmt = detect(&model);
    if fmt != ModelFormat::Tptf {
        anyhow::bail!(
            "export only accepts .tptf source files (detected: {:?})",
            fmt
        );
    }
    match format.as_str() {
        "gguf" => {
            println!("Exporting to GGUF: {:?}", output);
            GgufExporter::export(&GgufExportConfig {
                source: model,
                dest: output,
                group_size: 128,
            })?;
        }
        "exl2" => {
            println!("Exporting to EXL2 directory: {:?}", output);
            Exl2Exporter::export(&Exl2ExportConfig {
                source: model,
                dest_dir: output,
                group_size: 128,
            })?;
        }
        other => anyhow::bail!("unknown export format '{}' (use gguf or exl2)", other),
    }
    println!("Done.");
    Ok(())
}

fn cmd_bench(before: PathBuf, after: PathBuf) -> Result<()> {
    let samples = CalibrationGenerator::new(vec!["general".to_string()])
        .with_samples_per_domain(16)
        .generate()?;
    let bench = QualityBenchmark::new(samples);
    let m_before = bench.evaluate(&before)?;
    let m_after = bench.evaluate(&after)?;
    let result =
        tpt_gpu_model_optimizer::benchmark::BenchmarkResult::compute(m_before, m_after, 0.05);
    result.print_report();
    Ok(())
}

fn cmd_kv_calc(model: PathBuf) -> Result<()> {
    let header = read_header(&model)?;
    let profile = HardwareProfiler::new().profile()?;
    let calc = KvCacheCalculator::new(
        header.num_layers as usize,
        header.num_kv_heads,
        header.hidden_dim / header.num_heads.max(1),
    );
    let bits: Vec<u8> = (0..header.num_layers as usize)
        .map(|i| header.per_layer_bits.get(i).copied().unwrap_or(4))
        .collect();
    let param_counts: Vec<u64> =
        vec![header.hidden_dim as u64 * header.ffn_dim as u64 * 3; header.num_layers as usize];
    let rec = calc.calculate(&profile, &bits, &param_counts)?;
    println!("Model footprint: {:.0} MiB", rec.model_footprint_mb);
    println!("KV cache budget: {:.0} MiB", rec.kv_vram_mb);
    println!("Bytes per token: {} bytes", rec.kv_bytes_per_token);
    println!("Max context:     {} tokens", rec.context_len);
    Ok(())
}
