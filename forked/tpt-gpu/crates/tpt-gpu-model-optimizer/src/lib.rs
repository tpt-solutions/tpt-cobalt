//! TPT Model Optimizer — hardware-aware LLM compression pipeline.
//!
//! # Pipeline
//! ```text
//! GGUF source
//!   → HardwareProfiler   (benchmark GPU; cache to ~/.tpt/hardware_profile.json)
//!   → LayerSensitivityMap (rank layers by quantization sensitivity)
//!   → DomainMapper        (identify neurons by domain via activation analysis)
//!   → SurgicalPruner      (zero unwanted domain neurons)
//!   → MixedPrecisionAllocator (5% loss frontier per layer)
//!   → KvCacheCalculator   (recommend context window for remaining VRAM)
//!   → TptfFormat writer   (output .tptf — self-contained: weights + tokenizer + chat template)
//!
//! Export: GgufExporter, Exl2Exporter
//! Quality: QualityBenchmark (perplexity + task accuracy)
//! Large models: StreamingLoader (70B+ via layer-by-layer mmap)
//! ```

pub mod activation_capture;
pub mod benchmark;
pub mod calibration;
pub mod domain_mapper;
pub mod export;
pub mod import;
pub mod kv_calculator;
pub mod profiler;
pub mod pruner;
pub mod quant_allocator;
pub mod sensitivity;
pub mod streaming;
pub mod tptf_format;

pub use activation_capture::{
    load_domain_maps_from_dir, ActivationCallback, ActivationCapture, ActivationMap,
    LayerActivations,
};
pub use benchmark::QualityBenchmark;
pub use calibration::CalibrationGenerator;
pub use domain_mapper::DomainMapper;
pub use export::detect::ModelFormat;
pub use import::gguf::{GgufImporter, GgufModel};
pub use kv_calculator::KvCacheCalculator;
pub use profiler::HardwareProfiler;
pub use pruner::SurgicalPruner;
pub use quant_allocator::MixedPrecisionAllocator;
pub use sensitivity::{LayerSensitivityMap, SensitivityConfig};
pub use streaming::StreamingLoader;
pub use tptf_format::{TensorBlock, TptfHeader, TptfWriter};

use std::path::PathBuf;

/// Configuration for a full optimization run.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Source model path (.gguf or .tptf).
    pub model_path: PathBuf,
    /// Output path (defaults to `<stem>.tptf`).
    pub output_path: PathBuf,
    /// Maximum acceptable quality loss as a fraction (0.05 = 5%).
    pub max_loss_fraction: f32,
    /// Domains to remove via surgical pruning (e.g. `["sql", "typescript"]`).
    pub domains_to_drop: Vec<String>,
    /// Force streaming mode (auto-enabled when model_size > 80% vram_free).
    pub force_streaming: bool,
}

impl OptimizerConfig {
    pub fn new(model_path: impl Into<PathBuf>, output_path: impl Into<PathBuf>) -> Self {
        OptimizerConfig {
            model_path: model_path.into(),
            output_path: output_path.into(),
            max_loss_fraction: 0.05,
            domains_to_drop: Vec::new(),
            force_streaming: false,
        }
    }
}

/// Summary produced after a full optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub output_path: PathBuf,
    pub per_layer_bits: Vec<u8>,
    pub pruned_domains: Vec<String>,
    pub size_before_gb: f64,
    pub size_after_gb: f64,
    pub compression_ratio: f32,
    pub ppl_delta_pct: f32,
}
