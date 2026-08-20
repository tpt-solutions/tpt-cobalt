//! TPT-GPU Vendor Certification Tool
//!
//! Runs compatibility, performance, and correctness tests for vendor backends.

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

mod report;
mod tests;

/// TPT-GPU Vendor Certification CLI
#[derive(Parser, Debug)]
#[command(
    name = "tpt-gpu-vendor-cert",
    version,
    about = "TPT-GPU vendor certification test suite"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run certification tests
    Certify {
        /// Vendor name
        #[arg(short, long)]
        vendor: String,
        /// Target certification tier (1, 2, or 3)
        #[arg(short, long, default_value = "1")]
        tier: u32,
        /// Path to vendor profile JSON
        #[arg(short, long)]
        profile: Option<PathBuf>,
        /// Output directory for test results
        #[arg(short, long, default_value = "target/vendor-cert")]
        output: PathBuf,
    },
    /// List registered vendors
    ListVendors {
        /// Vendor profiles directory
        #[arg(short, long, default_value = "tuning/vendor")]
        dir: PathBuf,
    },
    /// Validate vendor profile
    ValidateProfile {
        /// Path to vendor profile JSON
        profile: PathBuf,
    },
    /// Generate vendor template
    GenerateTemplate {
        /// Vendor name
        #[arg(short, long)]
        vendor: String,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare vendor performance
    Compare {
        /// Vendor profiles to compare
        #[arg(short, long, num_args = 2..)]
        vendors: Vec<String>,
        /// Vendor profiles directory
        #[arg(short, long, default_value = "tuning/vendor")]
        dir: PathBuf,
    },
}

/// Vendor profile structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorProfile {
    pub vendor: String,
    pub gpu_model: String,
    pub driver_version: String,
    #[serde(default)]
    pub certification_tier: u32,
    #[serde(default)]
    pub certification_date: Option<String>,
    #[serde(default)]
    pub hardware_specs: HardwareSpecs,
    #[serde(default)]
    pub supported_operations: SupportedOperations,
    #[serde(default)]
    pub performance_baselines: HashMap<String, f64>,
    #[serde(default)]
    pub tuning_parameters: serde_json::Value,
    #[serde(default)]
    pub contact: Option<ContactInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareSpecs {
    #[serde(default)]
    pub memory_gb: u32,
    #[serde(default)]
    pub memory_bandwidth_gbps: u32,
    #[serde(default)]
    pub compute_tflops_fp32: u32,
    #[serde(default)]
    pub compute_tflops_fp16: u32,
    #[serde(default)]
    pub tensor_cores: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SupportedOperations {
    #[serde(default)]
    pub gemm: bool,
    #[serde(default)]
    pub attention: bool,
    #[serde(default)]
    pub conv2d: bool,
    #[serde(default)]
    pub conv3d: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContactInfo {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub github: Option<String>,
}

impl VendorProfile {
    /// Load vendor profile from JSON file
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read vendor profile: {:?}", path))?;
        let profile: VendorProfile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse vendor profile: {:?}", path))?;
        Ok(profile)
    }

    /// Save vendor profile to JSON file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content =
            serde_json::to_string_pretty(self).context("Failed to serialize vendor profile")?;
        fs::write(path, content)
            .with_context(|| format!("Failed to write vendor profile: {:?}", path))?;
        Ok(())
    }

    /// Validate the vendor profile
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        if self.vendor.is_empty() {
            issues.push("Vendor name is required".to_string());
        }

        if self.gpu_model.is_empty() {
            issues.push("GPU model is required".to_string());
        }

        if self.driver_version.is_empty() {
            issues.push("Driver version is required".to_string());
        }

        if self.certification_tier > 3 {
            issues.push("Certification tier must be 1, 2, or 3".to_string());
        }

        if self.certification_tier >= 1 && self.hardware_specs.memory_gb == 0 {
            issues.push("Memory size is required for Tier 1+ certification".to_string());
        }

        if self.certification_tier >= 2
            && !self.supported_operations.gemm
            && !self.supported_operations.attention
            && !self.supported_operations.conv2d
        {
            issues.push(
                "At least one operation must be supported for Tier 2+ certification".to_string(),
            );
        }

        if self.certification_tier >= 3
            && (!self.supported_operations.gemm
                || !self.supported_operations.attention
                || !self.supported_operations.conv2d
                || !self.supported_operations.conv3d)
        {
            issues.push("All operations must be supported for Tier 3 certification".to_string());
        }

        Ok(issues)
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Certify {
            vendor,
            tier,
            profile,
            output,
        } => {
            info!(
                "Running certification for vendor: {} (Tier {})",
                vendor, tier
            );
            run_certification(&vendor, tier, profile.as_ref(), &output)?;
        }
        Commands::ListVendors { dir } => {
            info!("Listing registered vendors from: {:?}", dir);
            list_vendors(&dir)?;
        }
        Commands::ValidateProfile { profile } => {
            info!("Validating vendor profile: {:?}", profile);
            validate_profile(&profile)?;
        }
        Commands::GenerateTemplate { vendor, output } => {
            info!("Generating vendor template for: {}", vendor);
            generate_template(&vendor, &output)?;
        }
        Commands::Compare { vendors, dir } => {
            info!("Comparing vendors: {:?}", vendors);
            compare_vendors(&vendors, &dir)?;
        }
    }

    Ok(())
}

/// Run certification tests for a vendor
fn run_certification(
    vendor: &str,
    tier: u32,
    profile: Option<&PathBuf>,
    output: &PathBuf,
) -> Result<()> {
    info!("Starting certification process for {}", vendor);

    // Load or create vendor profile
    let mut vendor_profile = if let Some(path) = profile {
        VendorProfile::load(path)?
    } else {
        VendorProfile {
            vendor: vendor.to_string(),
            gpu_model: String::new(),
            driver_version: String::new(),
            certification_tier: tier,
            certification_date: None,
            hardware_specs: HardwareSpecs::default(),
            supported_operations: SupportedOperations::default(),
            performance_baselines: HashMap::new(),
            tuning_parameters: serde_json::Value::Null,
            contact: None,
        }
    };

    // Validate profile
    let issues = vendor_profile.validate()?;
    if !issues.is_empty() {
        for issue in &issues {
            warn!("Profile validation issue: {}", issue);
        }
        if issues.iter().any(|i| i.contains("required")) {
            anyhow::bail!("Profile validation failed with required field issues");
        }
    }

    // Create output directory
    fs::create_dir_all(output)?;

    // Run compatibility tests
    info!("Running compatibility tests...");
    let compat_results = tests::run_compatibility_tests(vendor, tier)?;

    // Run correctness tests
    info!("Running correctness tests...");
    let correctness_results = tests::run_correctness_tests(vendor, tier)?;

    // Run performance tests (Tier 2+)
    let performance_results = if tier >= 2 {
        info!("Running performance tests...");
        Some(tests::run_performance_tests(vendor, tier)?)
    } else {
        None
    };

    // Generate report
    info!("Generating certification report...");
    let report = report::generate_report(
        &vendor_profile,
        &compat_results,
        &correctness_results,
        performance_results.as_ref(),
    )?;

    // Save report
    let report_path = output.join(format!("{}_certification_report.json", vendor));
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    info!("Certification report saved to: {:?}", report_path);

    // Update vendor profile with certification date
    vendor_profile.certification_date = Some(Utc::now().to_rfc3339());
    let profile_path = output.join(format!("{}_profile.json", vendor));
    vendor_profile.save(&profile_path)?;
    info!("Vendor profile saved to: {:?}", profile_path);

    // Print summary
    println!("\n=== Certification Summary for {} ===", vendor);
    println!("Tier: {}", tier);
    println!(
        "Compatibility Tests: {}/{} passed",
        compat_results.passed, compat_results.total
    );
    println!(
        "Correctness Tests: {}/{} passed",
        correctness_results.passed, correctness_results.total
    );
    if let Some(ref perf) = performance_results {
        println!("Performance Tests: {}/{} passed", perf.passed, perf.total);
    }
    println!(
        "Overall Status: {}",
        if report.passed { "PASSED" } else { "FAILED" }
    );
    println!("Report: {:?}", report_path);

    Ok(())
}

/// List all registered vendors
fn list_vendors(dir: &PathBuf) -> Result<()> {
    if !dir.exists() {
        println!("No vendor profiles directory found at: {:?}", dir);
        return Ok(());
    }

    let entries = fs::read_dir(dir)?;
    let mut vendors = Vec::new();

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            match VendorProfile::load(&path.to_path_buf()) {
                Ok(profile) => vendors.push(profile),
                Err(e) => warn!("Failed to load profile {:?}: {}", path, e),
            }
        }
    }

    if vendors.is_empty() {
        println!("No vendor profiles found in: {:?}", dir);
    } else {
        println!("\n=== Registered Vendors ===");
        for vendor in vendors {
            println!(
                "  {} - {} (Tier {})",
                vendor.vendor, vendor.gpu_model, vendor.certification_tier
            );
            if let Some(date) = vendor.certification_date {
                println!("    Certified: {}", date);
            }
        }
    }

    Ok(())
}

/// Validate a vendor profile
fn validate_profile(path: &PathBuf) -> Result<()> {
    let profile = VendorProfile::load(path)?;
    let issues = profile.validate()?;

    if issues.is_empty() {
        println!("Profile is valid: {:?}", path);
    } else {
        println!("Profile validation issues for {:?}:", path);
        for issue in &issues {
            println!("  - {}", issue);
        }
    }

    Ok(())
}

/// Generate a vendor template
fn generate_template(vendor: &str, output: &PathBuf) -> Result<()> {
    let template = VendorProfile {
        vendor: vendor.to_string(),
        gpu_model: "Your GPU Model".to_string(),
        driver_version: "1.0.0".to_string(),
        certification_tier: 1,
        certification_date: None,
        hardware_specs: HardwareSpecs {
            memory_gb: 0,
            memory_bandwidth_gbps: 0,
            compute_tflops_fp32: 0,
            compute_tflops_fp16: 0,
            tensor_cores: false,
        },
        supported_operations: SupportedOperations {
            gemm: false,
            attention: false,
            conv2d: false,
            conv3d: false,
        },
        performance_baselines: HashMap::new(),
        tuning_parameters: serde_json::json!({
            "gemm": {
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "vec_width": 4,
                "unroll": 4
            }
        }),
        contact: Some(ContactInfo {
            name: Some("Your Name".to_string()),
            email: Some("your.email@example.com".to_string()),
            github: Some("yourgithub".to_string()),
        }),
    };

    template.save(output)?;
    println!("Vendor template generated: {:?}", output);

    Ok(())
}

/// Compare vendor performance
fn compare_vendors(vendors: &[String], dir: &Path) -> Result<()> {
    let mut profiles = Vec::new();

    for vendor in vendors {
        let path = dir.join(format!("{}.json", vendor));
        if path.exists() {
            match VendorProfile::load(&path) {
                Ok(profile) => profiles.push(profile),
                Err(e) => warn!("Failed to load profile for {}: {}", vendor, e),
            }
        } else {
            warn!("Profile not found for vendor: {}", vendor);
        }
    }

    if profiles.is_empty() {
        println!("No vendor profiles found to compare");
        return Ok(());
    }

    println!("\n=== Vendor Comparison ===");
    println!(
        "{:<20} {:<15} {:<10} {:<10} {:<10}",
        "Vendor", "GPU Model", "Memory", "FP32 TFLOPS", "Tier"
    );
    println!("{}", "-".repeat(65));

    for profile in profiles {
        println!(
            "{:<20} {:<15} {:<10} {:<10} {:<10}",
            profile.vendor,
            profile.gpu_model,
            format!("{} GB", profile.hardware_specs.memory_gb),
            format!("{} T", profile.hardware_specs.compute_tflops_fp32),
            profile.certification_tier
        );
    }

    Ok(())
}
