use anyhow::Result;
use clap::{Parser, Subcommand};
use tpt_gpu_model_registry::{hf::HfDownload, ModelEntry, ModelRegistry};

#[derive(Parser)]
#[command(
    name = "tpt-gpu-models",
    about = "Manage the ~/.tpt/models/ shared model registry"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all registered models.
    List,
    /// Show the registry directory path.
    Dir,
    /// Remove a model entry from the manifest (does not delete the file).
    Remove { name: String },
    /// Register a model file that is already on disk (no download).
    Add {
        /// Human-readable name / lookup key (e.g. llama-3-8b-q4).
        #[arg(long)]
        name: String,
        /// Filename relative to the registry directory.
        #[arg(long)]
        file: String,
        /// Architecture tag (e.g. llama3, mistral, phi3).
        #[arg(long)]
        arch: String,
        /// Approximate size in GiB.
        #[arg(long)]
        size_gb: f64,
        /// Optional SHA-256 hex digest for integrity checks.
        #[arg(long)]
        sha256: Option<String>,
        /// Optional source URL (informational).
        #[arg(long)]
        source: Option<String>,
    },
    /// Download a GGUF from a HuggingFace URL and register it.
    Fetch {
        /// Direct URL to the .gguf file (HuggingFace "Download" link).
        #[arg(long)]
        url: String,
        /// Name to register (e.g. llama-3-8b-q4).
        #[arg(long)]
        name: String,
        /// Architecture tag (e.g. llama3).
        #[arg(long)]
        arch: String,
        /// Approximate size in GiB.
        #[arg(long)]
        size_gb: f64,
        /// Optional expected SHA-256 for integrity verification.
        #[arg(long)]
        sha256: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut registry = ModelRegistry::open()?;

    match cli.cmd {
        Cmd::List => {
            let models = registry.models();
            if models.is_empty() {
                println!(
                    "No models registered. Run `tpt-gpu-models fetch --help` to download one."
                );
            } else {
                println!("NAME                     ARCH       SIZE_GB  FILE");
                for m in models {
                    println!(
                        "{:<30} {:<10} {:<8.1} {}",
                        m.name, m.arch, m.size_gb, m.file
                    );
                }
            }
        }
        Cmd::Dir => {
            println!("{}", registry.dir().display());
        }
        Cmd::Remove { name } => {
            if registry.unregister(&name)? {
                println!("Removed '{}'.", name);
            } else {
                println!("No model named '{}' found.", name);
            }
        }
        Cmd::Add {
            name,
            file,
            arch,
            size_gb,
            sha256,
            source,
        } => {
            registry.register(ModelEntry {
                name: name.clone(),
                file,
                arch,
                size_gb,
                sha256,
                source,
                quant_bits: None,
                pruned_domains: None,
                source_model: None,
            })?;
            println!("Registered '{}'.", name);
        }
        Cmd::Fetch {
            url,
            name,
            arch,
            size_gb,
            sha256,
        } => {
            println!("Downloading '{}' from {}…", name, url);
            let dest = tpt_gpu_model_registry::hf::download(
                &mut registry,
                HfDownload {
                    url,
                    name: name.clone(),
                    arch,
                    size_gb,
                    expected_sha256: sha256,
                },
            )?;
            println!(
                "Downloaded and registered '{}' at {}.",
                name,
                dest.display()
            );
        }
    }

    Ok(())
}
