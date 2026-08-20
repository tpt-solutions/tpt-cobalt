//! `tpt-serve` — OpenAI-compatible HTTP server for TPT GPU LLM inference.
//!
//! Usage:
//!   tpt-serve --model <path> [--host 127.0.0.1] [--port 8080]
//!
//! Loads a GGUF/TPTF model via the `tpt-gpu-runtime` `LlmInference` engine and
//! serves OpenAI-compatible endpoints (`/v1/models`, `/v1/completions`,
//! `/v1/chat/completions`). See `server.rs` for details.

mod server;
mod tokenizer;

use std::path::Path;
use std::sync::{Arc, Mutex};

use tpt_gpu_runtime::{GpuInferenceEngine, LlmInference};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8080;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if let Some(v) = args.get(i + 1) {
                    model_path = Some(v.clone());
                }
                i += 2;
            }
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    host = v.clone();
                }
                i += 2;
            }
            "--port" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    port = v;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }

    let model = match model_path {
        Some(m) => m,
        None => {
            eprintln!("tpt-serve: error: --model <path> is required");
            eprintln!("Usage: tpt-serve --model <path> [--host 127.0.0.1] [--port 8080]");
            std::process::exit(2);
        }
    };

    eprintln!("tpt-serve: loading model from {model}");
    let engine = match GpuInferenceEngine::load(Path::new(&model)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("tpt-serve: failed to load model: {e}");
            std::process::exit(1);
        }
    };
    let vocab = engine.model_info().vocab_size;
    eprintln!("tpt-serve: model loaded (vocab={vocab})");

    let state = Arc::new(Mutex::new(engine));
    if let Err(e) = server::run(state, &model, &host, port) {
        eprintln!("tpt-serve: server error: {e}");
        std::process::exit(1);
    }
}
