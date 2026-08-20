use crate::ir::{TptIr, TptIrError};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    PyTorch,
    Onnx,
    TensorFlow,
    Gguf,
    Tflite,
    Exl2,
    Llamafile,
    KerasH5,
    JaxFlax,
    AwqGptq,
    HuggingFace,
    SafeTensors,
}

impl ModelFormat {
    pub fn detect(path: &Path) -> Result<Self, TptIrError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Extension-based detection
        match ext.as_str() {
            "pt" | "pth" => return Ok(Self::PyTorch),
            "bin" => {
                // Could be PyTorch or SafeTensors — check magic bytes
                if let Ok(true) = check_magic(path, b"PK") {
                    return Ok(Self::SafeTensors);
                }
                return Ok(Self::PyTorch);
            }
            "onnx" => return Ok(Self::Onnx),
            "pb" | "savedmodel" => return Ok(Self::TensorFlow),
            "gguf" => return Ok(Self::Gguf),
            "tflite" => return Ok(Self::Tflite),
            "exl2" => return Ok(Self::Exl2),
            "llamafile" => return Ok(Self::Llamafile),
            "h5" | "keras" => return Ok(Self::KerasH5),
            "safetensors" => return Ok(Self::SafeTensors),
            _ => {}
        }

        // Magic byte detection for files without recognized extension
        if path.is_file() {
            if let Ok(true) = check_magic(path, b"TFL3") {
                return Ok(Self::Tflite);
            }
            if let Ok(true) = check_magic(path, b"EXL2") {
                return Ok(Self::Exl2);
            }
            if let Ok(true) = check_magic(path, b"\x7fELF") {
                // Could be llamafile — check for LlamaFile marker
                if let Ok(true) = check_magic_offset(path, b"LlamaFile", 8) {
                    return Ok(Self::Llamafile);
                }
            }
            if let Ok(true) = check_magic(path, b"\x89HDF") {
                return Ok(Self::KerasH5);
            }
            if let Ok(true) = check_magic(path, b"GGUF") {
                return Ok(Self::Gguf);
            }
            if let Ok(true) = check_magic(path, b"PK") {
                return Ok(Self::SafeTensors);
            }
        }

        // Directory-based detection
        if path.is_dir() {
            if path.join("config.json").exists() {
                if path.join("quantize_config.json").exists() {
                    return Ok(Self::AwqGptq);
                }
                if path.join("_metadata").exists() {
                    return Ok(Self::JaxFlax);
                }
                return Ok(Self::HuggingFace);
            }
            // Check for safetensors files
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    if let Some(ext) = entry.path().extension() {
                        if ext == "safetensors" {
                            return Ok(Self::SafeTensors);
                        }
                    }
                }
            }
        }

        if ext.is_empty() {
            Err(TptIrError::UnsupportedFormat("no extension".into()))
        } else {
            Err(TptIrError::UnsupportedFormat(ext))
        }
    }
}

fn check_magic(path: &Path, magic: &[u8]) -> Result<bool, TptIrError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut header = vec![0u8; magic.len()];
    file.read_exact(&mut header)?;
    Ok(header == magic)
}

fn check_magic_offset(path: &Path, magic: &[u8], offset: usize) -> Result<bool, TptIrError> {
    use std::io::Read;
    use std::io::Seek;
    let mut file = std::fs::File::open(path)?;
    file.seek(std::io::SeekFrom::Start(offset as u64))?;
    let mut header = vec![0u8; magic.len()];
    file.read_exact(&mut header)?;
    Ok(header == magic)
}

pub fn ingest_model(path: &Path) -> Result<TptIr, TptIrError> {
    let format = ModelFormat::detect(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    match format {
        ModelFormat::PyTorch => ingest_pytorch(path, name),
        ModelFormat::Onnx => ingest_onnx(path, name),
        ModelFormat::TensorFlow => ingest_tensorflow(path, name),
        ModelFormat::Gguf => ingest_format(path, name, "gguf"),
        ModelFormat::Tflite => ingest_format(path, name, "tflite"),
        ModelFormat::Exl2 => ingest_format(path, name, "exl2"),
        ModelFormat::Llamafile => ingest_format(path, name, "llamafile"),
        ModelFormat::KerasH5 => ingest_format(path, name, "keras_h5"),
        ModelFormat::JaxFlax => ingest_format(path, name, "jax_flax"),
        ModelFormat::AwqGptq => ingest_format(path, name, "awq_gptq"),
        ModelFormat::HuggingFace => ingest_format(path, name, "huggingface"),
        ModelFormat::SafeTensors => ingest_format(path, name, "safetensors"),
    }
}

fn ingest_pytorch(_path: &Path, name: String) -> Result<TptIr, TptIrError> {
    let ir = TptIr::new(name, "pytorch".into());
    tracing::info!("PyTorch ingestion stub — real implementation requires Python bridge");
    Ok(ir)
}

fn ingest_onnx(_path: &Path, name: String) -> Result<TptIr, TptIrError> {
    let ir = TptIr::new(name, "onnx".into());
    tracing::info!("ONNX ingestion stub — real implementation requires onnxruntime bindings");
    Ok(ir)
}

fn ingest_tensorflow(_path: &Path, name: String) -> Result<TptIr, TptIrError> {
    let ir = TptIr::new(name, "tensorflow".into());
    tracing::info!("TensorFlow ingestion stub — real implementation requires TF saved model parser");
    Ok(ir)
}

fn ingest_format(_path: &Path, name: String, format: &str) -> Result<TptIr, TptIrError> {
    let ir = TptIr::new(name, format.into());
    tracing::info!("{format} ingestion — routed to Python implementation");
    Ok(ir)
}