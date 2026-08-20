//! Model format detection via magic bytes.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Supported model file formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    Tptf,
    Gguf,
    Exl2,
    Unknown,
}

/// Detect format by reading the first 4 bytes.
pub fn detect(path: &Path) -> ModelFormat {
    let Ok(mut f) = File::open(path) else {
        return ModelFormat::Unknown;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return ModelFormat::Unknown;
    }
    match &magic {
        b"TPTF" => ModelFormat::Tptf,
        b"GGUF" => ModelFormat::Gguf,
        // EXL2 is a directory-based format; a single file won't have magic.
        // If the path ends in .exl2 and magic is JSON-like, treat as EXL2 config.
        [b'{', ..] if path.extension().and_then(|e| e.to_str()) == Some("exl2") => {
            ModelFormat::Exl2
        }
        _ => ModelFormat::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_tptf() {
        let dir = std::env::temp_dir();
        let p = dir.join("test_detect.tptf");
        let mut f = File::create(&p).unwrap();
        f.write_all(b"TPTF\x01\x00\x00\x00").unwrap();
        drop(f);
        assert_eq!(detect(&p), ModelFormat::Tptf);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn detects_gguf() {
        let dir = std::env::temp_dir();
        let p = dir.join("test_detect.gguf");
        let mut f = File::create(&p).unwrap();
        f.write_all(b"GGUF\x03\x00\x00\x00").unwrap();
        drop(f);
        assert_eq!(detect(&p), ModelFormat::Gguf);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn unknown_for_empty() {
        let dir = std::env::temp_dir();
        let p = dir.join("test_detect_empty.bin");
        File::create(&p).unwrap();
        assert_eq!(detect(&p), ModelFormat::Unknown);
        std::fs::remove_file(p).ok();
    }
}
