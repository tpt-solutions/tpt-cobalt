//! HuggingFace GGUF download helper.
//!
//! Downloads a GGUF file from a HuggingFace URL into `~/.tpt/models/` and
//! registers it in the manifest. Requires the `url` to be a direct download
//! link (e.g. the HuggingFace "Download" button URL).
//!
//! The download streams the response body straight to disk, optionally
//! verifies the SHA-256 of the resulting file when `expected_sha256` is set
//! (removing the partial file on mismatch), and only registers the manifest
//! entry after the file is confirmed on disk.

use std::fs::File;
use std::io::{copy, Read};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::{ModelEntry, ModelRegistry};

/// Parameters for a HuggingFace GGUF download.
pub struct HfDownload {
    /// Direct URL to the GGUF file.
    pub url: String,
    /// Model name to register (e.g. `"llama-3-8b-q4"`).
    pub name: String,
    /// Architecture tag (e.g. `"llama3"`).
    pub arch: String,
    /// Expected approximate size in GiB (used for the manifest).
    pub size_gb: f64,
    /// Optional expected SHA-256 for integrity checking.
    pub expected_sha256: Option<String>,
}

/// Compute the SHA-256 hex digest of a file on disk.
fn sha256_of(path: &std::path::Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("cannot open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("cannot read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Download a GGUF from HuggingFace into the registry dir and update the manifest.
///
/// Streams the response body to `~/.tpt/models/<filename>`, verifies the
/// SHA-256 when `expected_sha256` is provided (deleting the partial file on
/// mismatch), and only then registers the entry in `models.json`. The manifest
/// is never touched unless the file landed on disk intact.
pub fn download(registry: &mut ModelRegistry, params: HfDownload) -> Result<PathBuf> {
    if params.url.is_empty() {
        bail!("URL must not be empty");
    }
    if params.name.is_empty() {
        bail!("model name must not be empty");
    }
    if !params.url.ends_with(".gguf") {
        bail!(
            "URL does not look like a GGUF download link: {}",
            params.url
        );
    }

    let filename = params
        .url
        .split('/')
        .next_back()
        .unwrap_or("model.gguf")
        .to_string();

    let dest = registry.dir().join(&filename);

    let resp = ureq::get(&params.url)
        .call()
        .with_context(|| format!("GET {} failed", params.url))?;
    let mut reader = resp.into_body().into_reader();

    let mut file =
        File::create(&dest).with_context(|| format!("cannot create {}", dest.display()))?;
    copy(&mut reader, &mut file)
        .with_context(|| format!("failed while streaming to {}", dest.display()))?;
    drop(file);

    // Optional integrity check: remove the partial file on mismatch.
    if let Some(expected) = &params.expected_sha256 {
        let expected = expected.trim().to_ascii_lowercase();
        let actual = sha256_of(&dest)?;
        if actual != expected {
            let _ = std::fs::remove_file(&dest);
            bail!(
                "SHA-256 mismatch for {}: expected {} but computed {} (partial file removed)",
                dest.display(),
                expected,
                actual
            );
        }
    }

    // Only register once the file is confirmed on disk.
    registry.register(ModelEntry {
        name: params.name,
        file: filename,
        arch: params.arch,
        size_gb: params.size_gb,
        sha256: params.expected_sha256,
        source: Some(params.url),
        quant_bits: None,
        pruned_domains: None,
        source_model: None,
    })?;

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_encodes_lowercase() {
        assert_eq!(hex(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn rejects_bad_urls_and_names() {
        let dir = std::env::temp_dir().join(format!("tpt_hf_test_{}", std::process::id()));
        let mut reg = ModelRegistry::open_at(&dir).unwrap();

        let bad_url = HfDownload {
            url: "https://example.com/model.bin".to_string(),
            name: "m".to_string(),
            arch: "llama3".to_string(),
            size_gb: 1.0,
            expected_sha256: None,
        };
        assert!(download(&mut reg, bad_url).is_err());

        let bad_name = HfDownload {
            url: "https://example.com/model.gguf".to_string(),
            name: String::new(),
            arch: "llama3".to_string(),
            size_gb: 1.0,
            expected_sha256: None,
        };
        assert!(download(&mut reg, bad_name).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
