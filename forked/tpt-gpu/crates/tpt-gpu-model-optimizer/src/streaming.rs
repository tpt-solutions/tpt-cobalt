//! Streaming Loader — processes 70B+ models layer-by-layer via mmap.
//!
//! For models too large to load entirely into VRAM, `StreamingLoader` mmaps
//! the source GGUF file and yields one layer at a time. The caller quantizes
//! and writes each layer to the output `.tptf` immediately, so peak memory
//! equals the largest single layer (not the full model).
//!
//! Activation thresholds: if `model_size > vram_free * 0.8`, streaming is
//! enabled automatically.

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

/// A single layer's raw weight data, borrowed from the mmap.
#[derive(Debug)]
pub struct LayerView<'a> {
    pub layer_idx: usize,
    /// Raw byte slice pointing into the mmap. Caller is responsible for
    /// interpreting as the correct element type (f32, f16, etc.).
    pub weights: &'a [u8],
    /// Number of rows (hidden_dim for gate/up/down projections).
    pub rows: usize,
    /// Number of columns (ffn_dim for gate/up, hidden_dim for down).
    pub cols: usize,
}

/// Streams a GGUF model file layer-by-layer without loading it all into RAM.
pub struct StreamingLoader {
    path: std::path::PathBuf,
    num_layers: usize,
    /// If true, prints progress to stderr.
    verbose: bool,
}

impl StreamingLoader {
    pub fn new(path: &Path, num_layers: usize) -> Self {
        StreamingLoader {
            path: path.to_path_buf(),
            num_layers,
            verbose: false,
        }
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Returns true when streaming should be used.
    ///
    /// `model_size_mb`: total model weight size in MiB.
    /// `vram_free_mb`: free VRAM reported by the hardware profiler.
    pub fn should_stream(model_size_mb: f64, vram_free_mb: u64) -> bool {
        model_size_mb > vram_free_mb as f64 * 0.8
    }

    /// Iterate over all layers, calling `f` for each.
    ///
    /// The mmap is held open for the duration of the callback to allow
    /// zero-copy access to raw bytes. In production `f` would quantize the
    /// layer weights and write the compressed result to the output `.tptf`.
    pub fn for_each_layer<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(LayerView<'_>) -> Result<()>,
    {
        let file = File::open(&self.path).with_context(|| format!("opening {:?}", self.path))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.with_context(|| "mmap failed")?;

        // Validate GGUF or TPTF magic
        if mmap.len() < 8 {
            bail!("file too small to be a valid model");
        }
        let magic = &mmap[..4];
        if magic != b"GGUF" && magic != b"TPTF" {
            bail!("unknown file format (magic={:?})", magic);
        }

        // In production: parse the GGUF tensor index (key-value + tensor info
        // sections) to locate each layer's data offset + byte length. Then
        // call f() for each layer with a slice into `mmap`.
        //
        // This scaffold simulates that by dividing the file evenly.
        let header_bytes = 256usize;
        let payload = mmap.len().saturating_sub(header_bytes);
        let layer_bytes = payload / self.num_layers.max(1);

        for i in 0..self.num_layers {
            let start = header_bytes + i * layer_bytes;
            let end = (start + layer_bytes).min(mmap.len());
            let slice = &mmap[start..end];

            if self.verbose {
                eprintln!("[streaming] layer {}/{}", i + 1, self.num_layers);
            }

            f(LayerView {
                layer_idx: i,
                weights: slice,
                rows: 4096,
                cols: 11008,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_stream_threshold() {
        // 14 GiB model vs 16 GiB free — model is 87.5% of free, should stream
        assert!(StreamingLoader::should_stream(14336.0, 16384));
        // 8 GiB model vs 16 GiB free — 50%, should NOT stream
        assert!(!StreamingLoader::should_stream(8192.0, 16384));
    }
}
