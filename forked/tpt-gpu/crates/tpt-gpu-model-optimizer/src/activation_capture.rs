//! Activation Capture — hooks for capturing intermediate activations during inference.
//!
//! Provides a callback-based mechanism to intercept and record tensor activations
//! at key points in the forward pass (FFN intermediate outputs). This enables
//! real per-layer sensitivity analysis and domain mapping.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Activation statistics for a single layer's output.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayerActivations {
    /// Layer index in the model
    pub layer_idx: usize,
    /// Mean activation magnitude per neuron (length = ffn_dim)
    pub mean_magnitudes: Vec<f32>,
    /// Standard deviation per neuron
    pub stddev_magnitudes: Vec<f32>,
    /// Number of samples captured for this layer
    pub sample_count: usize,
}

impl LayerActivations {
    pub fn new(layer_idx: usize, ffn_dim: usize) -> Self {
        LayerActivations {
            layer_idx,
            mean_magnitudes: vec![0.0f32; ffn_dim],
            stddev_magnitudes: vec![0.0f32; ffn_dim],
            sample_count: 0,
        }
    }

    /// Add an activation sample and update running mean/std (Welford's algorithm)
    pub fn add_sample(&mut self, activations: &[f32]) {
        if activations.len() != self.mean_magnitudes.len() {
            return;
        }
        self.sample_count += 1;
        let n = self.sample_count as f32;

        for (i, &act) in activations.iter().enumerate() {
            let delta = act - self.mean_magnitudes[i];
            self.mean_magnitudes[i] += delta / n;

            // Update variance using Welford's online algorithm
            if self.sample_count > 1 {
                let delta2 = act - self.mean_magnitudes[i];
                self.stddev_magnitudes[i] =
                    (self.stddev_magnitudes[i] * (n - 1.0) + delta * delta2) / n;
            }
        }
    }

    /// Get the mean absolute activation per neuron (for importance scoring)
    pub fn mean_abs(&self) -> Vec<f32> {
        self.mean_magnitudes.iter().map(|&m| m.abs()).collect()
    }
}

/// Full activation map: layer_idx → LayerActivations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivationMap {
    pub layers: HashMap<usize, LayerActivations>,
    pub ffn_dim: usize,
}

impl ActivationMap {
    /// Save this activation map to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load an activation map from a JSON file written by [`save_to_file`].
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let map = serde_json::from_str(&raw)?;
        Ok(map)
    }
}

/// Load per-domain activation maps from a directory of `<domain>.json` files.
///
/// The file stem of each `*.json` file becomes the domain name. Files that do
/// not parse as an [`ActivationMap`] are skipped.
pub fn load_domain_maps_from_dir(dir: &Path) -> Result<HashMap<String, ActivationMap>> {
    let mut maps = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map(|e| e == "json").unwrap_or(false) {
            let domain = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if !domain.is_empty() {
                if let Ok(map) = ActivationMap::load_from_file(&path) {
                    maps.insert(domain, map);
                }
            }
        }
    }
    Ok(maps)
}

/// Callback type for activation capture
pub type ActivationCallback = Box<dyn FnMut(usize, &[f32]) + Send>;

/// Context for activation capture during inference
pub struct ActivationCapture {
    pub ffn_dim: usize,
    pub activations: ActivationMap,
    /// Per-domain activation maps: domain → ActivationMap. Populated via
    /// [`record_domain`](ActivationCapture::record_domain).
    pub domain_activations: HashMap<String, ActivationMap>,
    pub callback: Option<ActivationCallback>,
}

impl ActivationCapture {
    pub fn new(ffn_dim: usize) -> Self {
        ActivationCapture {
            ffn_dim,
            activations: ActivationMap::default(),
            domain_activations: HashMap::new(),
            callback: None,
        }
    }

    /// Set an activation callback
    pub fn with_callback(mut self, callback: ActivationCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Record activations for a given layer
    pub fn record(&mut self, layer_idx: usize, activations: &[f32]) {
        // Call the external callback if present
        if let Some(ref mut cb) = self.callback {
            cb(layer_idx, activations);
        }

        // Update internal statistics
        let layer_acts = self
            .activations
            .layers
            .entry(layer_idx)
            .or_insert_with(|| LayerActivations::new(layer_idx, self.ffn_dim));
        layer_acts.add_sample(activations);
    }

    /// Record activations for a given domain + layer.
    ///
    /// Used by the production domain-mapping path: each calibration prompt is
    /// tagged with its domain, and per-domain activation statistics are kept
    /// separately so neurons can be assigned to the domain that activates them
    /// most strongly.
    pub fn record_domain(&mut self, domain: &str, layer_idx: usize, activations: &[f32]) {
        // Call the external callback if present
        if let Some(ref mut cb) = self.callback {
            cb(layer_idx, activations);
        }

        let map = self
            .domain_activations
            .entry(domain.to_string())
            .or_default();
        map.ffn_dim = self.ffn_dim;
        let layer_acts = map
            .layers
            .entry(layer_idx)
            .or_insert_with(|| LayerActivations::new(layer_idx, self.ffn_dim));
        layer_acts.add_sample(activations);
    }

    /// Finalize and return the activation map
    pub fn finalize(self) -> ActivationMap {
        self.activations
    }

    /// Finalize and return the per-domain activation maps.
    pub fn finalize_domains(self) -> HashMap<String, ActivationMap> {
        self.domain_activations
    }

    /// Save activation map to JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        self.activations.save_to_file(path)
    }

    /// Save per-domain activation maps to a directory as `<domain>.json`.
    pub fn save_domains_to_dir(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        for (domain, map) in &self.domain_activations {
            map.save_to_file(&dir.join(format!("{domain}.json")))?;
        }
        Ok(())
    }
}

/// Extension trait for GpuInferenceEngine to add activation capture
pub trait ActivationCaptureExt {
    /// Enable activation capture with the given callback
    fn with_activation_capture(ffn_dim: usize, callback: ActivationCallback) -> Self;

    /// Get the captured activations
    fn activation_map(&self) -> &ActivationMap;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_layer_activations() {
        let mut capture = ActivationCapture::new(8);

        // Record some fake activations
        capture.record(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        capture.record(0, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let map = capture.finalize();
        assert!(map.layers.contains_key(&0));
        let layer0 = &map.layers[&0];
        assert_eq!(layer0.sample_count, 2);
    }

    #[test]
    fn callback_is_invoked() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut capture =
            ActivationCapture::new(4).with_callback(Box::new(move |_layer, _acts| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
            }));

        capture.record(0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn captures_per_domain_activations() {
        let mut capture = ActivationCapture::new(4);
        capture.record_domain("sql", 0, &[0.0, 2.0, 0.0, 0.0]);
        capture.record_domain("sql", 0, &[0.0, 2.0, 0.0, 0.0]);
        capture.record_domain("python", 0, &[0.0, 0.0, 3.0, 0.0]);

        let domains = capture.finalize_domains();
        assert_eq!(domains.len(), 2);
        let sql = &domains["sql"];
        let python = &domains["python"];
        assert_eq!(sql.layers[&0].sample_count, 2);
        assert_eq!(python.layers[&0].sample_count, 1);
        // Neuron 1 is strongly activated by sql prompts, neuron 2 by python.
        assert!(sql.layers[&0].mean_abs()[1] > 0.0);
        assert!(python.layers[&0].mean_abs()[2] > 0.0);
    }

    #[test]
    fn domain_maps_round_trip_via_dir() {
        let mut capture = ActivationCapture::new(4);
        capture.record_domain("sql", 0, &[0.0, 2.0, 0.0, 0.0]);
        capture.record_domain("python", 0, &[0.0, 0.0, 3.0, 0.0]);

        let dir = std::env::temp_dir().join("tpt_act_capture_dir_test");
        let _ = std::fs::remove_dir_all(&dir);
        capture.save_domains_to_dir(&dir).unwrap();

        let loaded = load_domain_maps_from_dir(&dir).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("sql"));
        assert!(loaded.contains_key("python"));
        assert_eq!(loaded["sql"].layers[&0].mean_abs()[1], 2.0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
