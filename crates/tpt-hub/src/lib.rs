//! # tpt-hub — Model & Weight Hub (Phase 2, spec §5.4)
//!
//! Starting point: `tpt-rust6::tpt-io` + `tpt-crucible`'s Catalyst (ONNX/GGUF/
//! SafeTensors ingestion). This scaffold implements a **SafeTensors** reader/
//! writer over [`tpt_tensor::Tensor`]: the dominant format for shipping trained
//! weights. ONNX/GGUF parsers are deferred (see status note) — SafeTensors is
//! enough to load/round-trip model state produced by `tpt-ml` and friends.

pub mod safetensors;

pub use safetensors::{HubError, load_safetensors, save_safetensors};
