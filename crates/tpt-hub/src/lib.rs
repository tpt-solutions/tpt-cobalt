//! # tpt-hub — Model & Weight Hub (Phase 2, spec §5.4)
//!
//! Starting point: `tpt-rust6::tpt-io` + `tpt-crucible`'s Catalyst (ONNX/GGUF/
//! SafeTensors ingestion). This scaffold implements a **SafeTensors** reader/
//! writer over [`tpt_tensor::Tensor`]: the dominant format for shipping trained
//! weights. ONNX/GGUF parsers are deferred (see status note) — SafeTensors is
//! enough to load/round-trip model state produced by `tpt-ml` and friends.

pub mod safetensors;
pub mod serialize;

pub use safetensors::{HubError, load_safetensors, save_safetensors};
pub use serialize::{
    TPTB_MAGIC, load_tptb, save_tptb, tensor_from_json_debug, tensor_to_json_debug,
};
