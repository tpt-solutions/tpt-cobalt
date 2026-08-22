# tpt-hub

The model and weight hub — serialization formats and tensor sharing for the
`tpt-cobalt` stack.

`tpt-hub` reads and writes the formats trained models actually ship in, all
over [`tpt-tensor::Tensor`](https://crates.io/crates/tpt-tensor):

- **SafeTensors** — full reader/writer (the dominant weight-shipping format).
- **TPTB** — a minimal self-describing binary container (magic + version +
  dtype + rank + dims + raw LE data) for checkpoints and IPC frames.
- **JSON debug** — human-readable `{dtype, shape, values}` documents for
  inspection and small fixtures.
- **GGUF** — writer plus parser (metadata + tensor layout info).
- **ONNX** — a minimal `ModelProto` walker extracting initializers, graph
  inputs/outputs, IR version, and node ops (hand-rolled protobuf wire-format
  reader; no prost dependency).
- **Arrow IPC** — named tensors to/from Arrow IPC *files*, shapes preserved as
  schema-field metadata.
- **Cross-process sharing** — an atomic-rename mailbox: `publish` / `load` /
  `wait_for` / `available`. Readers never see a partial tensor.

## Installation

```toml
[dependencies]
tpt-hub = "0.1"
```

## Quick start

```rust
use std::collections::HashMap;
use tpt_hub::{load_safetensors, save_safetensors};
use tpt_tensor::Tensor;

fn main() {
    let w = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();

    // Serialize named tensors to SafeTensors bytes (interop with PyTorch/HF).
    let bytes = save_safetensors(&[("weights", &w)]);
    let loaded: HashMap<String, Tensor> = load_safetensors(&bytes).unwrap();

    assert_eq!(
        loaded["weights"].to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
```

## Examples

```sh
cargo run -p tpt-hub --example safetensors_roundtrip
cargo run -p tpt-hub --example ipc_tensor_mailbox
```

## Error handling

All fallible entry points return [`HubError`], covering short buffers, bad
header lengths, malformed JSON, out-of-range offsets, and unknown dtypes.

## Status

SafeTensors, TPTB, JSON debug, GGUF, ONNX (subset), Arrow IPC, and the IPC
mailbox are functional. Full ONNX graph execution is out of scope — this crate
is about weights and transport, not inference runtimes.

## License

Dual-licensed under the workspace license (see repository root).
