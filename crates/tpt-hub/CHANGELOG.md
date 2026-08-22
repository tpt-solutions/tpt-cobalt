# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- SafeTensors reader/writer over 	pt-tensor::Tensor with typed HubError diagnostics.
- TPTB minimal self-describing binary container (save_tptb / load_tptb).
- JSON debug format for human-readable tensor inspection and fixtures.
- GGUF writer and parser (GgufFile, GgufTensorInfo, GgufValue, 32-byte alignment).
- Minimal ONNX ModelProto walker: initializers, graph I/O, IR version, node ops (no protobuf dependency).
- Arrow IPC file bridge with shape-preserving schema metadata.
- Cross-process tensor mailbox: publish / load / wait_for / vailable with atomic rename semantics.
- Two runnable examples (safetensors_roundtrip, ipc_tensor_mailbox) and a comprehensive README.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-hub-v0.1.0
