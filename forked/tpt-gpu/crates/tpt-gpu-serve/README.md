# tpt-gpu-serve

`tpt-serve` — an OpenAI-compatible HTTP server for TPT GPU LLM inference.

## Overview

`tpt-gpu-serve` loads a GGUF or TPTF model via the `tpt-gpu-runtime` `LlmInference`
engine and serves OpenAI-shaped endpoints over plain `std::net::TcpListener` (no web
framework dependency). It supports both non-streaming and SSE streaming responses, and
uses the model's real GGUF tokenizer when present (falling back to a session-local
placeholder for tokenizer-less models).

## Installation

```bash
cargo install tpt-gpu-serve
# or, from the repo root:
cargo build --release -p tpt-gpu-serve
```

## Usage

```bash
tpt-serve --model path/to/model.gguf [--host 127.0.0.1] [--port 8080]
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/models` | List the loaded model |
| `POST` | `/v1/completions` | Text completion (supports `stream: true`) |
| `POST` | `/v1/chat/completions` | Chat completion (supports `stream: true`) |

Requests accept `prompt`/`messages` (text) or `prompt_tokens` (raw token ids), and the
responses follow the OpenAI JSON schema.

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
