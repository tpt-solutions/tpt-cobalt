//! End-to-end test: spawn the real `tpt-serve` binary, load a minimal GGUF
//! model, and exercise the OpenAI-compatible `/v1/completions` endpoint.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::Duration;

/// Write a minimal valid GGUF v2 file that also carries a small real tokenizer
/// (vocab + merges + special-token ids) so `tpt-serve` emits real decoded text
/// instead of the `<id>` placeholder tokens.
fn write_minimal_gguf(path: &std::path::Path) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count

    // (key, type, value) — type: 4=U32, 8=STRING, 9=STRING_ARRAY, 7=BOOL
    // value: for 8/9 a string (or comma-joined for arrays), for 7 "0"/"1".
    let kv: &[(&str, u8, &str)] = &[
        ("general.architecture", 8, "llama3"),
        ("llm.context_length", 4, "64"),
        ("llm.embedding_length", 4, "64"),
        ("llm.attention.head_count", 4, "4"),
        ("llm.attention.head_count_kv", 4, "2"),
        ("llm.feed_forward_length", 4, "128"),
        ("llm.block_count", 4, "2"),
        ("tokenizer.ggml.model", 8, "llama"),
        ("tokenizer.ggml.tokens", 9, "the,cat,hello,world"),
        ("tokenizer.ggml.merges", 9, ""),
        ("tokenizer.ggml.bos_token_id", 4, "0"),
        ("tokenizer.ggml.eos_token_id", 4, "0"),
        ("tokenizer.ggml.unknown_token_id", 4, "0"),
        ("tokenizer.ggml.add_bos_token", 7, "0"),
        ("tokenizer.ggml.add_eos_token", 7, "0"),
    ];
    buf.extend_from_slice(&(kv.len() as u64).to_le_bytes()); // kv_count
    for (key, ty, val) in kv {
        let kb = key.as_bytes();
        buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        buf.extend_from_slice(kb);
        match *ty {
            4 => {
                buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
                let v: u32 = val.parse().unwrap_or(0);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            7 => {
                buf.extend_from_slice(&7u32.to_le_bytes()); // BOOL
                buf.push(if val == &"1" { 1 } else { 0 });
            }
            8 => {
                buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
                let sb = val.as_bytes();
                buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                buf.extend_from_slice(sb);
            }
            9 => {
                buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
                buf.extend_from_slice(&8u32.to_le_bytes()); // elem_type STRING
                let items: Vec<&str> = if val.is_empty() {
                    vec![]
                } else {
                    val.split(',').collect()
                };
                buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    let ib = it.as_bytes();
                    buf.extend_from_slice(&(ib.len() as u64).to_le_bytes());
                    buf.extend_from_slice(ib);
                }
            }
            _ => {}
        }
    }
    std::fs::write(path, &buf).unwrap();
}

struct ServerProcess {
    child: Child,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn wait_for_server(addr: &str) -> TcpStream {
    for _ in 0..100 {
        if let Ok(s) = TcpStream::connect(addr) {
            return s;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("tpt-serve did not start listening on {addr}");
}

fn http_request(method: &str, addr: &str, path: &str, body: &str) -> String {
    let mut stream = wait_for_server(addr);
    let req = if body.is_empty() {
        format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    };
    stream.write_all(req.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    match resp.find("\r\n\r\n") {
        Some(p) => resp[p + 4..].to_string(),
        None => resp,
    }
}

fn http_post(addr: &str, path: &str, body: &str) -> String {
    http_request("POST", addr, path, body)
}

fn http_get(addr: &str, path: &str) -> String {
    http_request("GET", addr, path, "")
}

#[test]
fn serve_completions_returns_openai_shape() {
    let dir = std::env::temp_dir();
    let model = dir.join(format!("tpt_serve_test_{}.gguf", std::process::id()));
    write_minimal_gguf(&model);

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");

    let child = Command::new(env!("CARGO_BIN_EXE_tpt-serve"))
        .args([
            "--model",
            &model.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn tpt-serve");
    let _server = ServerProcess { child };

    // /v1/models
    let models = http_get(&addr, "/v1/models");
    let models_json: serde_json::Value = serde_json::from_str(&models).unwrap();
    assert_eq!(models_json["object"], "list");
    assert_eq!(
        models_json["data"][0]["id"],
        model.to_string_lossy().to_string()
    );

    // /v1/completions with explicit token ids
    let body = r#"{"model":"x","prompt_tokens":[1,2,3],"max_tokens":4}"#;
    let resp = http_post(&addr, "/v1/completions", body);
    let json: serde_json::Value = serde_json::from_str(&resp).unwrap();

    assert_eq!(json["object"], "text_completion");
    assert!(json["choices"].is_array());
    let choice = &json["choices"][0];
    assert!(choice["text"].is_string(), "expected a text field: {resp}");
    assert_eq!(choice["finish_reason"], "length");
    assert!(json["usage"]["completion_tokens"].as_u64().unwrap_or(0) <= 4);

    // The model carries a real tokenizer, so decoded output must be real text,
    // never the `<id>` placeholder emitted by the old WordTokenizer fallback.
    // With zeroed weights the deterministic llama3 sampler yields token 3
    // ("world"), repeated `max_tokens` (4) times.
    let text = choice["text"].as_str().unwrap_or("");
    assert!(
        !text.contains('<'),
        "expected real decoded text, got placeholder output: {text:?}"
    );
    assert!(
        text == "world".repeat(4),
        "expected real decoded vocab text, got: {text:?}"
    );

    // /v1/chat/completions
    let chat_body =
        r#"{"model":"x","messages":[{"role":"user","content":"hello"}],"max_tokens":3}"#;
    let chat = http_post(&addr, "/v1/chat/completions", chat_body);
    let chat_json: serde_json::Value = serde_json::from_str(&chat).unwrap();
    assert_eq!(chat_json["object"], "chat.completion");
    assert_eq!(chat_json["choices"][0]["message"]["role"], "assistant");

    let _ = std::fs::remove_file(&model);
}
