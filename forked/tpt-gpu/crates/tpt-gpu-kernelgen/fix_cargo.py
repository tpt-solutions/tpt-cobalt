"""Fix Cargo.toml by removing duplicate sections."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\Cargo.toml")

content = """[package]
name = "tptc-rs"
version = "0.1.0"
edition = "2021"
description = "Rust port of TPTIR compiler stack"

[lib]
name = "tptc_rs"
crate-type = ["lib", "cdylib"]

[features]
default = ["ffi"]
ffi = []

[dependencies]
libc = "0.2"
serde = { workspace = true }
serde_json = { workspace = true }
"""

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
