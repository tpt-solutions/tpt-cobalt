//! # tpt-fusion — the Fusion (FPGA) backend, native Rust (Phase 5)
//!
//! Native Rust build per `docs/phase5-exotic-backends.md` (no port of the
//! upstream Python module, and no raw-bitstream path — that is explicitly
//! declined there). The honest deliverable is **artifact emission**: Catalyst
//! IR in → HLS-C++ kernel sources + a vendor toolchain invocation manifest
//! out, with a **memory-fit proof computed before anything is emitted** (the
//! artifact-level version of the spec's "compiler proves memory fit before
//! execution" success criterion).
//!
//! Scope of the first slice:
//! - tiled GEMM kernels (the LLM-inference workhorse) with per-buffer
//!   on-chip accounting (A/B tiles double-buffered; C accumulates in f32
//!   unless already f32),
//! - Xilinx `v++` command templates; other vendors error honestly instead of
//!   emitting pretend command lines,
//! - ops outside the supported set are reported, never silently skipped.
//!
//! The generated kernel is a correctness-shaped reference skeleton (tiled
//! loads into local buffers, pipelined inner loop, fixed bounds from the IR
//! shapes); performance tuning is vendor-toolchain work and is deliberately
//! out of scope here.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tpt_catalyst::ir::TptIr;

pub mod proof;
pub use proof::{alloc_region, allocs_from_module, build_manifest_proved, AllocTensor, Dim, MemoryProof};

// ------------------------------- device model ------------------------------

/// FPGA vendor family (drives the toolchain manifest; see [`FusionError`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vendor {
    Xilinx,
    Intel,
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vendor::Xilinx => write!(f, "xilinx"),
            Vendor::Intel => write!(f, "intel"),
        }
    }
}

/// Target device: an identity plus the two numbers the memory-fit proof
/// actually needs — the on-chip buffer budget and the clock. Hardware
/// defaults are intentionally *not* hardcoded; budgets are per-project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub name: String,
    pub vendor: Vendor,
    /// On-chip memory available for kernel local buffers, in bytes.
    pub buffer_budget_bytes: usize,
    /// Global memory (DDR) the *model* must provably fit — consumed by
    /// [`proof::build_manifest_proved`]'s UIR memory-bound proof.
    pub global_mem_bytes: usize,
    pub clock_mhz: f64,
}

impl DeviceConfig {
    pub fn new(
        name: impl Into<String>,
        vendor: Vendor,
        buffer_budget_bytes: usize,
        global_mem_bytes: usize,
        clock_mhz: f64,
    ) -> Self {
        DeviceConfig {
            name: name.into(),
            vendor,
            buffer_budget_bytes,
            global_mem_bytes,
            clock_mhz,
        }
    }
}

/// Element type of the GEMM (C always accumulates in f32 unless dtype is f32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I8,
}

impl DType {
    pub fn bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
        }
    }

    pub fn hls_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::I8 => "ap_int<8>",
        }
    }

    /// Accumulator element type (f32 for every dtype).
    pub fn accum_bytes(self) -> usize {
        4
    }
}

// --------------------------------- gemm spec --------------------------------

/// Tiled GEMM: `C[M,N] = A[M,K] · B[K,N]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmSpec {
    pub name: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: DType,
}

/// Tile sizes for the local buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Double-buffer the A/B tile loads against compute.
    pub double_buffer: bool,
}

/// Per-buffer byte accounting for one GEMM kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitReport {
    /// Named per-buffer byte costs (`a_buf`, `b_buf`, `c_buf`).
    pub breakdown: Vec<(String, usize)>,
    pub total_bytes: usize,
    pub budget_bytes: usize,
    pub fits: bool,
}

impl FitReport {
    /// `name (n bytes)` lines for error messages.
    pub fn describe(&self) -> String {
        self.breakdown
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// On-chip buffer bytes for one kernel: A tile (m×k) + B tile (k×n) + C tile
/// (m×n). A/B double-buffer when configured; C accumulates in f32 bytes.
pub fn check_memory_fit(spec: &GemmSpec, tile: &TileConfig, device: &DeviceConfig) -> FitReport {
    let dt = spec.dtype.bytes();
    let mult = if tile.double_buffer { 2 } else { 1 };
    let a = tile.m * tile.k * dt * mult;
    let b = tile.k * tile.n * dt * mult;
    let c = tile.m * tile.n * spec.dtype.accum_bytes();
    let total = a + b + c;
    FitReport {
        breakdown: vec![
            ("a_buf".to_string(), a),
            ("b_buf".to_string(), b),
            ("c_buf".to_string(), c),
        ],
        total_bytes: total,
        budget_bytes: device.buffer_budget_bytes,
        fits: total <= device.buffer_budget_bytes,
    }
}

// ------------------------------ HLS emission -------------------------------

/// Emit an HLS-C++ tiled GEMM kernel for `spec`. Deterministic: the same
/// inputs always produce byte-identical output (tested).
pub fn emit_hls_gemm(spec: &GemmSpec, tile: &TileConfig) -> String {
    let t = spec.dtype.hls_type();
    let acc = spec.dtype.accum_bytes();
    let _ = acc;
    let mut s = String::new();
    s.push_str("// Generated by tpt-fusion — tiled GEMM kernel\n");
    s.push_str(&format!(
        "// shapes: M={} N={} K={}  tiles: MI={} NJ={} KI={}  db={}\n",
        spec.m, spec.n, spec.k, tile.m, tile.n, tile.k, tile.double_buffer
    ));
    s.push_str(&format!(
        "// budget check: caller must run check_memory_fit before emitting\n"
    ));
    s.push_str("#include <hls_stream.h>\n");
    s.push_str("#include <ap_int.h>\n\n");
    s.push_str(&format!("#define M {}\n", spec.m));
    s.push_str(&format!("#define N {}\n", spec.n));
    s.push_str(&format!("#define K {}\n", spec.k));
    s.push_str(&format!("#define MI {}\n", tile.m));
    s.push_str(&format!("#define NJ {}\n", tile.n));
    s.push_str(&format!("#define KI {}\n", tile.k));
    s.push_str(&format!("typedef {} T;\n", t));
    s.push_str("typedef float ACC;\n\n");
    s.push_str(&format!(
        "void {}(const T a[M][K], const T b[K][N], T c[M][N]) {{\n",
        spec.name
    ));
    s.push_str("#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0\n");
    s.push_str("#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1\n");
    s.push_str("#pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem0\n");
    s.push_str("#pragma HLS INTERFACE s_axilite port=return\n\n");
    s.push_str("  T a_buf[MI][KI];\n#pragma HLS BIND_STORAGE variable=a_buf type=ram_2p\n");
    s.push_str("  T b_buf[KI][NJ];\n#pragma HLS BIND_STORAGE variable=b_buf type=ram_2p\n");
    s.push_str("  ACC c_buf[MI][NJ];\n#pragma HLS BIND_STORAGE variable=c_buf type=ram_2p\n\n");
    s.push_str("  for (int mi = 0; mi < M; mi += MI) {\n");
    s.push_str("  for (int nj = 0; nj < N; nj += NJ) {\n");
    s.push_str("    for (int i = 0; i < MI; i++)\n");
    s.push_str("      for (int j = 0; j < NJ; j++) c_buf[i][j] = 0;\n");
    s.push_str("    for (int ki = 0; ki < K; ki += KI) {\n");
    s.push_str("      // load A/B tiles (double-buffered by the toolchain when db=1)\n");
    s.push_str("      T a_next[MI][KI], b_next[KI][NJ];\n");
    s.push_str("#pragma HLS BIND_STORAGE variable=a_next type=ram_2p\n");
    s.push_str("#pragma HLS BIND_STORAGE variable=b_next type=ram_2p\n");
    s.push_str("      load_a: for (int i = 0; i < MI; i++)\n");
    s.push_str("        for (int p = 0; p < KI; p++)\n");
    s.push_str("          a_next[i][p] = a[mi + i][ki + p];\n");
    s.push_str("      load_b: for (int p = 0; p < KI; p++)\n");
    s.push_str("        for (int j = 0; j < NJ; j++)\n");
    s.push_str("          b_next[p][j] = b[ki + p][nj + j];\n\n");
    s.push_str("      compute: for (int i = 0; i < MI; i++) {\n");
    s.push_str("#pragma HLS PIPELINE II=1\n");
    s.push_str("        for (int p = 0; p < KI; p++)\n");
    s.push_str("          for (int j = 0; j < NJ; j++)\n");
    s.push_str("            c_buf[i][j] += (ACC)a_next[i][p] * (ACC)b_next[p][j];\n");
    s.push_str("      }\n");
    s.push_str("    }\n");
    s.push_str("    store_c: for (int i = 0; i < MI; i++)\n");
    s.push_str("      for (int j = 0; j < NJ; j++)\n");
    s.push_str("        c[mi + i][nj + j] = (T)c_buf[i][j];\n");
    s.push_str("  }\n");
    s.push_str("  }\n");
    s.push_str("}\n");
    s
}

// -------------------------------- manifest ---------------------------------

/// One emitted kernel in the toolchain manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArtifact {
    pub name: String,
    pub kind: String,
    pub gemm: GemmSpec,
    pub tile: TileConfig,
    pub fit: FitReport,
    pub hls_file: String,
}

/// A complete, toolchain-runnable artifact set: HLS sources plus the vendor
/// command lines that consume them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainManifest {
    pub model_name: String,
    pub device: String,
    pub vendor: Vendor,
    pub clock_mhz: f64,
    pub kernels: Vec<KernelArtifact>,
    pub commands: Vec<String>,
}

impl ToolchainManifest {
    /// Serialize to pretty JSON (for `manifest.json`).
    pub fn to_json(&self) -> Result<String, FusionError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| FusionError::Serialization(e.to_string()))
    }

    /// Write `manifest.json` plus one `.cpp` per kernel into `dir`.
    pub fn write_out(&self, dir: &Path) -> Result<(), FusionError> {
        std::fs::create_dir_all(dir).map_err(FusionError::io)?;
        for k in &self.kernels {
            let text = emit_hls_gemm(&k.gemm, &k.tile);
            std::fs::write(dir.join(&k.hls_file), text).map_err(FusionError::io)?;
        }
        std::fs::write(dir.join("manifest.json"), self.to_json()?).map_err(FusionError::io)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// IR nodes Fusion cannot lower (reported together, never skipped).
    UnsupportedOps(Vec<String>),
    /// A supported op is missing its shape attributes.
    MissingShape { node: String, attr: String },
    /// A kernel does not fit the device's on-chip buffer budget.
    DoesNotFit { kernel: String, report: FitReport },
    /// The UIR global-memory proof found an overflowing assignment.
    ProofFailed { detail: String },
    /// The UIR global-memory proof could not be decided.
    ProofInconclusive(String),
    UnsupportedVendor(Vendor),
    Serialization(String),
    Io(String),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionError::UnsupportedOps(ops) => {
                write!(f, "unsupported ops for FPGA lowering: {}", ops.join(", "))
            }
            FusionError::MissingShape { node, attr } => {
                write!(f, "node '{node}' is missing required shape attribute '{attr}'")
            }
            FusionError::DoesNotFit { kernel, report } => write!(
                f,
                "kernel '{kernel}' needs {} on-chip bytes but device budget is {}: {}",
                report.total_bytes,
                report.budget_bytes,
                report.describe()
            ),
            FusionError::UnsupportedVendor(v) => {
                write!(f, "no toolchain command template for vendor '{v}' yet")
            }
            FusionError::ProofFailed { detail } => {
                write!(f, "memory-bound proof failed: {detail}")
            }
            FusionError::ProofInconclusive(r) => {
                write!(f, "memory-bound proof inconclusive: {r}")
            }
            FusionError::Serialization(e) => write!(f, "serialization error: {e}"),
            FusionError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for FusionError {}

impl FusionError {
    fn io(e: std::io::Error) -> Self {
        FusionError::Io(e.to_string())
    }
}

/// Op types in Catalyst IR that lower to a GEMM kernel.
fn is_gemm_op(op_type: &str) -> bool {
    matches!(op_type, "matmul" | "gemm" | "Gemm" | "MatMul")
}

fn shape_attr(
    attrs: &HashMap<String, serde_json::Value>,
    node: &str,
    key: &str,
) -> Result<usize, FusionError> {
    attrs
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| FusionError::MissingShape {
            node: node.to_string(),
            attr: key.to_string(),
        })
}

/// Lower every node of `ir` to FPGA kernel artifacts, proving each kernel
/// fits `device`'s on-chip budget **before** emission. Fails on the first
/// unsupported op, missing shape attribute, or non-fitting kernel.
pub fn build_manifest(
    ir: &TptIr,
    device: &DeviceConfig,
    tile: TileConfig,
    dtype: DType,
) -> Result<ToolchainManifest, FusionError> {
    let unsupported: Vec<String> = ir
        .graph
        .nodes
        .iter()
        .filter(|n| !is_gemm_op(&n.op_type))
        .map(|n| n.op_type.clone())
        .collect();
    if !unsupported.is_empty() {
        return Err(FusionError::UnsupportedOps(unsupported));
    }

    let mut kernels = Vec::new();
    for node in &ir.graph.nodes {
        let spec = GemmSpec {
            name: node.name.clone(),
            m: shape_attr(&node.attributes, &node.name, "m")?,
            n: shape_attr(&node.attributes, &node.name, "n")?,
            k: shape_attr(&node.attributes, &node.name, "k")?,
            dtype,
        };
        let fit = check_memory_fit(&spec, &tile, device);
        if !fit.fits {
            return Err(FusionError::DoesNotFit {
                kernel: spec.name.clone(),
                report: fit,
            });
        }
        kernels.push(KernelArtifact {
            name: spec.name.clone(),
            kind: "gemm".to_string(),
            gemm: spec,
            tile,
            fit,
            hls_file: format!("{}.cpp", node.name),
        });
    }

    let mut commands = Vec::new();
    match device.vendor {
        Vendor::Xilinx => {
            for k in &kernels {
                commands.push(format!(
                    "v++ -c -t hw -k {} --platform {} --clock {}MHz -I. {} -o {}.xo",
                    k.name, device.name, device.clock_mhz, k.hls_file, k.name
                ));
            }
            let xos: Vec<String> = kernels.iter().map(|k| format!("{}.xo", k.name)).collect();
            commands.push(format!(
                "v++ -l -t hw --platform {} --clock {}MHz {} -o fused.xclbin",
                device.name,
                device.clock_mhz,
                xos.join(" ")
            ));
        }
        other => return Err(FusionError::UnsupportedVendor(other)),
    }

    Ok(ToolchainManifest {
        model_name: ir.metadata.name.clone(),
        device: device.name.clone(),
        vendor: device.vendor,
        clock_mhz: device.clock_mhz,
        kernels,
        commands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_catalyst::ir::{ComputationalGraph, Edge, ModelMetadata, OpNode};

    fn device() -> DeviceConfig {
        DeviceConfig::new("test.platform", Vendor::Xilinx, 1 << 20, 16 << 20, 300.0)
    }

    fn gemm_ir() -> TptIr {
        let mut attrs = HashMap::new();
        attrs.insert("m".to_string(), serde_json::json!(512u64));
        attrs.insert("n".to_string(), serde_json::json!(512u64));
        attrs.insert("k".to_string(), serde_json::json!(512u64));
        TptIr {
            version: "1.0.0".into(),
            metadata: ModelMetadata {
                name: "toy-gemm".into(),
                source_format: "synthetic".into(),
                parameter_count: 0,
            },
            graph: ComputationalGraph {
                nodes: vec![OpNode {
                    id: 0,
                    op_type: "matmul".into(),
                    name: "qkv_proj".into(),
                    attributes: attrs,
                }],
                edges: vec![Edge {
                    from: 0,
                    to: 1,
                    tensor_name: "qkv".into(),
                }],
            },
        }
    }

    #[test]
    fn buffer_accounting_matches_formula() {
        let spec = GemmSpec {
            name: "g".into(),
            m: 512,
            n: 512,
            k: 512,
            dtype: DType::F32,
        };
        let tile = TileConfig {
            m: 32,
            n: 32,
            k: 32,
            double_buffer: false,
        };
        // (32·32 A + 32·32 B)·4B + 32·32 C·4B = 3·4096
        let r = check_memory_fit(&spec, &tile, &device());
        assert_eq!(r.total_bytes, 12_288);
        assert_eq!(
            r.breakdown,
            vec![
                ("a_buf".into(), 4096),
                ("b_buf".into(), 4096),
                ("c_buf".into(), 4096)
            ]
        );
        assert!(r.fits);
        // double buffering doubles only A and B
        let db = TileConfig { double_buffer: true, ..tile };
        let r = check_memory_fit(&spec, &db, &device());
        assert_eq!(r.total_bytes, 2 * 8_192 + 4_096);
        // f16 accumulates in f32: C tile unchanged, A/B halve
        let spec16 = GemmSpec { dtype: DType::F16, ..spec };
        let r16 = check_memory_fit(&spec16, &tile, &device());
        assert_eq!(r16.total_bytes, 2 * 2_048 + 4_096);
    }

    #[test]
    fn memory_fit_rejects_over_budget() {
        let tiny = DeviceConfig::new("tiny", Vendor::Xilinx, 1_000, 16 << 20, 200.0);
        let spec = GemmSpec {
            name: "big".into(),
            m: 512,
            n: 512,
            k: 512,
            dtype: DType::F32,
        };
        let tile = TileConfig {
            m: 64,
            n: 64,
            k: 64,
            double_buffer: true,
        };
        let r = check_memory_fit(&spec, &tile, &tiny);
        assert!(!r.fits);
        let err = build_manifest(&gemm_ir(), &tiny, tile, DType::F32).unwrap_err();
        match err {
            FusionError::DoesNotFit { kernel, report } => {
                assert_eq!(kernel, "qkv_proj");
                assert!(!report.fits);
                assert!(report.describe().contains("a_buf"));
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn hls_text_is_structured_and_deterministic() {
        let spec = GemmSpec {
            name: "qkv_proj".into(),
            m: 512,
            n: 512,
            k: 512,
            dtype: DType::F32,
        };
        let tile = TileConfig {
            m: 32,
            n: 32,
            k: 32,
            double_buffer: true,
        };
        let a = emit_hls_gemm(&spec, &tile);
        let b = emit_hls_gemm(&spec, &tile);
        assert_eq!(a, b, "emission must be deterministic");
        for needle in [
            "void qkv_proj(",
            "#pragma HLS INTERFACE m_axi",
            "#pragma HLS PIPELINE II=1",
            "#define M 512",
            "#define KI 32",
            "typedef float T;",
        ] {
            assert!(a.contains(needle), "missing `{needle}` in HLS output");
        }
    }

    #[test]
    fn manifest_from_catalyst_ir_and_unsupported_ops() {
        // a relu node is reported, never skipped
        let mut ir = gemm_ir();
        ir.graph.nodes.push(OpNode {
            id: 1,
            op_type: "relu".into(),
            name: "act".into(),
            attributes: HashMap::new(),
        });
        let err = build_manifest(&ir, &device(), TileConfig { m: 32, n: 32, k: 32, double_buffer: true }, DType::F32)
            .unwrap_err();
        assert_eq!(err, FusionError::UnsupportedOps(vec!["relu".into()]));

        // pure-GEMM graph lowers end to end
        let manifest = build_manifest(
            &gemm_ir(),
            &device(),
            TileConfig {
                m: 32,
                n: 32,
                k: 32,
                double_buffer: true,
            },
            DType::F32,
        )
        .unwrap();
        assert_eq!(manifest.kernels.len(), 1);
        assert_eq!(manifest.kernels[0].name, "qkv_proj");
        assert!(manifest.kernels[0].fit.fits);
        assert_eq!(manifest.commands.len(), 2, "one compile + one link");
        assert!(manifest.commands[0].starts_with("v++ -c"));
        assert!(manifest.commands[1].starts_with("v++ -l"));
        assert!(manifest.commands[1].contains("fused.xclbin"));

        // manifest JSON round-trips
        let json = manifest.to_json().unwrap();
        let back: ToolchainManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, "toy-gemm");
        assert_eq!(back.kernels.len(), 1);
    }

    #[test]
    fn missing_shape_attribute_names_the_node() {
        let mut ir = gemm_ir();
        ir.graph.nodes[0].attributes.remove("k");
        let err = build_manifest(
            &ir,
            &device(),
            TileConfig {
                m: 32,
                n: 32,
                k: 32,
                double_buffer: true,
            },
            DType::F32,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FusionError::MissingShape {
                node: "qkv_proj".into(),
                attr: "k".into()
            }
        );
    }

    #[test]
    fn write_out_creates_sources_and_manifest() {
        let manifest = build_manifest(
            &gemm_ir(),
            &device(),
            TileConfig {
                m: 32,
                n: 32,
                k: 32,
                double_buffer: true,
            },
            DType::F32,
        )
        .unwrap();
        let dir = std::env::temp_dir().join("tpt-fusion-test-out");
        manifest.write_out(&dir).unwrap();
        let cpp = std::fs::read_to_string(dir.join("qkv_proj.cpp")).unwrap();
        assert!(cpp.contains("void qkv_proj("));
        let json = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(json.contains("toy-gemm"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn intel_vendor_is_honest_about_missing_templates() {
        let dev = DeviceConfig::new("intel.platform", Vendor::Intel, 1 << 20, 16 << 20, 300.0);
        let err = build_manifest(
            &gemm_ir(),
            &dev,
            TileConfig {
                m: 32,
                n: 32,
                k: 32,
                double_buffer: true,
            },
            DType::F32,
        )
        .unwrap_err();
        assert_eq!(err, FusionError::UnsupportedVendor(Vendor::Intel));
    }
}
