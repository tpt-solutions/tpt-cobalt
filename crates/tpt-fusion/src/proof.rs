//! Memory-bound proofs in the compiler pipeline (Phase 5: *"Wire
//! `tpt-telos-uir-bridge` into the compiler pipeline for memory-bound
//! proofs"*).
//!
//! The artifact-level tile fit ([`crate::check_memory_fit`]) only covers the
//! on-chip buffers of a single kernel. This module proves the **global**
//! claim — that the model's full operand tensors (weights + activations)
//! fit the target device's global memory — by lowering the allocations to a
//! TPT-UIR region (`tpt_memory.alloc` ops) and delegating to
//! `tpt_telos_uir_bridge::prove_memory_bounds`, which decides the linear
//! arithmetic with its built-in Fourier–Motzkin engine. Symbolic dimensions
//! (e.g. a batch dim) are quantified over their declared bounds: the proof
//! must hold for *every* admissible assignment, and a failure returns a
//! concrete counterexample witness.
//!
//! Real models enter through [`allocs_from_module`], which walks any
//! `tpt_ml::Module`'s parameters. The proof runs at artifact precision (the
//! F32/F16/I8 the FPGA kernel stores), not the f64 the tensors live in on
//! the host — documented and deliberate.

use tpt_ml::Module;
use tpt_telos_uir_bridge::{prove_memory_bounds, MemoryLimits, ProofResult};
use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::ir::Region;
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType, Type};
use tpt_uir_core::{Block, Operation};

use crate::{DeviceConfig, DType, FusionError};

/// One dimension of an allocation: fixed, or symbolic with an upper bound
/// the proof quantifies over (`0 ≤ b ≤ max`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Fixed(usize),
    Bounded { symbol: String, max: usize },
}

/// A named tensor allocation in the proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocTensor {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<Dim>,
}

impl AllocTensor {
    /// A fully fixed-shape allocation.
    pub fn fixed(name: impl Into<String>, dtype: DType, dims: &[usize]) -> Self {
        AllocTensor {
            name: name.into(),
            dtype,
            dims: dims.iter().map(|&d| Dim::Fixed(d)).collect(),
        }
    }

    /// Total bytes when every dimension is fixed (`None` if symbolic).
    pub fn bytes_static(&self) -> Option<usize> {
        let mut bytes = self.dtype.bytes();
        for d in &self.dims {
            match d {
                Dim::Fixed(n) => bytes *= n,
                Dim::Bounded { .. } => return None,
            }
        }
        Some(bytes)
    }
}

fn scalar_type(dtype: DType) -> ScalarType {
    match dtype {
        DType::F32 => ScalarType::F32,
        DType::F16 => ScalarType::F16,
        DType::I8 => ScalarType::I8,
    }
}

/// Lower the allocations to a TPT-UIR region: one `tpt_memory.alloc` per
/// tensor, all in the `"model"` memory scope.
pub fn alloc_region(allocs: &[AllocTensor]) -> Region {
    let ops: Vec<Operation> = allocs
        .iter()
        .enumerate()
        .map(|(i, a)| Operation {
            id: (i + 1) as u32,
            op_name: OpName::new("tpt_memory", "alloc"),
            operands: vec![],
            results: vec![],
            regions: vec![],
            attributes: vec![
                Attribute::string("scope", "model"),
                Attribute {
                    key: "tensor".into(),
                    value: AttributeValue::Type(Type::Tensor(TensorType {
                        dtype: scalar_type(a.dtype),
                        shape: Some(ShapeSpec {
                            dimensions: a
                                .dims
                                .iter()
                                .map(|d| match d {
                                    Dim::Fixed(n) => Dimension::Fixed(*n),
                                    Dim::Bounded { symbol, max } => Dimension::Bounded {
                                        symbol: symbol.clone(),
                                        max_value: *max,
                                    },
                                })
                                .collect(),
                        }),
                    })),
                },
            ],
        })
        .collect();
    Region {
        blocks: vec![Block {
            arguments: vec![],
            operations: ops,
        }],
    }
}

/// The proof outcome over a device's global-memory budget.
#[derive(Debug, Clone)]
pub struct MemoryProof {
    pub result: ProofResult,
    pub budget_bytes: usize,
    /// Total bytes when every dimension is fixed (`None` if any symbolic).
    pub static_bytes: Option<usize>,
}

/// Prove that `allocs` fit `device`'s global memory for **all** admissible
/// symbolic-dimension assignments.
pub fn prove_model_memory(
    device: &DeviceConfig,
    allocs: &[AllocTensor],
) -> Result<MemoryProof, FusionError> {
    let region = alloc_region(allocs);
    let result = prove_memory_bounds(
        &region,
        &MemoryLimits::with_default(device.global_mem_bytes as i64),
    );
    let static_bytes = allocs
        .iter()
        .map(|a| a.bytes_static())
        .collect::<Option<Vec<_>>>()
        .map(|v| v.iter().sum());
    Ok(MemoryProof {
        result,
        budget_bytes: device.global_mem_bytes,
        static_bytes,
    })
}

/// The compiler-pipeline entry point: tile-fit + emit (like
/// [`crate::build_manifest`]) **and** the global-memory proof over every
/// kernel's full operand tensors plus `extra_allocs` (activations etc.).
/// Emission only succeeds on a `Valid` proof.
pub fn build_manifest_proved(
    ir: &tpt_catalyst::ir::TptIr,
    device: &DeviceConfig,
    tile: crate::TileConfig,
    dtype: DType,
    extra_allocs: &[AllocTensor],
) -> Result<(crate::ToolchainManifest, MemoryProof), FusionError> {
    let manifest = crate::build_manifest(ir, device, tile, dtype)?;
    let mut allocs: Vec<AllocTensor> = Vec::new();
    for k in &manifest.kernels {
        let (m, n, k_dim) = (k.gemm.m, k.gemm.n, k.gemm.k);
        allocs.push(AllocTensor::fixed(format!("{}_A", k.name), dtype, &[m, k_dim]));
        allocs.push(AllocTensor::fixed(format!("{}_B", k.name), dtype, &[k_dim, n]));
        allocs.push(AllocTensor::fixed(format!("{}_C", k.name), dtype, &[m, n]));
    }
    allocs.extend_from_slice(extra_allocs);
    let proof = prove_model_memory(device, &allocs)?;
    match &proof.result {
        ProofResult::Valid => Ok((manifest, proof)),
        ProofResult::Counterexample {
            scope,
            total_bytes,
            limit_bytes,
            ..
        } => Err(FusionError::ProofFailed {
            detail: format!(
                "scope '{scope}' can reach {total_bytes} bytes against a {}-byte device budget",
                limit_bytes
            ),
        }),
        ProofResult::Inconclusive { reason } => {
            Err(FusionError::ProofInconclusive(reason.clone()))
        }
    }
}

/// Walk a real model's parameters into fixed-shape allocations (proof at
/// artifact precision: each host f64 parameter tensor maps to its F32
/// artifact footprint).
pub fn allocs_from_module(model: &dyn Module) -> Vec<AllocTensor> {
    model
        .parameters()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            AllocTensor::fixed(format!("param_{i}"), DType::F32, t.shape())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TileConfig, Vendor};
    use tpt_catalyst::ir::{ComputationalGraph, Edge, ModelMetadata, OpNode};
    use std::collections::HashMap;

    fn device(global: usize) -> DeviceConfig {
        DeviceConfig::new("test.platform", Vendor::Xilinx, 1 << 20, global, 300.0)
    }

    fn gemm_ir() -> tpt_catalyst::ir::TptIr {
        let mut attrs = HashMap::new();
        attrs.insert("m".to_string(), serde_json::json!(512u64));
        attrs.insert("n".to_string(), serde_json::json!(512u64));
        attrs.insert("k".to_string(), serde_json::json!(512u64));
        tpt_catalyst::ir::TptIr {
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

    fn tile() -> TileConfig {
        TileConfig {
            m: 32,
            n: 32,
            k: 32,
            double_buffer: true,
        }
    }

    #[test]
    fn fixed_allocs_prove_valid_and_counterexample() {
        // 4 MiB of weights: fits 16 MiB, not 1 MiB
        let weights = AllocTensor::fixed("weights", DType::F32, &[1024, 1024]);
        let proof = prove_model_memory(&device(16 << 20), &[weights.clone()]).unwrap();
        assert!(matches!(proof.result, ProofResult::Valid));
        assert_eq!(proof.static_bytes, Some(4 << 20));

        let proof = prove_model_memory(&device(1 << 20), &[weights]).unwrap();
        match proof.result {
            ProofResult::Counterexample {
                total_bytes,
                limit_bytes,
                ..
            } => {
                assert_eq!(total_bytes, 4 << 20);
                assert_eq!(limit_bytes, 1 << 20);
            }
            other => panic!("expected counterexample, got {other:?}"),
        }
    }

    #[test]
    fn symbolic_batch_is_quantified_over_its_bound() {
        // weights 1 MiB + activation [batch, 512, 512] f32 with batch ≤ 8:
        // worst case 1 + 8 = 9 MiB < 16 MiB for EVERY admissible batch.
        let allocs = vec![
            AllocTensor::fixed("weights", DType::F32, &[256, 1024]),
            AllocTensor {
                name: "activations".into(),
                dtype: DType::F32,
                dims: vec![
                    Dim::Bounded {
                        symbol: "batch".into(),
                        max: 8,
                    },
                    Dim::Fixed(512),
                    Dim::Fixed(512),
                ],
            },
        ];
        let proof = prove_model_memory(&device(16 << 20), &allocs).unwrap();
        assert!(matches!(proof.result, ProofResult::Valid), "{:?}", proof.result);
        assert_eq!(proof.static_bytes, None);

        // batch ≤ 32 → 33 MiB worst case: must overflow with a witness
        let allocs = vec![
            AllocTensor::fixed("weights", DType::F32, &[256, 1024]),
            AllocTensor {
                name: "activations".into(),
                dtype: DType::F32,
                dims: vec![
                    Dim::Bounded {
                        symbol: "batch".into(),
                        max: 32,
                    },
                    Dim::Fixed(512),
                    Dim::Fixed(512),
                ],
            },
        ];
        let proof = prove_model_memory(&device(16 << 20), &allocs).unwrap();
        match proof.result {
            ProofResult::Counterexample {
                model,
                total_bytes,
                limit_bytes,
                ..
            } => {
                // any witness must genuinely overflow: 1 MiB weights +
                // batch·1 MiB activations > 16 MiB (minimal witness: batch 16)
                let batch = model["batch"];
                assert!(batch >= 16, "witness batch {batch} does not overflow");
                assert_eq!(
                    (1 << 20) + batch * (512 * 512 * 4i64),
                    total_bytes,
                    "witness total must match the allocation arithmetic"
                );
                assert_eq!(limit_bytes, 16 << 20);
            }
            other => panic!("expected counterexample, got {other:?}"),
        }
    }

    #[test]
    fn manifest_emission_is_gated_on_the_global_proof() {
        // one 512³ f32 GEMM = 3 MiB of operands + 1 MiB activation spare
        let extra = AllocTensor::fixed("kv_cache", DType::F32, &[128, 2048]);
        let (manifest, proof) =
            build_manifest_proved(&gemm_ir(), &device(16 << 20), tile(), DType::F32, &[extra])
                .unwrap();
        assert!(matches!(proof.result, ProofResult::Valid));
        // A/B/C at 512·512·4 B each + kv_cache 128·2048·4 B
        assert_eq!(
            proof.static_bytes,
            Some(3 * (512 * 512 * 4) + 128 * 2048 * 4)
        );
        assert_eq!(manifest.kernels.len(), 1);

        // 1 MiB device: tile fit passes (20 KiB on-chip) but the global
        // proof must still refuse emission
        let err = build_manifest_proved(
            &gemm_ir(),
            &device(1 << 20),
            tile(),
            DType::F32,
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, FusionError::ProofFailed { .. }), "{err:?}");
    }

    #[test]
    fn real_model_allocations_prove_fit_or_exceed() {
        // a real (tiny) transformer-ish stack from tpt-ml
        let mut model = tpt_ml::Sequential::new();
        model.push(tpt_ml::Linear::new(512, 2048, true));
        model.push(tpt_ml::Linear::new(2048, 512, true));
        model.push(tpt_ml::Linear::new(512, 512, true));
        let allocs = allocs_from_module(&model);
        assert_eq!(allocs.len(), 6, "2 params per Linear");
        // (512·2048 + 2048 + 2048·512 + 512 + 512·512 + 512) params × 4 B
        let expected = (512 * 2048 + 2048 + 2048 * 512 + 512 + 512 * 512 + 512) * 4;
        assert_eq!(allocs.iter().map(|a| a.bytes_static().unwrap()).sum::<usize>(), expected);

        let proof = prove_model_memory(&device(16 << 20), &allocs).unwrap();
        assert!(matches!(proof.result, ProofResult::Valid));
        let proof = prove_model_memory(&device(1 << 20), &allocs).unwrap();
        assert!(matches!(proof.result, ProofResult::Counterexample { .. }));
    }
}
