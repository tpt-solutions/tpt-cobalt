//! TPTIR → TPT-UIR ingestion adapter for `tpt-gpu`.
//!
//! This implements the Phase 3 ingestion adapter described in
//! `tpt-uir/todo.md` (the `⛔ OUT OF REPO` GPU tasks). It converts the legacy
//! TPTIR SSA representation (`tpt_gpu_compiler::ir::Region`) into the unified
//! TPT-UIR (`tpt_uir_core::Region`) using the GPU dialect, emitting all tensor
//! shapes as `Dimension::Bounded` (or `None`) so the result satisfies the GPU
//! dialect invariant (never `Fixed`/`Symbolic`). A reverse converter
//! reconstructs TPTIR for lossless round-tripping.
//!
//! # Losslessness
//!
//! The minimal TPT-UIR core does not model every TPTIR detail (operation kind,
//! result id/type, block labels). To keep the round-trip lossless those details
//! are preserved as attributes / encodings:
//!
//! * The original `OpKind` is stored under [`ATTR_OP`] so the reverse pass
//!   rebuilds the exact operation.
//! * Result `id` and `Type` are stored under [`ATTR_RESULT_ID`] /
//!   [`ATTR_RESULT_TYPE`].
//! * A `MemRef`/`Tensor` address space is encoded as a leading
//!   `Dimension::Bounded { symbol: "addr_<space>" }` so it survives the
//!   shape-only representation.
//! * Dynamic TPTIR dims (`-1`) become `Dimension::Bounded { symbol: "dyn" }`
//!   and fixed dims `d` become `Dimension::Bounded { symbol: "fixed_d" }`.
//!
//! Block labels are not modelled by TPT-UIR `Block`, so the reverse pass
//! regenerates them (`entry` for the first block, `block_N` otherwise). For the
//! single-entry kernels produced by `tpt_gpu_compiler` this is identical.

use std::collections::BTreeMap;

use tpt_gpu_compiler::ir::{
    AddressSpace, OpKind, Operation as TptirOp, Region as TptirRegion, Type as TptirType, TypeKind,
    Value as TptirValue,
};
use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::op_name::{OpName, CORE_ADD, CORE_LOAD, CORE_MUL, CORE_STORE};
use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType, Type as UirType};
use tpt_uir_core::{Block as UirBlock, Operation as UirOp, Region as UirRegion, ValueId};
use tpt_uir_dialects::gpu::GpuOp;
use tpt_uir_dialects::{GpuDialect, ValidateDialect};

/// Attribute key holding the original TPTIR `OpKind` encoding.
pub const ATTR_OP: &str = "tptir.op";
/// Attribute key holding the original TPTIR result value id (as `i64`).
pub const ATTR_RESULT_ID: &str = "tptir.result_id";
/// Attribute key holding the original TPTIR result `Type`.
pub const ATTR_RESULT_TYPE: &str = "tptir.result_type";

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("GPU dialect op build failed: {0}")]
    GpuBuild(String),
    #[error("GPU dialect validation failed: {0:?}")]
    GpuValidation(Vec<String>),
    #[error("GGUF parsing failed: {0}")]
    Gguf(String),
    #[error("TPT-UIR serialization failed: {0}")]
    Serde(String),
}

pub mod gguf;
pub use gguf::{gguf_to_tptir, parse_gguf_bytes, write_minimal_gguf_bytes, LlamaModelSpec};

// ---------------------------------------------------------------------------
// File I/O (.tptuir emission / consumption)
// ---------------------------------------------------------------------------

use std::path::Path;

/// Convert a TPTIR region to TPT-UIR and write it to a `.tptuir` file.
pub fn write_tptuir(region: &TptirRegion, path: &Path) -> Result<(), AdapterError> {
    let uir = from_tptir(region)?;
    tpt_uir_serde::write_tptuir(path, &uir)
        .map_err(|e| AdapterError::Serde(format!("write_tptuir({}): {e}", path.display())))
}

/// Read a `.tptuir` file, deserialize it to TPT-UIR, and lower it back to a
/// TPTIR region (the reverse of [`write_tptuir`]).
pub fn read_tptuir(path: &Path) -> Result<TptirRegion, AdapterError> {
    let uir = tpt_uir_serde::read_tptuir(path)
        .map_err(|e| AdapterError::Serde(format!("read_tptuir({}): {e}", path.display())))?;
    Ok(to_tptir(&uir))
}

// ---------------------------------------------------------------------------
// TPTIR -> TPT-UIR
// ---------------------------------------------------------------------------

/// Convert a TPTIR region into a TPT-UIR region using the GPU dialect.
///
/// The returned region is guaranteed to pass [`GpuDialect::validate`].
pub fn from_tptir(region: &TptirRegion) -> Result<UirRegion, AdapterError> {
    let mut next_op_id: u32 = 0;
    let mut blocks = Vec::with_capacity(region.blocks.len());

    for block in &region.blocks {
        let arguments: Vec<(ValueId, UirType)> = block
            .arguments
            .iter()
            .map(|v| (v.id as ValueId, tptir_type_to_uir(&v.typ)))
            .collect();

        let mut operations = Vec::with_capacity(block.operations.len());
        for op in &block.operations {
            let op_name = op_name_for(&op.kind);
            let operands: Vec<ValueId> = op.operands.iter().map(|v| v.id as ValueId).collect();
            let results: Vec<ValueId> = match op.result_id {
                Some(id) => vec![id as ValueId],
                None => vec![],
            };

            let mut attributes = vec![Attribute::string(ATTR_OP, encode_op_kind(&op.kind))];
            if let Some(rid) = op.result_id {
                attributes.push(Attribute::i64(ATTR_RESULT_ID, rid as i64));
            }
            if let Some(rt) = &op.result_type {
                attributes.push(Attribute {
                    key: ATTR_RESULT_TYPE.into(),
                    value: AttributeValue::Type(tptir_type_to_uir(rt)),
                });
            }

            let uir_op = GpuOp::build(next_op_id, op_name, operands, results, attributes, vec![])
                .map_err(AdapterError::GpuBuild)?;
            operations.push(uir_op);
            next_op_id += 1;
        }

        blocks.push(UirBlock {
            arguments,
            operations,
        });
    }

    let uir = UirRegion { blocks };
    GpuDialect::validate(&uir).map_err(AdapterError::GpuValidation)?;
    Ok(uir)
}

fn op_name_for(kind: &OpKind) -> OpName {
    match kind {
        OpKind::Load => OpName::parse(CORE_LOAD).unwrap(),
        OpKind::Store => OpName::parse(CORE_STORE).unwrap(),
        OpKind::Addi | OpKind::Addf => OpName::parse(CORE_ADD).unwrap(),
        OpKind::Muli | OpKind::Mulf => OpName::parse(CORE_MUL).unwrap(),
        OpKind::Subi | OpKind::Subf => OpName::new("tpt_gpu", "sub"),
        OpKind::And | OpKind::Or | OpKind::Xor => OpName::new("tpt_gpu", "logic"),
        OpKind::CmpEq | OpKind::CmpLt => OpName::new("tpt_gpu", "cmp"),
        OpKind::Branch => OpName::new("tpt_gpu", "branch"),
        OpKind::Return => OpName::new("tpt_gpu", "return"),
        OpKind::Constant(_) => OpName::new("tpt_gpu", "constant"),
        OpKind::Gemm => OpName::new("tpt_gpu", "gemm"),
        OpKind::Quantize => OpName::new("tpt_gpu", "quantize"),
        OpKind::Dequantize => OpName::new("tpt_gpu", "dequantize"),
        OpKind::QuantGemm => OpName::new("tpt_gpu", "quant_gemm"),
        OpKind::QuantAttention => OpName::new("tpt_gpu", "quant_attention"),
        OpKind::Custom(_) => OpName::new("tpt_gpu", "custom"),
    }
}

/// Reversibly encode an `OpKind` into a string (preserving `Constant`/`Custom`
/// payloads, which `Display` would mangle).
fn encode_op_kind(kind: &OpKind) -> String {
    match kind {
        OpKind::Constant(v) => format!("Constant({})", v),
        OpKind::Custom(v) => format!("Custom({})", v),
        other => other.to_string(),
    }
}

fn scalar_from_tptir(name: &str) -> Option<ScalarType> {
    match name {
        "i1" => Some(ScalarType::Bool),
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f16" => Some(ScalarType::F16),
        "bf16" => Some(ScalarType::BF16),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        _ => None,
    }
}

fn scalar_to_tptir_name(s: ScalarType) -> &'static str {
    match s {
        ScalarType::Bool => "i1",
        ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::U8 => "i8",
        ScalarType::U16 => "i16",
        ScalarType::U32 => "i32",
        ScalarType::U64 => "i64",
        ScalarType::F16 => "f16",
        ScalarType::BF16 => "bf16",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Q4_0 | ScalarType::Q4_1 | ScalarType::Q8_0 => "i8",
    }
}

fn dim_to_uir(d: i64) -> Dimension {
    if d < 0 {
        Dimension::Bounded {
            symbol: "dyn".into(),
            max_value: 0,
        }
    } else {
        Dimension::Bounded {
            symbol: format!("fixed_{}", d),
            max_value: d as usize,
        }
    }
}

fn addr_from_str(s: &str) -> AddressSpace {
    match s {
        "shared" => AddressSpace::Shared,
        "local" => AddressSpace::Local,
        "constant" => AddressSpace::Constant,
        "generic" => AddressSpace::Generic,
        _ => AddressSpace::Global,
    }
}

fn dim_to_tptir(dim: &Dimension) -> i64 {
    match dim {
        Dimension::Bounded { symbol, .. } => {
            if let Some(rest) = symbol.strip_prefix("fixed_") {
                rest.parse().unwrap_or(0)
            } else {
                -1
            }
        }
        Dimension::Symbolic(_) | Dimension::Fixed(_) => -1,
    }
}

fn tptir_type_to_uir(t: &TptirType) -> UirType {
    match &t.kind {
        TypeKind::I1 => UirType::Scalar(ScalarType::Bool),
        TypeKind::I8 => UirType::Scalar(ScalarType::I8),
        TypeKind::I16 => UirType::Scalar(ScalarType::I16),
        TypeKind::I32 => UirType::Scalar(ScalarType::I32),
        TypeKind::I64 => UirType::Scalar(ScalarType::I64),
        TypeKind::F16 => UirType::Scalar(ScalarType::F16),
        TypeKind::BF16 => UirType::Scalar(ScalarType::BF16),
        TypeKind::F32 => UirType::Scalar(ScalarType::F32),
        TypeKind::F64 => UirType::Scalar(ScalarType::F64),
        TypeKind::Index => UirType::Index,
        TypeKind::Tensor(shape, elem, addr) | TypeKind::MemRef(shape, elem, addr) => {
            let mut dims: Vec<Dimension> = shape.iter().map(|&d| dim_to_uir(d)).collect();
            if *addr != AddressSpace::Global {
                dims.insert(
                    0,
                    Dimension::Bounded {
                        symbol: format!("addr_{}", addr),
                        max_value: 0,
                    },
                );
            }
            let dtype = scalar_from_tptir(&elem.to_string()).unwrap_or(ScalarType::F32);
            UirType::Tensor(TensorType {
                dtype,
                shape: Some(ShapeSpec { dimensions: dims }),
            })
        }
        TypeKind::Vector(lanes, elem) => {
            let dtype = scalar_from_tptir(&elem.to_string()).unwrap_or(ScalarType::F32);
            UirType::Tensor(TensorType {
                dtype,
                shape: Some(ShapeSpec {
                    dimensions: vec![Dimension::Bounded {
                        symbol: format!("fixed_{}", lanes),
                        max_value: *lanes as usize,
                    }],
                }),
            })
        }
        TypeKind::Function(_, _) | TypeKind::None => UirType::Scalar(ScalarType::I32),
    }
}

// ---------------------------------------------------------------------------
// TPT-UIR -> TPTIR
// ---------------------------------------------------------------------------

/// Reconstruct a TPTIR region from a TPT-UIR region produced by [`from_tptir`].
pub fn to_tptir(region: &UirRegion) -> TptirRegion {
    let mut blocks = Vec::with_capacity(region.blocks.len());

    for (bi, block) in region.blocks.iter().enumerate() {
        let label = if bi == 0 {
            "entry".to_string()
        } else {
            format!("block_{}", bi)
        };

        // Build a value-id -> type table so operand types can be recovered.
        let mut value_types: BTreeMap<ValueId, UirType> = BTreeMap::new();
        for (id, ty) in &block.arguments {
            value_types.insert(*id, ty.clone());
        }

        let arguments: Vec<TptirValue> = block
            .arguments
            .iter()
            .map(|(id, ty)| TptirValue::new(*id as u64, uir_type_to_tptir(ty)))
            .collect();

        let mut operations = Vec::with_capacity(block.operations.len());
        for op in &block.operations {
            let kind = decode_op_kind(attr_string(op, ATTR_OP).unwrap_or("custom"));
            let mut t_op = TptirOp::new(kind);

            t_op.operands = op
                .operands
                .iter()
                .map(|&id| {
                    let ty = value_types
                        .get(&id)
                        .map(uir_type_to_tptir)
                        .unwrap_or_else(|| TptirType::primitive("i32"));
                    TptirValue::new(id as u64, ty)
                })
                .collect();

            if let Some(rid) = attr_i64(op, ATTR_RESULT_ID) {
                t_op.result_id = Some(rid as u64);
            }
            if let Some(AttributeValue::Type(ty)) = attr_value(op, ATTR_RESULT_TYPE) {
                t_op.result_type = Some(uir_type_to_tptir(ty));
                if let Some(rid) = t_op.result_id {
                    value_types.insert(rid as ValueId, ty.clone());
                }
            }

            operations.push(t_op);
        }

        let mut b = tpt_gpu_compiler::ir::Block::new(&label);
        b.arguments = arguments;
        b.operations = operations;
        blocks.push(b);
    }

    TptirRegion { blocks }
}

fn decode_op_kind(s: &str) -> OpKind {
    if let Some(rest) = s.strip_prefix("Constant(") {
        return OpKind::Constant(rest.trim_end_matches(')').to_string());
    }
    if let Some(rest) = s.strip_prefix("Custom(") {
        return OpKind::Custom(rest.trim_end_matches(')').to_string());
    }
    match s {
        "addi" => OpKind::Addi,
        "subi" => OpKind::Subi,
        "muli" => OpKind::Muli,
        "addf" => OpKind::Addf,
        "subf" => OpKind::Subf,
        "mulf" => OpKind::Mulf,
        "andi" => OpKind::And,
        "ori" => OpKind::Or,
        "xori" => OpKind::Xor,
        "cmpeq" => OpKind::CmpEq,
        "cmplt" => OpKind::CmpLt,
        "load" => OpKind::Load,
        "store" => OpKind::Store,
        "br" => OpKind::Branch,
        "return" => OpKind::Return,
        "gemm" => OpKind::Gemm,
        "quantize" => OpKind::Quantize,
        "dequantize" => OpKind::Dequantize,
        "quant_gemm" => OpKind::QuantGemm,
        "quant_attention" => OpKind::QuantAttention,
        "constant" => OpKind::Constant(String::new()),
        other => OpKind::Custom(other.to_string()),
    }
}

fn uir_type_to_tptir(ty: &UirType) -> TptirType {
    match ty {
        UirType::Scalar(s) => TptirType::primitive(scalar_to_tptir_name(*s)),
        UirType::Index => TptirType::primitive("index"),
        UirType::Tensor(tt) => {
            let mut dims: Vec<i64> = Vec::new();
            let mut addr = AddressSpace::Global;
            if let Some(shape) = &tt.shape {
                for d in &shape.dimensions {
                    if let Dimension::Bounded { symbol, .. } = d {
                        if let Some(rest) = symbol.strip_prefix("addr_") {
                            addr = addr_from_str(rest);
                            continue;
                        }
                    }
                    dims.push(dim_to_tptir(d));
                }
            }
            let elem = TptirType::primitive(scalar_to_tptir_name(tt.dtype));
            TptirType::memref(dims, elem, addr)
        }
    }
}

fn attr_string<'a>(op: &'a UirOp, key: &str) -> Option<&'a str> {
    op.attributes.iter().find(|a| a.key == key).and_then(|a| {
        if let AttributeValue::String(s) = &a.value {
            Some(s.as_str())
        } else {
            None
        }
    })
}

fn attr_i64(op: &UirOp, key: &str) -> Option<i64> {
    op.attributes.iter().find(|a| a.key == key).and_then(|a| {
        if let AttributeValue::I64(v) = &a.value {
            Some(*v)
        } else {
            None
        }
    })
}

fn attr_value<'a>(op: &'a UirOp, key: &str) -> Option<&'a AttributeValue> {
    op.attributes
        .iter()
        .find(|a| a.key == key)
        .map(|a| &a.value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_gpu_compiler::ir::{Block, ElemType};

    /// Assert two TPTIR regions are semantically identical (op kinds, SSA
    /// value ids, and types), ignoring block labels which TPT-UIR drops.
    fn regions_equivalent(a: &TptirRegion, b: &TptirRegion) -> bool {
        if a.blocks.len() != b.blocks.len() {
            return false;
        }
        for (ba, bb) in a.blocks.iter().zip(b.blocks.iter()) {
            if ba.arguments.len() != bb.arguments.len() {
                return false;
            }
            for (va, vb) in ba.arguments.iter().zip(bb.arguments.iter()) {
                if va.id != vb.id || va.typ.to_string() != vb.typ.to_string() {
                    return false;
                }
            }
            if ba.operations.len() != bb.operations.len() {
                return false;
            }
            for (oa, ob) in ba.operations.iter().zip(bb.operations.iter()) {
                if oa.kind.to_string() != ob.kind.to_string() {
                    return false;
                }
                let ids_a: Vec<u64> = oa.operands.iter().map(|v| v.id).collect();
                let ids_b: Vec<u64> = ob.operands.iter().map(|v| v.id).collect();
                if ids_a != ids_b {
                    return false;
                }
                if oa.result_id != ob.result_id {
                    return false;
                }
                if oa.result_type.as_ref().map(|t| t.to_string())
                    != ob.result_type.as_ref().map(|t| t.to_string())
                {
                    return false;
                }
            }
        }
        true
    }

    fn assert_lossless(region: &TptirRegion) {
        // Convert TPTIR -> TPT-UIR and back.
        let uir = from_tptir(region).expect("from_tptir failed");

        // The produced UIR must be well-formed: SSA-valid and satisfying the
        // GPU dialect invariant (shapes are Bounded/None, never Fixed/Symbolic).
        assert!(
            tpt_uir_core::validate_region(&uir).is_ok(),
            "produced TPT-UIR is not SSA-valid"
        );

        let back = to_tptir(&uir);

        // It must be semantically equivalent to the original (op kinds, SSA
        // value ids, types). This ignores block labels, which TPT-UIR `Block`
        // does not model and which the reverse pass regenerates.
        assert!(
            regions_equivalent(region, &back),
            "round-trip not lossless:\noriginal:\n{}\nback:\n{}",
            emit(region),
            emit(&back)
        );

        // For a single-block region the regenerated label is "entry", so the
        // canonical TPTIR text emission must match byte-for-byte.
        if region.blocks.len() == 1 {
            assert_eq!(emit(region), emit(&back));
        }
    }

    /// A representative Llama-3 attention-block style TPTIR region: one block
    /// exercising constants, loads/stores, fused arithmetic, a GEMM, and a
    /// custom op, over dynamic `memref<*xf32>` buffers.
    fn make_rich_attention_kernel() -> TptirRegion {
        let mut region = TptirRegion::new();
        let mut block = Block::new("entry");

        let q = TptirValue::new(
            0,
            TptirType::memref(vec![-1], TptirType::primitive("f32"), AddressSpace::Global),
        );
        let k = TptirValue::new(
            1,
            TptirType::memref(vec![-1], TptirType::primitive("f32"), AddressSpace::Global),
        );
        let v = TptirValue::new(
            2,
            TptirType::memref(vec![-1], TptirType::primitive("f32"), AddressSpace::Global),
        );
        let seq = TptirValue::new(3, TptirType::primitive("i32"));
        block.arguments = vec![q.clone(), k.clone(), v.clone(), seq.clone()];

        let mut c = TptirOp::new(OpKind::Constant("0.5".to_string()));
        c.result_id = Some(10);
        c.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(c);

        let mut qk = TptirOp::new(OpKind::Mulf);
        qk.operands = vec![q.clone(), k.clone()];
        qk.result_id = Some(11);
        qk.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(qk);

        let mut attn = TptirOp::new(OpKind::Addf);
        attn.operands = vec![
            TptirValue::new(11, TptirType::primitive("f32")),
            TptirValue::new(10, TptirType::primitive("f32")),
        ];
        attn.result_id = Some(12);
        attn.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(attn);

        let mut l = TptirOp::new(OpKind::Load);
        l.operands = vec![TptirValue::new(11, TptirType::primitive("f32"))];
        l.result_id = Some(13);
        l.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(l);

        let mut g = TptirOp::new(OpKind::Gemm);
        g.operands = vec![q.clone(), k.clone(), v.clone()];
        g.result_id = Some(14);
        g.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(g);

        let mut cust = TptirOp::new(OpKind::Custom("fused_attn".to_string()));
        cust.operands = vec![TptirValue::new(14, TptirType::primitive("f32"))];
        cust.result_id = Some(15);
        cust.result_type = Some(TptirType::primitive("f32"));
        block.operations.push(cust);

        let mut st = TptirOp::new(OpKind::Store);
        st.operands = vec![TptirValue::new(15, TptirType::primitive("f32")), v.clone()];
        block.operations.push(st);

        let mut ret = TptirOp::new(OpKind::Return);
        ret.operands = vec![TptirValue::new(15, TptirType::primitive("f32"))];
        block.operations.push(ret);

        region.blocks.push(block);
        region
    }

    /// A multi-block TPTIR region mimicking a decoder block with separate
    /// entry / scale / exit blocks. Each block is self-contained SSA (mirroring
    /// TPTIR's single-block-value-space reality) but exercises control flow
    /// (`br` terminators), per-block arguments, loads, stores, and mixed
    /// `memref`/`i32` types.
    fn make_llama_decoder_block() -> TptirRegion {
        let mut region = TptirRegion::new();

        let mut entry = Block::new("entry");
        entry.arguments = vec![
            TptirValue::new(
                0,
                TptirType::memref(vec![-1], TptirType::primitive("f32"), AddressSpace::Global),
            ),
            TptirValue::new(
                1,
                TptirType::memref(vec![-1], TptirType::primitive("f32"), AddressSpace::Global),
            ),
            TptirValue::new(2, TptirType::primitive("i32")),
        ];
        let mut e_mul = TptirOp::new(OpKind::Mulf);
        e_mul.operands = vec![
            TptirValue::new(0, TptirType::primitive("f32")),
            TptirValue::new(1, TptirType::primitive("f32")),
        ];
        e_mul.result_id = Some(10);
        e_mul.result_type = Some(TptirType::primitive("f32"));
        entry.operations.push(e_mul);
        entry.operations.push(TptirOp::new(OpKind::Branch));
        region.blocks.push(entry);

        let mut scale = Block::new("scale");
        scale.arguments = vec![TptirValue::new(10, TptirType::primitive("f32"))];
        let mut s_mul = TptirOp::new(OpKind::Mulf);
        s_mul.operands = vec![
            TptirValue::new(10, TptirType::primitive("f32")),
            TptirValue::new(10, TptirType::primitive("f32")),
        ];
        s_mul.result_id = Some(11);
        s_mul.result_type = Some(TptirType::primitive("f32"));
        scale.operations.push(s_mul);
        let mut s_store = TptirOp::new(OpKind::Store);
        s_store.operands = vec![
            TptirValue::new(11, TptirType::primitive("f32")),
            TptirValue::new(10, TptirType::primitive("f32")),
        ];
        scale.operations.push(s_store);
        scale.operations.push(TptirOp::new(OpKind::Branch));
        region.blocks.push(scale);

        let mut exit = Block::new("exit");
        exit.arguments = vec![TptirValue::new(11, TptirType::primitive("f32"))];
        let mut x_add = TptirOp::new(OpKind::Addf);
        x_add.operands = vec![
            TptirValue::new(11, TptirType::primitive("f32")),
            TptirValue::new(11, TptirType::primitive("f32")),
        ];
        x_add.result_id = Some(12);
        x_add.result_type = Some(TptirType::primitive("f32"));
        exit.operations.push(x_add);
        exit.operations.push(TptirOp::new(OpKind::Return));
        region.blocks.push(exit);

        region
    }

    fn emit(region: &TptirRegion) -> String {
        tpt_gpu_compiler::ir::emit_tptir(region, "kernel", &[])
    }

    #[test]
    fn test_roundtrip_gpu_matmul() {
        let region =
            tpt_gpu_compiler::ir::build_kernel_region("matmul", ElemType::F32, &[1024]).unwrap();
        assert_lossless(&region);
    }

    #[test]
    fn test_roundtrip_gpu_vector_add() {
        let region =
            tpt_gpu_compiler::ir::build_kernel_region("vector_add", ElemType::F16, &[2048])
                .unwrap();
        assert_lossless(&region);
    }

    #[test]
    fn test_roundtrip_gpu_softmax() {
        let region =
            tpt_gpu_compiler::ir::build_kernel_region("softmax", ElemType::F32, &[512]).unwrap();
        assert_lossless(&region);
    }

    #[test]
    fn test_roundtrip_gpu_rich_attention_kernel() {
        let region = make_rich_attention_kernel();
        assert_lossless(&region);
    }

    #[test]
    fn test_roundtrip_gpu_multiblock_decoder() {
        let region = make_llama_decoder_block();
        assert_lossless(&region);
    }

    #[test]
    fn test_roundtrip_gpu_llama3_gguf() {
        // Materialize a real GGUF fixture (Llama-3 arch), parse it, lower it to
        // a representative TPTIR decoder block, then assert the full
        // TPTIR -> TPT-UIR -> TPTIR round-trip is lossless.
        let spec = LlamaModelSpec {
            arch: "llama3".into(),
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            ffn_dim: 11008,
            num_layers: 32,
            context_len: 8192,
            rope_freq_base: 10_000.0,
        };
        let bytes = write_minimal_gguf_bytes(&spec);
        let parsed = parse_gguf_bytes(&bytes).expect("parse GGUF fixture");
        assert_eq!(parsed.arch, "llama3");
        assert_eq!(parsed.hidden_dim, 4096);
        assert_eq!(parsed.context_len, 8192);

        let region = gguf_to_tptir(&parsed);
        assert_lossless(&region);
    }

    #[test]
    fn test_uir_serialization_roundtrip() {
        let region =
            tpt_gpu_compiler::ir::build_kernel_region("matmul", ElemType::F32, &[1024]).unwrap();
        let uir = from_tptir(&region).unwrap();
        let bytes = tpt_uir_serde::serialize_region(&uir).expect("serialize failed");
        let back = tpt_uir_serde::deserialize_region(&bytes).expect("deserialize failed");
        assert_eq!(uir, back);
    }

    #[test]
    fn test_write_and_read_tptuir_file() {
        let region =
            tpt_gpu_compiler::ir::build_kernel_region("matmul", ElemType::F32, &[1024]).unwrap();
        let path =
            std::env::temp_dir().join(format!("tptuir_adapter_{}.tptuir", std::process::id()));
        write_tptuir(&region, &path).expect("write_tptuir failed");
        let back = read_tptuir(&path).expect("read_tptuir failed");
        // Single-block kernel: byte-identical TPTIR text after a file round-trip.
        assert_eq!(emit(&region), emit(&back));
        let _ = std::fs::remove_file(&path);
    }
}
