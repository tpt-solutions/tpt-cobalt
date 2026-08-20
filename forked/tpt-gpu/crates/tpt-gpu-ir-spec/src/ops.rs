use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Broad category for grouping ops in the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OpCategory {
    Arithmetic,
    Bitwise,
    Memory,
    Control,
    Conversion,
    Matrix,
    Reduction,
    Custom,
}

/// Canonical opcode identifiers for the TPTIR dialect.
///
/// These string names are the stable text-format identifiers and must not be
/// renamed without a major-version bump.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Op {
    // Integer arithmetic
    Addi,
    Subi,
    Muli,
    Divi,
    Modi,
    // Float arithmetic
    Addf,
    Subf,
    Mulf,
    Divf,
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // Comparison
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    // Memory
    Load,
    Store,
    Alloc,
    Dealloc,
    Memcpy,
    // Control
    Branch,
    CondBranch,
    Return,
    Call,
    // Constant
    Constant,
    // Conversion
    Trunc,
    Extend,
    Bitcast,
    IntToFloat,
    FloatToInt,
    // Matrix / tensor
    Gemm,
    Matmul,
    Transpose,
    // Reduction
    ReduceSum,
    ReduceMax,
    ReduceMin,
    // Quantization (mixed-precision model optimization)
    Quantize,
    Dequantize,
    QuantGemm,
    QuantAttention,
    // Escape hatch for custom / future ops
    Custom(String),
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Op::Addi => "addi",
            Op::Subi => "subi",
            Op::Muli => "muli",
            Op::Divi => "divi",
            Op::Modi => "modi",
            Op::Addf => "addf",
            Op::Subf => "subf",
            Op::Mulf => "mulf",
            Op::Divf => "divf",
            Op::And => "andi",
            Op::Or => "ori",
            Op::Xor => "xori",
            Op::Shl => "shl",
            Op::Shr => "shr",
            Op::CmpEq => "cmpeq",
            Op::CmpNe => "cmpne",
            Op::CmpLt => "cmplt",
            Op::CmpLe => "cmple",
            Op::CmpGt => "cmpgt",
            Op::CmpGe => "cmpge",
            Op::Load => "load",
            Op::Store => "store",
            Op::Alloc => "alloc",
            Op::Dealloc => "dealloc",
            Op::Memcpy => "memcpy",
            Op::Branch => "br",
            Op::CondBranch => "cond_br",
            Op::Return => "return",
            Op::Call => "call",
            Op::Constant => "constant",
            Op::Trunc => "trunc",
            Op::Extend => "extend",
            Op::Bitcast => "bitcast",
            Op::IntToFloat => "itof",
            Op::FloatToInt => "ftoi",
            Op::Gemm => "gemm",
            Op::Matmul => "matmul",
            Op::Transpose => "transpose",
            Op::ReduceSum => "reduce_sum",
            Op::ReduceMax => "reduce_max",
            Op::ReduceMin => "reduce_min",
            Op::Quantize => "quantize",
            Op::Dequantize => "dequantize",
            Op::QuantGemm => "quant_gemm",
            Op::QuantAttention => "quant_attention",
            Op::Custom(name) => return write!(f, "{}", name),
        };
        write!(f, "{}", s)
    }
}

impl Op {
    pub fn category(&self) -> OpCategory {
        match self {
            Op::Addi
            | Op::Subi
            | Op::Muli
            | Op::Divi
            | Op::Modi
            | Op::Addf
            | Op::Subf
            | Op::Mulf
            | Op::Divf
            | Op::CmpEq
            | Op::CmpNe
            | Op::CmpLt
            | Op::CmpLe
            | Op::CmpGt
            | Op::CmpGe => OpCategory::Arithmetic,
            Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr => OpCategory::Bitwise,
            Op::Load | Op::Store | Op::Alloc | Op::Dealloc | Op::Memcpy => OpCategory::Memory,
            Op::Branch | Op::CondBranch | Op::Return | Op::Call => OpCategory::Control,
            Op::Constant => OpCategory::Arithmetic,
            Op::Trunc | Op::Extend | Op::Bitcast | Op::IntToFloat | Op::FloatToInt => {
                OpCategory::Conversion
            }
            Op::Gemm | Op::Matmul | Op::Transpose => OpCategory::Matrix,
            Op::ReduceSum | Op::ReduceMax | Op::ReduceMin => OpCategory::Reduction,
            Op::Quantize | Op::Dequantize | Op::QuantGemm | Op::QuantAttention => {
                OpCategory::Matrix
            }
            Op::Custom(_) => OpCategory::Custom,
        }
    }

    /// Parse an op from its stable text-format name.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "addi" => Op::Addi,
            "subi" => Op::Subi,
            "muli" => Op::Muli,
            "divi" => Op::Divi,
            "modi" => Op::Modi,
            "addf" => Op::Addf,
            "subf" => Op::Subf,
            "mulf" => Op::Mulf,
            "divf" => Op::Divf,
            "andi" => Op::And,
            "ori" => Op::Or,
            "xori" => Op::Xor,
            "shl" => Op::Shl,
            "shr" => Op::Shr,
            "cmpeq" => Op::CmpEq,
            "cmpne" => Op::CmpNe,
            "cmplt" => Op::CmpLt,
            "cmple" => Op::CmpLe,
            "cmpgt" => Op::CmpGt,
            "cmpge" => Op::CmpGe,
            "load" => Op::Load,
            "store" => Op::Store,
            "alloc" => Op::Alloc,
            "dealloc" => Op::Dealloc,
            "memcpy" => Op::Memcpy,
            "br" => Op::Branch,
            "cond_br" => Op::CondBranch,
            "return" => Op::Return,
            "call" => Op::Call,
            "constant" => Op::Constant,
            "trunc" => Op::Trunc,
            "extend" => Op::Extend,
            "bitcast" => Op::Bitcast,
            "itof" => Op::IntToFloat,
            "ftoi" => Op::FloatToInt,
            "gemm" => Op::Gemm,
            "matmul" => Op::Matmul,
            "transpose" => Op::Transpose,
            "reduce_sum" => Op::ReduceSum,
            "reduce_max" => Op::ReduceMax,
            "reduce_min" => Op::ReduceMin,
            "quantize" => Op::Quantize,
            "dequantize" => Op::Dequantize,
            "quant_gemm" => Op::QuantGemm,
            "quant_attention" => Op::QuantAttention,
            other => Op::Custom(other.to_string()),
        }
    }
}

/// Structured metadata for a single op definition — used for validation and
/// documentation generation.
#[derive(Debug, Clone)]
pub struct OpDef {
    pub op: Op,
    pub mnemonic: &'static str,
    pub num_operands: usize,
    pub has_result: bool,
    pub category: OpCategory,
    pub doc: &'static str,
}

/// Returns the full op table for the TPTIR core dialect.
pub fn core_op_table() -> Vec<OpDef> {
    vec![
        OpDef {
            op: Op::Addi,
            mnemonic: "addi",
            num_operands: 2,
            has_result: true,
            category: OpCategory::Arithmetic,
            doc: "Integer addition",
        },
        OpDef {
            op: Op::Addf,
            mnemonic: "addf",
            num_operands: 2,
            has_result: true,
            category: OpCategory::Arithmetic,
            doc: "Floating-point addition",
        },
        OpDef {
            op: Op::Muli,
            mnemonic: "muli",
            num_operands: 2,
            has_result: true,
            category: OpCategory::Arithmetic,
            doc: "Integer multiplication",
        },
        OpDef {
            op: Op::Mulf,
            mnemonic: "mulf",
            num_operands: 2,
            has_result: true,
            category: OpCategory::Arithmetic,
            doc: "Floating-point multiplication",
        },
        OpDef {
            op: Op::Load,
            mnemonic: "load",
            num_operands: 1,
            has_result: true,
            category: OpCategory::Memory,
            doc: "Load value from memref",
        },
        OpDef {
            op: Op::Store,
            mnemonic: "store",
            num_operands: 2,
            has_result: false,
            category: OpCategory::Memory,
            doc: "Store value into memref",
        },
        OpDef {
            op: Op::Gemm,
            mnemonic: "gemm",
            num_operands: 3,
            has_result: true,
            category: OpCategory::Matrix,
            doc: "General matrix multiply: C = alpha*A*B + beta*C",
        },
        OpDef {
            op: Op::Return,
            mnemonic: "return",
            num_operands: 0,
            has_result: false,
            category: OpCategory::Control,
            doc: "Return from function or kernel",
        },
        OpDef {
            op: Op::Quantize,
            mnemonic: "quantize",
            num_operands: 1,
            has_result: true,
            category: OpCategory::Matrix,
            doc: "Quantize f32 tensor to integer with scale and zero-point",
        },
        OpDef {
            op: Op::Dequantize,
            mnemonic: "dequantize",
            num_operands: 1,
            has_result: true,
            category: OpCategory::Matrix,
            doc: "Dequantize integer tensor to f32 with scale and zero-point",
        },
        OpDef {
            op: Op::QuantGemm,
            mnemonic: "quant_gemm",
            num_operands: 3,
            has_result: true,
            category: OpCategory::Matrix,
            doc:
                "Integer matrix multiply with inline dequantization: C_f32 = dequant(A_int * B_int)",
        },
        OpDef {
            op: Op::QuantAttention,
            mnemonic: "quant_attention",
            num_operands: 4,
            has_result: true,
            category: OpCategory::Matrix,
            doc: "Quantized flash-attention: softmax(Q*K^T/sqrt(d_k))*V with int4/int8 inputs",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_display_parse() {
        let ops = [
            Op::Addi,
            Op::Mulf,
            Op::Gemm,
            Op::Load,
            Op::Return,
            Op::Quantize,
            Op::Dequantize,
            Op::QuantGemm,
            Op::QuantAttention,
        ];
        for op in &ops {
            assert_eq!(Op::from_str(&op.to_string()), *op);
        }
    }

    #[test]
    fn custom_op_roundtrip() {
        let op = Op::Custom("tpt.simt_sync".to_string());
        assert_eq!(op.to_string(), "tpt.simt_sync");
    }
}
