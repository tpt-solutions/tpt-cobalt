//! TPT-IR MLIR Dialect definitions.
//!
//! This module defines the custom MLIR dialect for TPT-IR operations.
//! Requires the `mlir` feature flag and a valid MLIR installation.

#[cfg(feature = "mlir")]
pub mod compiler {
    use melior::{
        dialect::{arith, func, mlir::register_all_dialects, DialectRegistry},
        ir::{Attribute, Block, Location, Module, Region, Type},
        pass::{self, PassManager},
        utility::{register_all_dialects, register_all_passes},
        Context,
    };

    use crate::ir::TptIr;

    pub struct TptCompiler {
        context: Context,
    }

    impl TptCompiler {
        pub fn new() -> Self {
            let context = Context::new();
            Self { context }
        }

        pub fn compile_ir(&self, ir: &TptIr) -> Result<String, String> {
            let module = Module::new(Location::new(&self.context, "tpt", "", 0));

            // Build the MLIR module from TPT-IR
            let mut module = module.into_module();
            let body = module.body();

            // Convert each op node to MLIR operations
            for node in &ir.graph.nodes {
                let location = Location::new(&self.context, &node.name, "", 0);
                let block = Block::new(&[]);

                // Insert operation based on op_type
                match node.op_type.as_str() {
                    "matmul" | "fused_matmul_relu" | "fused_matmul_gelu" => {
                        // Insert tpt.matmul or tpt.fused.matmul_relu
                    }
                    "relu" | "gelu" => {
                        // Insert tpt.activation
                    }
                    _ => {
                        // Generic tpt.generic_op
                    }
                }
            }

            Ok(module.as_operation().to_string())
        }
    }

    impl Default for TptCompiler {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(not(feature = "mlir"))]
pub mod compiler {
    //! Stub compiler when MLIR feature is not enabled.
    use crate::ir::TptIr;

    pub struct TptCompiler;

    impl TptCompiler {
        pub fn new() -> Self {
            Self
        }

        pub fn compile_ir(&self, _ir: &TptIr) -> Result<String, String> {
            Err("MLIR feature not enabled. Rebuild with --features mlir and ensure LLVM/MLIR is installed.".into())
        }
    }

    impl Default for TptCompiler {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// TableGen-style definitions for the TPT-IR dialect.
///
/// These would normally be processed by `mlir-tblgen` to generate
/// C++/Rust dialect classes. For now, they serve as documentation
/// and a reference for the dialect design.
pub const TPT_DIALECT_TABLEGEN: &str = r#"
// TPT-IR MLIR Dialect Definition
//
// Dialect: tpt
// Description: Hardware-agnostic intermediate representation for AI model compilation
// Target: TPT Crucible hardware backends (FPGA, Analog, Swarm)

#ifndef TPT_IR_DIALECT
#define TPT_IR_DIALECT

include "mlir/IR/OpBase.td"

def TPT_Dialect : Dialect {
  let name = "tpt";
  let summary = "TPT-IR dialect for hardware-agnostic AI model compilation";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

// --- Operations ---

def TPT_MatmulOp : TPT_Op<"matmul", [NoSideEffect]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication of two input tensors.
    This is a fundamental building block for neural network computation.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def TPT_FusedMatmulReluOp : TPT_Op<"fused.matmul_relu", [NoSideEffect]> {
  let summary = "Fused matrix multiplication + ReLU operation";
  let description = [{
    Performs matmul followed by ReLU activation in a single fused operation.
    Reduces memory traffic by avoiding an intermediate materialization.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def TPT_FusedMatmulGeluOp : TPT_Op<"fused.matmul_gelu", [NoSideEffect]> {
  let summary = "Fused matrix multiplication + GELU operation";
  let description = [{
    Performs matmul followed by GELU activation in a single fused operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def TPT_FusedAddReluOp : TPT_Op<"fused.add_relu", [NoSideEffect]> {
  let summary = "Fused element-wise addition + ReLU operation";

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def TPT_ActivationOp : TPT_Op<"activation", [NoSideEffect]> {
  let summary = "Activation function operation";
  let description = [{ Applies an activation function (relu, gelu, sigmoid, tanh) }];

  let arguments = (ins
    AnyRankedTensor:$input,
    StrAttr:$function
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input) `->` type($result)
  }];
}

def TPT_ConstantOp : TPT_Op<"constant", [NoSideEffect, Pure]> {
  let summary = "Constant tensor operation";

  let arguments = (ins DenseElementsAttr:$value);
  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TPT_LoadTensorOp : TPT_Op<"load_tensor", [NoSideEffect]> {
  let summary = "Load a tensor from memory/HBM";

  let arguments = (ins
    IndexType:$offset,
    IndexType:$size,
    I64Attr:$rank
  );

  let results = (outs AnyRankedTensor:$result);
}

def TPT_StoreTensorOp : TPT_Op<"store_tensor", [MemWrite]> {
  let summary = "Store a tensor to memory/HBM";

  let arguments = (ins
    AnyRankedTensor:$input,
    IndexType:$offset
  );
}

#endif // TPT_IR_DIALECT
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tablegen_has_matmul() {
        assert!(TPT_DIALECT_TABLEGEN.contains("TPT_MatmulOp"));
        assert!(TPT_DIALECT_TABLEGEN.contains("def TPT_Dialect"));
    }

    #[test]
    fn test_compiler_stub() {
        let compiler = compiler::TptCompiler::new();
        let result = compiler.compile_ir(&crate::ir::TptIr::new("test".into(), "pytorch".into()));
        assert!(result.is_err());
    }
}
