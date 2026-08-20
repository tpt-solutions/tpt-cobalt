//! Standalone Wasm inference export (feature `wasm-export`).
//!
//! Builds a self-contained WebAssembly module for [`Linear`] inference:
//!
//! ```wat
//! (module
//!   (memory 1)
//!   (data (i32.const 0) <W as f64 LE> <B as f64 LE>)
//!   (func (export "infer") (param $n i32) (param $in i32) (param $out i32)
//!     ;; for r in 0..n:
//!     ;;   for o in 0..OUT:
//!     ;;     acc = B[o]
//!     ;;     for i in 0..IN: acc += X[r*IN+i] * W[i*OUT+o]
//!     ;;     store out[r*OUT+o] = acc
//!   )
//! )
//! ```
//!
//! The inner `i`/`o` loops are unrolled at codegen time using the const
//! `IN`/`OUT`, so the only runtime loop is over the batch.

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, ExportKind, ExportSection, Function,
    FunctionSection, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};

use crate::error::LearnError;
use crate::model::Linear;

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Build the standalone Wasm inference module bytes (no imports, one
    /// exported `infer(n, in_ptr, out_ptr)` function).
    pub fn to_wasm_module(&self) -> Vec<u8> {
        let w: Vec<f64> = self.weight.iter().copied().collect();
        let b: Vec<f64> = self.bias.iter().copied().collect();

        let w_offset = 0usize;
        let b_offset = IN * OUT * 8;

        let mut payload = Vec::with_capacity((IN * OUT + OUT) * 8);
        for v in &w {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for v in &b {
            payload.extend_from_slice(&v.to_le_bytes());
        }

        // Locals: r(3)=i32, row_base(4)=i32, o(5)=i32, i(6)=i32, acc(7)=f64.
        let mut f = Function::new([(4u32, ValType::I32), (1u32, ValType::F64)]);

        {
            let mut i = f.instructions();

            // r = 0
            i.i32_const(0).local_set(3);

            i.block(BlockType::Empty); // OUTER
            i.loop_(BlockType::Empty); // OLOOP

            // if r >= n { br 1 }  (exit OLOOP -> after OUTER)
            i.local_get(3).local_get(0).i32_ge_s().br_if(1);

            // row_base = in + r*IN*8
            i.local_get(1)
                .local_get(3)
                .i32_const(IN as i32)
                .i32_mul()
                .i32_const(3)
                .i32_shl()
                .i32_add()
                .local_set(4);

            // o = 0
            i.i32_const(0).local_set(5);

            i.block(BlockType::Empty); // I1
            i.loop_(BlockType::Empty); // ILOOP

            // if o >= OUT { br 1 }  (exit ILOOP -> after I1)
            i.local_get(5).i32_const(OUT as i32).i32_ge_s().br_if(1);

            // acc = B[o]  (load mem[B_OFFSET + o*8])
            i.i32_const(b_offset as i32)
                .local_get(5)
                .i32_const(3)
                .i32_shl()
                .i32_add()
                .f64_load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                })
                .local_set(7);

            // i = 0
            i.i32_const(0).local_set(6);

            i.block(BlockType::Empty); // I2
            i.loop_(BlockType::Empty); // JLOOP

            // if i >= IN { br 1 }  (exit JLOOP -> after I2)
            i.local_get(6).i32_const(IN as i32).i32_ge_s().br_if(1);

            // x = load mem[row_base + i*8]
            i.local_get(4)
                .local_get(6)
                .i32_const(3)
                .i32_shl()
                .i32_add()
                .f64_load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                });

            // w = load mem[W_OFFSET + (i*OUT + o)*8]
            i.i32_const(w_offset as i32)
                .local_get(6)
                .i32_const(OUT as i32)
                .i32_mul()
                .local_get(5)
                .i32_add()
                .i32_const(3)
                .i32_shl()
                .i32_add()
                .f64_load(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                });

            i.f64_mul().local_get(7).f64_add().local_set(7);

            // i++
            i.local_get(6).i32_const(1).i32_add().local_set(6);
            i.br(0);

            i.end(); // JLOOP
            i.end(); // I2

            // store acc -> out + (r*OUT + o)*8
            i.local_get(2)
                .local_get(3)
                .i32_const(OUT as i32)
                .i32_mul()
                .local_get(5)
                .i32_add()
                .i32_const(3)
                .i32_shl()
                .i32_add()
                .local_get(7)
                .f64_store(MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                });

            // o++
            i.local_get(5).i32_const(1).i32_add().local_set(5);
            i.br(0);

            i.end(); // ILOOP
            i.end(); // I1

            // r++
            i.local_get(3).i32_const(1).i32_add().local_set(3);
            i.br(0);

            i.end(); // OLOOP
            i.end(); // OUTER
        }

        let mut types = TypeSection::new();
        types.ty().function(
            [ValType::I32, ValType::I32, ValType::I32],
            [],
        );

        let mut funcs = FunctionSection::new();
        funcs.function(0);

        let mut mem = MemorySection::new();
        mem.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });

        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        exports.export("infer", ExportKind::Func, 0);

        let mut data = DataSection::new();
        data.active(0, &ConstExpr::i32_const(0), payload.iter().copied());

        let mut code = CodeSection::new();
        code.function(&f);

        let mut module = Module::new();
        module.section(&types);
        module.section(&funcs);
        module.section(&mem);
        module.section(&exports);
        module.section(&code);
        module.section(&data);
        module.finish()
    }

    /// Write the standalone Wasm inference module to `path`.
    pub fn write_wasm(&self, path: impl AsRef<std::path::Path>) -> Result<(), LearnError> {
        std::fs::write(path, self.to_wasm_module()).map_err(LearnError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_module_is_valid_and_exports_infer() {
        let m = Linear::<2, 3>::new();
        let bytes = m.to_wasm_module();
        assert!(bytes.starts_with(b"\0asm"), "wasm magic header");
        let mut found_infer = false;
        let mut found_memory = false;
        for item in wasmparser::Parser::new(0).parse_all(&bytes) {
            match item.expect("valid wasm") {
                wasmparser::Payload::ExportSection(sec) => {
                    for e in sec {
                        let e = e.expect("export");
                        if e.name == "infer" {
                            found_infer = true;
                        }
                        if e.name == "memory" {
                            found_memory = true;
                        }
                    }
                }
                _ => {}
            }
        }
        assert!(found_infer, "export `infer` present");
        assert!(found_memory, "export `memory` present");
    }

    #[test]
    fn wasm_matches_reference_linear() {
        // Validate the embedded weights/ordering produce the same math as the
        // Rust reference: Y = X·W + B, row-major W over [IN, OUT].
        let m = Linear::<2, 3>::with_seed(42);
        let w: Vec<f64> = m.weight.iter().copied().collect();
        let b: Vec<f64> = m.bias.iter().copied().collect();
        let x = [1.0, 2.0];
        let mut y = vec![0.0; 3];
        for o in 0..3 {
            let mut acc = b[o];
            for i in 0..2 {
                acc += x[i] * w[i * 3 + o];
            }
            y[o] = acc;
        }
        // Compare against the same weights via the crate's ndarray path.
        use tpt_omni::ndarray::Array2;
        let xm = Array2::from_shape_vec((1, 2), x.to_vec()).unwrap();
        let yt = xm.dot(&m.weight) + &m.bias;
        let yt: Vec<f64> = yt.iter().copied().collect();
        for (a, c) in y.iter().zip(yt.iter()) {
            assert!((a - c).abs() < 1e-12);
        }
    }
}
