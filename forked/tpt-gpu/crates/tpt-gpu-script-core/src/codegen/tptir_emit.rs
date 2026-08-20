/// Lowers GPU-kernel TPT Script functions to TPTIR text.
///
/// Only functions annotated `@requires_gpu(true)` are emitted; all others
/// are silently skipped (they go to the Rust emitter instead).
///
/// The emitted TPTIR text can be fed directly into the layer3 compiler:
///
/// ```ignore
/// let isa = layer3_tptc::compile_native(&tptir_source, "tptisa")?;
/// let llvm = layer3_tptc::compile_native(&tptir_source, "llvmir")?;
/// ```
///
/// # TPTIR conventions used
/// - Functions: `func @name(%p: type, ...) -> ret_type { ^entry: ... }`
/// - Tensor params: `memref<*x*xf32, global>` (symbolic dims → `*`)
/// - `tpt.xxx(a, b, c)` → `tpt.xxx %a, %b, %c {named_key = val, ...}`
/// - `let x = expr` → SSA binding `%x = ...`
/// - `return expr` → `tpt.return %val`
/// - Integer arithmetic:  `addi`, `subi`, `muli`, `divi`, `remi`
/// - Float arithmetic:    `addf`, `subf`, `mulf`, `divf`, `remf`
///   (operand types are not yet tracked; the default is float ops for
///   GPU kernels which overwhelmingly use fp math)
use std::collections::HashMap;

use crate::ast::*;
use crate::semantic::metadata::extract_function_metadata;

pub struct TptIrEmitter {
    /// SSA counter for fresh temporaries.
    counter: u64,
    /// Maps TPT Script local name → its TPTIR SSA name (e.g. `%x`).
    locals: HashMap<String, String>,
    /// Compilation errors accumulated during emission.
    errors: Vec<String>,
}

impl Default for TptIrEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TptIrEmitter {
    pub fn new() -> Self {
        Self {
            counter: 0,
            locals: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Drain and return all accumulated errors.
    pub fn take_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.errors)
    }

    pub fn emit_program(&mut self, program: &Program) -> String {
        let mut out = String::from("; Generated TPTIR from TPT Script\n");
        out.push_str("; Compile: tptc::compile_native(this, \"tptisa\" | \"llvmir\")\n\n");

        let mut has_any = false;
        for item in &program.items {
            if let Item::Function(f) = item {
                let meta = extract_function_metadata(f);
                if meta.hardware.requires_gpu {
                    out.push_str(&self.emit_kernel_function(f));
                    out.push('\n');
                    has_any = true;
                }
            }
        }

        if !has_any {
            out.push_str("; (no GPU kernel functions found)\n");
        }

        out
    }

    // -----------------------------------------------------------------------
    // Function emission
    // -----------------------------------------------------------------------

    fn emit_kernel_function(&mut self, f: &FunctionDecl) -> String {
        self.counter = 0;
        self.locals.clear();

        let mut s = String::new();

        // Metadata as TPTIR comments
        let meta = extract_function_metadata(f);
        if let Some(doc) = &meta.doc {
            s.push_str(&format!("; {doc}\n"));
        }
        for c in &meta.constraints {
            s.push_str(&format!("; @constraint: {}\n", c.expr_str));
        }
        if meta.hardware.requires_tensor_cores {
            s.push_str("; @requires_tensor_cores\n");
        }
        if meta.hardware.min_vram_gb > 0 {
            s.push_str(&format!("; @min_vram_gb: {}\n", meta.hardware.min_vram_gb));
        }
        if let Some(cplx) = &meta.complexity {
            s.push_str(&format!("; @complexity: {cplx}\n"));
        }

        // Register params as SSA names before emitting signature
        let params: Vec<String> = f
            .params
            .iter()
            .map(|p| {
                let ssa = format!("%{}", p.name);
                self.locals.insert(p.name.clone(), ssa.clone());
                format!("{ssa}: {}", self.tptir_type(&p.ty))
            })
            .collect();

        let ret_ty = f
            .return_type
            .as_ref()
            .map(|t| format!(" -> {}", self.tptir_type(t)))
            .unwrap_or_default();

        s.push_str(&format!(
            "func @{}({}){} {{\n",
            f.name,
            params.join(", "),
            ret_ty
        ));
        s.push_str("^entry:\n");

        // Body
        let body = self.emit_block_body(&f.body);
        s.push_str(&body);

        // Implicit return if the body has no explicit one
        let has_return = f.body.stmts.iter().any(|st| matches!(st, Stmt::Return(_)));
        if !has_return {
            s.push_str("  tpt.return\n");
        }

        s.push_str("}\n");
        s
    }

    // -----------------------------------------------------------------------
    // Block / statement emission
    // -----------------------------------------------------------------------

    fn emit_block_body(&mut self, block: &Block) -> String {
        let mut s = String::new();
        for stmt in &block.stmts {
            s.push_str(&self.emit_stmt(stmt));
        }
        s
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Let(l) => {
                // Prefer `%variable_name` as SSA name so the IR is readable.
                let ssa = format!("%{}", l.name);
                let rhs = self.emit_expr_rhs(&l.value);
                // If the RHS is just a plain SSA alias (e.g. `let y = x`),
                // skip the instruction and remap the name directly.
                if rhs.starts_with('%') && !rhs.contains(' ') {
                    self.locals.insert(l.name.clone(), rhs);
                    return String::new();
                }
                self.locals.insert(l.name.clone(), ssa.clone());
                format!("  {ssa} = {rhs}\n")
            }

            Stmt::Return(r) => match &r.value {
                Some(e) => {
                    let val = self.emit_expr_val(e);
                    format!("  tpt.return {val}\n")
                }
                None => "  tpt.return\n".to_string(),
            },

            Stmt::Expr(e) => {
                let line = self.emit_bare_call(e);
                format!("  {line}\n")
            }

            Stmt::Break(_) | Stmt::Continue(_) => {
                let msg = "break/continue requires control-flow lowering — \
                           not yet supported in GPU kernels";
                self.errors.push(msg.to_string());
                format!("  ; ERROR: {msg}\n")
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression emission
    // -----------------------------------------------------------------------

    /// Emit an expression as a **right-hand side** (after `= `).
    /// Returns something like `tpt.relu %x` or `addf %a, %b` or `constant 1 : i64`.
    fn emit_expr_rhs(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            // Literals → TPTIR constant ops
            ExprKind::IntLit(n) => format!("constant {n} : i64"),
            ExprKind::FloatLit(f) => format!("constant {f} : f64"),
            ExprKind::BoolLit(b) => format!("constant {} : i1", if *b { 1 } else { 0 }),

            // Ident: SSA alias (handled in emit_stmt above, but also here as fallback)
            ExprKind::Ident(name) => self.resolve(name),

            // Array literal → TPTIR array/shape literal
            ExprKind::ArrayLit(elems) => {
                let parts: Vec<_> = elems.iter().map(|e| self.emit_expr_val(e)).collect();
                format!("[{}]", parts.join(", "))
            }

            ExprKind::Paren(inner) => self.emit_expr_rhs(inner),

            // Binary ops
            ExprKind::BinaryOp { op, left, right } => {
                let lv = self.emit_expr_val(left);
                let rv = self.emit_expr_val(right);
                format!("{} {lv}, {rv}", tptir_binop(op))
            }

            // `tpt.xxx(pos_args, named_key=val)` → `tpt.xxx %a, %b {key = val}`
            // Parser may produce either Call(FieldAccess) or MethodCall depending on form.
            ExprKind::Call { callee, args } => {
                if let Some(fn_name) = tpt_fn(callee) {
                    return self.emit_tpt_op(fn_name, args);
                }
                let callee_v = self.emit_expr_val(callee);
                let arg_strs = self.emit_args_as_values(args);
                format!("call {callee_v}({})", arg_strs.join(", "))
            }

            ExprKind::MethodCall {
                expr: obj,
                method,
                args,
            } => {
                // `tpt.xxx(args)` parsed as MethodCall by the parser
                if let ExprKind::Ident(root) = &obj.kind {
                    if root == "tpt" {
                        return self.emit_tpt_op(method, args);
                    }
                }
                let obj_v = self.emit_expr_val(obj);
                let arg_strs = self.emit_args_as_values(args);
                format!("{obj_v}.{method}({})", arg_strs.join(", "))
            }

            ExprKind::Index { expr: obj, indices } => {
                let obj_v = self.emit_expr_val(obj);
                let idx_parts: Vec<_> = indices.iter().map(|i| self.emit_expr_val(i)).collect();
                format!("{obj_v}[{}]", idx_parts.join(", "))
            }

            // Complex sub-expressions that cannot be lowered inline.
            _ => {
                let msg = format!("unsupported sub-expression in GPU kernel at {}", expr.span);
                self.errors.push(msg.clone());
                format!("; ERROR: {msg}")
            }
        }
    }

    /// Emit an expression as a simple **value reference** (SSA name or literal).
    /// Used when a value is needed inline (e.g. as a function argument).
    fn emit_expr_val(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Ident(name) => {
                // Primitive type names used as value arguments (e.g. `dtype=f32`)
                // are bare type literals in TPTIR, not SSA values.
                if is_type_name(name) {
                    return name.clone();
                }
                self.resolve(name)
            }
            ExprKind::IntLit(n) => n.to_string(),
            ExprKind::FloatLit(f) => {
                if f.fract() == 0.0 {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
            ExprKind::BoolLit(b) => if *b { "1" } else { "0" }.to_string(),
            ExprKind::Paren(inner) => self.emit_expr_val(inner),
            ExprKind::ArrayLit(elems) => {
                let parts: Vec<_> = elems.iter().map(|e| self.emit_expr_val(e)).collect();
                format!("[{}]", parts.join(", "))
            }
            ExprKind::FieldAccess { expr: obj, field } => {
                format!("{}.{field}", self.emit_expr_val(obj))
            }
            ExprKind::Index { expr: obj, indices } => {
                let obj_v = self.emit_expr_val(obj);
                let idx_parts: Vec<_> = indices.iter().map(|i| self.emit_expr_val(i)).collect();
                format!("{obj_v}[{}]", idx_parts.join(", "))
            }
            // Non-trivial sub-expressions in value position cannot be lowered
            // to a single SSA value without emitting additional instructions.
            // Record an error and return a sentinel name so downstream IR
            // remains syntactically valid (though semantically wrong).
            _ => {
                let msg = format!(
                    "unsupported expression in value position at {} — \
                     value cannot be inlined; hoist to a let-binding",
                    expr.span
                );
                self.errors.push(msg.clone());
                "%_error".to_string()
            }
        }
    }

    /// Emit a **bare** expression statement (call without capturing the result).
    fn emit_bare_call(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Call { callee, args } => {
                if let Some(fn_name) = tpt_fn(callee) {
                    return self.emit_tpt_op(fn_name, args);
                }
                let callee_v = self.emit_expr_val(callee);
                let arg_strs = self.emit_args_as_values(args);
                format!("call {callee_v}({})", arg_strs.join(", "))
            }
            ExprKind::MethodCall {
                expr: obj,
                method,
                args,
            } => {
                // `tpt.xxx(args)` parsed as MethodCall
                if let ExprKind::Ident(root) = &obj.kind {
                    if root == "tpt" {
                        return self.emit_tpt_op(method, args);
                    }
                }
                let obj_v = self.emit_expr_val(obj);
                let arg_strs = self.emit_args_as_values(args);
                format!("{obj_v}.{method}({})", arg_strs.join(", "))
            }
            _ => self.emit_expr_rhs(expr),
        }
    }

    /// Emit a `tpt.<fn_name>` operation (no LHS).
    fn emit_tpt_op(&mut self, fn_name: &str, args: &[CallArg]) -> String {
        let pos_args: Vec<String> = args
            .iter()
            .filter_map(|a| {
                if let CallArg::Positional(e) = a {
                    Some(self.emit_expr_val(e))
                } else {
                    None
                }
            })
            .collect();

        let named_attrs: Vec<String> = args
            .iter()
            .filter_map(|a| {
                if let CallArg::Named { name, value, .. } = a {
                    Some(format!("{name} = {}", self.emit_expr_val(value)))
                } else {
                    None
                }
            })
            .collect();

        let mut op = format!("tpt.{fn_name}");
        if !pos_args.is_empty() {
            op.push(' ');
            op.push_str(&pos_args.join(", "));
        }
        if !named_attrs.is_empty() {
            op.push_str(&format!(" {{{}}}", named_attrs.join(", ")));
        }
        op
    }

    fn emit_args_as_values(&mut self, args: &[CallArg]) -> Vec<String> {
        args.iter()
            .map(|a| match a {
                CallArg::Positional(e) => self.emit_expr_val(e),
                CallArg::Named { name, value, .. } => {
                    format!("{name} = {}", self.emit_expr_val(value))
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Type emission
    // -----------------------------------------------------------------------

    fn tptir_type(&self, ty: &Type) -> String {
        match ty {
            Type::Primitive(p, _) => tptir_primitive(p).to_string(),

            Type::Tensor { dtype, dims, .. } => {
                let shape: Vec<String> = dims
                    .iter()
                    .map(|d| match d {
                        Dim::Concrete(n) => n.to_string(),
                        Dim::Named(_) => "*".to_string(),
                        Dim::Dynamic => "*".to_string(),
                    })
                    .collect();
                format!(
                    "memref<{}x{}, global>",
                    shape.join("x"),
                    tptir_primitive(dtype)
                )
            }

            Type::Tuple(ts, _) => {
                let parts: Vec<_> = ts.iter().map(|t| self.tptir_type(t)).collect();
                format!("({})", parts.join(", "))
            }

            Type::Array { elem, size, .. } => {
                format!("memref<{}x{}, local>", size, self.tptir_type(elem))
            }

            Type::Slice(elem, _) => format!("memref<*x{}, global>", self.tptir_type(elem)),

            Type::Named(name, _) => format!("!tpt.{}", name.to_lowercase()),
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    fn fresh(&mut self) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("%t{n}")
    }

    fn resolve(&self, name: &str) -> String {
        self.locals
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("%{name}"))
    }
}

// -----------------------------------------------------------------------
// Free functions
// -----------------------------------------------------------------------

/// Returns true for primitive type names used as bare value arguments (e.g. `dtype=f32`).
fn is_type_name(name: &str) -> bool {
    matches!(
        name,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "f16"
            | "bf16"
            | "f32"
            | "f64"
            | "bool"
            | "index"
    )
}

/// If `callee` is `tpt.<fn>`, return `<fn>`. Otherwise `None`.
fn tpt_fn(callee: &Expr) -> Option<&str> {
    if let ExprKind::FieldAccess { expr: base, field } = &callee.kind {
        if let ExprKind::Ident(root) = &base.kind {
            if root == "tpt" {
                return Some(field.as_str());
            }
        }
    }
    None
}

fn tptir_primitive(p: &PrimitiveType) -> &'static str {
    match p {
        PrimitiveType::I8 => "i8",
        PrimitiveType::I16 => "i16",
        PrimitiveType::I32 => "i32",
        PrimitiveType::I64 => "i64",
        // TPTIR uses signed integer types; unsigned TPT Script types are
        // represented as same-width signed integers in TPTIR (widths match).
        PrimitiveType::U8 => "i8",
        PrimitiveType::U16 => "i16",
        PrimitiveType::U32 => "i32",
        PrimitiveType::U64 => "i64",
        PrimitiveType::F16 => "f16",
        PrimitiveType::Bf16 => "bf16",
        PrimitiveType::F32 => "f32",
        PrimitiveType::F64 => "f64",
        PrimitiveType::Bool => "i1",
        PrimitiveType::Index => "index",
    }
}

/// TPTIR binary op mnemonic.
/// GPU kernels are predominantly float; integer variants are also listed for
/// completeness. The caller may refine this based on inferred operand type.
fn tptir_binop(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "addf",
        BinOp::Sub => "subf",
        BinOp::Mul => "mulf",
        BinOp::Div => "divf",
        BinOp::Mod => "remf",
        BinOp::Eq => "cmpeq",
        BinOp::Ne => "cmpne",
        BinOp::Lt => "cmplt",
        BinOp::Gt => "cmpgt",
        BinOp::Le => "cmple",
        BinOp::Ge => "cmpge",
        BinOp::And => "andi",
        BinOp::Or => "ori",
        BinOp::Range => "range",
        BinOp::RangeEq => "range_inclusive",
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_str;

    fn tptir_emit(src: &str) -> String {
        let prog = compile_str(src).expect("compile_str");
        TptIrEmitter::new().emit_program(&prog)
    }

    #[test]
    fn test_gpu_kernel_emitted() {
        let out = tptir_emit(
            "@requires_gpu(true) fn relu_kernel(x: Tensor[f32, m, n]) -> Tensor[f32, m, n] { \
                 let r = tpt.relu(x) \
                 return r \
             }",
        );
        assert!(out.contains("func @relu_kernel"), "{out}");
        assert!(out.contains("%x: memref<*x*xf32, global>"), "{out}");
        assert!(out.contains("tpt.relu"), "{out}");
        assert!(out.contains("tpt.return"), "{out}");
    }

    #[test]
    fn test_host_fn_not_emitted() {
        let out = tptir_emit("fn host(x: f32) -> f32 { return x }");
        assert!(!out.contains("func @host"), "{out}");
        assert!(out.contains("no GPU kernel functions found"), "{out}");
    }

    #[test]
    fn test_tensor_type_lowering() {
        let out = tptir_emit("@requires_gpu(true) fn k(a: Tensor[f32, 128, 256]) { }");
        // Concrete dims should be preserved
        assert!(out.contains("memref<128x256xf32, global>"), "{out}");
    }

    #[test]
    fn test_symbolic_dims_become_star() {
        let out = tptir_emit("@requires_gpu(true) fn k(a: Tensor[f32, m, n]) { }");
        assert!(out.contains("memref<*x*xf32, global>"), "{out}");
    }

    #[test]
    fn test_metadata_as_comments() {
        let src = r#"
@doc("Fast attention")
@constraint("q.shape[1] == k.shape[1]", error="head dim must match")
@requires_gpu(true)
fn attn(q: Tensor[f32, b, h], k: Tensor[f32, b, h]) { }
"#;
        let out = tptir_emit(src);
        assert!(out.contains("; Fast attention"), "{out}");
        assert!(
            out.contains("; @constraint: q.shape[1] == k.shape[1]"),
            "{out}"
        );
    }

    #[test]
    fn test_tpt_op_with_named_args() {
        let out = tptir_emit("@requires_gpu(true) fn k() { let z = tpt.zeros([4, 4], dtype=f32) }");
        assert!(out.contains("tpt.zeros"), "{out}");
        assert!(out.contains("dtype = f32"), "{out}");
    }

    #[test]
    fn test_bare_call_no_lhs() {
        let out = tptir_emit(
            "@requires_gpu(true) fn k(a: Tensor[f32, m, k], b: Tensor[f32, k, n], c: Tensor[f32, m, n]) { \
                 tpt.gemm(a, b, c) \
             }",
        );
        // Bare call should not have an `= ` assignment
        assert!(out.contains("tpt.gemm %a, %b, %c"), "{out}");
        assert!(!out.contains("= tpt.gemm"), "{out}");
    }

    #[test]
    fn test_full_matmul_kernel() {
        let src = r#"
@doc("Matrix multiply A × B → C")
@constraint("a.shape[1] == b.shape[0]", error="Inner dims must match")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(8)
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    let result = tpt.zeros([m, n], dtype=f32)
    tpt.gemm(a, b, result)
    return result
}
"#;
        let out = tptir_emit(src);
        assert!(out.contains("func @matmul"), "{out}");
        assert!(out.contains("%result = tpt.zeros"), "{out}");
        assert!(out.contains("tpt.gemm %a, %b, %result"), "{out}");
        assert!(out.contains("tpt.return %result"), "{out}");
        assert!(out.contains("; @requires_tensor_cores"), "{out}");
    }

    #[test]
    fn test_multiple_gpu_kernels() {
        let src = r#"
@requires_gpu(true) fn k1(x: Tensor[f32, n]) -> Tensor[f32, n] { return x }
@requires_gpu(true) fn k2(x: Tensor[f32, n]) -> Tensor[f32, n] { return x }
"#;
        let out = tptir_emit(src);
        assert!(out.contains("func @k1"), "{out}");
        assert!(out.contains("func @k2"), "{out}");
    }

    #[test]
    fn test_let_alias_no_instruction() {
        // `let y = x` — pure alias, should not emit an instruction
        let out = tptir_emit(
            "@requires_gpu(true) fn k(x: Tensor[f32, n]) -> Tensor[f32, n] { \
                 let y = x \
                 return y \
             }",
        );
        // return should reference %x (the resolved alias), not %y
        assert!(out.contains("tpt.return %x"), "{out}");
    }
}
