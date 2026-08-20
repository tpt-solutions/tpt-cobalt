pub mod builtins;
pub mod constraints;
pub mod metadata;
pub mod types;

use std::collections::HashMap;

use crate::ast::*;
use crate::errors::{ErrorCode, ErrorContext, TptError};
use crate::lexer::Span;

use builtins::{ident_as_dtype, infer_builtin};
use constraints::{eval_constraint, ConstraintResult, EvalEnv};
use metadata::extract_function_metadata;
use types::{DimVal, TptType, TypeAnnotation};

pub use crate::errors::TptError as TypeError;
pub use metadata::{extract_function_metadata as fn_metadata, FunctionMeta};
pub use types::TptType as SemType;

// ---------------------------------------------------------------------------
// Type environment (scope chain)
// ---------------------------------------------------------------------------

/// A scope chain mapping names to their inferred types.
#[derive(Default, Clone)]
pub struct TypeEnv {
    scopes: Vec<HashMap<String, TptType>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn define(&mut self, name: impl Into<String>, ty: TptType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), ty);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&TptType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// DimEnv — dimension bindings for constraint checking
// ---------------------------------------------------------------------------

/// Maps symbolic dimension names (e.g. `m`, `batch`) to concrete values when
/// available.  Populated from call-site arguments.
#[derive(Default, Clone)]
struct DimEnv {
    dims: HashMap<String, i64>,
    shapes: HashMap<String, Vec<i64>>,
}

// ---------------------------------------------------------------------------
// TypeChecker
// ---------------------------------------------------------------------------

/// Walks a `Program` AST and:
///  1. Infers types for all expressions.
///  2. Checks parameter and return types against declared types.
///  3. Evaluates `@constraint` expressions when possible.
///  4. Collects errors into `errors`.
///  5. Builds a `type_map` from span → inferred type for IDE integration.
pub struct TypeChecker {
    pub errors: Vec<TptError>,
    /// Maps (span.start, span.end) → inferred type for every expression.
    pub type_map: Vec<TypeAnnotation>,
    /// Top-level function signatures visible for forward references.
    global_fns: HashMap<String, (Vec<TptType>, TptType)>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            type_map: Vec::new(),
            global_fns: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    pub fn check_program(&mut self, program: &Program) {
        // First pass: collect all top-level function signatures so that forward
        // references within the same file work.
        for item in &program.items {
            if let Item::Function(f) = item {
                let params: Vec<TptType> =
                    f.params.iter().map(|p| TptType::from_ast(&p.ty)).collect();
                let ret = f
                    .return_type
                    .as_ref()
                    .map(TptType::from_ast)
                    .unwrap_or(TptType::Unit);
                self.global_fns.insert(f.name.clone(), (params, ret));
            }
        }

        // Second pass: check each item.
        for item in &program.items {
            match item {
                Item::Import(_) => {} // nothing to type-check
                Item::Function(f) => self.check_function(f),
                Item::TypeAlias(_) => {} // aliases resolved on use
            }
        }
    }

    // -----------------------------------------------------------------------
    // Function checking
    // -----------------------------------------------------------------------

    fn check_function(&mut self, f: &FunctionDecl) {
        let mut env = TypeEnv::new();

        // `tpt` is auto-imported (spec §9, §12.2) — treat it as an opaque
        // module so that `tpt.zeros(...)` etc. do not produce UNDEFINED_VARIABLE.
        env.define("tpt", TptType::Unknown);

        // Collect dimension names introduced in the signature and bind them as
        // `index` values so that code like `tpt.zeros([m, n], ...)` can reference
        // the symbolic dimension variables without triggering UNDEFINED_VARIABLE.
        let mut dim_names: HashMap<String, ()> = HashMap::new();
        for param in &f.params {
            collect_dim_names(&param.ty, &mut dim_names);
        }
        for dim_name in dim_names.keys() {
            env.define(dim_name, TptType::Index);
        }

        // Bind parameters in scope.
        for param in &f.params {
            env.define(&param.name, TptType::from_ast(&param.ty));
        }

        // Evaluate constraints if possible.
        let meta = extract_function_metadata(f);
        let dim_env = DimEnv::default(); // no concrete bindings at declaration time
        let eval_env = EvalEnv {
            dims: &dim_env.dims,
            shapes: &dim_env.shapes,
        };
        for c in &meta.constraints {
            match &c.expr {
                Ok(expr) => {
                    // At declaration time all dims are symbolic → Symbolic result.
                    // A Known(false) here would be a compile-time contradiction.
                    if eval_constraint(expr, &eval_env) == ConstraintResult::Known(false) {
                        self.errors.push(
                            TptError::new(
                                ErrorCode::ConstraintViolation,
                                format!(
                                    "Constraint '{}' is statically false{}",
                                    c.expr_str,
                                    c.error_msg
                                        .as_deref()
                                        .map(|m| format!(": {m}"))
                                        .unwrap_or_default()
                                ),
                                f.span.clone(),
                            )
                            .with_context(
                                ErrorContext::new().with("constraint_expr", c.expr_str.clone()),
                            ),
                        );
                    }
                }
                Err(e) => {
                    self.errors.push(TptError::new(
                        ErrorCode::ParseError,
                        format!("Could not parse constraint '{}': {e}", c.expr_str),
                        f.span.clone(),
                    ));
                }
            }
        }

        // Check body.
        let declared_ret = f
            .return_type
            .as_ref()
            .map(TptType::from_ast)
            .unwrap_or(TptType::Unit);
        self.check_block(&f.body, &mut env, &declared_ret);
    }

    // -----------------------------------------------------------------------
    // Block / statement checking
    // -----------------------------------------------------------------------

    fn check_block(&mut self, block: &Block, env: &mut TypeEnv, expected_ret: &TptType) {
        env.push_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt, env, expected_ret);
        }
        env.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &Stmt, env: &mut TypeEnv, expected_ret: &TptType) {
        match stmt {
            Stmt::Let(l) => {
                let inferred = self.infer_expr(&l.value, env);
                if let Some(declared) = &l.ty {
                    let declared_ty = TptType::from_ast(declared);
                    if !inferred.compatible(&declared_ty) {
                        let (code, ctx) = mismatch_code_and_ctx(&inferred, &declared_ty, &l.name);
                        self.errors.push(
                            TptError::new(
                                code,
                                format!(
                                    "Type mismatch in `let {name}`: declared `{declared_ty}`, \
                                 inferred `{inferred}`",
                                    name = l.name,
                                ),
                                l.span.clone(),
                            )
                            .with_context(ctx),
                        );
                    }
                    env.define(&l.name, declared_ty);
                } else {
                    env.define(&l.name, inferred);
                }
            }

            Stmt::Return(r) => {
                let actual = r
                    .value
                    .as_ref()
                    .map(|e| self.infer_expr(e, env))
                    .unwrap_or(TptType::Unit);
                if !actual.compatible(expected_ret) {
                    let ctx = ErrorContext::new()
                        .with("expected_type", format!("{expected_ret}"))
                        .with("found_type", format!("{actual}"));
                    self.errors.push(
                        TptError::new(
                            ErrorCode::ReturnTypeMismatch,
                            format!(
                                "Return type mismatch: expected `{expected_ret}`, found `{actual}`"
                            ),
                            r.span.clone(),
                        )
                        .with_context(ctx),
                    );
                }
            }

            Stmt::Expr(e) => {
                self.infer_expr(e, env);
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    // -----------------------------------------------------------------------
    // Expression type inference
    // -----------------------------------------------------------------------

    /// Infer the type of `expr`, record it in `self.type_map`, and return it.
    pub fn infer_expr(&mut self, expr: &Expr, env: &TypeEnv) -> TptType {
        let ty = self.infer_expr_inner(expr, env);
        self.type_map.push(TypeAnnotation {
            span: expr.span.clone(),
            ty: ty.clone(),
        });
        ty
    }

    fn infer_expr_inner(&mut self, expr: &Expr, env: &TypeEnv) -> TptType {
        match &expr.kind {
            ExprKind::IntLit(_) => TptType::I64,
            ExprKind::FloatLit(_) => TptType::F64,
            ExprKind::BoolLit(_) => TptType::Bool,
            ExprKind::StringLit(_) => TptType::Slice(Box::new(TptType::U8)),

            ExprKind::Ident(name) => {
                // Check if it's a dtype name used as a value (e.g. `dtype=f32`)
                if let Some(ty) = ident_as_dtype(name) {
                    return ty;
                }
                match env.lookup(name) {
                    Some(ty) => ty.clone(),
                    None => {
                        // Check global functions
                        if self.global_fns.contains_key(name) {
                            return TptType::Unknown; // function reference
                        }
                        self.errors.push(
                            TptError::new(
                                ErrorCode::UndefinedVariable,
                                format!("Undefined variable `{name}`"),
                                expr.span.clone(),
                            )
                            .with_context(ErrorContext::new().with("name", name))
                            .with_suggestion(format!(
                                "Did you mean to declare `let {name} = ...`?"
                            )),
                        );
                        TptType::Unknown
                    }
                }
            }

            ExprKind::Paren(inner) => self.infer_expr(inner, env),

            ExprKind::ArrayLit(elems) => {
                if elems.is_empty() {
                    return TptType::Slice(Box::new(TptType::Unknown));
                }
                let first_ty = self.infer_expr(&elems[0], env);
                for elem in &elems[1..] {
                    let ty = self.infer_expr(elem, env);
                    if !ty.compatible(&first_ty) {
                        self.errors.push(
                            TptError::new(
                                ErrorCode::TypeError,
                                format!("Array literal has mixed types: `{first_ty}` and `{ty}`"),
                                elem.span.clone(),
                            )
                            .with_context(
                                ErrorContext::new()
                                    .with("expected_type", format!("{first_ty}"))
                                    .with("found_type", format!("{ty}")),
                            ),
                        );
                    }
                }
                TptType::Slice(Box::new(first_ty))
            }

            ExprKind::UnaryOp { op, operand } => {
                let ty = self.infer_expr(operand, env);
                match op {
                    UnOp::Neg => {
                        if !ty.is_numeric() && !ty.is_tensor() && ty != TptType::Unknown {
                            self.errors.push(
                                TptError::new(
                                    ErrorCode::TypeError,
                                    format!("Unary `-` requires a numeric type, found `{ty}`"),
                                    expr.span.clone(),
                                )
                                .with_context(
                                    ErrorContext::new().with("found_type", format!("{ty}")),
                                ),
                            );
                        }
                        ty
                    }
                    UnOp::Not => {
                        if ty != TptType::Bool && !ty.is_tensor() && ty != TptType::Unknown {
                            self.errors.push(
                                TptError::new(
                                    ErrorCode::TypeError,
                                    format!("Unary `!` requires bool, found `{ty}`"),
                                    expr.span.clone(),
                                )
                                .with_context(
                                    ErrorContext::new()
                                        .with("expected_type", "bool")
                                        .with("found_type", format!("{ty}")),
                                ),
                            );
                        }
                        TptType::Bool
                    }
                }
            }

            ExprKind::BinaryOp { op, left, right } => {
                let lt = self.infer_expr(left, env);
                let rt = self.infer_expr(right, env);
                self.infer_binop_type(op, &lt, &rt, &expr.span)
            }

            ExprKind::FieldAccess { expr: obj, field } => {
                let obj_ty = self.infer_expr(obj, env);
                self.infer_field_access(&obj_ty, field, &expr.span)
            }

            ExprKind::MethodCall {
                expr: obj,
                method,
                args,
            } => {
                let obj_ty = self.infer_expr(obj, env);
                let (arg_tys, named_tys) = self.collect_call_arg_types(args, env);
                self.infer_method_call(&obj_ty, method, &arg_tys, &named_tys, &expr.span)
            }

            ExprKind::Call { callee, args } => {
                let callee_ty = self.infer_expr(callee, env);
                let (arg_tys, named_tys) = self.collect_call_arg_types(args, env);

                // If the callee is a field access on `tpt`, dispatch to the
                // builtin registry.
                if let ExprKind::FieldAccess {
                    expr: base,
                    field: name,
                } = &callee.kind
                {
                    if let ExprKind::Ident(root) = &base.kind {
                        if root == "tpt" {
                            let named_refs: Vec<(&str, TptType)> = named_tys
                                .iter()
                                .map(|(k, v)| (k.as_str(), v.clone()))
                                .collect();
                            return infer_builtin(name, &arg_tys, &named_refs);
                        }
                    }
                }

                // Generic function call
                match &callee_ty {
                    TptType::Fn { ret, .. } => *ret.clone(),
                    TptType::Unknown => TptType::Unknown,
                    _ => {
                        self.errors.push(
                            TptError::new(
                                ErrorCode::NotCallable,
                                format!("Expression of type `{callee_ty}` is not callable"),
                                callee.span.clone(),
                            )
                            .with_context(
                                ErrorContext::new().with("found_type", format!("{callee_ty}")),
                            ),
                        );
                        TptType::Unknown
                    }
                }
            }

            ExprKind::Index { expr: obj, indices } => {
                let obj_ty = self.infer_expr(obj, env);
                for idx in indices {
                    let idx_ty = self.infer_expr(idx, env);
                    if !matches!(
                        idx_ty,
                        TptType::I8
                            | TptType::I16
                            | TptType::I32
                            | TptType::I64
                            | TptType::U8
                            | TptType::U16
                            | TptType::U32
                            | TptType::U64
                            | TptType::Index
                            | TptType::Unknown
                    ) {
                        self.errors.push(
                            TptError::new(
                                ErrorCode::InvalidIndexType,
                                format!("Index must be an integer type, found `{idx_ty}`"),
                                idx.span.clone(),
                            )
                            .with_context(
                                ErrorContext::new().with("found_type", format!("{idx_ty}")),
                            ),
                        );
                    }
                }
                // Subscript of a tensor reduces rank by number of full indices.
                match &obj_ty {
                    TptType::Tensor { dtype, shape } => {
                        if indices.len() >= shape.len() {
                            *dtype.clone() // fully indexed → scalar element
                        } else {
                            let remaining = shape[indices.len()..].to_vec();
                            TptType::Tensor {
                                dtype: dtype.clone(),
                                shape: remaining,
                            }
                        }
                    }
                    TptType::Slice(elem) | TptType::Array(elem, _) => *elem.clone(),
                    _ => TptType::Unknown,
                }
            }

            ExprKind::Block(block) => {
                // Type of a block is the type of its last expression-statement,
                // or Unit if the block is empty / last statement is non-expression.
                let mut env2 = env.clone();
                let mut last_ty = TptType::Unit;
                for stmt in &block.stmts {
                    match stmt {
                        Stmt::Expr(e) => {
                            last_ty = self.infer_expr(e, &env2);
                        }
                        other => {
                            self.check_stmt(other, &mut env2, &TptType::Unknown);
                            last_ty = TptType::Unit;
                        }
                    }
                }
                last_ty
            }

            ExprKind::If {
                condition,
                then_block,
                else_branch,
            } => {
                let cond_ty = self.infer_expr(condition, env);
                if cond_ty != TptType::Bool && cond_ty != TptType::Unknown {
                    self.errors.push(
                        TptError::new(
                            ErrorCode::TypeError,
                            format!("If condition must be bool, found `{cond_ty}`"),
                            condition.span.clone(),
                        )
                        .with_context(
                            ErrorContext::new()
                                .with("expected_type", "bool")
                                .with("found_type", format!("{cond_ty}")),
                        ),
                    );
                }
                let mut env2 = env.clone();
                self.check_block(then_block, &mut env2, &TptType::Unknown);
                if let Some(eb) = else_branch {
                    self.infer_expr(eb, env)
                } else {
                    TptType::Unit
                }
            }

            ExprKind::For {
                var,
                iterable,
                body,
            } => {
                let iter_ty = self.infer_expr(iterable, env);
                // Infer element type from the iterable.
                let elem_ty = match &iter_ty {
                    TptType::Slice(e) | TptType::Array(e, _) => *e.clone(),
                    TptType::Tensor { dtype, .. } => *dtype.clone(),
                    TptType::DataLoader => TptType::Unknown,
                    _ => TptType::Unknown,
                };
                let mut loop_env = env.clone();
                loop_env.define(var, elem_ty);
                self.check_block(body, &mut loop_env, &TptType::Unknown);
                TptType::Unit
            }

            ExprKind::While { condition, body } => {
                let cond_ty = self.infer_expr(condition, env);
                if cond_ty != TptType::Bool && cond_ty != TptType::Unknown {
                    self.errors.push(
                        TptError::new(
                            ErrorCode::TypeError,
                            format!("While condition must be bool, found `{cond_ty}`"),
                            condition.span.clone(),
                        )
                        .with_context(
                            ErrorContext::new()
                                .with("expected_type", "bool")
                                .with("found_type", format!("{cond_ty}")),
                        ),
                    );
                }
                let mut env2 = env.clone();
                self.check_block(body, &mut env2, &TptType::Unknown);
                TptType::Unit
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn collect_call_arg_types(
        &mut self,
        args: &[CallArg],
        env: &TypeEnv,
    ) -> (Vec<TptType>, Vec<(String, TptType)>) {
        let mut positional = Vec::new();
        let mut named = Vec::new();
        for arg in args {
            match arg {
                CallArg::Positional(e) => {
                    positional.push(self.infer_expr(e, env));
                }
                CallArg::Named { name, value, .. } => {
                    let ty = self.infer_expr(value, env);
                    named.push((name.clone(), ty));
                }
            }
        }
        (positional, named)
    }

    fn infer_binop_type(&mut self, op: &BinOp, lt: &TptType, rt: &TptType, span: &Span) -> TptType {
        match op {
            // Comparison → bool (or tensor of bool for tensor operands)
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                if lt.is_tensor() || rt.is_tensor() {
                    if let TptType::Tensor { shape, .. } = lt {
                        return TptType::Tensor {
                            dtype: Box::new(TptType::Bool),
                            shape: shape.clone(),
                        };
                    }
                }
                TptType::Bool
            }
            // Logical → bool
            BinOp::And | BinOp::Or => {
                if lt != &TptType::Bool && *lt != TptType::Unknown {
                    self.errors.push(
                        TptError::new(
                            ErrorCode::TypeError,
                            format!("`{op}` requires bool operands, left is `{lt}`"),
                            span.clone(),
                        )
                        .with_context(
                            ErrorContext::new()
                                .with("expected_type", "bool")
                                .with("found_type", format!("{lt}")),
                        ),
                    );
                }
                TptType::Bool
            }
            // Range → Slice<index>
            BinOp::Range | BinOp::RangeEq => TptType::Slice(Box::new(TptType::Index)),
            // Arithmetic: for tensors, broadcast; for scalars, usual rules.
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                // If either side is a tensor, output is tensor with same shape/dtype.
                if lt.is_tensor() {
                    return lt.clone();
                }
                if rt.is_tensor() {
                    return rt.clone();
                }
                if !lt.compatible(rt) && *lt != TptType::Unknown && *rt != TptType::Unknown {
                    self.errors.push(
                        TptError::new(
                            ErrorCode::TypeError,
                            format!("Type mismatch: `{lt}` {op} `{rt}`"),
                            span.clone(),
                        )
                        .with_context(
                            ErrorContext::new()
                                .with("expected_type", format!("{lt}"))
                                .with("found_type", format!("{rt}")),
                        )
                        .with_suggestion("Use `tpt.cast(x, dtype=...)` to align types"),
                    );
                }
                if *lt != TptType::Unknown {
                    lt.clone()
                } else {
                    rt.clone()
                }
            }
        }
    }

    fn infer_field_access(&mut self, obj_ty: &TptType, field: &str, span: &Span) -> TptType {
        match (obj_ty, field) {
            // tensor.shape → [index]
            (TptType::Tensor { .. }, "shape") => TptType::Slice(Box::new(TptType::Index)),
            // tensor.dtype / tensor.device
            (TptType::Tensor { .. }, "dtype") => TptType::Unknown,
            (TptType::Tensor { .. }, "device") => TptType::Index,
            // Unknown → don't error (field access on Unknown is valid)
            (TptType::Unknown, _) => TptType::Unknown,
            // Anything else: warn but allow for forward compat.
            (ty, field_name) => {
                self.errors.push(
                    TptError::new(
                        ErrorCode::TypeError,
                        format!("Type `{ty}` has no field `{field_name}`"),
                        span.clone(),
                    )
                    .with_context(
                        ErrorContext::new()
                            .with("type_name", format!("{ty}"))
                            .with("field", field_name),
                    ),
                );
                TptType::Unknown
            }
        }
    }

    fn infer_method_call(
        &mut self,
        obj_ty: &TptType,
        method: &str,
        arg_tys: &[TptType],
        named_tys: &[(String, TptType)],
        span: &Span,
    ) -> TptType {
        let named_refs: Vec<(&str, TptType)> = named_tys
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        match (obj_ty, method) {
            // `loss.backward()` → ()
            (TptType::Tensor { .. }, "backward") => TptType::Unit,
            // `model.forward(...)` → Unknown (depends on model architecture)
            (TptType::Model, "forward") => TptType::Unknown,
            // `model.step()` → ()
            (TptType::Model | TptType::Tensor { .. }, "step") => TptType::Unit,

            // For any `tpt` module method on the tpt object — dispatched by caller.
            (TptType::Unknown, name) => infer_builtin(name, arg_tys, &named_refs),

            _ => {
                // Try builtin registry as a fallback (covers model/tensor methods).
                let result = infer_builtin(method, arg_tys, &named_refs);
                if result == TptType::Unknown {
                    self.errors.push(
                        TptError::new(
                            ErrorCode::UndefinedOperation,
                            format!("Method `{method}` not found on type `{obj_ty}`"),
                            span.clone(),
                        )
                        .with_context(
                            ErrorContext::new()
                                .with("method_name", method)
                                .with("type_name", format!("{obj_ty}")),
                        ),
                    );
                }
                result
            }
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Classify a type mismatch into the most specific [`ErrorCode`] and build
/// the corresponding [`ErrorContext`] so the auto-fix engine can act on it.
fn mismatch_code_and_ctx(
    found: &TptType,
    expected: &TptType,
    var_name: &str,
) -> (ErrorCode, ErrorContext) {
    match (found, expected) {
        (
            TptType::Tensor {
                dtype: d1,
                shape: s1,
            },
            TptType::Tensor {
                dtype: d2,
                shape: s2,
            },
        ) => {
            if d1 != d2 {
                (
                    ErrorCode::DtypeMismatch,
                    ErrorContext::new()
                        .with("var_name", var_name)
                        .with("expected_dtype", format!("{d2}"))
                        .with("found_dtype", format!("{d1}")),
                )
            } else {
                let fmt_shape = |s: &[DimVal]| {
                    format!(
                        "[{}]",
                        s.iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                (
                    ErrorCode::ShapeMismatch,
                    ErrorContext::new()
                        .with("var_name", var_name)
                        .with("expected_shape", fmt_shape(s2))
                        .with("found_shape", fmt_shape(s1)),
                )
            }
        }
        _ => (
            ErrorCode::TypeError,
            ErrorContext::new()
                .with("var_name", var_name)
                .with("expected_type", format!("{expected}"))
                .with("found_type", format!("{found}")),
        ),
    }
}

fn collect_dim_names(ty: &Type, out: &mut HashMap<String, ()>) {
    if let Type::Tensor { dims, .. } = ty {
        for d in dims {
            if let Dim::Named(name) = d {
                out.insert(name.clone(), ());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the type checker over a parsed `Program`.
/// Returns `Ok(TypeChecker)` with the populated `errors` and `type_map` fields.
pub fn type_check(program: &Program) -> TypeChecker {
    let mut checker = TypeChecker::new();
    checker.check_program(program);
    checker
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ErrorCode;
    use crate::{lexer::tokenize, parser::parse};

    fn check(src: &str) -> TypeChecker {
        let prog = parse(tokenize(src).unwrap()).unwrap();
        type_check(&prog)
    }

    #[test]
    fn test_simple_fn_no_errors() {
        let c = check("fn add(a: f32, b: f32) -> f32 { return a + b }");
        assert!(c.errors.is_empty(), "{:?}", c.errors);
    }

    #[test]
    fn test_undefined_variable() {
        let c = check("fn f() { let x = y }");
        assert!(c
            .errors
            .iter()
            .any(|e| e.code == ErrorCode::UndefinedVariable));
    }

    #[test]
    fn test_type_annotation_in_type_map() {
        let c = check("fn f() { let x = 42 }");
        assert!(!c.type_map.is_empty());
    }

    #[test]
    fn test_tensor_param_type() {
        let c = check("fn f(x: Tensor[f32, m, n]) { }");
        assert!(c.errors.is_empty(), "{:?}", c.errors);
    }

    #[test]
    fn test_return_type_mismatch() {
        // f32 return declared, but we return an i64 literal
        let c = check("fn f() -> f32 { return 42 }");
        // 42 is inferred as i64; f32 declared → mismatch
        assert!(
            c.errors
                .iter()
                .any(|e| e.code == ErrorCode::ReturnTypeMismatch),
            "expected RETURN_TYPE_MISMATCH, got: {:?}",
            c.errors
        );
    }

    #[test]
    fn test_if_condition_not_bool() {
        let c = check("fn f() { if 42 { } }");
        assert!(
            c.errors.iter().any(|e| e.code == ErrorCode::TypeError),
            "expected TYPE_ERROR for non-bool condition"
        );
    }

    #[test]
    fn test_let_type_mismatch() {
        let c = check("fn f() { let x: f32 = true }");
        assert!(c.errors.iter().any(|e| e.code == ErrorCode::TypeError));
    }

    #[test]
    fn test_for_range_loop() {
        let c = check("fn f() { for i in 0..10 { } }");
        assert!(c.errors.is_empty(), "{:?}", c.errors);
    }

    #[test]
    fn test_full_matmul_fn() {
        let src = r#"
@doc("Multiply two matrices")
@constraint("a.shape[1] == b.shape[0]", error="Inner dimensions must match")
@complexity("O(m * n * k)")
@differentiable(true)
@requires_gpu(true)
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    let result = tpt.zeros([m, n], dtype=f32)
    tpt.gemm(a, b, result)
    return result
}
"#;
        let c = check(src);
        assert!(c.errors.is_empty(), "errors: {:?}", c.errors);
        // Should have produced type annotations
        assert!(!c.type_map.is_empty());
    }

    #[test]
    fn test_unary_not_on_bool() {
        let c = check("fn f(b: bool) { let x = !b }");
        assert!(c.errors.is_empty(), "{:?}", c.errors);
    }

    #[test]
    fn test_index_expr() {
        let c = check("fn f(x: Tensor[f32, m, n]) { let v = x[0] }");
        assert!(c.errors.is_empty(), "{:?}", c.errors);
        // x[0] reduces rank; should be inferred as Tensor[f32, n]
        let indexed = c
            .type_map
            .iter()
            .find(|a| matches!(&a.ty, TptType::Tensor { .. }));
        assert!(indexed.is_some());
    }

    #[test]
    fn test_constraint_extraction_and_check() {
        // Statically false constraint should produce an error.
        let src = r#"
@constraint("0 == 1", error="always false")
fn broken() {}
"#;
        let c = check(src);
        assert!(
            c.errors
                .iter()
                .any(|e| e.code == ErrorCode::ConstraintViolation),
            "expected CONSTRAINT_VIOLATION"
        );
    }

    #[test]
    fn test_compile_str_roundtrip() {
        let prog = crate::compile_str(
            r#"import tpt
fn f(x: Tensor[f32, batch, seq]) -> Tensor[f32, batch, seq] {
    let y = tpt.relu(x)
    return y
}"#,
        )
        .expect("compile_str failed");
        let c = type_check(&prog);
        assert!(c.errors.is_empty(), "{:?}", c.errors);
    }
}
