use crate::lexer::Span;

// ---------------------------------------------------------------------------
// Top-level program
// ---------------------------------------------------------------------------

/// A parsed TPT Script source file.
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Top-level declaration.
#[derive(Debug, Clone)]
pub enum Item {
    Import(ImportDecl),
    Function(FunctionDecl),
    TypeAlias(TypeDecl),
}

impl Item {
    pub fn span(&self) -> &Span {
        match self {
            Item::Import(d) => &d.span,
            Item::Function(d) => &d.span,
            Item::TypeAlias(d) => &d.span,
        }
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// `import tpt::nn` or `import model::transformer as tr`
///
/// Note: both `.` and `::` are accepted as path separators during parsing so
/// that `import tpt.introspect` (spec §6.2 examples) and
/// `import model::transformer` (spec §12.1) both work.
#[derive(Debug, Clone)]
pub struct ImportDecl {
    /// Segments of the module path, e.g. `["tpt", "introspect"]`.
    pub path: Vec<String>,
    /// Optional `as <ident>` alias.
    pub alias: Option<String>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Function declaration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Type alias declaration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Annotations  (@doc("..."), @requires_gpu(true), ...)
// ---------------------------------------------------------------------------

/// `@name` or `@name(arg, key=value, ...)`
#[derive(Debug, Clone)]
pub struct Annotation {
    pub name: String,
    pub args: Vec<AnnotationArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AnnotationArg {
    /// `key = value`
    Named {
        key: String,
        value: AnnotationValue,
        span: Span,
    },
    /// bare positional value
    Positional { value: AnnotationValue, span: Span },
}

impl AnnotationArg {
    pub fn span(&self) -> &Span {
        match self {
            AnnotationArg::Named { span, .. } => span,
            AnnotationArg::Positional { span, .. } => span,
        }
    }
}

/// The literal kinds that may appear inside an annotation argument list.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Type {
    /// Named primitive: `i32`, `f32`, `bool`, `index`, etc.
    Primitive(PrimitiveType, Span),
    /// `Tensor[dtype, dim, dim, ...]`
    Tensor {
        dtype: PrimitiveType,
        dims: Vec<Dim>,
        span: Span,
    },
    /// `(T1, T2, ...)` – zero-element `()` is the unit / void type.
    Tuple(Vec<Type>, Span),
    /// `[T; N]` – fixed-size array.
    Array {
        elem: Box<Type>,
        size: i64,
        span: Span,
    },
    /// `[T]` – dynamically-sized slice.
    Slice(Box<Type>, Span),
    /// Any other identifier used as a type name (user-defined aliases, `Model`,
    /// `DataLoader`, `Optimizer`, etc.).
    Named(String, Span),
}

impl Type {
    pub fn span(&self) -> &Span {
        match self {
            Type::Primitive(_, s) => s,
            Type::Tensor { span, .. } => span,
            Type::Tuple(_, s) => s,
            Type::Array { span, .. } => span,
            Type::Slice(_, s) => s,
            Type::Named(_, s) => s,
        }
    }
}

/// Primitive numeric / logical types (spec §5.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
    Index,
}

impl PrimitiveType {
    /// Try to parse an identifier string as a primitive type name.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "i8" => PrimitiveType::I8,
            "i16" => PrimitiveType::I16,
            "i32" => PrimitiveType::I32,
            "i64" => PrimitiveType::I64,
            "u8" => PrimitiveType::U8,
            "u16" => PrimitiveType::U16,
            "u32" => PrimitiveType::U32,
            "u64" => PrimitiveType::U64,
            "f16" => PrimitiveType::F16,
            "bf16" => PrimitiveType::Bf16,
            "f32" => PrimitiveType::F32,
            "f64" => PrimitiveType::F64,
            "bool" => PrimitiveType::Bool,
            "index" => PrimitiveType::Index,
            _ => return None,
        })
    }
}

/// A tensor dimension: either a concrete integer, a named symbolic variable,
/// or `*` for a fully dynamic dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Concrete(i64),
    Named(String),
    Dynamic,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    Return(ReturnStmt),
    Break(Span),
    Continue(Span),
    /// Expression used as a statement (call, method call, etc.).
    Expr(Expr),
}

impl Stmt {
    pub fn span(&self) -> &Span {
        match self {
            Stmt::Let(s) => &s.span,
            Stmt::Return(s) => &s.span,
            Stmt::Break(s) => s,
            Stmt::Continue(s) => s,
            Stmt::Expr(e) => &e.span,
        }
    }
}

/// `let name [: Type] = expr`
#[derive(Debug, Clone)]
pub struct LetStmt {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expr,
    pub span: Span,
}

/// `return [expr]`
#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),

    /// Variable / function reference.
    Ident(String),

    /// `[expr, expr, ...]` — array / list / tensor literal.
    ArrayLit(Vec<Expr>),

    /// `(expr)` — parenthesised expression.
    Paren(Box<Expr>),

    /// Infix binary operation.
    BinaryOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Prefix unary operation.
    UnaryOp {
        op: UnOp,
        operand: Box<Expr>,
    },

    /// `expr.field`
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },

    /// `expr.method(args)` — method call.
    MethodCall {
        expr: Box<Expr>,
        method: String,
        args: Vec<CallArg>,
    },

    /// `callee(args)` — free-function call or any callable expression.
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
    },

    /// `expr[i, j, ...]` — subscript / index access.
    Index {
        expr: Box<Expr>,
        indices: Vec<Expr>,
    },

    /// `if cond { ... } [else { ... }]` or `else if ...`
    If {
        condition: Box<Expr>,
        then_block: Block,
        else_branch: Option<Box<Expr>>,
    },

    /// `for var in iterable { ... }`
    For {
        var: String,
        iterable: Box<Expr>,
        body: Block,
    },

    /// `while cond { ... }`
    While {
        condition: Box<Expr>,
        body: Block,
    },

    /// Block used as an expression: `{ stmts... }`
    Block(Block),
}

// ---------------------------------------------------------------------------
// Binary and unary operators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    // Logical
    And,
    Or,
    // Range
    Range,   // ..
    RangeEq, // ..=
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Range => write!(f, ".."),
            BinOp::RangeEq => write!(f, "..="),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    /// Unary `-`
    Neg,
    /// Unary `!`
    Not,
}

// ---------------------------------------------------------------------------
// Call arguments
// ---------------------------------------------------------------------------

/// A single argument in a call expression: either positional or named (`key=expr`).
#[derive(Debug, Clone)]
pub enum CallArg {
    Named {
        name: String,
        value: Expr,
        span: Span,
    },
    Positional(Expr),
}

impl CallArg {
    pub fn span(&self) -> &Span {
        match self {
            CallArg::Named { span, .. } => span,
            CallArg::Positional(e) => &e.span,
        }
    }
}
