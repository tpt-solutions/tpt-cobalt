use thiserror::Error;

use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("expected {expected} but found '{found}' at {span}")]
    Expected {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("expected identifier but found '{found}' at {span}")]
    ExpectedIdent { found: String, span: Span },

    #[error("unexpected token '{found}' at {span}")]
    Unexpected { found: String, span: Span },

    #[error("integer literal too large at {span}")]
    IntLitTooLarge { span: Span },
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct Parser {
    tokens: Vec<Token>,
    /// Index of the current token. Never exceeds `tokens.len() - 1` (last
    /// token is always `Eof`).
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        // Guarantee there is always an Eof sentinel.
        assert!(!tokens.is_empty(), "token stream must be non-empty");
        Self { tokens, pos: 0 }
    }

    // -----------------------------------------------------------------------
    // Cursor helpers
    // -----------------------------------------------------------------------

    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn current_kind(&self) -> &TokenKind {
        &self.current().kind
    }

    fn current_span(&self) -> Span {
        self.current().span.clone()
    }

    /// Advance past the current token and return a clone of it.
    fn bump(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    /// Peek at the token `offset` positions ahead (0 = current).
    fn peek_ahead(&self, offset: usize) -> &TokenKind {
        let idx = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[idx].kind
    }

    /// Return true if the current token matches `kind` **exactly**.
    fn at(&self, kind: &TokenKind) -> bool {
        self.current_kind() == kind
    }

    /// Consume the current token if it matches `kind`; otherwise error.
    fn eat(&mut self, kind: TokenKind) -> Result<Span, ParseError> {
        if self.current_kind() == &kind {
            Ok(self.bump().span)
        } else {
            Err(ParseError::Expected {
                expected: kind.to_string(),
                found: self.current_kind().to_string(),
                span: self.current_span(),
            })
        }
    }

    /// Consume an identifier token and return its name.
    fn eat_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(name) => {
                let span = self.bump().span;
                Ok((name, span))
            }
            // `as` is used as a contextual keyword (not fully reserved).
            _ => Err(ParseError::ExpectedIdent {
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Top-level: program
    // -----------------------------------------------------------------------

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        // Collect leading annotations.
        let mut annotations = Vec::new();
        while self.at(&TokenKind::At) {
            annotations.push(self.parse_annotation()?);
        }

        match self.current_kind() {
            TokenKind::KwImport => {
                if !annotations.is_empty() {
                    return Err(ParseError::Unexpected {
                        found: "annotations before import".to_string(),
                        span: self.current_span(),
                    });
                }
                Ok(Item::Import(self.parse_import()?))
            }
            TokenKind::KwFn => Ok(Item::Function(self.parse_fn(annotations)?)),
            TokenKind::KwType => Ok(Item::TypeAlias(self.parse_type_decl(annotations)?)),
            _ => Err(ParseError::Unexpected {
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Import declaration
    // -----------------------------------------------------------------------

    fn parse_import(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.eat(TokenKind::KwImport)?;

        // module_path = IDENT { ("::" | ".") IDENT }
        let mut path = Vec::new();
        let (first, _) = self.eat_ident()?;
        path.push(first);

        // Accept both `::` and `.` as path separators per spec examples.
        while let TokenKind::ColonColon | TokenKind::Dot = self.current_kind() {
            self.bump();
            let (seg, _) = self.eat_ident()?;
            path.push(seg);
        }

        // Optional `as <ident>` alias.
        let alias = if matches!(self.current_kind(), TokenKind::Ident(s) if s == "as") {
            self.bump(); // consume `as`
            let (name, _) = self.eat_ident()?;
            Some(name)
        } else {
            None
        };

        let span = start.merge(&self.tokens[self.pos.saturating_sub(1)].span);
        Ok(ImportDecl { path, alias, span })
    }

    // -----------------------------------------------------------------------
    // Function declaration
    // -----------------------------------------------------------------------

    fn parse_fn(&mut self, annotations: Vec<Annotation>) -> Result<FunctionDecl, ParseError> {
        let fn_span = self.eat(TokenKind::KwFn)?;
        let (name, _) = self.eat_ident()?;
        self.eat(TokenKind::LParen)?;

        let params = if self.at(&TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_param_list()?
        };
        self.eat(TokenKind::RParen)?;

        let return_type = if self.at(&TokenKind::Arrow) {
            self.bump();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let span = fn_span.merge(&body.span);
        Ok(FunctionDecl {
            annotations,
            name,
            params,
            return_type,
            body,
            span,
        })
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        params.push(self.parse_param()?);
        while self.at(&TokenKind::Comma) {
            self.bump();
            // Trailing comma before `)` is allowed.
            if self.at(&TokenKind::RParen) {
                break;
            }
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let (name, name_span) = self.eat_ident()?;
        self.eat(TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let span = name_span.merge(ty.span());
        Ok(Param { name, ty, span })
    }

    // -----------------------------------------------------------------------
    // Type alias declaration
    // -----------------------------------------------------------------------

    fn parse_type_decl(&mut self, annotations: Vec<Annotation>) -> Result<TypeDecl, ParseError> {
        let type_span = self.eat(TokenKind::KwType)?;
        let (name, _) = self.eat_ident()?;
        self.eat(TokenKind::Eq)?;
        let ty = self.parse_type()?;
        let span = type_span.merge(ty.span());
        Ok(TypeDecl {
            annotations,
            name,
            ty,
            span,
        })
    }

    // -----------------------------------------------------------------------
    // Annotations
    // -----------------------------------------------------------------------

    fn parse_annotation(&mut self) -> Result<Annotation, ParseError> {
        let at_span = self.eat(TokenKind::At)?;
        let (name, _) = self.eat_ident()?;

        let args = if self.at(&TokenKind::LParen) {
            self.bump();
            let mut args = Vec::new();
            while !self.at(&TokenKind::RParen) && !self.at(&TokenKind::Eof) {
                args.push(self.parse_annotation_arg()?);
                if self.at(&TokenKind::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.eat(TokenKind::RParen)?;
            args
        } else {
            Vec::new()
        };

        let end_span = self.tokens[self.pos.saturating_sub(1)].span.clone();
        let span = at_span.merge(&end_span);
        Ok(Annotation { name, args, span })
    }

    fn parse_annotation_arg(&mut self) -> Result<AnnotationArg, ParseError> {
        let start_span = self.current_span();

        // Named arg: `IDENT "=" ann_value`
        if matches!(self.current_kind(), TokenKind::Ident(_))
            && self.peek_ahead(1) == &TokenKind::Eq
        {
            let (key, key_span) = self.eat_ident()?;
            self.eat(TokenKind::Eq)?;
            let (value, val_span) = self.parse_annotation_value()?;
            let span = key_span.merge(&val_span);
            return Ok(AnnotationArg::Named { key, value, span });
        }

        // Positional arg
        let (value, val_span) = self.parse_annotation_value()?;
        let span = start_span.merge(&val_span);
        Ok(AnnotationArg::Positional { value, span })
    }

    fn parse_annotation_value(&mut self) -> Result<(AnnotationValue, Span), ParseError> {
        match self.current_kind().clone() {
            TokenKind::StringLit(s) => {
                let span = self.bump().span;
                Ok((AnnotationValue::String(s), span))
            }
            TokenKind::IntLit(n) => {
                let span = self.bump().span;
                Ok((AnnotationValue::Int(n), span))
            }
            TokenKind::FloatLit(n) => {
                let span = self.bump().span;
                Ok((AnnotationValue::Float(n), span))
            }
            TokenKind::BoolLit(b) => {
                let span = self.bump().span;
                Ok((AnnotationValue::Bool(b), span))
            }
            _ => Err(ParseError::Expected {
                expected: "annotation value (string, int, float, or bool)".to_string(),
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        match self.current_kind().clone() {
            // Tuple or unit type: `(...)` / `()`
            TokenKind::LParen => self.parse_tuple_type(),

            // Array or slice: `[T; N]` / `[T]`
            TokenKind::LBracket => self.parse_array_or_slice_type(),

            // Identifier: primitive type, `Tensor[...]`, or named alias.
            TokenKind::Ident(name) => {
                let span = self.bump().span;

                // Tensor type
                if name == "Tensor" {
                    return self.parse_tensor_type_tail(span);
                }

                // Primitive type
                if let Some(prim) = PrimitiveType::from_str(&name) {
                    return Ok(Type::Primitive(prim, span));
                }

                // Named alias (Model, DataLoader, user-defined, etc.)
                Ok(Type::Named(name, span))
            }

            _ => Err(ParseError::Expected {
                expected: "type".to_string(),
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    /// Parse `"[" dtype "," dim { "," dim } "]"` after `Tensor` was consumed.
    fn parse_tensor_type_tail(&mut self, tensor_span: Span) -> Result<Type, ParseError> {
        self.eat(TokenKind::LBracket)?;

        // dtype must be a primitive type identifier
        let dtype = self.parse_dtype()?;
        self.eat(TokenKind::Comma)?;

        let mut dims = Vec::new();
        dims.push(self.parse_dim()?);
        while self.at(&TokenKind::Comma) {
            self.bump();
            // Allow trailing comma before `]`
            if self.at(&TokenKind::RBracket) {
                break;
            }
            dims.push(self.parse_dim()?);
        }

        let end_span = self.eat(TokenKind::RBracket)?;
        let span = tensor_span.merge(&end_span);
        Ok(Type::Tensor { dtype, dims, span })
    }

    fn parse_dtype(&mut self) -> Result<PrimitiveType, ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(name) => {
                if let Some(prim) = PrimitiveType::from_str(&name) {
                    self.bump();
                    Ok(prim)
                } else {
                    Err(ParseError::Expected {
                        expected: "primitive dtype (e.g. f32, i64)".to_string(),
                        found: name,
                        span: self.current_span(),
                    })
                }
            }
            _ => Err(ParseError::Expected {
                expected: "dtype identifier".to_string(),
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    fn parse_dim(&mut self) -> Result<Dim, ParseError> {
        match self.current_kind().clone() {
            TokenKind::IntLit(n) => {
                self.bump();
                Ok(Dim::Concrete(n))
            }
            TokenKind::Ident(s) => {
                self.bump();
                Ok(Dim::Named(s))
            }
            TokenKind::Star => {
                self.bump();
                Ok(Dim::Dynamic)
            }
            _ => Err(ParseError::Expected {
                expected: "dimension (integer, identifier, or *)".to_string(),
                found: self.current_kind().to_string(),
                span: self.current_span(),
            }),
        }
    }

    fn parse_tuple_type(&mut self) -> Result<Type, ParseError> {
        let lp_span = self.eat(TokenKind::LParen)?;

        // Unit type `()`
        if self.at(&TokenKind::RParen) {
            let rp_span = self.bump().span;
            return Ok(Type::Tuple(vec![], lp_span.merge(&rp_span)));
        }

        let mut types = Vec::new();
        types.push(self.parse_type()?);

        // Require at least one comma to distinguish from parenthesised type.
        while self.at(&TokenKind::Comma) {
            self.bump();
            if self.at(&TokenKind::RParen) {
                break;
            } // trailing comma
            types.push(self.parse_type()?);
        }

        let rp_span = self.eat(TokenKind::RParen)?;
        Ok(Type::Tuple(types, lp_span.merge(&rp_span)))
    }

    fn parse_array_or_slice_type(&mut self) -> Result<Type, ParseError> {
        let lb_span = self.eat(TokenKind::LBracket)?;
        let elem = self.parse_type()?;

        if self.at(&TokenKind::Semicolon) {
            // Array type: `[T; N]`
            self.bump();
            let (n, _n_span) = match self.current_kind().clone() {
                TokenKind::IntLit(n) => (n, self.bump().span),
                _ => {
                    return Err(ParseError::Expected {
                        expected: "integer size for array type".to_string(),
                        found: self.current_kind().to_string(),
                        span: self.current_span(),
                    })
                }
            };
            let rb_span = self.eat(TokenKind::RBracket)?;
            let span = lb_span.merge(&rb_span);
            Ok(Type::Array {
                elem: Box::new(elem),
                size: n,
                span,
            })
        } else {
            // Slice type: `[T]`
            let rb_span = self.eat(TokenKind::RBracket)?;
            Ok(Type::Slice(Box::new(elem), lb_span.merge(&rb_span)))
        }
    }

    // -----------------------------------------------------------------------
    // Block
    // -----------------------------------------------------------------------

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let lb_span = self.eat(TokenKind::LBrace)?;
        let mut stmts = Vec::new();

        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
            // Optional semicolons between statements.
            while self.at(&TokenKind::Semicolon) {
                self.bump();
            }
        }

        let rb_span = self.eat(TokenKind::RBrace)?;
        Ok(Block {
            stmts,
            span: lb_span.merge(&rb_span),
        })
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.current_kind() {
            TokenKind::KwLet => self.parse_let_stmt().map(Stmt::Let),
            TokenKind::KwReturn => self.parse_return_stmt().map(Stmt::Return),
            TokenKind::KwBreak => {
                let s = self.bump().span;
                Ok(Stmt::Break(s))
            }
            TokenKind::KwContinue => {
                let s = self.bump().span;
                Ok(Stmt::Continue(s))
            }
            _ => self.parse_expr(0).map(Stmt::Expr),
        }
    }

    fn parse_let_stmt(&mut self) -> Result<LetStmt, ParseError> {
        let let_span = self.eat(TokenKind::KwLet)?;
        let (name, _) = self.eat_ident()?;

        let ty = if self.at(&TokenKind::Colon) {
            self.bump();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.eat(TokenKind::Eq)?;
        let value = self.parse_expr(0)?;
        let span = let_span.merge(&value.span);
        Ok(LetStmt {
            name,
            ty,
            value,
            span,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<ReturnStmt, ParseError> {
        let ret_span = self.eat(TokenKind::KwReturn)?;

        // A `return` with no value is indicated by a `}`, `;`, or EOF next.
        let (value, span) = if self.at(&TokenKind::RBrace)
            || self.at(&TokenKind::Semicolon)
            || self.at(&TokenKind::Eof)
        {
            (None, ret_span.clone())
        } else {
            let e = self.parse_expr(0)?;
            let span = ret_span.merge(&e.span);
            (Some(e), span)
        };

        Ok(ReturnStmt { value, span })
    }

    // -----------------------------------------------------------------------
    // Expression parsing — Pratt / precedence-climbing hybrid
    //
    // Precedence levels (higher = tighter binding):
    //   0  — `..` `..=`  (range, lowest)
    //   1  — `||`
    //   2  — `&&`
    //   3  — `==` `!=` `<` `>` `<=` `>=`
    //   4  — `+` `-`
    //   5  — `*` `/` `%`
    //   postfix — field access, method call, call, index (handled separately)
    //   7  — unary `!` `-`
    // -----------------------------------------------------------------------

    fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let (op, prec, right_assoc) = match self.current_kind() {
                TokenKind::DotDot => (BinOp::Range, 0u8, false),
                TokenKind::DotDotEq => (BinOp::RangeEq, 0, false),
                TokenKind::PipePipe => (BinOp::Or, 1, false),
                TokenKind::AmpAmp => (BinOp::And, 2, false),
                TokenKind::EqEq => (BinOp::Eq, 3, false),
                TokenKind::BangEq => (BinOp::Ne, 3, false),
                TokenKind::Lt => (BinOp::Lt, 3, false),
                TokenKind::Gt => (BinOp::Gt, 3, false),
                TokenKind::LtEq => (BinOp::Le, 3, false),
                TokenKind::GtEq => (BinOp::Ge, 3, false),
                TokenKind::Plus => (BinOp::Add, 4, false),
                TokenKind::Minus => (BinOp::Sub, 4, false),
                TokenKind::Star => (BinOp::Mul, 5, false),
                TokenKind::Slash => (BinOp::Div, 5, false),
                TokenKind::Percent => (BinOp::Mod, 5, false),
                _ => break,
            };

            if prec < min_prec {
                break;
            }
            self.bump();

            let next_min = if right_assoc { prec } else { prec + 1 };
            let right = self.parse_expr(next_min)?;
            let span = left.span.merge(&right.span);
            left = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.current_kind() {
            TokenKind::Bang => {
                let span_start = self.bump().span;
                let operand = self.parse_unary()?;
                let span = span_start.merge(&operand.span);
                Ok(Expr {
                    kind: ExprKind::UnaryOp {
                        op: UnOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::Minus => {
                let span_start = self.bump().span;
                let operand = self.parse_unary()?;
                let span = span_start.merge(&operand.span);
                Ok(Expr {
                    kind: ExprKind::UnaryOp {
                        op: UnOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            _ => self.parse_postfix(),
        }
    }

    // -----------------------------------------------------------------------
    // Postfix: field access, method call, free call, subscript
    // -----------------------------------------------------------------------

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.current_kind() {
                TokenKind::Dot => {
                    self.bump();
                    let (field, field_span) = self.eat_ident()?;

                    if self.at(&TokenKind::LParen) {
                        // Method call: `expr.method(args)`
                        self.bump();
                        let args = self.parse_call_args()?;
                        let end = self.eat(TokenKind::RParen)?;
                        let span = expr.span.merge(&end);
                        expr = Expr {
                            kind: ExprKind::MethodCall {
                                expr: Box::new(expr),
                                method: field,
                                args,
                            },
                            span,
                        };
                    } else {
                        // Field access: `expr.field`
                        let span = expr.span.merge(&field_span);
                        expr = Expr {
                            kind: ExprKind::FieldAccess {
                                expr: Box::new(expr),
                                field,
                            },
                            span,
                        };
                    }
                }

                TokenKind::LParen => {
                    // Free call: `callee(args)`
                    self.bump();
                    let args = self.parse_call_args()?;
                    let end = self.eat(TokenKind::RParen)?;
                    let span = expr.span.merge(&end);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }

                TokenKind::LBracket => {
                    // Subscript: `expr[i, j, ...]`
                    self.bump();
                    let mut indices = vec![self.parse_expr(0)?];
                    while self.at(&TokenKind::Comma) {
                        self.bump();
                        if self.at(&TokenKind::RBracket) {
                            break;
                        }
                        indices.push(self.parse_expr(0)?);
                    }
                    let end = self.eat(TokenKind::RBracket)?;
                    let span = expr.span.merge(&end);
                    expr = Expr {
                        kind: ExprKind::Index {
                            expr: Box::new(expr),
                            indices,
                        },
                        span,
                    };
                }

                _ => break,
            }
        }

        Ok(expr)
    }

    // -----------------------------------------------------------------------
    // Call argument list
    // -----------------------------------------------------------------------

    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ParseError> {
        let mut args = Vec::new();
        while !self.at(&TokenKind::RParen) && !self.at(&TokenKind::Eof) {
            args.push(self.parse_call_arg()?);
            if self.at(&TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        Ok(args)
    }

    fn parse_call_arg(&mut self) -> Result<CallArg, ParseError> {
        // Named arg: `IDENT "=" expr`
        if matches!(self.current_kind(), TokenKind::Ident(_))
            && self.peek_ahead(1) == &TokenKind::Eq
        {
            let (name, name_span) = self.eat_ident()?;
            self.eat(TokenKind::Eq)?;
            let value = self.parse_expr(0)?;
            let span = name_span.merge(&value.span);
            return Ok(CallArg::Named { name, value, span });
        }

        Ok(CallArg::Positional(self.parse_expr(0)?))
    }

    // -----------------------------------------------------------------------
    // Primary expressions
    // -----------------------------------------------------------------------

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.current_span();

        match self.current_kind().clone() {
            TokenKind::IntLit(n) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::IntLit(n),
                    span,
                })
            }
            TokenKind::FloatLit(n) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::FloatLit(n),
                    span,
                })
            }
            TokenKind::BoolLit(b) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::BoolLit(b),
                    span,
                })
            }
            TokenKind::StringLit(s) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::StringLit(s),
                    span,
                })
            }
            TokenKind::Ident(name) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span,
                })
            }

            // Array / list literal: `[expr, ...]`
            TokenKind::LBracket => {
                self.bump();
                let mut elems = Vec::new();
                while !self.at(&TokenKind::RBracket) && !self.at(&TokenKind::Eof) {
                    elems.push(self.parse_expr(0)?);
                    if self.at(&TokenKind::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                let end = self.eat(TokenKind::RBracket)?;
                Ok(Expr {
                    kind: ExprKind::ArrayLit(elems),
                    span: span.merge(&end),
                })
            }

            // Parenthesised expression: `( expr )`
            TokenKind::LParen => {
                self.bump();
                let inner = self.parse_expr(0)?;
                let end = self.eat(TokenKind::RParen)?;
                Ok(Expr {
                    kind: ExprKind::Paren(Box::new(inner)),
                    span: span.merge(&end),
                })
            }

            // Block expression: `{ stmts... }`
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                let s = block.span.clone();
                Ok(Expr {
                    kind: ExprKind::Block(block),
                    span: s,
                })
            }

            // Control flow expressions
            TokenKind::KwIf => self.parse_if_expr(),
            TokenKind::KwFor => self.parse_for_expr(),
            TokenKind::KwWhile => self.parse_while_expr(),

            _ => Err(ParseError::Unexpected {
                found: self.current_kind().to_string(),
                span,
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.eat(TokenKind::KwIf)?;
        let condition = self.parse_expr(0)?;
        let then_block = self.parse_block()?;

        let else_branch = if self.at(&TokenKind::KwElse) {
            self.bump();
            if self.at(&TokenKind::KwIf) {
                // `else if ...`
                Some(Box::new(self.parse_if_expr()?))
            } else {
                // `else { ... }`
                let block = self.parse_block()?;
                let s = block.span.clone();
                Some(Box::new(Expr {
                    kind: ExprKind::Block(block),
                    span: s,
                }))
            }
        } else {
            None
        };

        let end_span = else_branch
            .as_ref()
            .map(|e| e.span.clone())
            .unwrap_or_else(|| then_block.span.clone());
        let span = start.merge(&end_span);

        Ok(Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_block,
                else_branch,
            },
            span,
        })
    }

    fn parse_for_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.eat(TokenKind::KwFor)?;
        let (var, _) = self.eat_ident()?;
        self.eat(TokenKind::KwIn)?;
        let iterable = self.parse_expr(0)?;
        let body = self.parse_block()?;
        let span = start.merge(&body.span);
        Ok(Expr {
            kind: ExprKind::For {
                var,
                iterable: Box::new(iterable),
                body,
            },
            span,
        })
    }

    fn parse_while_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.eat(TokenKind::KwWhile)?;
        let condition = self.parse_expr(0)?;
        let body = self.parse_block()?;
        let span = start.merge(&body.span);
        Ok(Expr {
            kind: ExprKind::While {
                condition: Box::new(condition),
                body,
            },
            span,
        })
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a token stream (produced by the lexer) into a `Program` AST.
///
/// The token stream **must** end with an `Eof` token.
pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    Parser::new(tokens).parse_program()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    fn parse_src(src: &str) -> Program {
        let tokens = tokenize(src).expect("lex error");
        parse(tokens).expect("parse error")
    }

    #[test]
    fn test_import_simple() {
        let prog = parse_src("import tpt");
        assert_eq!(prog.items.len(), 1);
        if let Item::Import(imp) = &prog.items[0] {
            assert_eq!(imp.path, ["tpt"]);
            assert!(imp.alias.is_none());
        } else {
            panic!("expected Import");
        }
    }

    #[test]
    fn test_import_dotted() {
        let prog = parse_src("import tpt.introspect");
        if let Item::Import(imp) = &prog.items[0] {
            assert_eq!(imp.path, ["tpt", "introspect"]);
        } else {
            panic!("expected Import");
        }
    }

    #[test]
    fn test_import_colons_with_alias() {
        let prog = parse_src("import model::transformer as tr");
        if let Item::Import(imp) = &prog.items[0] {
            assert_eq!(imp.path, ["model", "transformer"]);
            assert_eq!(imp.alias.as_deref(), Some("tr"));
        } else {
            panic!("expected Import");
        }
    }

    #[test]
    fn test_fn_no_params_no_return() {
        let prog = parse_src("fn foo() {}");
        if let Item::Function(f) = &prog.items[0] {
            assert_eq!(f.name, "foo");
            assert!(f.params.is_empty());
            assert!(f.return_type.is_none());
        } else {
            panic!("expected Function");
        }
    }

    #[test]
    fn test_fn_with_params_and_return() {
        let prog = parse_src("fn add(a: f32, b: f32) -> f32 { return a + b }");
        if let Item::Function(f) = &prog.items[0] {
            assert_eq!(f.name, "add");
            assert_eq!(f.params.len(), 2);
            assert_eq!(f.params[0].name, "a");
            matches!(f.return_type, Some(Type::Primitive(PrimitiveType::F32, _)));
        }
    }

    #[test]
    fn test_tensor_type() {
        let prog = parse_src("fn f(x: Tensor[f32, m, k]) {}");
        if let Item::Function(f) = &prog.items[0] {
            if let Type::Tensor { dtype, dims, .. } = &f.params[0].ty {
                assert_eq!(*dtype, PrimitiveType::F32);
                assert_eq!(dims.len(), 2);
                assert_eq!(dims[0], Dim::Named("m".to_string()));
                assert_eq!(dims[1], Dim::Named("k".to_string()));
            } else {
                panic!("expected Tensor type");
            }
        }
    }

    #[test]
    fn test_annotation() {
        let prog = parse_src(r#"@doc("hello") @requires_gpu(true) fn f() {}"#);
        if let Item::Function(f) = &prog.items[0] {
            assert_eq!(f.annotations.len(), 2);
            assert_eq!(f.annotations[0].name, "doc");
            assert_eq!(f.annotations[1].name, "requires_gpu");
        }
    }

    #[test]
    fn test_let_stmt() {
        let prog = parse_src("fn f() { let x = 42 }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Let(l) = &f.body.stmts[0] {
                assert_eq!(l.name, "x");
                matches!(l.value.kind, ExprKind::IntLit(42));
            }
        }
    }

    #[test]
    fn test_binary_expr_precedence() {
        // `1 + 2 * 3` should parse as `1 + (2 * 3)`.
        let prog = parse_src("fn f() { 1 + 2 * 3 }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Expr(Expr {
                kind: ExprKind::BinaryOp { op, right, .. },
                ..
            }) = &f.body.stmts[0]
            {
                assert_eq!(*op, BinOp::Add);
                if let ExprKind::BinaryOp { op: op2, .. } = &right.kind {
                    assert_eq!(*op2, BinOp::Mul);
                }
            }
        }
    }

    #[test]
    fn test_method_call() {
        let prog = parse_src("fn f() { loss.backward() }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Expr(Expr {
                kind: ExprKind::MethodCall { method, .. },
                ..
            }) = &f.body.stmts[0]
            {
                assert_eq!(method, "backward");
            }
        }
    }

    #[test]
    fn test_named_call_arg() {
        let prog = parse_src("fn f() { tpt.zeros([m, n], dtype=f32) }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Expr(Expr {
                kind: ExprKind::MethodCall { args, .. },
                ..
            }) = &f.body.stmts[0]
            {
                // Second arg should be named `dtype=f32`
                matches!(&args[1], CallArg::Named { name, .. } if name == "dtype");
            }
        }
    }

    #[test]
    fn test_for_loop() {
        let prog = parse_src("fn f() { for i in 0..n { } }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Expr(Expr {
                kind: ExprKind::For { var, .. },
                ..
            }) = &f.body.stmts[0]
            {
                assert_eq!(var, "i");
            }
        }
    }

    #[test]
    fn test_if_else() {
        let prog = parse_src("fn f() { if x { } else { } }");
        if let Item::Function(f) = &prog.items[0] {
            if let Stmt::Expr(Expr {
                kind: ExprKind::If { else_branch, .. },
                ..
            }) = &f.body.stmts[0]
            {
                assert!(else_branch.is_some());
            }
        }
    }

    #[test]
    fn test_type_alias() {
        let prog = parse_src("type MatF32 = Tensor[f32, m, n]");
        if let Item::TypeAlias(td) = &prog.items[0] {
            assert_eq!(td.name, "MatF32");
        }
    }

    #[test]
    fn test_full_matmul_example() {
        let src = r#"
@doc("Multiply two matrices")
@constraint("a.shape[1] == b.shape[0]", error="Inner dimensions must match")
@complexity("O(m * n * k)")
@differentiable(true)
@gpu_optimized(true)
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    let result = tpt.zeros([m, n], dtype=f32)
    tpt.gemm(a, b, result)
    return result
}
"#;
        let prog = parse_src(src);
        assert_eq!(prog.items.len(), 1);
        if let Item::Function(f) = &prog.items[0] {
            assert_eq!(f.name, "matmul");
            assert_eq!(f.annotations.len(), 5);
            assert_eq!(f.params.len(), 2);
            assert!(f.return_type.is_some());
            assert_eq!(f.body.stmts.len(), 3);
        }
    }
}
