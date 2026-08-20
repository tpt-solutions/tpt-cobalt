use std::collections::HashMap;

use crate::lexer::{tokenize, TokenKind};

// ---------------------------------------------------------------------------
// Constraint expression — a small subset of expressions used inside
// `@constraint("expr", error="...")` annotations.
//
// Grammar (simplified):
//   constraint_expr = or_expr
//   or_expr  = and_expr { "||" and_expr }
//   and_expr = cmp_expr { "&&" cmp_expr }
//   cmp_expr = arith_expr [ cmp_op arith_expr ]
//   arith_expr = unary_expr { ("+" | "-" | "*" | "/" | "%") unary_expr }
//   unary_expr = ["!"] primary
//   primary  = INT_LIT | IDENT | IDENT ".shape[" INT "]" | "(" expr ")"
//
// Values are signed 64-bit integers; boolean true = 1, false = 0.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintExpr {
    IntLit(i64),
    BoolLit(bool),
    /// A symbolic dimension name introduced in the function signature, e.g. `m`.
    DimVar(String),
    /// `tensor_param.shape[N]` — access the N-th dimension of a tensor arg.
    ShapeAccess {
        param: String,
        index: usize,
    },
    BinOp {
        op: ConstraintOp,
        left: Box<ConstraintExpr>,
        right: Box<ConstraintExpr>,
    },
    UnaryNot(Box<ConstraintExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

// ---------------------------------------------------------------------------
// Parse a constraint string into a ConstraintExpr
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConstraintParseError(pub String);

impl std::fmt::Display for ConstraintParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "constraint parse error: {}", self.0)
    }
}

struct ConstraintParser {
    tokens: Vec<TokenKind>,
    pos: usize,
}

impl ConstraintParser {
    fn new(src: &str) -> Result<Self, ConstraintParseError> {
        let toks = tokenize(src).map_err(|e| ConstraintParseError(e.to_string()))?;
        let kinds: Vec<_> = toks.into_iter().map(|t| t.kind).collect();
        Ok(Self {
            tokens: kinds,
            pos: 0,
        })
    }

    fn peek(&self) -> &TokenKind {
        self.tokens.get(self.pos).unwrap_or(&TokenKind::Eof)
    }

    fn bump(&mut self) -> TokenKind {
        let t = self.tokens.get(self.pos).cloned().unwrap_or(TokenKind::Eof);
        self.pos += 1;
        t
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == kind {
            self.bump();
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------

    fn parse(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        let mut left = self.parse_and()?;
        while self.peek() == &TokenKind::PipePipe {
            self.bump();
            let right = self.parse_and()?;
            left = ConstraintExpr::BinOp {
                op: ConstraintOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        let mut left = self.parse_cmp()?;
        while self.peek() == &TokenKind::AmpAmp {
            self.bump();
            let right = self.parse_cmp()?;
            left = ConstraintExpr::BinOp {
                op: ConstraintOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        let left = self.parse_arith()?;
        let op = match self.peek() {
            TokenKind::EqEq => ConstraintOp::Eq,
            TokenKind::BangEq => ConstraintOp::Ne,
            TokenKind::Lt => ConstraintOp::Lt,
            TokenKind::LtEq => ConstraintOp::Le,
            TokenKind::Gt => ConstraintOp::Gt,
            TokenKind::GtEq => ConstraintOp::Ge,
            _ => return Ok(left),
        };
        self.bump();
        let right = self.parse_arith()?;
        Ok(ConstraintExpr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn parse_arith(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => ConstraintOp::Add,
                TokenKind::Minus => ConstraintOp::Sub,
                TokenKind::Star => ConstraintOp::Mul,
                TokenKind::Slash => ConstraintOp::Div,
                TokenKind::Percent => ConstraintOp::Mod,
                _ => break,
            };
            self.bump();
            let right = self.parse_unary()?;
            left = ConstraintExpr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        if self.peek() == &TokenKind::Bang {
            self.bump();
            let inner = self.parse_unary()?;
            return Ok(ConstraintExpr::UnaryNot(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<ConstraintExpr, ConstraintParseError> {
        match self.peek().clone() {
            TokenKind::IntLit(n) => {
                self.bump();
                Ok(ConstraintExpr::IntLit(n))
            }
            TokenKind::BoolLit(b) => {
                self.bump();
                Ok(ConstraintExpr::BoolLit(b))
            }
            TokenKind::LParen => {
                self.bump();
                let inner = self.parse()?;
                if !self.eat(&TokenKind::RParen) {
                    return Err(ConstraintParseError("expected ')'".to_string()));
                }
                Ok(inner)
            }
            TokenKind::Ident(name) => {
                self.bump();
                // `name.shape[N]`
                if self.peek() == &TokenKind::Dot {
                    self.bump();
                    match self.bump() {
                        TokenKind::Ident(field) if field == "shape" => {
                            if !self.eat(&TokenKind::LBracket) {
                                return Err(ConstraintParseError(
                                    "expected '[' after .shape".to_string(),
                                ));
                            }
                            let idx = match self.bump() {
                                TokenKind::IntLit(n) => n as usize,
                                _ => {
                                    return Err(ConstraintParseError(
                                        "expected integer index in .shape[N]".to_string(),
                                    ))
                                }
                            };
                            if !self.eat(&TokenKind::RBracket) {
                                return Err(ConstraintParseError(
                                    "expected ']' after shape index".to_string(),
                                ));
                            }
                            Ok(ConstraintExpr::ShapeAccess {
                                param: name,
                                index: idx,
                            })
                        }
                        other => Err(ConstraintParseError(format!(
                            "unexpected field '{other}' after '{name}'"
                        ))),
                    }
                } else {
                    Ok(ConstraintExpr::DimVar(name))
                }
            }
            other => Err(ConstraintParseError(format!(
                "unexpected token '{other}' in constraint"
            ))),
        }
    }
}

/// Parse a constraint expression string (e.g. `"a.shape[1] == b.shape[0]"`).
pub fn parse_constraint(src: &str) -> Result<ConstraintExpr, ConstraintParseError> {
    ConstraintParser::new(src)?.parse()
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Result of evaluating a constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult {
    /// The constraint evaluated to a concrete boolean.
    Known(bool),
    /// Not all variables were bound — cannot evaluate at compile time.
    Symbolic,
    /// Evaluation error (e.g., division by zero).
    Error(String),
}

/// Bindings for dimension variables and parameter shapes.
///
/// `dims`: symbolic dim name → concrete value (e.g. `m → 128`)
/// `shapes`: parameter name → shape vector (e.g. `a → [10, 20]`)
pub struct EvalEnv<'a> {
    pub dims: &'a HashMap<String, i64>,
    pub shapes: &'a HashMap<String, Vec<i64>>,
}

pub fn eval_constraint(expr: &ConstraintExpr, env: &EvalEnv<'_>) -> ConstraintResult {
    match eval_int(expr, env) {
        Some(n) => ConstraintResult::Known(n != 0),
        None => ConstraintResult::Symbolic,
    }
}

fn eval_int(expr: &ConstraintExpr, env: &EvalEnv<'_>) -> Option<i64> {
    match expr {
        ConstraintExpr::IntLit(n) => Some(*n),
        ConstraintExpr::BoolLit(b) => Some(*b as i64),

        ConstraintExpr::DimVar(name) => env.dims.get(name).copied(),

        ConstraintExpr::ShapeAccess { param, index } => {
            env.shapes.get(param).and_then(|s| s.get(*index)).copied()
        }

        ConstraintExpr::UnaryNot(inner) => eval_int(inner, env).map(|n| if n == 0 { 1 } else { 0 }),

        ConstraintExpr::BinOp { op, left, right } => {
            let l = eval_int(left, env)?;
            let r = eval_int(right, env)?;
            Some(match op {
                ConstraintOp::Add => l.wrapping_add(r),
                ConstraintOp::Sub => l.wrapping_sub(r),
                ConstraintOp::Mul => l.wrapping_mul(r),
                ConstraintOp::Div => {
                    if r == 0 {
                        return None;
                    } else {
                        l / r
                    }
                }
                ConstraintOp::Mod => {
                    if r == 0 {
                        return None;
                    } else {
                        l % r
                    }
                }
                ConstraintOp::Eq => (l == r) as i64,
                ConstraintOp::Ne => (l != r) as i64,
                ConstraintOp::Lt => (l < r) as i64,
                ConstraintOp::Le => (l <= r) as i64,
                ConstraintOp::Gt => (l > r) as i64,
                ConstraintOp::Ge => (l >= r) as i64,
                ConstraintOp::And => ((l != 0) && (r != 0)) as i64,
                ConstraintOp::Or => ((l != 0) || (r != 0)) as i64,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
    fn shapes(pairs: &[(&str, Vec<i64>)]) -> HashMap<String, Vec<i64>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_parse_simple_eq() {
        let expr = parse_constraint("a.shape[1] == b.shape[0]").unwrap();
        let env = EvalEnv {
            dims: &dims(&[]),
            shapes: &shapes(&[("a", vec![10, 20]), ("b", vec![20, 30])]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(true));
    }

    #[test]
    fn test_parse_fails_mismatch() {
        let expr = parse_constraint("a.shape[1] == b.shape[0]").unwrap();
        let env = EvalEnv {
            dims: &dims(&[]),
            shapes: &shapes(&[("a", vec![10, 20]), ("b", vec![15, 30])]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(false));
    }

    #[test]
    fn test_dim_var_concrete() {
        let expr = parse_constraint("batch > 0").unwrap();
        let env = EvalEnv {
            dims: &dims(&[("batch", 32)]),
            shapes: &shapes(&[]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(true));
    }

    #[test]
    fn test_dim_var_symbolic() {
        let expr = parse_constraint("m == k").unwrap();
        let env = EvalEnv {
            dims: &dims(&[]),
            shapes: &shapes(&[]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Symbolic);
    }

    #[test]
    fn test_arithmetic_constraint() {
        let expr = parse_constraint("2 * m * n * k > 0").unwrap();
        let env = EvalEnv {
            dims: &dims(&[("m", 4), ("n", 4), ("k", 4)]),
            shapes: &shapes(&[]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(true));
    }

    #[test]
    fn test_logical_and() {
        let expr = parse_constraint("batch > 0 && batch <= 1024").unwrap();
        let env = EvalEnv {
            dims: &dims(&[("batch", 512)]),
            shapes: &shapes(&[]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(true));
    }

    #[test]
    fn test_not() {
        let expr = parse_constraint("!(batch == 0)").unwrap();
        let env = EvalEnv {
            dims: &dims(&[("batch", 1)]),
            shapes: &shapes(&[]),
        };
        assert_eq!(eval_constraint(&expr, &env), ConstraintResult::Known(true));
    }
}
