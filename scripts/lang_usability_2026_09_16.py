"""Language usability pass: for/elif, ranges, kwargs, interpolation,
chained member assign, tensor multi-index + slicing. Applied to tpt-lang."""

p = r"crates/tpt-lang/src/interp.rs"
src = open(p, encoding="utf-8").read()
count = 0

def rep(old, new):
    global src, count
    assert old in src, "MISSING ANCHOR: " + old[:90]
    src = src.replace(old, new, 1)
    count += 1

# ------------------------------ AST ---------------------------------------

rep(
    """    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    /// Attribute access `base.name`: module members (and dict keys as sugar).
    Member(Box<Expr>, String),
}""",
    """    Call(Box<Expr>, Vec<CallArg>),
    Index(Box<Expr>, Vec<IdxArg>),
    /// Attribute access `base.name`: module members (and dict keys as sugar).
    Member(Box<Expr>, String),
}

/// One argument in a call: positional, or `name = expr` keyword.
#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<String>,
    pub expr: Expr,
}

/// One bracket index: a plain expression, or an `a:b` slice (either side
/// optional).
#[derive(Debug, Clone)]
pub enum IdxArg {
    Expr(Expr),
    Range(Option<Expr>, Option<Expr>),
}""",
)

rep(
    """    /// `module Name { ... }`: execute the body in a child scope, then bind
    /// its bindings as a [`crate::value::Module`] under `Name`.
    Module(String, Vec<Stmt>),""",
    """    /// `module Name { ... }`: execute the body in a child scope, then bind
    /// its bindings as a [`crate::value::Module`] under `Name`.
    Module(String, Vec<Stmt>),
    /// `for name in iterable { body }`. Iterates lists, tensors (flat),
    /// strings (chars), and dicts (sorted keys). The loop variable is bound
    /// in the current scope (Python-like).
    For(String, Expr, Vec<Stmt>),""",
)

# ------------------------------ lexer --------------------------------------

rep(
    """            '+' | '-' | '*' | '/' | '%' | '<' | '>' | '=' => {
                let two: String = [c, cs.get(i + 1).copied().unwrap_or(' ')].iter().collect();
                if ["==", "!=", "<=", ">="].contains(&two.as_str()) {
                    out.push(Tok::Op(two));
                    i += 2;
                } else {
                    out.push(Tok::Op(c.to_string()));
                    i += 1;
                }
            }""",
    """            '+' | '-' | '*' | '/' | '%' | '<' | '>' | '=' => {
                let two: String = [c, cs.get(i + 1).copied().unwrap_or(' ')].iter().collect();
                if ["==", "!=", "<=", ">="].contains(&two.as_str()) {
                    out.push(Tok::Op(two));
                    i += 2;
                } else {
                    out.push(Tok::Op(c.to_string()));
                    i += 1;
                }
            }
            '.' if cs.get(i + 1) == Some(&'.') => {
                out.push(Tok::Op("..".into()));
                i += 2;
            }""",
)

# ------------------------------ parser -------------------------------------

rep(
    """                "def" | "fn" => {""",
    """                "for" => {
                    self.pos += 1;
                    let name = self.expect_ident()?;
                    if !self.at_kw("in") {
                        return Err(InterpreterError::new(
                            "SyntaxError",
                            "expected 'in' in for loop",
                        ));
                    }
                    self.pos += 1;
                    let iterable = self.parse_expr()?;
                    self.expect_op("{")?;
                    let body = self.parse_block()?;
                    self.expect_op("}")?;
                    Ok(Stmt::For(name, iterable, body))
                }
                "elif" => {
                    // `elif` desugars to a nested if in the else branch
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let then_body = self.parse_block()?;
                    self.expect_op("}")?;
                    let mut else_body = Vec::new();
                    self.skip_newlines();
                    if self.at_kw("else") || self.at_kw("elif") {
                        else_body = self.parse_tail_else()?;
                    }
                    Ok(Stmt::If(cond, then_body, else_body))
                }
                "def" | "fn" => {""",
)

# if: share the else/elif tail helper
rep(
    """                "if" => {
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let then_body = self.parse_block()?;
                    self.expect_op("}")?;
                    let mut else_body = Vec::new();
                    self.skip_newlines();
                    if self.at_kw("else") {
                        self.pos += 1;
                        self.expect_op("{")?;
                        else_body = self.parse_block()?;
                        self.expect_op("}")?;
                    }
                    Ok(Stmt::If(cond, then_body, else_body))
                }""",
    """                "if" => {
                    self.pos += 1;
                    let cond = self.parse_expr()?;
                    self.expect_op("{")?;
                    let then_body = self.parse_block()?;
                    self.expect_op("}")?;
                    let mut else_body = Vec::new();
                    self.skip_newlines();
                    if self.at_kw("else") || self.at_kw("elif") {
                        else_body = self.parse_tail_else()?;
                    }
                    Ok(Stmt::If(cond, then_body, else_body))
                }""",
)

rep(
    """    fn parse_assign_or_expr(&mut self) -> Res<Stmt> {""",
    """    /// The `else { ... }` / `else if ...` / `elif ...` tail of an if.
    fn parse_tail_else(&mut self) -> Res<Vec<Stmt>> {
        if self.at_kw("elif") || self.peek_is_ident("if") {
            // recurse: the tail is itself an if-statement
            return Ok(vec![self.parse_stmt()?]);
        }
        self.pos += 1; // 'else'
        self.expect_op("{")?;
        let body = self.parse_block()?;
        self.expect_op("}")?;
        Ok(body)
    }

    fn peek_is_ident(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn parse_assign_or_expr(&mut self) -> Res<Stmt> {""",
)

# chained member assignment lookahead: IDENT ('.' IDENT)+ '='
rep(
    """                Some(Tok::Op(op)) if op == "." => {
                    if let Some(Tok::Ident(_)) = self.toks.get(self.pos + 2) {
                        if let Some(Tok::Op(eq)) = self.toks.get(self.pos + 3) {
                            if eq == "=" {
                                let base = self.expect_ident()?;
                                self.pos += 1; // '.'
                                let member = self.expect_ident()?;
                                self.pos += 1; // '='
                                let e = self.parse_expr()?;
                                return Ok(Stmt::MemberAssign(
                                    Box::new(Expr::Ident(base)),
                                    member,
                                    e,
                                ));
                            }
                        }
                    }
                }""",
    """                Some(Tok::Op(op)) if op == "." => {
                    // IDENT ('.' IDENT)+ '=' → member(-chain) assignment
                    let mut path = vec![self.expect_ident()?];
                    let mut j = self.pos;
                    while matches!(self.toks.get(j), Some(Tok::Op(o)) if o == ".")
                        && matches!(self.toks.get(j + 1), Some(Tok::Ident(_)))
                        && matches!(self.toks.get(j + 2), Some(Tok::Op(o)) if o == "=")
                    {
                        j += 1; // '.'
                        let seg = match self.toks.get(j).cloned() {
                            Some(Tok::Ident(s)) => s,
                            _ => unreachable!(),
                        };
                        path.push(seg);
                        j += 2; // past segment and '='
                    }
                    if path.len() >= 2 {
                        // rebuild the chain as nested Member exprs
                        let mut base = Expr::Ident(path[0].clone());
                        for seg in &path[1..path.len() - 1] {
                            base = Expr::Member(Box::new(base), seg.clone());
                        }
                        self.pos = j; // at '='
                        self.pos += 1;
                        let e = self.parse_expr()?;
                        return Ok(Stmt::MemberAssign(
                            Box::new(base),
                            path.last().unwrap().clone(),
                            e,
                        ));
                    }
                }""",
)

# range expression: a..b (non-associative, loosest)
rep(
    """    pub fn parse_expr(&mut self) -> Res<Expr> {
        self.parse_comparison()
    }""",
    """    pub fn parse_expr(&mut self) -> Res<Expr> {
        let lhs = self.parse_comparison()?;
        if self.at_op("..") {
            self.pos += 1;
            let rhs = self.parse_comparison()?;
            return Ok(Expr::Binary("..".into(), Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }""",
)

# index brackets: comma-separated expr / slice args
rep(
    """            if self.at_op(".") {
                self.pos += 1;
                let name = self.expect_ident()?;
                e = Expr::Member(Box::new(e), name);
            } else if self.at_op("[") {
                self.pos += 1;
                let idx = self.parse_expr()?;
                self.expect_op("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            } else if self.at_op("(") {""",
    """            if self.at_op(".") {
                self.pos += 1;
                let name = self.expect_ident()?;
                e = Expr::Member(Box::new(e), name);
            } else if self.at_op("[") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op("]") {
                    args.push(self.parse_idx_arg()?);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op("]")?;
                e = Expr::Index(Box::new(e), args);
            } else if self.at_op("(") {""",
)

rep(
    """    fn parse_atom(&mut self) -> Res<Expr> {""",
    """    /// One bracket index: `expr`, `expr? : expr?` (slice), with the slice
    /// form recognized by a leading ':' or a ':' after the first expression.
    fn parse_idx_arg(&mut self) -> Res<IdxArg> {
        let mut start = None;
        if !self.at_op(":") {
            start = Some(self.parse_expr()?);
        }
        if !self.at_op(":") {
            let start = start.ok_or_else(|| {
                InterpreterError::new("SyntaxError", "empty index")
            })?;
            return Ok(IdxArg::Expr(start));
        }
        self.pos += 1; // ':'
        let mut end = None;
        if !self.at_op("]") && !self.at_op(",") {
            end = Some(self.parse_expr()?);
        }
        Ok(IdxArg::Range(start, end))
    }

    fn parse_atom(&mut self) -> Res<Expr> {""",
)

# call arguments: optional `name =`
rep(
    """            } else if self.at_op("(") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op(")") {
                    args.push(self.parse_expr()?);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op(")")?;
                e = Expr::Call(Box::new(e), args);
            } else {""",
    """            } else if self.at_op("(") {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.at_op(")") {
                    let arg = if matches!(self.peek(), Some(Tok::Ident(_)))
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Op(o)) if o == "=")
                    {
                        let name = self.expect_ident()?;
                        self.pos += 1; // '='
                        CallArg {
                            name: Some(name),
                            expr: self.parse_expr()?,
                        }
                    } else {
                        CallArg {
                            name: None,
                            expr: self.parse_expr()?,
                        }
                    };
                    args.push(arg);
                    if !self.eat_op(",")? {
                        break;
                    }
                }
                self.expect_op(")")?;
                e = Expr::Call(Box::new(e), args);
            } else {""",
)

open(p, "w", encoding="utf-8", newline="").write(src)
print("patched parser/lexer:", count, "edits")
