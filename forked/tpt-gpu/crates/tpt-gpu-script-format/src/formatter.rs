use tpt_gpu_script_core;
use tpt_gpu_script_core::TokenKind;

pub fn format_source(source: &str) -> Result<String, tpt_gpu_script_core::CompileError> {
    let tokens = tpt_gpu_script_core::tokenize(source)?;
    let formatter = Formatter::new(&tokens);
    Ok(formatter.format())
}

struct Formatter<'a> {
    tokens: &'a [tpt_gpu_script_core::Token],
    pos: usize,
    output: String,
    indent_level: u32,
    indent_size: u32,
}

impl<'a> Formatter<'a> {
    fn new(tokens: &'a [tpt_gpu_script_core::Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            output: String::new(),
            indent_level: 0,
            indent_size: 4,
        }
    }

    fn format(mut self) -> String {
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos].kind {
                TokenKind::KwFn => self.format_fn_decl(),
                TokenKind::KwImport => self.format_import(),
                TokenKind::KwType => self.format_type_decl(),
                TokenKind::At => self.format_annotation(),
                TokenKind::Eof => break,
                _ => {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
        }
        self.output
    }

    fn push_token(&mut self, token: tpt_gpu_script_core::Token) {
        self.output.push_str(&token_text(&token));
    }
    fn push_str(&mut self, s: &str) {
        self.output.push_str(s);
    }
    fn indent_str(&mut self) {
        self.output
            .push_str(&" ".repeat((self.indent_level * self.indent_size) as usize));
    }
    fn newline(&mut self) {
        self.output.push('\n');
        self.indent_str();
    }
    fn current(&self) -> tpt_gpu_script_core::Token {
        self.tokens[self.pos].clone()
    }
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn format_fn_decl(&mut self) {
        while !self.at_end() && matches!(self.current().kind, TokenKind::At) {
            self.format_annotation();
        }
        self.push_token(self.current()); // fn
        self.push_str(" ");
        self.pos += 1;
        if let TokenKind::Ident(_) = &self.current().kind {
            self.push_token(self.current());
            self.pos += 1;
            if !matches!(self.current().kind, TokenKind::LParen) {
                self.push_str(" ");
            }
        }
        if matches!(self.current().kind, TokenKind::LParen) {
            self.push_token(self.current());
            self.pos += 1;
            self.format_params();
            if matches!(self.current().kind, TokenKind::RParen) {
                self.push_token(self.current());
                self.push_str(" ");
                self.pos += 1;
            }
        }
        if matches!(self.current().kind, TokenKind::Arrow) {
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
            self.format_type();
            self.push_str(" ");
        }
        if matches!(self.current().kind, TokenKind::LBrace) {
            self.push_token(self.current());
            self.pos += 1;
            self.indent_level += 1;
            self.format_block();
            self.indent_level -= 1;
            if matches!(self.current().kind, TokenKind::RBrace) {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn format_block(&mut self) {
        while !self.at_end() {
            match self.current().kind {
                TokenKind::RBrace => break,
                _ => {
                    self.newline();
                    self.format_statement();
                }
            }
        }
    }

    fn format_statement(&mut self) {
        match &self.current().kind {
            TokenKind::KwLet => self.format_let(),
            TokenKind::KwReturn => self.format_return(),
            TokenKind::KwIf => self.format_if(),
            TokenKind::KwFor => self.format_for(),
            TokenKind::KwWhile => self.format_while(),
            TokenKind::KwBreak | TokenKind::KwContinue => {
                self.push_token(self.current());
                self.pos += 1;
                self.consume_semicolon();
            }
            TokenKind::LBrace => {
                self.push_token(self.current());
                self.pos += 1;
                self.indent_level += 1;
                self.format_block();
                self.indent_level -= 1;
                self.newline();
                if matches!(self.current().kind, TokenKind::RBrace) {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
            _ => {
                self.format_expression();
                self.consume_semicolon();
            }
        }
    }

    fn format_let(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        if let TokenKind::Ident(_) = &self.current().kind {
            self.push_token(self.current());
            self.pos += 1;
        }
        if matches!(self.current().kind, TokenKind::Colon) {
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
            self.format_type();
        }
        if matches!(self.current().kind, TokenKind::Eq) {
            self.push_str(" ");
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
            self.format_expression();
        }
        self.consume_semicolon();
    }

    fn format_if(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        self.format_expression();
        self.push_str(" ");
        if matches!(self.current().kind, TokenKind::LBrace) {
            self.push_token(self.current());
            self.pos += 1;
            self.indent_level += 1;
            self.format_block();
            self.indent_level -= 1;
            self.newline();
            if matches!(self.current().kind, TokenKind::RBrace) {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
        if matches!(self.current().kind, TokenKind::KwElse) {
            self.push_str(" ");
            self.push_token(self.current());
            self.pos += 1;
            if matches!(self.current().kind, TokenKind::KwIf) {
                self.push_str(" ");
                self.push_token(self.current());
                self.pos += 1;
                self.format_if();
            } else if matches!(self.current().kind, TokenKind::LBrace) {
                self.push_str(" ");
                self.push_token(self.current());
                self.pos += 1;
                self.indent_level += 1;
                self.format_block();
                self.indent_level -= 1;
                self.newline();
                if matches!(self.current().kind, TokenKind::RBrace) {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
        }
    }

    fn format_for(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        if let TokenKind::Ident(_) = &self.current().kind {
            self.push_token(self.current());
            self.pos += 1;
        }
        if matches!(self.current().kind, TokenKind::KwIn) {
            self.push_str(" ");
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
        }
        self.format_expression();
        self.push_str(" ");
        if matches!(self.current().kind, TokenKind::LBrace) {
            self.push_token(self.current());
            self.pos += 1;
            self.indent_level += 1;
            self.format_block();
            self.indent_level -= 1;
            self.newline();
            if matches!(self.current().kind, TokenKind::RBrace) {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn format_while(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        self.format_expression();
        self.push_str(" ");
        if matches!(self.current().kind, TokenKind::LBrace) {
            self.push_token(self.current());
            self.pos += 1;
            self.indent_level += 1;
            self.format_block();
            self.indent_level -= 1;
            self.newline();
            if matches!(self.current().kind, TokenKind::RBrace) {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn format_return(&mut self) {
        self.push_token(self.current());
        self.pos += 1;
        if !matches!(
            self.current().kind,
            TokenKind::Semicolon | TokenKind::RBrace
        ) {
            self.push_str(" ");
            self.format_expression();
        }
        self.consume_semicolon();
    }

    fn format_expression(&mut self) {
        self.format_primary();
        while !self.at_end() {
            match &self.current().kind {
                TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::EqEq
                | TokenKind::BangEq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::LtEq
                | TokenKind::GtEq
                | TokenKind::AmpAmp
                | TokenKind::PipePipe
                | TokenKind::DotDot
                | TokenKind::DotDotEq
                | TokenKind::Eq => {
                    self.push_str(" ");
                    self.push_token(self.current());
                    self.push_str(" ");
                    self.pos += 1;
                    self.format_primary();
                }
                TokenKind::Dot => {
                    self.push_token(self.current());
                    self.pos += 1;
                    if let TokenKind::Ident(_) = &self.current().kind {
                        self.push_token(self.current());
                        self.pos += 1;
                    }
                    if matches!(self.current().kind, TokenKind::LParen) {
                        self.push_token(self.current());
                        self.pos += 1;
                        while !self.at_end() && !matches!(self.current().kind, TokenKind::RParen) {
                            self.format_expression();
                            if matches!(self.current().kind, TokenKind::Comma) {
                                self.push_token(self.current());
                                self.push_str(" ");
                                self.pos += 1;
                            }
                        }
                        if !self.at_end() {
                            self.push_token(self.current());
                            self.pos += 1;
                        }
                    }
                }
                TokenKind::LParen => {
                    self.push_token(self.current());
                    self.pos += 1;
                    while !self.at_end() && !matches!(self.current().kind, TokenKind::RParen) {
                        self.format_expression();
                        if matches!(self.current().kind, TokenKind::Comma) {
                            self.push_token(self.current());
                            self.push_str(" ");
                            self.pos += 1;
                        }
                    }
                    if !self.at_end() {
                        self.push_token(self.current());
                        self.pos += 1;
                    }
                }
                TokenKind::LBracket => {
                    self.push_token(self.current());
                    self.pos += 1;
                    while !self.at_end() && !matches!(self.current().kind, TokenKind::RBracket) {
                        self.format_expression();
                        if matches!(self.current().kind, TokenKind::Comma) {
                            self.push_token(self.current());
                            self.push_str(" ");
                            self.pos += 1;
                        }
                    }
                    if !self.at_end() {
                        self.push_token(self.current());
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn format_primary(&mut self) {
        match &self.current().kind {
            TokenKind::IntLit(_)
            | TokenKind::FloatLit(_)
            | TokenKind::BoolLit(_)
            | TokenKind::StringLit(_)
            | TokenKind::Ident(_) => {
                self.push_token(self.current());
                self.pos += 1;
            }
            TokenKind::LParen => {
                self.push_token(self.current());
                self.pos += 1;
                self.format_expression();
                if matches!(self.current().kind, TokenKind::RParen) {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
            TokenKind::LBracket => {
                self.push_token(self.current());
                self.pos += 1;
                while !self.at_end() && !matches!(self.current().kind, TokenKind::RBracket) {
                    self.format_expression();
                    if matches!(self.current().kind, TokenKind::Comma) {
                        self.push_token(self.current());
                        self.push_str(" ");
                        self.pos += 1;
                    }
                }
                if !self.at_end() {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
            TokenKind::Bang | TokenKind::Minus => {
                self.push_token(self.current());
                self.pos += 1;
                self.format_primary();
            }
            _ => {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn format_import(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        while !self.at_end() {
            match &self.current().kind {
                TokenKind::Semicolon => {
                    self.pos += 1;
                    break;
                }
                TokenKind::ColonColon | TokenKind::Dot | TokenKind::Ident(_) => {
                    self.push_token(self.current());
                    self.pos += 1;
                }
                _ => break,
            }
        }
        self.consume_semicolon();
    }

    fn format_type_decl(&mut self) {
        self.push_token(self.current());
        self.push_str(" ");
        self.pos += 1;
        if let TokenKind::Ident(_) = &self.current().kind {
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
        }
        if matches!(self.current().kind, TokenKind::Eq) {
            self.push_token(self.current());
            self.push_str(" ");
            self.pos += 1;
            self.format_type();
        }
        self.consume_semicolon();
    }

    fn format_annotation(&mut self) {
        self.push_token(self.current());
        self.pos += 1;
        if let TokenKind::Ident(_) = &self.current().kind {
            self.push_token(self.current());
            self.pos += 1;
        }
        if matches!(self.current().kind, TokenKind::LParen) {
            self.push_token(self.current());
            self.pos += 1;
            while !self.at_end() && !matches!(self.current().kind, TokenKind::RParen) {
                match &self.current().kind {
                    TokenKind::StringLit(_) => {
                        self.push_token(self.current());
                        self.pos += 1;
                    }
                    TokenKind::Ident(_) => {
                        self.push_token(self.current());
                        self.pos += 1;
                        if matches!(self.current().kind, TokenKind::Eq) {
                            self.push_str(" ");
                            self.push_token(self.current());
                            self.push_str(" ");
                            self.pos += 1;
                            if matches!(
                                self.current().kind,
                                TokenKind::Ident(_) | TokenKind::IntLit(_) | TokenKind::BoolLit(_)
                            ) {
                                self.push_token(self.current());
                                self.pos += 1;
                            }
                        }
                    }
                    TokenKind::Comma => {
                        self.push_token(self.current());
                        self.push_str(" ");
                        self.pos += 1;
                    }
                    _ => {
                        self.push_token(self.current());
                        self.pos += 1;
                    }
                }
            }
            if !self.at_end() {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn format_params(&mut self) {
        while !self.at_end() {
            match self.current().kind {
                TokenKind::RParen => break,
                TokenKind::Comma => {
                    self.push_token(self.current());
                    self.push_str(" ");
                    self.pos += 1;
                }
                TokenKind::Ident(_) => {
                    self.push_token(self.current());
                    self.pos += 1;
                    if matches!(self.current().kind, TokenKind::Colon) {
                        self.push_token(self.current());
                        self.push_str(" ");
                        self.pos += 1;
                        self.format_type();
                    }
                }
                _ => {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
        }
    }

    fn format_type(&mut self) {
        match &self.current().kind {
            TokenKind::Ident(_) => {
                self.push_token(self.current());
                self.pos += 1;
            }
            TokenKind::LBracket => {
                self.push_token(self.current());
                self.pos += 1;
                while !self.at_end() && !matches!(self.current().kind, TokenKind::RBracket) {
                    self.push_token(self.current());
                    self.pos += 1;
                }
                if !self.at_end() {
                    self.push_token(self.current());
                    self.pos += 1;
                }
            }
            _ => {
                self.push_token(self.current());
                self.pos += 1;
            }
        }
    }

    fn consume_semicolon(&mut self) {
        if matches!(self.current().kind, TokenKind::Semicolon) {
            self.push_token(self.current());
            self.pos += 1;
        }
    }
}

fn token_text(token: &tpt_gpu_script_core::Token) -> String {
    match &token.kind {
        TokenKind::KwBreak => "break".into(),
        TokenKind::KwContinue => "continue".into(),
        TokenKind::KwElse => "else".into(),
        TokenKind::KwFn => "fn".into(),
        TokenKind::KwFor => "for".into(),
        TokenKind::KwIf => "if".into(),
        TokenKind::KwImport => "import".into(),
        TokenKind::KwIn => "in".into(),
        TokenKind::KwLet => "let".into(),
        TokenKind::KwReturn => "return".into(),
        TokenKind::KwType => "type".into(),
        TokenKind::KwWhile => "while".into(),
        TokenKind::IntLit(n) => n.to_string(),
        TokenKind::FloatLit(f) => f.to_string(),
        TokenKind::BoolLit(b) => b.to_string(),
        TokenKind::StringLit(s) => format!("\"{}\"", s),
        TokenKind::Ident(s) => s.clone(),
        TokenKind::Plus => "+".into(),
        TokenKind::Minus => "-".into(),
        TokenKind::Star => "*".into(),
        TokenKind::Slash => "/".into(),
        TokenKind::Percent => "%".into(),
        TokenKind::EqEq => "==".into(),
        TokenKind::BangEq => "!=".into(),
        TokenKind::Lt => "<".into(),
        TokenKind::Gt => ">".into(),
        TokenKind::LtEq => "<=".into(),
        TokenKind::GtEq => ">=".into(),
        TokenKind::AmpAmp => "&&".into(),
        TokenKind::PipePipe => "||".into(),
        TokenKind::Bang => "!".into(),
        TokenKind::Eq => "=".into(),
        TokenKind::Arrow => "->".into(),
        TokenKind::DotDot => "..".into(),
        TokenKind::DotDotEq => "..=".into(),
        TokenKind::ColonColon => "::".into(),
        TokenKind::LParen => "(".into(),
        TokenKind::RParen => ")".into(),
        TokenKind::LBracket => "[".into(),
        TokenKind::RBracket => "]".into(),
        TokenKind::LBrace => "{".into(),
        TokenKind::RBrace => "}".into(),
        TokenKind::Comma => ",".into(),
        TokenKind::Dot => ".".into(),
        TokenKind::Colon => ":".into(),
        TokenKind::Semicolon => ";".into(),
        TokenKind::At => "@".into(),
        TokenKind::Eof => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_simple_fn() {
        let input = "fn f(x:f32)->f32{return x}";
        let result = format_source(input).unwrap();
        assert!(result.contains("fn f(x: f32) -> f32"));
        assert!(result.contains("return x"));
    }

    #[test]
    fn test_format_adds_indentation() {
        let input = "fn f() { let x = 42 return x }";
        let result = format_source(input).unwrap();
        assert!(result.contains("    let x = 42"));
        assert!(result.contains("    return x"));
    }

    #[test]
    fn test_format_import() {
        let input = "import tpt::nn";
        let result = format_source(input).unwrap();
        assert!(result.contains("import tpt::nn"));
    }
}
