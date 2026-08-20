use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// Byte-offset span for a token, plus the line/column of the start position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Byte offset of first character (inclusive).
    pub start: usize,
    /// Byte offset one past the last character (exclusive).
    pub end: usize,
    /// 1-based line number at `start`.
    pub line: u32,
    /// 1-based column number at `start`.
    pub col: u32,
}

impl Span {
    pub fn new(start: usize, end: usize, line: u32, col: u32) -> Self {
        Self {
            start,
            end,
            line,
            col,
        }
    }

    pub fn dummy() -> Self {
        Self {
            start: 0,
            end: 0,
            line: 0,
            col: 0,
        }
    }

    /// Return the smallest span that covers both `self` and `other`.
    pub fn merge(&self, other: &Span) -> Span {
        let (first, second) = if self.start <= other.start {
            (self, other)
        } else {
            (other, self)
        };
        Span {
            start: first.start,
            end: second.end.max(first.end),
            line: first.line,
            col: first.col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ---------------------------------------------------------------------------
// TokenKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // --- Keywords ---
    KwBreak,
    KwContinue,
    KwElse,
    KwFn,
    KwFor,
    KwIf,
    KwImport,
    KwIn,
    KwLet,
    KwReturn,
    KwType,
    KwWhile,

    // --- Literals ---
    IntLit(i64),
    FloatLit(f64),
    /// `true` and `false` are keywords that produce BoolLit tokens.
    BoolLit(bool),
    StringLit(String),

    // --- Identifier ---
    Ident(String),

    // --- Operators ---
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=
    AmpAmp,     // &&
    PipePipe,   // ||
    Bang,       // !
    Eq,         // =
    Arrow,      // ->
    DotDot,     // ..
    DotDotEq,   // ..=
    ColonColon, // ::

    // --- Punctuation ---
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    Comma,     // ,
    Dot,       // .
    Colon,     // :
    Semicolon, // ;
    At,        // @

    // --- Sentinel ---
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::KwBreak => write!(f, "break"),
            TokenKind::KwContinue => write!(f, "continue"),
            TokenKind::KwElse => write!(f, "else"),
            TokenKind::KwFn => write!(f, "fn"),
            TokenKind::KwFor => write!(f, "for"),
            TokenKind::KwIf => write!(f, "if"),
            TokenKind::KwImport => write!(f, "import"),
            TokenKind::KwIn => write!(f, "in"),
            TokenKind::KwLet => write!(f, "let"),
            TokenKind::KwReturn => write!(f, "return"),
            TokenKind::KwType => write!(f, "type"),
            TokenKind::KwWhile => write!(f, "while"),
            TokenKind::IntLit(n) => write!(f, "{n}"),
            TokenKind::FloatLit(n) => write!(f, "{n}"),
            TokenKind::BoolLit(b) => write!(f, "{b}"),
            TokenKind::StringLit(s) => write!(f, "\"{s}\""),
            TokenKind::Ident(s) => write!(f, "{s}"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::BangEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::AmpAmp => write!(f, "&&"),
            TokenKind::PipePipe => write!(f, "||"),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::DotDotEq => write!(f, "..="),
            TokenKind::ColonColon => write!(f, "::"),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Eof => write!(f, "<eof>"),
        }
    }
}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

// ---------------------------------------------------------------------------
// LexError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Error)]
pub enum LexError {
    #[error("unexpected character '{c}' at {line}:{col}")]
    UnexpectedChar { c: char, line: u32, col: u32 },

    #[error("unterminated string literal starting at {line}:{col}")]
    UnterminatedString { line: u32, col: u32 },

    #[error("unterminated block comment starting at {line}:{col}")]
    UnterminatedBlockComment { line: u32, col: u32 },

    #[error("invalid escape sequence '\\{c}' at {line}:{col}")]
    InvalidEscape { c: char, line: u32, col: u32 },

    #[error("invalid numeric literal '{s}' at {line}:{col}")]
    InvalidNumber { s: String, line: u32, col: u32 },
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

pub struct Lexer<'src> {
    #[allow(dead_code)]
    source: &'src str,
    /// Iterator over the characters that come *after* `current`.
    chars: std::str::Chars<'src>,
    /// The lookahead character (None = EOF).
    current: Option<char>,
    /// Byte offset of `current`.
    pos: usize,
    line: u32,
    col: u32,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        let mut chars = source.chars();
        let current = chars.next();
        Self {
            source,
            chars,
            current,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Low-level character helpers
    // -----------------------------------------------------------------------

    fn peek(&self) -> Option<char> {
        self.current
    }

    /// Peek at the character *after* the current one without consuming anything.
    fn peek2(&self) -> Option<char> {
        self.chars.clone().next()
    }

    /// Consume the current character and return it. Returns `None` at EOF.
    fn advance(&mut self) -> Option<char> {
        let c = self.current?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.pos += c.len_utf8();
        self.current = self.chars.next();
        Some(c)
    }

    // -----------------------------------------------------------------------
    // Comment / whitespace skipping
    // -----------------------------------------------------------------------

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r' | '\n')) {
            self.advance();
        }
    }

    fn skip_line_comment(&mut self) {
        // Caller already consumed `//`.
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        // Caller already consumed `/*`.
        let (err_line, err_col) = (self.line, self.col);
        loop {
            match self.peek() {
                None => {
                    return Err(LexError::UnterminatedBlockComment {
                        line: err_line,
                        col: err_col,
                    })
                }
                Some('*') if self.peek2() == Some('/') => {
                    self.advance(); // *
                    self.advance(); // /
                    return Ok(());
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Literal helpers
    // -----------------------------------------------------------------------

    fn lex_string(&mut self, err_line: u32, err_col: u32) -> Result<String, LexError> {
        // Caller already consumed the opening `"`.
        let mut s = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(LexError::UnterminatedString {
                        line: err_line,
                        col: err_col,
                    })
                }
                Some('"') => return Ok(s),
                Some('\\') => {
                    let (esc_line, esc_col) = (self.line, self.col);
                    match self.advance() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('\\') => s.push('\\'),
                        Some('"') => s.push('"'),
                        Some(c) => {
                            return Err(LexError::InvalidEscape {
                                c,
                                line: esc_line,
                                col: esc_col,
                            })
                        }
                        None => {
                            return Err(LexError::UnterminatedString {
                                line: err_line,
                                col: err_col,
                            })
                        }
                    }
                }
                Some(c) => s.push(c),
            }
        }
    }

    fn lex_number(
        &mut self,
        first: char,
        err_line: u32,
        err_col: u32,
    ) -> Result<TokenKind, LexError> {
        let mut s = String::new();
        s.push(first);
        let mut is_float = false;

        // Consume integer digits (underscores allowed as separators).
        while matches!(self.peek(), Some('0'..='9' | '_')) {
            let c = self.advance().unwrap();
            if c != '_' {
                s.push(c);
            }
        }

        // Decimal point — but only if NOT followed by another `.` (range operator).
        if self.peek() == Some('.') && self.peek2() != Some('.') {
            // Only treat as float decimal if followed by a digit.
            if matches!(self.peek2(), Some('0'..='9')) {
                is_float = true;
                self.advance(); // consume '.'
                s.push('.');
                while matches!(self.peek(), Some('0'..='9' | '_')) {
                    let c = self.advance().unwrap();
                    if c != '_' {
                        s.push(c);
                    }
                }
            }
        }

        // Optional exponent: `e` or `E`, optional sign, digits.
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            s.push(self.advance().unwrap()); // e/E
            if matches!(self.peek(), Some('+' | '-')) {
                s.push(self.advance().unwrap());
            }
            while matches!(self.peek(), Some('0'..='9')) {
                s.push(self.advance().unwrap());
            }
        }

        if is_float {
            s.parse::<f64>()
                .map(TokenKind::FloatLit)
                .map_err(|_| LexError::InvalidNumber {
                    s,
                    line: err_line,
                    col: err_col,
                })
        } else {
            s.parse::<i64>()
                .map(TokenKind::IntLit)
                .map_err(|_| LexError::InvalidNumber {
                    s,
                    line: err_line,
                    col: err_col,
                })
        }
    }

    // -----------------------------------------------------------------------
    // Keyword / identifier resolution
    // -----------------------------------------------------------------------

    fn classify_ident(s: &str) -> TokenKind {
        match s {
            "break" => TokenKind::KwBreak,
            "continue" => TokenKind::KwContinue,
            "else" => TokenKind::KwElse,
            "false" => TokenKind::BoolLit(false),
            "fn" => TokenKind::KwFn,
            "for" => TokenKind::KwFor,
            "if" => TokenKind::KwIf,
            "import" => TokenKind::KwImport,
            "in" => TokenKind::KwIn,
            "let" => TokenKind::KwLet,
            "return" => TokenKind::KwReturn,
            "true" => TokenKind::BoolLit(true),
            "type" => TokenKind::KwType,
            "while" => TokenKind::KwWhile,
            _ => TokenKind::Ident(s.to_string()),
        }
    }

    // -----------------------------------------------------------------------
    // Main tokenization loop
    // -----------------------------------------------------------------------

    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            let start_pos = self.pos;
            let start_line = self.line;
            let start_col = self.col;

            let c = match self.peek() {
                None => {
                    tokens.push(Token::new(
                        TokenKind::Eof,
                        Span::new(start_pos, start_pos, start_line, start_col),
                    ));
                    break;
                }
                Some(c) => c,
            };

            let kind = match c {
                // Comments and division
                '/' => {
                    self.advance();
                    match self.peek() {
                        Some('/') => {
                            self.advance();
                            self.skip_line_comment();
                            continue;
                        }
                        Some('*') => {
                            self.advance();
                            self.skip_block_comment()?;
                            continue;
                        }
                        _ => TokenKind::Slash,
                    }
                }

                // String literal
                '"' => {
                    self.advance();
                    TokenKind::StringLit(self.lex_string(start_line, start_col)?)
                }

                // Numeric literals
                '0'..='9' => {
                    let first = self.advance().unwrap();
                    self.lex_number(first, start_line, start_col)?
                }

                // Identifiers and keywords
                'A'..='Z' | 'a'..='z' | '_' => {
                    let mut s = String::new();
                    while matches!(self.peek(), Some('A'..='Z' | 'a'..='z' | '0'..='9' | '_')) {
                        s.push(self.advance().unwrap());
                    }
                    Self::classify_ident(&s)
                }

                // Single-character or disambiguated multi-character operators
                '+' => {
                    self.advance();
                    TokenKind::Plus
                }

                '-' => {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        TokenKind::Arrow
                    } else {
                        TokenKind::Minus
                    }
                }

                '*' => {
                    self.advance();
                    TokenKind::Star
                }
                '%' => {
                    self.advance();
                    TokenKind::Percent
                }

                '=' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::EqEq
                    } else {
                        TokenKind::Eq
                    }
                }

                '!' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::BangEq
                    } else {
                        TokenKind::Bang
                    }
                }

                '<' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::LtEq
                    } else {
                        TokenKind::Lt
                    }
                }

                '>' => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::GtEq
                    } else {
                        TokenKind::Gt
                    }
                }

                '&' => {
                    self.advance();
                    if self.peek() == Some('&') {
                        self.advance();
                        TokenKind::AmpAmp
                    } else {
                        return Err(LexError::UnexpectedChar {
                            c: '&',
                            line: start_line,
                            col: start_col,
                        });
                    }
                }

                '|' => {
                    self.advance();
                    if self.peek() == Some('|') {
                        self.advance();
                        TokenKind::PipePipe
                    } else {
                        return Err(LexError::UnexpectedChar {
                            c: '|',
                            line: start_line,
                            col: start_col,
                        });
                    }
                }

                '.' => {
                    self.advance();
                    if self.peek() == Some('.') {
                        self.advance();
                        if self.peek() == Some('=') {
                            self.advance();
                            TokenKind::DotDotEq
                        } else {
                            TokenKind::DotDot
                        }
                    } else {
                        TokenKind::Dot
                    }
                }

                ':' => {
                    self.advance();
                    if self.peek() == Some(':') {
                        self.advance();
                        TokenKind::ColonColon
                    } else {
                        TokenKind::Colon
                    }
                }

                '(' => {
                    self.advance();
                    TokenKind::LParen
                }
                ')' => {
                    self.advance();
                    TokenKind::RParen
                }
                '[' => {
                    self.advance();
                    TokenKind::LBracket
                }
                ']' => {
                    self.advance();
                    TokenKind::RBracket
                }
                '{' => {
                    self.advance();
                    TokenKind::LBrace
                }
                '}' => {
                    self.advance();
                    TokenKind::RBrace
                }
                ',' => {
                    self.advance();
                    TokenKind::Comma
                }
                ';' => {
                    self.advance();
                    TokenKind::Semicolon
                }
                '@' => {
                    self.advance();
                    TokenKind::At
                }

                c => {
                    self.advance();
                    return Err(LexError::UnexpectedChar {
                        c,
                        line: start_line,
                        col: start_col,
                    });
                }
            };

            let end_pos = self.pos;
            tokens.push(Token::new(
                kind,
                Span::new(start_pos, end_pos, start_line, start_col),
            ));
        }

        Ok(tokens)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Tokenize `source` and return the token list (always ends with `Eof`).
pub fn tokenize(source: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(source).tokenize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_keywords() {
        let ks = kinds("fn let return if else for while break continue import in type");
        assert_eq!(ks[0], TokenKind::KwFn);
        assert_eq!(ks[1], TokenKind::KwLet);
        assert_eq!(ks[2], TokenKind::KwReturn);
        assert_eq!(ks[3], TokenKind::KwIf);
        assert_eq!(ks[4], TokenKind::KwElse);
        assert_eq!(ks[5], TokenKind::KwFor);
        assert_eq!(ks[6], TokenKind::KwWhile);
        assert_eq!(ks[7], TokenKind::KwBreak);
        assert_eq!(ks[8], TokenKind::KwContinue);
        assert_eq!(ks[9], TokenKind::KwImport);
        assert_eq!(ks[10], TokenKind::KwIn);
        assert_eq!(ks[11], TokenKind::KwType);
    }

    #[test]
    fn test_bool_literals() {
        let ks = kinds("true false");
        assert_eq!(ks[0], TokenKind::BoolLit(true));
        assert_eq!(ks[1], TokenKind::BoolLit(false));
    }

    #[test]
    fn test_int_literals() {
        let ks = kinds("0 42 1_000_000");
        assert_eq!(ks[0], TokenKind::IntLit(0));
        assert_eq!(ks[1], TokenKind::IntLit(42));
        assert_eq!(ks[2], TokenKind::IntLit(1_000_000));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_float_literals() {
        let ks = kinds("3.14 1.0e-5 0.5");
        assert_eq!(ks[0], TokenKind::FloatLit(3.14));
        assert_eq!(ks[1], TokenKind::FloatLit(1.0e-5));
        assert_eq!(ks[2], TokenKind::FloatLit(0.5));
    }

    #[test]
    fn test_string_literal() {
        let ks = kinds(r#""hello\nworld""#);
        assert_eq!(ks[0], TokenKind::StringLit("hello\nworld".to_string()));
    }

    #[test]
    fn test_operators() {
        let ks = kinds("+ - * / % == != < > <= >= && || ! = -> .. ..= ::");
        use TokenKind::*;
        let expected = [
            Plus, Minus, Star, Slash, Percent, EqEq, BangEq, Lt, Gt, LtEq, GtEq, AmpAmp, PipePipe,
            Bang, Eq, Arrow, DotDot, DotDotEq, ColonColon,
        ];
        for (k, e) in ks.iter().zip(expected.iter()) {
            assert_eq!(k, e);
        }
    }

    #[test]
    fn test_range_not_float() {
        // `1..n` should be IntLit(1) + DotDot + Ident("n"), not a float parse attempt.
        let ks = kinds("1..n");
        assert_eq!(ks[0], TokenKind::IntLit(1));
        assert_eq!(ks[1], TokenKind::DotDot);
        assert_eq!(ks[2], TokenKind::Ident("n".to_string()));
    }

    #[test]
    fn test_line_comment_skipped() {
        let ks = kinds("let // this is a comment\nx");
        assert_eq!(ks[0], TokenKind::KwLet);
        assert_eq!(ks[1], TokenKind::Ident("x".to_string()));
    }

    #[test]
    fn test_block_comment_skipped() {
        let ks = kinds("let /* comment */ x");
        assert_eq!(ks[0], TokenKind::KwLet);
        assert_eq!(ks[1], TokenKind::Ident("x".to_string()));
    }

    #[test]
    fn test_span_line_col() {
        let tokens = tokenize("let x").unwrap();
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[0].span.col, 1);
        assert_eq!(tokens[1].span.line, 1);
        assert_eq!(tokens[1].span.col, 5);
    }
}
