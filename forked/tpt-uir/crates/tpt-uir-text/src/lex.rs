//! Lexer for the TPT-UIR textual format.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Caret,   // ^
    Hash,    // #
    Percent, // %
    Colon,   // :
    Equal,   // =
    Comma,   // ,
    Less,    // <
    Greater, // >
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
}

#[derive(Debug)]
pub struct LexError {
    pub pos: usize,
    pub message: String,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, LexError> {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();
    let mut out = Vec::new();

    while i < n {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            b'#' => {
                out.push(Tok::Hash);
                i += 1;
            }
            b'%' => {
                out.push(Tok::Percent);
                i += 1;
            }
            b':' => {
                out.push(Tok::Colon);
                i += 1;
            }
            b'=' => {
                out.push(Tok::Equal);
                i += 1;
            }
            b',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            b'<' => {
                out.push(Tok::Less);
                i += 1;
            }
            b'>' => {
                out.push(Tok::Greater);
                i += 1;
            }
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b'{' => {
                out.push(Tok::LBrace);
                i += 1;
            }
            b'}' => {
                out.push(Tok::RBrace);
                i += 1;
            }
            b'[' => {
                out.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                out.push(Tok::RBracket);
                i += 1;
            }
            b'"' => {
                i += 1;
                let mut s = String::new();
                while i < n && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < n {
                        i += 1;
                        match bytes[i] {
                            b'"' => s.push('"'),
                            b'\\' => s.push('\\'),
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            other => s.push(other as char),
                        }
                    } else {
                        s.push(bytes[i] as char);
                    }
                    i += 1;
                }
                if i >= n {
                    return Err(LexError {
                        pos: i,
                        message: "unterminated string".to_string(),
                    });
                }
                i += 1; // closing quote
                out.push(Tok::Str(s));
            }
            b'-' if i + 1 < n && bytes[i + 1].is_ascii_digit() => {
                let (tok, ni) = lex_number(src, i)?;
                out.push(tok);
                i = ni;
            }
            _ if c.is_ascii_digit() => {
                let (tok, ni) = lex_number(src, i)?;
                out.push(tok);
                i = ni;
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < n && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                out.push(Tok::Ident(src[start..i].to_string()));
            }
            other => {
                return Err(LexError {
                    pos: i,
                    message: format!("unexpected character '{}'", other as char),
                });
            }
        }
    }
    Ok(out)
}

fn lex_number(src: &str, start: usize) -> Result<(Tok, usize), LexError> {
    let bytes = src.as_bytes();
    let mut i = start;
    if bytes[i] == b'-' {
        i += 1;
    }
    let int_start = start;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let f: f64 = src[start..i].parse().map_err(|_| LexError {
            pos: start,
            message: "invalid float".to_string(),
        })?;
        Ok((Tok::Float(f), i))
    } else {
        let v: i64 = src[int_start..i].parse().map_err(|_| LexError {
            pos: start,
            message: "invalid integer".to_string(),
        })?;
        Ok((Tok::Int(v), i))
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_continue(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.'
}
