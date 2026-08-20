//! BibTeX entries, a small BibTeX parser, and duplicate-key detection.

use std::collections::BTreeMap;

use crate::error::DocError;

/// One bibliography entry: `@article{key, field = {value}, ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BibEntry {
    /// Entry type, lowercased (`article`, `book`, `inproceedings`, ...).
    pub kind: String,
    /// Citation key, as used by `cite("key")`.
    pub key: String,
    /// Fields, lowercased names, in deterministic (sorted) order.
    pub fields: BTreeMap<String, String>,
}

impl BibEntry {
    pub fn new(kind: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            kind: kind.into().to_lowercase(),
            key: key.into(),
            fields: BTreeMap::new(),
        }
    }

    /// Builder-style field setter (used by the [`bib!`](crate::bib) macro).
    pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(name.into().to_lowercase(), value.into());
        self
    }

    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(|s| s.as_str())
    }
    pub fn title(&self) -> Option<&str> {
        self.field("title")
    }
    pub fn author(&self) -> Option<&str> {
        self.field("author")
    }
    pub fn year(&self) -> Option<&str> {
        self.field("year")
    }

    /// A deterministic single-line rendering used by the reference lists.
    pub fn format(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(a) = self.author() {
            parts.push(a.to_string());
        }
        if let Some(t) = self.title() {
            parts.push(t.to_string());
        }
        for f in ["journal", "booktitle", "publisher"] {
            if let Some(v) = self.field(f) {
                parts.push(v.to_string());
                break;
            }
        }
        if let Some(y) = self.year() {
            parts.push(y.to_string());
        }
        if parts.is_empty() {
            self.key.clone()
        } else {
            format!("{}.", parts.join(". "))
        }
    }
}

/// An ordered set of [`BibEntry`] values.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bibliography {
    entries: Vec<BibEntry>,
}

impl Bibliography {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append without checking; duplicates are reported by
    /// [`Bibliography::check_duplicates`] (and therefore by `Document::validate`).
    pub fn push(&mut self, entry: BibEntry) {
        self.entries.push(entry);
    }

    /// Append, rejecting a key that is already present.
    pub fn try_push(&mut self, entry: BibEntry) -> Result<(), DocError> {
        if self.contains(&entry.key) {
            return Err(DocError::DuplicateKey(entry.key));
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Merge another bibliography, rejecting duplicate keys.
    pub fn try_extend(&mut self, other: Bibliography) -> Result<(), DocError> {
        for e in other.entries {
            self.try_push(e)?;
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&BibEntry> {
        self.entries.iter().find(|e| e.key == key)
    }
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    pub fn entries(&self) -> &[BibEntry] {
        &self.entries
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `Err(DuplicateKey)` if any key occurs twice.
    pub fn check_duplicates(&self) -> Result<(), DocError> {
        for (i, e) in self.entries.iter().enumerate() {
            if self.entries[..i].iter().any(|p| p.key == e.key) {
                return Err(DocError::DuplicateKey(e.key.clone()));
            }
        }
        Ok(())
    }

    /// Parse a `.bib` source.
    ///
    /// Supports `@type{key, field = {braced}, field = "quoted", field = bare}`,
    /// `(...)`-delimited entries, `#` concatenation, and skips `@comment`,
    /// `@preamble` and `@string` blocks. Whitespace inside values is collapsed
    /// so output is deterministic. Duplicate keys are an error.
    pub fn parse(src: &str) -> Result<Self, DocError> {
        let s: Vec<char> = src.chars().collect();
        let mut i = 0usize;
        let mut bib = Bibliography::new();
        while i < s.len() {
            if s[i] != '@' {
                i += 1;
                continue;
            }
            i += 1;
            let kind = read_ident(&s, &mut i).to_lowercase();
            if kind.is_empty() {
                return Err(err(i, "expected an entry type after `@`"));
            }
            skip_ws(&s, &mut i);
            let close = match s.get(i) {
                Some('{') => '}',
                Some('(') => ')',
                _ => return Err(err(i, format!("expected `{{` after `@{kind}`"))),
            };
            i += 1;
            if matches!(kind.as_str(), "comment" | "preamble" | "string") {
                skip_group(&s, &mut i, close);
                continue;
            }
            skip_ws(&s, &mut i);
            let mut key = String::new();
            while let Some(&c) = s.get(i) {
                if c == ',' || c == close {
                    break;
                }
                key.push(c);
                i += 1;
            }
            let key = key.trim().to_string();
            if key.is_empty() {
                return Err(err(i, format!("`@{kind}` entry has no citation key")));
            }
            let mut entry = BibEntry::new(kind, key);
            loop {
                skip_ws(&s, &mut i);
                match s.get(i) {
                    None => return Err(err(i, format!("unterminated entry `{}`", entry.key))),
                    Some(&c) if c == close => {
                        i += 1;
                        break;
                    }
                    Some(',') => {
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
                let name = read_ident(&s, &mut i).to_lowercase();
                if name.is_empty() {
                    return Err(err(i, format!("expected a field name in `{}`", entry.key)));
                }
                skip_ws(&s, &mut i);
                if s.get(i) != Some(&'=') {
                    return Err(err(i, format!("expected `=` after field `{name}`")));
                }
                i += 1;
                let value = read_value(&s, &mut i)?;
                entry.fields.insert(name, value);
            }
            bib.try_push(entry)?;
        }
        Ok(bib)
    }
}

fn err(pos: usize, msg: impl Into<String>) -> DocError {
    DocError::Bibtex {
        pos,
        msg: msg.into(),
    }
}

fn skip_ws(s: &[char], i: &mut usize) {
    while matches!(s.get(*i), Some(c) if c.is_whitespace()) {
        *i += 1;
    }
}

fn read_ident(s: &[char], i: &mut usize) -> String {
    skip_ws(s, i);
    let start = *i;
    while matches!(s.get(*i), Some(c) if c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.' || *c == ':')
    {
        *i += 1;
    }
    s[start..*i].iter().collect()
}

/// Consume tokens up to and including the matching `close` delimiter.
fn skip_group(s: &[char], i: &mut usize, close: char) {
    let open = if close == '}' { '{' } else { '(' };
    let mut depth = 1usize;
    while let Some(&c) = s.get(*i) {
        *i += 1;
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return;
            }
        }
    }
}

/// Read one (possibly `#`-concatenated) field value.
fn read_value(s: &[char], i: &mut usize) -> Result<String, DocError> {
    let mut out = String::new();
    loop {
        skip_ws(s, i);
        match s.get(*i) {
            Some('{') => {
                *i += 1;
                let mut depth = 1usize;
                while let Some(&c) = s.get(*i) {
                    *i += 1;
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    if depth > 0 || c != '}' {
                        out.push(c);
                    }
                }
                if depth != 0 {
                    return Err(err(*i, "unterminated `{` in field value"));
                }
            }
            Some('"') => {
                *i += 1;
                let mut depth = 0usize;
                loop {
                    match s.get(*i) {
                        None => return Err(err(*i, "unterminated `\"` in field value")),
                        Some('{') => {
                            depth += 1;
                            out.push('{');
                        }
                        Some('}') => {
                            depth = depth.saturating_sub(1);
                            out.push('}');
                        }
                        Some('"') if depth == 0 => {
                            *i += 1;
                            break;
                        }
                        Some(&c) => out.push(c),
                    }
                    *i += 1;
                }
            }
            Some(c) if c.is_alphanumeric() || *c == '_' => out.push_str(&read_ident(s, i)),
            _ => return Err(err(*i, "expected a field value")),
        }
        skip_ws(s, i);
        if s.get(*i) == Some(&'#') {
            *i += 1;
            continue;
        }
        break;
    }
    Ok(collapse_ws(&out))
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
