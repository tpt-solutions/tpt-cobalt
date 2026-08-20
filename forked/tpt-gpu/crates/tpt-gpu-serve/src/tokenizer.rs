//! Minimal, reversible-within-a-session tokenizer.
//!
//! The TPT GPU runtime does not currently expose a model vocabulary, so this
//! is a *placeholder* tokenizer that lets `tpt-serve` accept and emit text for
//! demos, CI, and OpenAI client compatibility. It is intentionally simple:
//!
//! - `encode` splits text on whitespace and ASCII punctuation, assigning each
//!   distinct token a sequential session-local id starting at 1.
//! - `decode` maps ids back to the words seen during the most recent `encode`;
//!   unknown ids render as `<id>`.
//!
//! A real GGUF/SentencePiece vocabulary should replace this (tracked as a
//! follow-up) so generated tokens map back to real text.

use std::collections::HashMap;

pub struct WordTokenizer {
    vocab_size: u32,
    forward: HashMap<String, u32>,
    reverse: HashMap<u32, String>,
    next: u32,
}

impl WordTokenizer {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size: vocab_size.max(1),
            forward: HashMap::new(),
            reverse: HashMap::new(),
            next: 1,
        }
    }

    fn alloc(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.forward.get(word) {
            return id;
        }
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        if self.next >= self.vocab_size {
            self.next = 1;
        }
        self.forward.insert(word.to_string(), id);
        self.reverse.insert(id, word.to_string());
        id
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .flat_map(split_punct)
            .filter(|w| !w.is_empty())
            .map(|w| self.alloc(&w))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                self.reverse
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<{id}>"))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Split a token on ASCII punctuation boundaries so `"hello,"` becomes
/// `["hello", ","]`.
fn split_punct(tok: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in tok.chars() {
        if ch.is_ascii_punctuation() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_is_deterministic() {
        let mut tok = WordTokenizer::new(1024);
        assert_eq!(tok.encode("the cat sat"), tok.encode("the cat sat"));
    }

    #[test]
    fn decode_round_trips_known_tokens() {
        let mut tok = WordTokenizer::new(1024);
        let ids = tok.encode("hello world");
        assert_eq!(tok.decode(&ids), "hello world");
    }

    #[test]
    fn unknown_ids_render_as_angle_bracket() {
        let tok = WordTokenizer::new(1024);
        assert_eq!(tok.decode(&[999]), "<999>");
    }

    #[test]
    fn punctuation_is_split() {
        let mut tok = WordTokenizer::new(1024);
        let ids = tok.encode("hi, there!");
        // "hi" and "there" and the two punctuation marks -> 4 tokens
        assert_eq!(ids.len(), 4);
    }
}
