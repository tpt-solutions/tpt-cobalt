//! Real model tokenizer parsed from a GGUF file's `tokenizer.ggml.*` metadata.
//!
//! GGUF stores the vocabulary as `tokenizer.ggml.tokens` (a STRING array where
//! the index is the token id), BPE merge rules as `tokenizer.ggml.merges`
//! (`"a b"` pairs, applied in rank order), plus bos/eos/unknown ids and the
//! `add_bos_token` / `add_eos_token` flags. This lets `tpt-serve` map real
//! model text to and from token ids instead of the session-local placeholder
//! scheme used by `tpt-gpu-serve`'s `WordTokenizer`.

use std::collections::HashMap;

/// A GGUF/SentencePiece-style tokenizer: vocab + BPE merges + special tokens.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Underlying model family (e.g. `"llama"`, `"gpt2"`).
    pub model: String,
    /// `vocab[id]` is the surface string for token `id`.
    pub vocab: Vec<String>,
    /// BPE merge rules as `(left, right)` pairs, applied in rank (list) order.
    pub merges: Vec<(String, String)>,
    /// Begin-of-sequence token id.
    pub bos: u32,
    /// End-of-sequence token id.
    pub eos: u32,
    /// Unknown/unk token id (used for byte-fallback gaps).
    pub unk: u32,
    /// Whether to prepend `bos` during `encode`.
    pub add_bos: bool,
    /// Whether to append `eos` during `encode`.
    pub add_eos: bool,
    /// Reverse map `vocab_string -> id` for `encode`.
    vocab_id: HashMap<String, u32>,
}

impl Tokenizer {
    /// Build a tokenizer from its raw components, deriving the reverse index.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: impl Into<String>,
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        bos: u32,
        eos: u32,
        unk: u32,
        add_bos: bool,
        add_eos: bool,
    ) -> Self {
        let vocab_id = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self {
            model: model.into(),
            vocab,
            merges,
            bos,
            eos,
            unk,
            add_bos,
            add_eos,
            vocab_id,
        }
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Surface string for a token id, if in range.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Split text into pre-tokens, attaching a single leading space to every
    /// word after the first so a space-delimited BPE vocabulary (tokens stored
    /// with an explicit space prefix like `" the"`) aligns correctly.
    fn pretokenize(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut first = true;
        for c in text.chars() {
            if c.is_whitespace() {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur = " ".to_string();
            } else if cur.is_empty() && !first {
                cur = " ".to_string();
                cur.push(c);
            } else {
                cur.push(c);
            }
            first = false;
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Encode text into token ids: pre-tokenize, run BPE merges in rank order,
    /// then map each resulting symbol to its vocab id (byte-fallback for any
    /// symbol absent from the vocabulary). Prepends/appends bos/eos if enabled.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.push(self.bos);
        }

        let merge_rank: HashMap<(String, String), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        for piece in Self::pretokenize(text) {
            // Fast path: the whole pre-token is itself a vocab entry (common for
            // whole-word tokens stored directly in the vocabulary).
            if let Some(&id) = self.vocab_id.get(&piece) {
                out.push(id);
                continue;
            }
            // Otherwise start with one symbol per Unicode char and merge.
            let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();

            // Greedily apply the highest-priority (lowest rank) adjacent merge
            // until no listed pair remains.
            loop {
                let mut best_rank: Option<usize> = None;
                let mut best_idx: Option<usize> = None;
                for i in 0..symbols.len().saturating_sub(1) {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    if let Some(&rank) = merge_rank.get(&pair) {
                        if best_rank.is_none_or(|r| rank < r) {
                            best_rank = Some(rank);
                            best_idx = Some(i);
                        }
                    }
                }
                match best_idx {
                    Some(i) => {
                        symbols[i] = format!("{}{}", symbols[i], symbols[i + 1]);
                        symbols.remove(i + 1);
                    }
                    None => break,
                }
            }

            // Map each symbol to a token id, with byte-fallback for unknowns.
            for sym in symbols {
                if let Some(&id) = self.vocab_id.get(&sym) {
                    out.push(id);
                } else {
                    for b in sym.bytes() {
                        let tok = format!("<0x{:02X}>", b);
                        if let Some(&id) = self.vocab_id.get(&tok) {
                            out.push(id);
                        } else {
                            out.push(self.unk);
                        }
                    }
                }
            }
        }

        if self.add_eos {
            out.push(self.eos);
        }
        out
    }

    /// Decode token ids back to text: concatenate the surface strings and
    /// collapse any `<0xXX>` byte-fallback tokens into their raw bytes (so the
    /// result may contain UTF-8 multibyte sequences).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let mut out = String::new();
        for &id in ids {
            let Some(tok) = self.vocab.get(id as usize) else {
                continue;
            };
            if let Some(b) = byte_fallback_value(tok) {
                bytes.push(b);
            } else {
                if !bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    bytes.clear();
                }
                out.push_str(tok);
            }
        }
        if !bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&bytes));
        }
        out
    }
}

/// If `tok` is a byte-fallback token of the form `<0xXX>`, return the byte.
fn byte_fallback_value(tok: &str) -> Option<u8> {
    let rest = tok.strip_prefix("<0x")?;
    let rest = rest.strip_suffix('>')?;
    if rest.len() != 2 {
        return None;
    }
    u8::from_str_radix(rest, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Tokenizer {
        // Vocabulary covers single chars plus one merged digraph, so BPE and
        // byte-fallback both get exercised.
        let vocab = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "ll".to_string(),
            " ".to_string(),
        ];
        let merges = vec![("l".to_string(), "l".to_string())];
        Tokenizer::new("gpt2", vocab, merges, 0, 0, 0, false, false)
    }

    #[test]
    fn encode_applies_bpe_merge() {
        let t = sample();
        // "hello" -> h, e, l, l, o -> merge "l"+"l" -> "ll" -> [h, e, ll, o]
        let ids = t.encode("hello");
        assert_eq!(ids, vec![0, 1, 4, 3]);
        assert_eq!(t.decode(&ids), "hello");
    }

    #[test]
    fn decode_round_trips_text() {
        let t = sample();
        let text = "hello";
        let ids = t.encode(text);
        assert_eq!(t.decode(&ids), text);
    }

    #[test]
    fn byte_fallback_for_unknown_symbol() {
        let t = sample();
        // 'x' is not in the vocab, so it falls back to the <0xXX> byte token.
        // There is no <0x78> token in the sample vocab, so it maps to unk (0).
        let ids = t.encode("x");
        assert_eq!(ids, vec![0]); // 'x' -> unk (0); 'h' id is 0

        // Use a vocab with the byte token to verify real byte fallback.
        let vocab = vec!["a".into(), "b".into(), "<0x78>".into()];
        let t2 = Tokenizer::new("gpt2", vocab, vec![], 0, 0, 0, false, false);
        assert_eq!(t2.encode("x"), vec![2]);
        assert_eq!(t2.decode(&[2]), "x");
    }

    #[test]
    fn add_bos_eos_wraps_output() {
        let vocab = vec!["<s>".into(), "the".into(), "</s>".into()];
        let t = Tokenizer::new("llama", vocab, vec![], 0, 2, 0, true, true);
        assert_eq!(t.encode("the"), vec![0, 1, 2]);
        // sanity: decode ignores bos/eos values beyond string concat
        assert_eq!(t.decode(&[0, 1, 2]), "<s>the</s>");
    }
}
