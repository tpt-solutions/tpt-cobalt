//! Stable TPTIR text-format emitter and parser.
//!
//! The text format is stable from v0.1.0: tools that consume TPTIR (e.g.
//! tpt-crucible Catalyst) can rely on this module for round-trippable output.
//!
//! # Grammar (informally)
//!
//! ```text
//! region   ::= "func" "@" ident "(" arglist ")" ["->" typelist] "{" block+ "}"
//! arglist  ::= (ident ":" type ("," ident ":" type)*)?
//! typelist ::= type | "(" type ("," type)* ")"
//! block    ::= "^" ident ":" instr+
//! instr    ::= ["%" ident "="] op_name [operands] ["{" attrs "}"] [":" type]
//! operands ::= "%" ident ("," "%" ident)*
//! attrs    ::= ident "=" value ("," ident "=" value)*
//! ```

use crate::dialect::TPTIR_DIALECT_VERSION;
use crate::ops::Op;
use crate::types::{AddressSpace, ElemType, Type};

// ---------------------------------------------------------------------------
// Emitter types
// ---------------------------------------------------------------------------

/// Configuration for the text emitter.
#[derive(Debug, Clone)]
pub struct EmitOptions {
    /// Emit a version header comment at the top of the output.
    pub emit_version_header: bool,
    /// Indent string for nested regions (default: two spaces).
    pub indent: String,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            emit_version_header: true,
            indent: "  ".to_string(),
        }
    }
}

/// A basic IR node used for text-format emission and parsing.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub result: Option<String>,
    pub op: Op,
    pub operands: Vec<String>,
    pub attrs: Vec<(String, String)>,
    pub result_type: Option<Type>,
}

/// A basic block containing a label and a list of instructions.
#[derive(Debug, Clone)]
pub struct Block {
    pub label: String,
    pub instructions: Vec<Instruction>,
}

/// A function/kernel region for emission and parsing.
#[derive(Debug, Clone)]
pub struct Region {
    pub name: String,
    pub args: Vec<(String, Type)>,
    pub return_types: Vec<Type>,
    pub blocks: Vec<Block>,
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

/// Emit a region to TPTIR text format.
pub fn emit(region: &Region, opts: &EmitOptions) -> String {
    let mut out = String::new();

    if opts.emit_version_header {
        out.push_str(&format!(
            "// tptir-spec v{} — stable text format\n",
            TPTIR_DIALECT_VERSION
        ));
    }

    let args: Vec<String> = region
        .args
        .iter()
        .map(|(name, ty)| format!("{}: {}", name, ty))
        .collect();
    let rets: Vec<String> = region.return_types.iter().map(|t| t.to_string()).collect();

    let ret_str = if rets.is_empty() {
        String::new()
    } else {
        format!(
            " -> {}",
            if rets.len() == 1 {
                rets[0].clone()
            } else {
                format!("({})", rets.join(", "))
            }
        )
    };

    out.push_str(&format!(
        "func @{}({}){} {{\n",
        region.name,
        args.join(", "),
        ret_str
    ));

    for block in &region.blocks {
        out.push_str(&format!("^{}:\n", block.label));
        for instr in &block.instructions {
            out.push_str(&opts.indent);
            if let Some(res) = &instr.result {
                out.push_str(&format!("%{} = ", res));
            }
            out.push_str(&instr.op.to_string());
            if !instr.operands.is_empty() {
                let ops: Vec<String> = instr.operands.iter().map(|o| format!("%{}", o)).collect();
                out.push_str(&format!(" {}", ops.join(", ")));
            }
            if !instr.attrs.is_empty() {
                let pairs: Vec<String> = instr
                    .attrs
                    .iter()
                    .map(|(k, v)| format!("{} = {}", k, v))
                    .collect();
                out.push_str(&format!(" {{{}}}", pairs.join(", ")));
            }
            if let Some(ty) = &instr.result_type {
                out.push_str(&format!(" : {}", ty));
            }
            out.push('\n');
        }
    }

    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse error with a line number and message.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse TPTIR text into a `Region`.
///
/// This is the primary entry point for cross-tool consumption (e.g. tpt-crucible
/// Catalyst). The output is guaranteed to round-trip through `emit()` for any
/// text produced by this crate.
pub fn parse_region(source: &str) -> Result<Region, ParseError> {
    let lines: Vec<&str> = source.lines().collect();
    let mut pos = 0;

    // Skip leading comments and blank lines to find `func @name(...)`
    while pos < lines.len() {
        let trimmed = lines[pos].trim();
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            break;
        }
        pos += 1;
    }

    if pos >= lines.len() {
        return Err(ParseError {
            line: pos + 1,
            message: "expected 'func @name(...)' but reached end of input".into(),
        });
    }

    let func_line = lines[pos].trim();
    pos += 1;

    if !func_line.starts_with("func @") {
        return Err(ParseError {
            line: pos,
            message: format!("expected 'func @name(...)', got '{}'", func_line),
        });
    }

    // Parse: func @name(args...) -> ret { or func @name(args...) {
    let after_func = &func_line["func @".len()..];

    // Extract function name (up to '(')
    let paren_open = after_func.find('(').ok_or_else(|| ParseError {
        line: pos,
        message: "expected '(' in function signature".into(),
    })?;
    let name = after_func[..paren_open].to_string();

    // Extract arg list (between the outermost parentheses)
    let after_name = &after_func[paren_open + 1..];
    let paren_close = find_matching_paren(after_name).ok_or_else(|| ParseError {
        line: pos,
        message: "unmatched '(' in function signature".into(),
    })?;
    let args_str = &after_name[..paren_close];
    let rest = after_name[paren_close + 1..].trim();

    // Parse args
    let args = parse_arg_list(args_str, pos)?;

    // Parse optional return types: -> type  or  -> (t1, t2)
    let return_types = if let Some(arrow) = rest.strip_prefix("->") {
        let ret_part = arrow.trim().trim_end_matches('{').trim();
        parse_return_types(ret_part, pos)?
    } else {
        vec![]
    };

    // Now parse blocks until closing '}'
    let mut blocks: Vec<Block> = Vec::new();
    let mut current_block: Option<Block> = None;

    while pos < lines.len() {
        let raw = lines[pos];
        let trimmed = raw.trim();
        pos += 1;

        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        if trimmed == "}" {
            break;
        }

        if trimmed.starts_with('^') {
            // New block label: ^label:
            if let Some(b) = current_block.take() {
                blocks.push(b);
            }
            let label = trimmed
                .trim_start_matches('^')
                .trim_end_matches(':')
                .to_string();
            current_block = Some(Block {
                label,
                instructions: Vec::new(),
            });
            continue;
        }

        // It's an instruction line
        let instr = parse_instruction(trimmed, pos)?;
        if let Some(ref mut b) = current_block {
            b.instructions.push(instr);
        } else {
            return Err(ParseError {
                line: pos,
                message: "instruction outside of any block".into(),
            });
        }
    }

    if let Some(b) = current_block {
        blocks.push(b);
    }

    Ok(Region {
        name,
        args,
        return_types,
        blocks,
    })
}

/// Legacy parse entry point — returns block labels found in the source.
/// Prefer `parse_region()` for full structural access.
pub fn parse(source: &str) -> Result<Vec<String>, String> {
    parse_region(source)
        .map(|r| r.blocks.iter().map(|b| b.label.clone()).collect())
        .map_err(|e| e.message)
}

// ---------------------------------------------------------------------------
// Type helpers (public for cross-tool use)
// ---------------------------------------------------------------------------

/// Parse a TPTIR type string into a `Type`.
///
/// Supports the minimal subset needed for cross-tool consumption:
/// scalars, vectors, tensors, and memrefs.
pub fn parse_type(s: &str) -> Result<Type, String> {
    let s = s.trim();
    match s {
        "i1" => return Ok(Type::scalar(ElemType::I1)),
        "i8" => return Ok(Type::scalar(ElemType::I8)),
        "i16" => return Ok(Type::scalar(ElemType::I16)),
        "i32" => return Ok(Type::scalar(ElemType::I32)),
        "i64" => return Ok(Type::scalar(ElemType::I64)),
        "f16" => return Ok(Type::scalar(ElemType::F16)),
        "bf16" => return Ok(Type::scalar(ElemType::BF16)),
        "f32" => return Ok(Type::scalar(ElemType::F32)),
        "f64" => return Ok(Type::scalar(ElemType::F64)),
        "index" => return Ok(Type::scalar(ElemType::Index)),
        "none" => return Ok(Type::none()),
        _ => {}
    }

    if s.starts_with("vector<") && s.ends_with('>') {
        let inner = &s[7..s.len() - 1];
        if let Some(x_pos) = inner.find('x') {
            let lanes: u32 = inner[..x_pos]
                .parse()
                .map_err(|_| format!("bad vector lanes in '{}'", s))?;
            let elem = parse_type(&inner[x_pos + 1..])?;
            return Ok(Type::vector(lanes, elem));
        }
    }

    if s.starts_with("tensor<") && s.ends_with('>') {
        return parse_shaped("tensor", &s[7..s.len() - 1]);
    }

    if s.starts_with("memref<") && s.ends_with('>') {
        return parse_shaped("memref", &s[7..s.len() - 1]);
    }

    Err(format!("unrecognized type '{}'", s))
}

fn parse_shaped(kind: &str, inner: &str) -> Result<Type, String> {
    let (shape_and_elem, addr) = if let Some(comma) = inner.rfind(',') {
        let addr_str = inner[comma + 1..].trim();
        let addr = match addr_str {
            "shared" => AddressSpace::Shared,
            "local" => AddressSpace::Local,
            "constant" => AddressSpace::Constant,
            "generic" => AddressSpace::Generic,
            _ => AddressSpace::Global,
        };
        (&inner[..comma], addr)
    } else {
        (inner, AddressSpace::Global)
    };

    let parts: Vec<&str> = shape_and_elem.split('x').collect();
    if parts.is_empty() {
        return Err(format!("empty {} shape", kind));
    }

    let mut dims = Vec::new();
    let mut elem_idx = parts.len() - 1;
    for (i, part) in parts.iter().enumerate() {
        if part.parse::<i64>().is_ok() || *part == "?" {
            dims.push(if *part == "?" {
                -1
            } else {
                part.parse::<i64>().unwrap()
            });
        } else {
            elem_idx = i;
            break;
        }
    }
    let elem_str = parts[elem_idx..].join("x");
    let elem = parse_type(elem_str.trim())?;

    match kind {
        "tensor" => Ok(Type::tensor(dims, elem, addr)),
        "memref" => Ok(Type::memref(dims, elem, addr)),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Internal parser helpers
// ---------------------------------------------------------------------------

/// Find the index of the ')' that closes the '(' at position 0 of `s`.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Find the index of '}' that closes the '{' at position 0 of `s`, respecting
/// nested angle brackets (for generic types like `tensor<...>`).
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Split a comma-separated list, ignoring commas inside `<>` and `()`.
fn split_comma_aware(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth_angle = 0usize;
    let mut depth_paren = 0usize;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth_angle += 1,
            '>' => depth_angle = depth_angle.saturating_sub(1),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            ',' if depth_angle == 0 && depth_paren == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        parts.push(last);
    }
    parts
}

/// Parse `name: type, name: type, ...` into `Vec<(String, Type)>`.
fn parse_arg_list(s: &str, line: usize) -> Result<Vec<(String, Type)>, ParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }
    let mut args = Vec::new();
    for part in split_comma_aware(s) {
        let colon = part.find(':').ok_or_else(|| ParseError {
            line,
            message: format!("expected 'name: type' in arg, got '{}'", part),
        })?;
        let name = part[..colon].trim().to_string();
        let ty =
            parse_type(part[colon + 1..].trim()).map_err(|e| ParseError { line, message: e })?;
        args.push((name, ty));
    }
    Ok(args)
}

/// Parse `type` or `(type, type, ...)` return type annotation.
fn parse_return_types(s: &str, line: usize) -> Result<Vec<Type>, ParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        return split_comma_aware(inner)
            .into_iter()
            .map(|t| parse_type(t).map_err(|e| ParseError { line, message: e }))
            .collect();
    }
    Ok(vec![
        parse_type(s).map_err(|e| ParseError { line, message: e })?
    ])
}

/// Parse a single instruction line.
///
/// Format: `[%result =] op_name [%op1, %op2, ...] [{k=v, ...}] [: type]`
fn parse_instruction(s: &str, line: usize) -> Result<Instruction, ParseError> {
    let mut rest = s;

    // Extract optional result: `%ident =`
    let result = if rest.starts_with('%') {
        if let Some(eq) = rest.find('=') {
            // Make sure it's `%name =`, not an attr value `key = val`
            let candidate = rest[1..eq].trim();
            if !candidate.contains(' ') && !candidate.is_empty() {
                let r = candidate.to_string();
                rest = rest[eq + 1..].trim();
                Some(r)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Extract op name (first whitespace-delimited token)
    let (op_tok, after_op) = match rest.find(|c: char| c.is_whitespace()) {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    };
    let op = parse_op(op_tok);

    let mut remaining = after_op;

    // Extract operands: `%a, %b, ...` before `{` or `:`
    let mut operands: Vec<String> = Vec::new();
    if remaining.starts_with('%') {
        // Collect operand tokens until we hit `{` or `:`
        let operand_end = remaining.find(['{', ':']).unwrap_or(remaining.len());
        let operand_str = remaining[..operand_end].trim();
        remaining = remaining[operand_end..].trim();
        for tok in split_comma_aware(operand_str) {
            let tok = tok.trim().trim_start_matches('%');
            if !tok.is_empty() {
                operands.push(tok.to_string());
            }
        }
    }

    // Extract attrs: `{key = val, key = val}`
    let mut attrs: Vec<(String, String)> = Vec::new();
    if remaining.starts_with('{') {
        let close = find_matching_brace(&remaining[1..]).ok_or_else(|| ParseError {
            line,
            message: format!("unmatched '{{' in instruction: '{}'", s),
        })?;
        let attr_str = &remaining[1..close + 1];
        remaining = remaining[close + 2..].trim();
        for pair in split_comma_aware(attr_str) {
            if let Some(eq) = pair.find('=') {
                let k = pair[..eq].trim().to_string();
                let v = pair[eq + 1..].trim().to_string();
                attrs.push((k, v));
            }
        }
    }

    // Extract result type: `: type`
    let result_type = if let Some(colon) = remaining.strip_prefix(':') {
        let ty_str = colon.trim();
        Some(parse_type(ty_str).map_err(|e| ParseError { line, message: e })?)
    } else {
        None
    };

    Ok(Instruction {
        result,
        op,
        operands,
        attrs,
        result_type,
    })
}

/// Map a text-format op name to an `Op`, falling back to `Op::Custom`.
pub fn parse_op(s: &str) -> Op {
    match s {
        "addi" => Op::Addi,
        "subi" => Op::Subi,
        "muli" => Op::Muli,
        "divi" => Op::Divi,
        "modi" => Op::Modi,
        "addf" => Op::Addf,
        "subf" => Op::Subf,
        "mulf" => Op::Mulf,
        "divf" => Op::Divf,
        "andi" => Op::And,
        "ori" => Op::Or,
        "xori" => Op::Xor,
        "shl" => Op::Shl,
        "shr" => Op::Shr,
        "cmpeq" => Op::CmpEq,
        "cmpne" => Op::CmpNe,
        "cmplt" => Op::CmpLt,
        "cmple" => Op::CmpLe,
        "cmpgt" => Op::CmpGt,
        "cmpge" => Op::CmpGe,
        "load" => Op::Load,
        "store" => Op::Store,
        "alloc" => Op::Alloc,
        "dealloc" => Op::Dealloc,
        "memcpy" => Op::Memcpy,
        "br" => Op::Branch,
        "cond_br" => Op::CondBranch,
        "return" => Op::Return,
        "call" => Op::Call,
        "constant" => Op::Constant,
        "trunc" => Op::Trunc,
        "extend" => Op::Extend,
        "bitcast" => Op::Bitcast,
        "itof" => Op::IntToFloat,
        "ftoi" => Op::FloatToInt,
        "gemm" => Op::Gemm,
        "matmul" => Op::Matmul,
        "transpose" => Op::Transpose,
        "reduce_sum" => Op::ReduceSum,
        "reduce_max" => Op::ReduceMax,
        "reduce_min" => Op::ReduceMin,
        "quantize" => Op::Quantize,
        "dequantize" => Op::Dequantize,
        "quant_gemm" => Op::QuantGemm,
        "quant_attention" => Op::QuantAttention,
        other => Op::Custom(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ElemType;

    fn f32_tensor(dims: Vec<i64>) -> Type {
        Type::tensor(dims, Type::scalar(ElemType::F32), AddressSpace::Global)
    }

    #[test]
    fn scalar_type_parse() {
        assert_eq!(parse_type("f32").unwrap(), Type::scalar(ElemType::F32));
        assert_eq!(parse_type("i64").unwrap(), Type::scalar(ElemType::I64));
    }

    #[test]
    fn vector_type_parse() {
        let t = parse_type("vector<8xf32>").unwrap();
        assert_eq!(t, Type::vector(8, Type::scalar(ElemType::F32)));
    }

    #[test]
    fn tensor_type_parse() {
        let t = parse_type("tensor<4x4xf16>").unwrap();
        assert_eq!(
            t,
            Type::tensor(
                vec![4, 4],
                Type::scalar(ElemType::F16),
                AddressSpace::Global
            )
        );
    }

    #[test]
    fn emit_round_trip() {
        let region = Region {
            name: "vector_add".to_string(),
            args: vec![
                ("a".to_string(), f32_tensor(vec![1024])),
                ("b".to_string(), f32_tensor(vec![1024])),
            ],
            return_types: vec![f32_tensor(vec![1024])],
            blocks: vec![Block {
                label: "entry".to_string(),
                instructions: vec![
                    Instruction {
                        result: Some("c".to_string()),
                        op: Op::Addf,
                        operands: vec!["a".to_string(), "b".to_string()],
                        attrs: vec![],
                        result_type: Some(f32_tensor(vec![1024])),
                    },
                    Instruction {
                        result: None,
                        op: Op::Return,
                        operands: vec!["c".to_string()],
                        attrs: vec![],
                        result_type: None,
                    },
                ],
            }],
        };

        let opts = EmitOptions::default();
        let text = emit(&region, &opts);
        assert!(text.contains("func @vector_add"));
        assert!(text.contains("^entry:"));
        assert!(text.contains("addf"));
        assert!(text.contains("return"));
    }

    #[test]
    fn parse_region_basic() {
        let src = r#"
func @add(a: tensor<4xf32>, b: tensor<4xf32>) -> tensor<4xf32> {
^entry:
  %c = addf %a, %b : tensor<4xf32>
  return %c
}
"#;
        let region = parse_region(src).unwrap();
        assert_eq!(region.name, "add");
        assert_eq!(region.args.len(), 2);
        assert_eq!(region.args[0].0, "a");
        assert_eq!(region.return_types.len(), 1);
        assert_eq!(region.blocks.len(), 1);
        assert_eq!(region.blocks[0].label, "entry");
        assert_eq!(region.blocks[0].instructions.len(), 2);

        let addf = &region.blocks[0].instructions[0];
        assert_eq!(addf.op, Op::Addf);
        assert_eq!(addf.result.as_deref(), Some("c"));
        assert_eq!(addf.operands, vec!["a", "b"]);

        let ret = &region.blocks[0].instructions[1];
        assert_eq!(ret.op, Op::Return);
        assert_eq!(ret.operands, vec!["c"]);
    }

    #[test]
    fn parse_region_no_return() {
        let src = r#"func @store_only(ptr: memref<1xf32>) {
^bb0:
  store %ptr
}
"#;
        let region = parse_region(src).unwrap();
        assert_eq!(region.name, "store_only");
        assert!(region.return_types.is_empty());
        assert_eq!(region.blocks[0].instructions[0].op, Op::Store);
    }

    #[test]
    fn parse_region_with_attrs() {
        let src = r#"func @gemm_kernel(a: tensor<4x4xf32>, b: tensor<4x4xf32>) -> tensor<4x4xf32> {
^entry:
  %out = gemm %a, %b {tile_m = 4, tile_n = 4} : tensor<4x4xf32>
  return %out
}
"#;
        let region = parse_region(src).unwrap();
        let instr = &region.blocks[0].instructions[0];
        assert_eq!(instr.op, Op::Gemm);
        assert_eq!(instr.attrs.len(), 2);
        assert_eq!(instr.attrs[0], ("tile_m".into(), "4".into()));
        assert_eq!(instr.attrs[1], ("tile_n".into(), "4".into()));
    }

    #[test]
    fn full_round_trip() {
        let original = Region {
            name: "matmul_fused".to_string(),
            args: vec![
                ("q".to_string(), f32_tensor(vec![128, 64])),
                ("k".to_string(), f32_tensor(vec![64, 128])),
            ],
            return_types: vec![f32_tensor(vec![128, 128])],
            blocks: vec![Block {
                label: "bb0".to_string(),
                instructions: vec![
                    Instruction {
                        result: Some("scores".to_string()),
                        op: Op::Gemm,
                        operands: vec!["q".to_string(), "k".to_string()],
                        attrs: vec![("transpose_b".into(), "true".into())],
                        result_type: Some(f32_tensor(vec![128, 128])),
                    },
                    Instruction {
                        result: Some("out".to_string()),
                        op: Op::ReduceMax,
                        operands: vec!["scores".to_string()],
                        attrs: vec![],
                        result_type: Some(f32_tensor(vec![128, 128])),
                    },
                    Instruction {
                        result: None,
                        op: Op::Return,
                        operands: vec!["out".to_string()],
                        attrs: vec![],
                        result_type: None,
                    },
                ],
            }],
        };

        let opts = EmitOptions::default();
        let text = emit(&original, &opts);
        let parsed = parse_region(&text).unwrap();

        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.args.len(), original.args.len());
        assert_eq!(parsed.return_types.len(), original.return_types.len());
        assert_eq!(parsed.blocks.len(), original.blocks.len());
        assert_eq!(
            parsed.blocks[0].instructions.len(),
            original.blocks[0].instructions.len()
        );

        let pi = &parsed.blocks[0].instructions[0];
        let oi = &original.blocks[0].instructions[0];
        assert_eq!(pi.op, oi.op);
        assert_eq!(pi.result, oi.result);
        assert_eq!(pi.operands, oi.operands);
        assert_eq!(pi.attrs, oi.attrs);
    }

    #[test]
    fn parse_op_known_and_unknown() {
        assert_eq!(parse_op("addf"), Op::Addf);
        assert_eq!(parse_op("gemm"), Op::Gemm);
        assert_eq!(parse_op("quant_attention"), Op::QuantAttention);
        assert_eq!(parse_op("my_custom_op"), Op::Custom("my_custom_op".into()));
    }

    #[test]
    fn error_on_missing_func() {
        let err = parse_region("// just a comment").unwrap_err();
        assert!(err.message.contains("func @name"));
    }

    #[test]
    fn legacy_parse_returns_labels() {
        let src = r#"func @f(x: f32) {
^bb0:
  return %x
^bb1:
  return %x
}
"#;
        let labels = parse(src).unwrap();
        assert_eq!(labels, vec!["bb0", "bb1"]);
    }
}
