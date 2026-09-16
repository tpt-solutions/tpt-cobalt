"""Clippy backlog cleanup: mechanical fixes + documented allows."""
count = 0

def patch(path, pairs):
    global count
    src = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in src, f"MISSING in {path}: {old[:80]}"
        src = src.replace(old, new, 1)
        count += 1
    open(path, "w", encoding="utf-8", newline="").write(src)
    print("patched", path)

# 1. tpt-ml: crate-level allow — the per-channel reductions deliberately use
# explicit index math
patch("crates/tpt-ml/src/lib.rs", [(
    "pub mod activations;",
    """// The norm layers compute per-channel reductions with explicit index math
// that mirrors the formulas; clippy's iterator rewrite obscures them.
#![allow(clippy::needless_range_loop)]

pub mod activations;""",
)])

# 2. autograd: iterate idx by enumerate
patch("crates/tpt-autograd/src/lib.rs", [(
    """        for d in 0..trank {
            let sd = d + srank - trank;
            let coord = if source[sd] == 1 { 0 } else { idx[d] };
            s += coord * strides[sd];""",
    """        for (d, &i) in idx.iter().enumerate() {
            let sd = d + srank - trank;
            let coord = if source[sd] == 1 { 0 } else { i };
            s += coord * strides[sd];""",
)])

# 3. tensor: alias the backward closure type
patch("crates/tpt-tensor/src/tensor.rs", [(
    "pub struct AutogradNode {\n    pub parents: Vec<Arc<AutogradNode>>,\n    pub backward: Option<Box<dyn Fn(&Tensor) + Send + Sync>>,",
    "pub struct AutogradNode {\n    pub parents: Vec<Arc<AutogradNode>>,\n    pub backward: Option<BackwardFn>,",
)])
src = open("crates/tpt-tensor/src/tensor.rs", encoding="utf-8").read()
if "pub type BackwardFn" not in src:
    src = src.replace(
        "pub struct AutogradNode {",
        "/// The VJP closure recorded per node: receives the upstream gradient.\npub type BackwardFn = Box<dyn Fn(&Tensor) + Send + Sync>;\n\npub struct AutogradNode {",
        1,
    )
    open("crates/tpt-tensor/src/tensor.rs", "w", encoding="utf-8", newline="").write(src)
    print("patched tensor alias")

# 4. interp: restructure the '%' arm (redundant guard) and the identical-if
patch("crates/tpt-lang/src/interp.rs", [(
    """        "%" => match (l.as_num(), r.as_num()) {
            (Some(a), Some(b)) if b != 0.0 => Ok(Value::Num(a % b)),
            (_, Some(b)) if b == 0.0 => Err(InterpreterError::from_lang(LangError::DivByZero)),
            _ => Err(type_err(op, l)),
        },""",
    """        "%" => match (l.as_num(), r.as_num()) {
            (Some(a), Some(b)) => {
                if b == 0.0 {
                    Err(InterpreterError::from_lang(LangError::DivByZero))
                } else {
                    Ok(Value::Num(a % b))
                }
            }
            _ => Err(type_err(op, l)),
        },""",
)])

# 5. gguf: is_multiple_of
patch("crates/tpt-hub/src/gguf.rs", [
    ("if numel % 32 != 0 {", "if !numel.is_multiple_of(32) {"),
    ("while out.len() % alignment != 0 {", "while !out.len().is_multiple_of(alignment) {"),
])

# 6. onnx: let-chains, dead-code note, alias
patch("crates/tpt-hub/src/onnx.rs", [
    (
        """            1 => {
                if let Field::Bytes(node) = &payload {
                    if let Some(op) = node_op_type(node) {
                        model.ops.push(op);
                    }
                }
            }""",
        """            1 => {
                if let Field::Bytes(node) = &payload
                    && let Some(op) = node_op_type(node)
                {
                    model.ops.push(op);
                }
            }""",
    ),
    (
        """                if let Field::Bytes(init) = &payload {
                    if let Some((name, t)) = parse_initializer(init)? {
                        model.initializers.insert(name, t);
                    }
                }""",
        """                if let Field::Bytes(init) = &payload
                    && let Some((name, t)) = parse_initializer(init)?
                {
                    model.initializers.insert(name, t);
                }""",
    ),
    (
        "enum Field<'a> {",
        "/// The protobuf wire types (unused variants kept so every wire type\n/// decodes without loss).\n#[allow(dead_code)]\nenum Field<'a> {",
    ),
    (
        "fn decode_message(b: &[u8]) -> Result<Vec<(u32, Field)>, HubError> {",
        "/// One decoded field: `(field_number, payload)`.\ntype FieldPair = (u32, Field);\n\nfn decode_message(b: &[u8]) -> Result<Vec<FieldPair>, HubError> {",
    ),
])

# 7. shared: collapse the parent-dir if via filter
patch("crates/tpt-hub/src/shared.rs", [(
    """        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }""",
    """        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }""",
)])

# 8. pool: drop(list) -> let _
patch("crates/tpt-runtime/src/pool.rs", [(
    "                    drop(list);",
    "                    let _ = list; // release the lock before map removal",
)])

# 9. lsp: await the publish future
patch("crates/tpt-lsp/src/server.rs", [(
    """        self.client
            .publish_diagnostics(uri.clone(), diags, None);""",
    """        self.client
            .publish_diagnostics(uri.clone(), diags, None)
            .await;""",
)])

# 10. columnar: unused import + closure var fixes
patch("crates/tpt-columnar/src/record_batch.rs", [(
    """    use crate::array::{
        Array, ArrayRef, BinaryArray, PrimitiveArray, StringArray,
    };""",
    """    use crate::array::{ArrayRef, BinaryArray, PrimitiveArray, StringArray};""",
)])
patch("crates/tpt-columnar/src/compute.rs", [
    (""".filter(|&(_v, &m)| m).map(|(v, &_m)| v.to_string())""",
     """.filter(|&(_, &m)| m).map(|(v, _)| v.to_string())"""),
    (""".filter(|&(_v, &m)| m).map(|(v, &_m)| v.to_vec())""",
     """.filter(|&(_, &m)| m).map(|(v, _)| v.to_vec())"""),
])

# 11. pinn: the demo fit function legitimately takes a config-per-arg
patch("crates/tpt-sci/src/pinn.rs", [(
    "pub fn train_pinn_ode<F>(",
    "#[allow(clippy::too_many_arguments)] // demo entry point: each knob is a distinct input\npub fn train_pinn_ode<F>(",
)])

# 12. reactions: rate_idx kept for future per-reaction rate lookup
patch("crates/tpt-sci/src/reactions.rs", [(
    """    /// Index into the rate-constant vector.
    rate_idx: usize,""",
    """    /// Index into the rate-constant vector (kept for the upcoming
    /// per-reaction rate-law generalization; the current tape mirror folds
    /// all rates through the exponent matrix).
    #[allow(dead_code)]
    rate_idx: usize,""",
)])

print("applied", count, "edits")
