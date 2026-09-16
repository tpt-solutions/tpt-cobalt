"""Interpreter-side changes for the usability pass."""

p = r"crates/tpt-lang/src/interp.rs"
src = open(p, encoding="utf-8").read()
count = 0

def rep(old, new):
    global src, count
    assert old in src, "MISSING ANCHOR: " + old[:90]
    src = src.replace(old, new, 1)
    count += 1

# Print: interpolate {expr} in the string operand
rep(
    """            Stmt::Print(e) => {
                let v = self.eval(e, env, dbg)?;
                self.out.push_str(&format!("{v}\\n"));
                Ok(Flow::Normal)
            }""",
    """            Stmt::Print(e) => {
                let v = self.eval(e, env, dbg)?;
                match &v {
                    Value::Str(s) if s.contains('{') => {
                        let rendered = self.interpolate(s, env, dbg)?;
                        self.out.push_str(&format!("{rendered}\\n"));
                    }
                    _ => self.out.push_str(&format!("{v}\\n")),
                }
                Ok(Flow::Normal)
            }""",
)

# MemberAssign: support Dict receivers too
rep(
    """            Stmt::MemberAssign(base, name, e) => {
                let b = self.eval(base, env, dbg)?;
                let v = self.eval(e, env, dbg)?;
                match b {
                    Value::Module(m) => {
                        m.set(name.clone(), v);
                        Ok(Flow::Normal)
                    }
                    other => Err(InterpreterError::new(
                        "TypeError",
                        format!("cannot assign member '{name}' on {}", other.type_name()),
                    )),
                }
            }""",
    """            Stmt::MemberAssign(base, name, e) => {
                let b = self.eval(base, env, dbg)?;
                let v = self.eval(e, env, dbg)?;
                match b {
                    Value::Module(m) => {
                        m.set(name.clone(), v);
                        Ok(Flow::Normal)
                    }
                    Value::Dict(d) => {
                        d.lock().unwrap().insert(name.clone(), v);
                        Ok(Flow::Normal)
                    }
                    other => Err(InterpreterError::new(
                        "TypeError",
                        format!("cannot assign member '{name}' on {}", other.type_name()),
                    )),
                }
            }""",
)

# For execution (placed before Stmt::Module arm)
rep(
    """            Stmt::Module(name, body) => {""",
    """            Stmt::For(name, iterable, body) => {
                let it = self.eval(iterable, env, dbg)?;
                let items: Vec<Value> = match it {
                    Value::List(l) => l.lock().unwrap().clone(),
                    Value::Tensor(t) => t
                        .to_vec::<f64>()
                        .unwrap()
                        .into_iter()
                        .map(Value::Num)
                        .collect(),
                    Value::Str(s) => {
                        s.chars().map(|c| Value::Str(c.to_string())).collect()
                    }
                    Value::Dict(d) => {
                        let mut keys: Vec<String> = d.lock().unwrap().keys().cloned().collect();
                        keys.sort(); // deterministic (HashMap order is not)
                        keys.into_iter().map(Value::Str).collect()
                    }
                    other => {
                        return Err(InterpreterError::new(
                            "TypeError",
                            format!("{} is not iterable in for", other.type_name()),
                        ))
                    }
                };
                for item in items {
                    env.define(name.clone(), item);
                    match self.exec_block(body, env, dbg)? {
                        Flow::Normal => {}
                        r @ Flow::Return(_) => return Ok(r),
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Module(name, body) => {""",
)

# range binary op evaluation
rep(
    """            Expr::Binary(op, lhs, rhs) => {
                let l = self.eval(lhs, env, dbg)?;
                let r = self.eval(rhs, env, dbg)?;
                apply_binary(op, &l, &r)
            }""",
    """            Expr::Binary(op, lhs, rhs) => {
                let l = self.eval(lhs, env, dbg)?;
                let r = self.eval(rhs, env, dbg)?;
                if op == ".." {
                    let (Some(a), Some(b)) = (l.as_num(), r.as_num()) else {
                        return Err(InterpreterError::new(
                            "TypeError",
                            "range '..' requires number endpoints",
                        ));
                    };
                    let (a, b) = (a as i64, b as i64);
                    return Ok(Value::List(Arc::new(Mutex::new(
                        (a..b).map(|x| Value::Num(x as f64)).collect(),
                    ))));
                }
                apply_binary(op, &l, &r)
            }""",
)

# Call evaluation with named args
rep(
    """            Expr::Call(fexpr, args) => {
                let f = self.eval(fexpr, env, dbg)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, env, dbg)?);
                }
                call_function(self, &f, &vals, dbg)
            }""",
    """            Expr::Call(fexpr, args) => {
                let f = self.eval(fexpr, env, dbg)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    let v = self.eval(&a.expr, env, dbg)?;
                    vals.push((a.name.clone(), v));
                }
                call_function(self, &f, &vals, dbg)
            }""",
)

# Index evaluation with multi-index / slices
rep(
    """            Expr::Index(base, idx) => {
                let b = self.eval(base, env, dbg)?;
                let i = self.eval(idx, env, dbg)?;
                index_value(&b, &i)
            }""",
    """            Expr::Index(base, idx_args) => {
                let b = self.eval(base, env, dbg)?;
                let mut idx = Vec::with_capacity(idx_args.len());
                for arg in idx_args {
                    match arg {
                        IdxArg::Expr(e) => idx.push(IndexArg::Value(self.eval(e, env, dbg)?)),
                        IdxArg::Range(start, end) => {
                            let s = match start {
                                Some(e) => Some(self.eval(e, env, dbg)?.as_num().ok_or_else(
                                    || {
                                        InterpreterError::new(
                                            "TypeError",
                                            "slice bounds must be numbers",
                                        )
                                    },
                                )? as i64),
                                None => None,
                            };
                            let e2 = match end {
                                Some(e) => Some(self.eval(e, env, dbg)?.as_num().ok_or_else(
                                    || {
                                        InterpreterError::new(
                                            "TypeError",
                                            "slice bounds must be numbers",
                                        )
                                    },
                                )? as i64),
                                None => None,
                            };
                            idx.push(IndexArg::Slice(s, e2));
                        }
                    }
                }
                index_value(&b, &idx)
            }""",
)

# call_function: kwargs binding
rep(
    """fn call_function(
    interp: &mut Interpreter,
    f: &Value,
    args: &[Value],
    dbg: &mut Option<DbgCb>,
) -> Res<Value> {
    let Value::Function(fun) = f else {
        return Err(InterpreterError::new(
            "TypeError",
            format!("{} is not callable", f.type_name()),
        ));
    };
    match fun.as_ref() {
        Function::Native { f, .. } => {
            f(args).map_err(|m| InterpreterError::new("RuntimeError", m))
        }""",
    """/// One evaluated call argument: positional or keyword.
pub type CallVal = (Option<String>, Value);

fn call_function(
    interp: &mut Interpreter,
    f: &Value,
    args: &[CallVal],
    dbg: &mut Option<DbgCb>,
) -> Res<Value> {
    let Value::Function(fun) = f else {
        return Err(InterpreterError::new(
            "TypeError",
            format!("{} is not callable", f.type_name()),
        ));
    };
    match fun.as_ref() {
        Function::Native { f, .. } => {
            if args.iter().any(|(name, _)| name.is_some()) {
                return Err(InterpreterError::new(
                    "TypeError",
                    "native functions take positional arguments only",
                ));
            }
            let positional: Vec<Value> = args.iter().map(|(_, v)| v.clone()).collect();
            f(&positional).map_err(|m| InterpreterError::new("RuntimeError", m))
        }""",
)

rep(
    """            let required = params_def
                .iter()
                .take_while(|p| p.default.is_none())
                .count();
            if args.len() < required {
                let missing: Vec<String> = params_def[args.len()..required]
                    .iter()
                    .map(|p| p.name.clone())
                    .collect();
                return Err(InterpreterError::new(
                    "TypeError",
                    format!(
                        "{name}() missing {} required argument(s): {}",
                        missing.len(),
                        missing.join(", ")
                    ),
                ));
            }
            if args.len() > params_def.len() {
                return Err(InterpreterError::new(
                    "TypeError",
                    format!(
                        "{name}() takes at most {} arguments but {} were given",
                        params_def.len(),
                        args.len()
                    ),
                ));
            }
            // calls evaluate in the *defining* scope (closures: a function
            // defined inside a `module` block sees that module's members)
            let parent = Arc::clone(env);
            let local = Arc::new(Environment::child(&parent));
            interp.call_depth += 1;
            for (i, p) in params_def.iter().enumerate() {
                if i < args.len() {
                    local.define(p.name.clone(), args[i].clone());
                } else if let Some(dv) = &p.default {
                    local.define(p.name.clone(), dv.clone());
                }
            }""",
    """            // bind: positionals fill slots left-to-right, keywords target
            // parameters by name, defaults cover whatever is still missing
            let mut slots: Vec<Option<Value>> = vec![None; params_def.len()];
            let mut positional = 0usize;
            for (cname, v) in args {
                match cname {
                    None => {
                        if positional >= params_def.len() {
                            return Err(InterpreterError::new(
                                "TypeError",
                                format!(
                                    "{name}() takes at most {} arguments but more were given",
                                    params_def.len()
                                ),
                            ));
                        }
                        slots[positional] = Some(v.clone());
                        positional += 1;
                    }
                    Some(cname) => {
                        let idx = params_def
                            .iter()
                            .position(|p| &p.name == cname)
                            .ok_or_else(|| {
                                InterpreterError::new(
                                    "TypeError",
                                    format!("{name}() has no parameter '{cname}'"),
                                )
                            })?;
                        if slots[idx].is_some() {
                            return Err(InterpreterError::new(
                                "TypeError",
                                format!("{name}() got multiple values for '{cname}'"),
                            ));
                        }
                        slots[idx] = Some(v.clone());
                    }
                }
            }
            let missing: Vec<String> = params_def
                .iter()
                .zip(&slots)
                .filter(|(p, s)| s.is_none() && p.default.is_none())
                .map(|(p, _)| p.name.clone())
                .collect();
            if !missing.is_empty() {
                return Err(InterpreterError::new(
                    "TypeError",
                    format!(
                        "{name}() missing {} required argument(s): {}",
                        missing.len(),
                        missing.join(", ")
                    ),
                ));
            }
            // calls evaluate in the *defining* scope (closures: a function
            // defined inside a `module` block sees that module's members)
            let parent = Arc::clone(env);
            let local = Arc::new(Environment::child(&parent));
            for (p, slot) in params_def.iter().zip(&slots) {
                match slot {
                    Some(v) => local.define(p.name.clone(), v.clone()),
                    None => {
                        if let Some(dv) = &p.default {
                            local.define(p.name.clone(), dv.clone());
                        }
                    }
                }
            }""",
)

# index_value: multi-index + slices
rep(
    """fn index_value(base: &Value, idx: &Value) -> Res<Value> {
    fn norm(i: f64, len: usize) -> Res<usize> {
        let i = i as i64;
        let i = if i < 0 { (len as i64 + i) as usize } else { i as usize };
        if i >= len {
            return Err(InterpreterError::new(
                "IndexError",
                format!("index {i} out of range (len {len})"),
            ));
        }
        Ok(i)
    }
    match (base, idx) {
        (Value::List(l), Value::Num(i)) => {
            let l = l.lock().unwrap();
            let i = norm(*i, l.len())?;
            Ok(l[i].clone())
        }
        (Value::Dict(d), Value::Str(k)) => d
            .lock()
            .unwrap()
            .get(k)
            .cloned()
            .ok_or_else(|| InterpreterError::new("KeyError", format!("key '{k}' not found"))),
        (Value::Tensor(t), Value::Num(i)) => {
            let v = t.to_vec::<f64>().unwrap();
            let i = norm(*i, v.len())?;
            Ok(Value::Num(v[i]))
        }
        (b, _) => Err(InterpreterError::new(
            "TypeError",
            format!("cannot index {}", b.type_name()),
        )),
    }
}""",
    """/// An evaluated bracket index: a value (number for tensors/lists, string
/// for dicts) or an `[a:b]` slice with optional bounds.
pub enum IndexArg {
    Value(Value),
    Slice(Option<i64>, Option<i64>),
}

fn norm(i: f64, len: usize) -> Res<usize> {
    let i = i as i64;
    let i = if i < 0 { (len as i64 + i) as usize } else { i as usize };
    if i >= len {
        return Err(InterpreterError::new(
            "IndexError",
            format!("index {i} out of range (len {len})"),
        ));
    }
    Ok(i)
}

/// Resolve an optional slice bound against `len` (None -> 0 or len).
fn slice_bound(v: Option<i64>, len: usize, is_start: bool) -> Res<usize> {
    match v {
        None => Ok(if is_start { 0 } else { len }),
        Some(x) => {
            let x = if x < 0 { (len as i64 + x).max(0) } else { x };
            let x = x.min(len as i64) as usize;
            Ok(x)
        }
    }
}

fn index_value(base: &Value, idx: &[IndexArg]) -> Res<Value> {
    match (base, idx) {
        (Value::List(l), [IndexArg::Value(Value::Num(i))]) => {
            let l = l.lock().unwrap();
            let i = norm(*i, l.len())?;
            Ok(l[i].clone())
        }
        (Value::List(l), [IndexArg::Slice(a, b)]) => {
            let l = l.lock().unwrap();
            let s = slice_bound(*a, l.len(), true)?;
            let e = slice_bound(*b, l.len(), false)?;
            if s > e {
                return Err(InterpreterError::new("IndexError", "slice start > end"));
            }
            Ok(Value::List(Arc::new(Mutex::new(l[s..e].to_vec()))))
        }
        (Value::Dict(d), [IndexArg::Value(Value::Str(k))]) => d
            .lock()
            .unwrap()
            .get(k)
            .cloned()
            .ok_or_else(|| InterpreterError::new("KeyError", format!("key '{k}' not found"))),
        (Value::Tensor(t), [IndexArg::Value(Value::Num(i))]) => {
            let v = t.to_vec::<f64>().unwrap();
            let i = norm(*i, v.len())?;
            Ok(Value::Num(v[i]))
        }
        // multi-index: one number per dimension, row-major offset
        (Value::Tensor(t), args)
            if !args.is_empty()
                && args.len() == t.ndim()
                && args
                    .iter()
                    .all(|a| matches!(a, IndexArg::Value(Value::Num(_)))) =>
        {
            let shape = t.shape();
            let nums: Vec<f64> = args
                .iter()
                .map(|a| match a {
                    IndexArg::Value(Value::Num(n)) => *n,
                    _ => unreachable!(),
                })
                .collect();
            let mut offset = 0usize;
            for (d, &n) in nums.iter().enumerate() {
                let i = norm(n, shape[d])?;
                offset = offset * shape[d] + i;
            }
            let v = t.to_vec::<f64>().unwrap();
            Ok(Value::Num(v[offset]))
        }
        // single slice over the first axis (flat for 1-D tensors)
        (Value::Tensor(t), [IndexArg::Slice(a, b)]) => {
            let shape = t.shape();
            let s = slice_bound(*a, shape[0], true)?;
            let e = slice_bound(*b, shape[0], false)?;
            if s > e {
                return Err(InterpreterError::new("IndexError", "slice start > end"));
            }
            let data = t.to_vec::<f64>().unwrap();
            let row_len: usize = shape[1..].iter().product();
            let mut out = Vec::new();
            for row in s..e {
                out.extend_from_slice(&data[row * row_len..(row + 1) * row_len]);
            }
            let mut new_shape = shape.to_vec();
            new_shape[0] = e - s;
            Ok(Value::Tensor(
                Tensor::from_typed(out).reshape(&new_shape).unwrap(),
            ))
        }
        (Value::Str(s), [IndexArg::Value(Value::Num(i))]) => {
            let chars: Vec<char> = s.chars().collect();
            let i = norm(*i, chars.len())?;
            Ok(Value::Str(chars[i].to_string()))
        }
        (b, _) => Err(InterpreterError::new(
            "TypeError",
            format!("cannot index {} with these arguments", b.type_name()),
        )),
    }
}

/// Render `{expr}` interpolations in a print string against `env`.
/// `{{` / `}}` are literal braces.
impl Interpreter {
    fn interpolate(&mut self, s: &str, env: &Arc<Environment>, dbg: &mut Option<DbgCb>) -> Res<String> {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            match c {
                '{' if s[i + 1..].starts_with('{') => {
                    out.push('{');
                    chars.next();
                }
                '}' if s[i + 1..].starts_with('}') => {
                    out.push('}');
                    chars.next();
                }
                '{' => {
                    let close = s[i..].find('}').ok_or_else(|| {
                        InterpreterError::new("SyntaxError", "unterminated '{' in print string")
                    })?;
                    let expr_src: String = s[i + 1..i + close].trim().to_string();
                    let toks = lex(&expr_src)?;
                    let mut parser = Parser::new(&toks);
                    let e = parser.parse_expr()?;
                    let v = self.eval(&e, env, dbg)?;
                    out.push_str(&v.to_string());
                    for _ in 0..close {
                        chars.next();
                    }
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }
}""",
)

open(p, "w", encoding="utf-8", newline="").write(src)
print("patched interpreter:", count, "edits")
