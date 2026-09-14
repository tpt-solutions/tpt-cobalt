# TPT Script — A Tutorial

TPT Script is the Pythonic scripting language of the Cobalt stack. It runs in
**eager mode** by default (fast iteration, full introspection), and **tensors
are native values** — not boxed afterthoughts. This tutorial walks through the
language using `tpt-lang`, the reference runtime.

---

## 1. The REPL

Start it from the workspace:

```text
$ cargo run -p tpt-lang --bin tpt-repl
TPT Script REPL — eager mode. Type :help for commands.
tpt> 2 + 3 * 4
= 14
```

State persists between entries. Multi-line entries accumulate until all
brackets balance:

```text
tpt> def add3(a, b, c) {
...     return a + b + c
... }
tpt> add3(1, 2, 3)
= 6
```

Magic commands start with `:`: `:help`, `:env` (list variables), `:clear`,
`:output`, `:quit`.

## 2. Values

The runtime value model has nil, booleans, numbers (single float type,
Python-like), strings, shared lists and dicts, and — most importantly —
**tensors**:

```text
tpt> let l = [10, 20, 30]
tpt> l[1]
= 20
tpt> let d = {"alpha": 1, "beta": 2}
tpt> d["beta"]
= 2
```

Truthiness is Python-style: `nil`, `false`, zero, empty strings/lists/dicts
are falsy.

## 3. Control flow and functions

```text
let total = 0
let i = 1
while i < 5 {
    total = total + i
    i = i + 1
}
print total          # 10

def square(x) {
    return x * x
}
```

Errors are raised, not returned: `undefined_name + 1` raises `NameError`,
failing `assert`s raise `AssertionError`, division by zero raises
`ZeroDivisionError`.

## 4. Tensors are first-class

Tensor literals come from constructors; arithmetic broadcasts scalars;
matmul is a builtin:

```text
let a = ones([2, 2])
let b = a + 1.0          # scalar broadcast
let m = matmul(b, b)
sum(m)                   # = 32
m[0]                     # flat indexing into tensors works too
```

Models are opaque values driven through natives (`mlp`, `transformer`,
`train_step`, `predict`) — see §5.

## 5. Training a network from script

```text
let xs = ...             # a [batch, features] tensor (from tpt-hub or natives)
let ys = ...
let net = mlp(1, 16, 1)
let i = 0
while i < 300 {
    train_step(net, xs, ys, 0.05)   # forward -> MSE -> backward -> AdamW
    i = i + 1
}
predict(net, xs)
```

`train_step` performs one full optimization step and returns the loss. The
optimizer state lives inside the model, so successive calls keep momentum.

## 6. Units — checked at compile time

Numeric literals can carry physical dimensions:

```text
let d = 3.0 m
let t = 5.0 s
let v = d / t            # fine: m/s
let bad = d + t          # UnitError: cannot add 'm' and 's'
```

Run `tpt_lang::typecheck(src)` (or the checker pass) *before* executing:
dimension algebra composes through `*` and `/` (`m/s * s == m`) and mismatched
`+`/`-`/comparisons are rejected **before any code runs**. This is one of the
Four Killer Features.

## 7. Shapes — inferred statically

The same static pass infers tensor shapes where they are knowable and rejects
impossible programs early:

```text
let a = ones([2, 3])
let b = ones([4, 5])
let c = matmul(a, b)     # ShapeError: matmul inner dims differ: 3 vs 4
```

Gradual by design: anything the checker cannot know is simply not constrained.

## 8. Profiling

Statement-level tracing exports the standard Chrome Trace format:

```rust
let mut it = tpt_lang::Interpreter::new();
it.enable_tracing();
it.run(src).unwrap();
std::fs::write("trace.json", it.take_chrome_trace_json()).unwrap();
// open chrome://tracing or https://ui.perfetto.dev and load trace.json
```

## 9. Modules, functions with defaults, and member access

The object model is deliberately small: modules and functions, with
parameters — no classes, inheritance, metaclasses, or descriptors.

Parameters can carry default values (evaluated once, at `def` time, like
Python), and calls may omit those arguments:

```text
def wave(amp, freq = 2.0) {
    return amp * freq
}
wave(3.0)        # 6.0
wave(3.0, 4.0)   # 12.0
```

Omitting a required argument is a `TypeError` that names the missing
parameters; a default placed before a required parameter is a `SyntaxError`.

A `module` statement runs its body in a fresh scope and binds the result as a
first-class module value. Functions defined inside a module close over it, so
they see its members; member access uses `.`:

```text
module geom {
    let pi = 3.14159
    def area(r) {
        return pi * r * r
    }
}
geom.area(2.0)   # 12.56636
geom.pi          # 3.14159
geom.pi = 3.0    # members are assignable
```

Two modules can both define `area` — bodies are keyed by identity, not name,
so namespaces never collide. Member access also works as sugar on dict keys
(`d.alpha` ≡ `d["alpha"]`), and an unknown member raises `AttributeError`.

## 10. Debugging

The interpreter has a built-in debugger: **breakpoints** on statement labels,
single-stepping, **watch expressions** evaluated in the paused scope (with
tensor-aware display), and full locals inspection. All of it is a library
API over `run_debug` — the callback decides at each pause whether to
continue, step, or abort:

```rust
use tpt_lang::Interpreter;
use tpt_lang::interp::DebugAction;

let mut it = Interpreter::new();
it.add_breakpoint("train_step");   // hits `call train_step`
it.add_watch("loss");              // evaluated in the paused scope
it.add_watch("net");               // tensors render with shape + elements
it.run_debug(src, &mut |frame| {
    println!("{}: loss = {:?}", frame.label, frame.lookup("loss"));
    println!("locals: {:?}", frame.locals.iter().map(|(n, _)| n).collect::<Vec<_>>());
    if frame.statement > 100 { DebugAction::Abort } else { DebugAction::Continue }
})?;
```

A pause happens *before* the statement runs; `DebugAction::Step` resumes and
pauses again before the next statement. `DebugAction::Abort` stops the
program with a `KeyboardInterrupt` error (and leaves globals untouched by
statements that never ran). Plain `run` is unaffected by any debug
configuration. DAP (editor) integration is the remaining Phase 6 item.

## 11. What's next

Traced execution (`@compile`), ahead-of-time compilation, the notebook
kernel, debugger, and LSP integration are on the Phase 6 roadmap (see
`todo.md`). The runtime shares the stack's single `Tensor` type and single
autograd tape with everything else in Cobalt — no glue, no copies.
