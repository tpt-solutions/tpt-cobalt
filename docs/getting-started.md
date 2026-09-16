# Getting started with TPT Cobalt

From clone to a trained model in about ten minutes. No Python anywhere.

## 1. Build

```sh
git clone <this repo> tpt-cobalt && cd tpt-cobalt
cargo build --release -p tpt-lang        # the CLI only (~1 min)
```

The full workspace (`cargo build --workspace`) compiles the forked pillars
too (~190 crates) — do it once if you plan to work on the Rust side.

## 2. Your first script

Save this as `first.tpt`:

```text
# unit-checked physics: the compiler verifies m/s^2 * s^2 = m
let g = 9.81 m/s^2
let t = 2.5 s
let drop = 0.5 * g * t * t
print "a {t}s fall drops you {drop}"

let i = 0
let acc = 0
for i in 0..5 {
    acc = acc + i
}
print "sum 0..5 = {acc}"
```

Run it:

```sh
cargo run --release -p tpt-lang --bin tpt -- run first.tpt
```

Try breaking it on purpose: change the last line to
`let bad = g + t` and run `cargo run --release -p tpt-lang --bin tpt --
check first.tpt` — the compiler rejects `m/s^2 + s` before the program
ever runs. That is one of the Four Killer Features.

## 3. Train a model

```text
let net = mlp(2, 16, 1)
let xs = ones([64, 2])
let ys = ones([64, 1])
let i = 0
while i < 300 {
    let loss = train_step(net, xs, ys, 0.01)
    i = i + 1
}
print "final loss = {loss}"
predict(net, xs)
```

`train_step` runs forward → MSE → backward → AdamW on the same tape every
Rust-side crate shares. Swap `xs`/`ys` for tensors loaded with `tpt-hub`
(SafeTensors / GGUF / ONNX) and the script is unchanged.

## 4. Explore interactively

```sh
cargo run --release -p tpt-lang --bin tpt -- repl
```

Expressions echo their value, tensors pretty-print with shape and
elements, `:help` lists the magic commands, and errors never poison the
session.

## 5. Where to go next

- **Tutorial** — [`docs/tpt-script-tutorial.md`](tpt-script-tutorial.md):
  the language end to end (units, shapes, modules, debugging).
- **Cookbook** — [`docs/tpt-script-cookbook.md`](tpt-script-cookbook.md):
  runnable examples and the adoption case studies.
- **Templates** — [`examples/templates/`](../examples/templates/): four
  ready-to-run starting points.
- **Architecture** — [`docs/architecture.md`](architecture.md): how the
  nine core crates and the forked pillars fit together.
- **Backends** — `tpt-runtime` executes on CPU, WGPU, and CUDA
  (`--features cuda`); FPGA/MCU artifact paths live in `tpt-fusion` and
  `tpt-alloy-deploy`.

## Building the Rust side

Every crate is a normal Cargo crate with README + CHANGELOG:

```sh
cargo test -p tpt-sci          # differentiable physics suite
cargo bench -p tpt-bench       # performance suite
cargo run -p tpt-lsp           # the language server (stdio)
```
