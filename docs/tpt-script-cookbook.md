# TPT Script Cookbook — community examples & case studies

Worked examples for the Phase 7 "community examples / case studies" line,
each pointing at the code path in this repo that makes it work. Every
snippet runs as-is through the interpreter (`cargo run -p tpt-repl`, or
`tpt_lang::Interpreter::run` in Rust) — no Python anywhere.

## Examples

### 1. Unit-checked physics: free fall with the checker watching

```text
let g = 9.81 m/s^2
let t = 2.5 s
let drop = 0.5 * g * t * t      # 30.66 m — the checker verifies m/s^2 * s^2 = m
print drop
```

The compile-time dimension algebra (`tpt_lang::check`) rejects `g + t`
before anything runs; composition through `*` and `/` propagates exponents.
Test: `check::tests::unit_mismatch_is_a_compile_error`.

### 2. Train an MLP from script

```text
let net = mlp(2, 16, 1)
let xs = ones([32, 2])
let ys = ones([32, 1])
let i = 0
while i < 300 {
    let loss = train_step(net, xs, ys, 0.01)
    i = i + 1
}
predict(net, xs)
```

`train_step` runs forward → MSE → backward → AdamW on the interpreter's
`Module` objects; optimizer state persists on the model between steps.
Swap `xs`/`ys` for `tpt-hub`-loaded tensors (SafeTensors/GGUF/ONNX) and the
script is unchanged. Tests: `ml::tests::script_trains_mlp_on_synthetic_regression`.

### 3. Namespace a model family with modules

```text
module rl {
    let gamma = 0.99
    def discount(rewards, i) {
        if i >= len(rewards) { return 0.0 }
        return rewards[i] + gamma * discount(rewards, i + 1)
    }
}
rl.discount([1.0, 1.0, 1.0], 0)
```

Module functions close over their namespace; bodies are identity-keyed so
two modules can both define `discount`. Tests:
`interp::tests::module_functions_do_not_collide_across_modules`.

### 4. Debug a training loop with watches

```rust,ignore
let mut it = Interpreter::new();
it.add_breakpoint("train_step");
it.add_watch("loss");
it.run_debug(src, &mut |frame| {
    println!("[{}] loss = {}", frame.statement, frame.watches[0].value.as_ref().unwrap());
    if frame.statement > 500 { DebugAction::Abort } else { DebugAction::Continue }
})?;
```

Tensor watches render shape + elements; `Abort` stops cleanly mid-run.
See the tutorial §10.

### 5. Notebook workflow with rich display

Cells share one kernel; tensor results come back as `text/plain` plus an
HTML table, and `completions("geom.ar")` completes into module members.
Tests: `notebook::tests::tensor_results_get_rich_html_display`.

## Case studies — the adoption wedges

The roadmap targets verticals where Python is *structurally* disqualifying
(safety-critical, embedded, formally verified). Each wedge below maps to
shipped, tested code in this workspace:

### Safety-critical simulation (grads through physics)

Differentiable FEA linear solves, ODE integration, Hertz–Mindlin contact,
and reaction kinetics all backprop through hand-derived adjoints
(`tpt-sci::{fea, ode, hertz, reactions}`), with the gradients checked
against finite differences in-tree. A crash-safety or structural engineer
gets `Linear`-layer ergonomics over a physics kernel without a Python
runtime in the deployed path — `tpt-sci::hertz`'s parity test even proves
the wrapped kernel matches the upstream DEM solver step-for-step.

### Embedded / MCU deployment

`tpt-alloy-deploy` takes a partitioned model to an RP2040 (UF2) or ESP32
(ROM-UART flash protocol) with a staged, digest-verified OTA rollout —
byte-level protocol tests run on mock transports, so the flashing path is
verifiable in CI without hardware. The Pulse SNN layer (`tpt-sci::snn`)
covers event-driven inference with surrogate gradients for future
neuromorphic targets.

### Formally verified memory budgets

`tpt-fusion::build_manifest_proved` refuses to emit FPGA artifacts until
the Fourier–Motzkin proof over the model's real allocations (weights,
activations, symbolic batch dims) shows every assignment fits the device —
with counterexample witnesses naming the overflowing assignment when it
doesn't. "The compiler proves memory fit before execution" is a checked
property, not a slogan; see `proof::tests::symbolic_batch_is_quantified_over_its_bound`.
