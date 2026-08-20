# tpt-eng-controls

Control-systems primitives: a discrete-time [PID controller](Pid) (with
derivative-on-measurement and clamped-integral anti-windup), a discrete
[transfer function](TransferFunction), and a [state-space](StateSpace) model
with a forward-Euler step.

All three are `std`-only numerical building blocks. The PID and transfer
function operate on plain `f64`; the state-space model wraps the
[`tpt_math_linalg`] dense matrix types for `A`, `B`, `C`, `D`.

## Features

- **PID** (`Pid`) — proportional/integral/derivative control with
  derivative-on-measurement (no setpoint "kick") and clamped integral
  anti-windup; symmetric or asymmetric output saturation.
- **Transfer function** (`TransferFunction`) — discrete `Y(z)/U(z)` in `z⁻¹`
  direct-form-II coefficients; advance one sample at a time with `step`.
- **State-space** (`StateSpace`) — continuous `ẋ = A x + B u`, `y = C x + D u`
  advanced with forward Euler; inspect/reset the state vector.

## Installation

```toml
[dependencies]
tpt-eng-controls = "0.1"
```

## Quick start

```rust
use tpt_eng_controls::{Pid, PidGains};

let mut pid = Pid::new(PidGains::new(2.0, 1.0, 0.0));
pid.set_setpoint(10.0);
let mut y = 0.0; // first-order plant, τ = 1 s
for _ in 0..600 {
    let u = pid.update(y, 0.01);
    y += 0.01 * u;
}
assert!((y - 10.0).abs() < 0.5);
```

A discrete transfer function and a state-space model integrate the same way:

```rust
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
use tpt_eng_controls::{StateSpace, TransferFunction};

// First-order lag ẋ = -x + u, y = x.
let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
let b = DMatrix::from_row_slice(1, 1, &[1.0]);
let c = DMatrix::from_row_slice(1, 1, &[1.0]);
let d = DMatrix::from_row_slice(1, 1, &[0.0]);
let mut ss = StateSpace::new(a, b, c, d);
let mut y = 0.0;
for _ in 0..1000 {
    y = ss.step(1.0, 0.001); // → 1 - e^-1 ≈ 0.632
}
assert!((y - (1.0 - (-1.0_f64).exp())).abs() < 0.01);

// y[n] = 0.5·y[n-1] + 0.5·u[n-1]; unit-step steady state → 1.
let mut tf = TransferFunction::new(&[0.0, 0.5], &[1.0, -0.5]);
let mut yt = 0.0;
for _ in 0..500 {
    yt = tf.step(1.0);
}
assert!((yt - 1.0).abs() < 0.01);
```

## Crate items

The crate is flat (no submodules). Key items:

| Item | Purpose |
| --- | --- |
| `Pid`, `PidGains` | Discrete PID controller and its gains. |
| `TransferFunction` | Discrete `z⁻¹` transfer function. |
| `StateSpace` | Continuous state-space model with forward-Euler step. |

## Related crates

- [tpt-eng-timeseries](../tpt-eng-timeseries/) — feed controller inputs/outputs
  from conditioned sensor streams.
- [tpt-eng-examples](../tpt-eng-examples/) — uses a PID in the thermal-loop
  scenario.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
