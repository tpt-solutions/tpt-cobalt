# tpt-for-deterministic-sim

Deterministic, seedable simulation harness.

A `Simulation` advances a `World` one step at a time from a seeded PRNG,
recording each step so a run can be replayed and inspected. Because the RNG is
deterministic, identical `(world, actions, seed)` inputs always produce
identical logs — essential for reproducible fault analysis and debugging.

## Features

- `SimRng` — the same fast xorshift64* generator used by `tpt-for-det-proptest`
  (`new`, `next_u64`, `gen_range`; a zero seed is remapped).
- `World` — implement `State` / `Action` / `Event` associated types and a `step`
  transition (`&self, &State, &Action, &mut SimRng -> (State, Event)`).
- `Simulation<W>` — `new(world, initial, seed)`, `apply(action)`, `run(actions)`;
  records every `StepRecord` and exposes `state()` / `steps()`.
- `StepRecord<S, A, E>` — a recorded transition (`action`, `state`, `event`,
  `index`).
- Fully determined by `(world, actions, seed)` — replay by re-running.

## Example

```rust
use tpt_for_deterministic_sim::{SimRng, Simulation, World};

#[derive(Clone, Copy)]
struct Queue { capacity: u32 }
#[derive(Clone, Debug, PartialEq)]
struct QState { waiting: u32, dropped: u32 }
#[derive(Clone, Debug, PartialEq)]
struct Arrivals(u8);
#[derive(Clone, Debug, PartialEq)]
struct Service { processed: u32, dropped_this_tick: u32 }

impl World for Queue {
    type State = QState;
    type Action = Arrivals;
    type Event = Service;
    fn step(&self, s: &QState, a: &Arrivals, rng: &mut SimRng) -> (QState, Service) {
        let candidate = s.waiting.saturating_add(a.0 as u32);
        let (waiting, dropped) = if candidate > self.capacity {
            (self.capacity, candidate - self.capacity)
        } else {
            (candidate, 0)
        };
        let processed = rng.gen_range((waiting as u64) + 1) as u32;
        (QState { waiting: waiting - processed, dropped: s.dropped.saturating_add(dropped) },
         Service { processed, dropped_this_tick: dropped })
    }
}

let mut sim = Simulation::new(Queue { capacity: 10 }, QState { waiting: 0, dropped: 0 }, 42);
sim.run([Arrivals(5), Arrivals(20), Arrivals(3), Arrivals(10)]);
assert_eq!(*sim.state(), QState { waiting: 0, dropped: 14 });

// Identical (world, actions, seed) reproduces the run byte-for-byte:
let mut replay = Simulation::new(Queue { capacity: 10 }, QState { waiting: 0, dropped: 0 }, 42);
replay.run([Arrivals(5), Arrivals(20), Arrivals(3), Arrivals(10)]);
assert_eq!(sim.steps(), replay.steps());
```

Run it with `cargo run --example deterministic_sim_basic -p tpt-for-deterministic-sim`.

## Cargo features

No optional features. `core`-only (`no_std`), no dependencies.

## Integration

Shares its RNG design with `tpt-for-det-proptest` and is the runtime harness
counterpart to the analysis crates (`tpt-for-model-check`,
`tpt-for-redundancy`) that reason about such traces.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
