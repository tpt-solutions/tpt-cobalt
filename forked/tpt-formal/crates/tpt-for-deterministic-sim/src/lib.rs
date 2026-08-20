//! Deterministic, seedable simulation harness.
//!
//! A [`Simulation`] advances a [`World`] one step at a time from a seeded PRNG,
//! recording each step so a run can be replayed and inspected. Because the
//! RNG is deterministic, identical `(world, actions, seed)` inputs always
//! produce identical logs — essential for reproducible fault analysis.

use core::num::Wrapping;

/// A small, fast, fully deterministic PRNG (xorshift64*).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimRng {
    state: Wrapping<u64>,
}

impl SimRng {
    /// Create a generator from a seed (a zero seed is remapped).
    #[inline]
    pub fn new(seed: u64) -> Self {
        SimRng {
            state: Wrapping(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            }),
        }
    }

    /// Next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = Wrapping(x);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A `u64` in `[0, n)`.
    #[inline]
    pub fn gen_range(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let m = (self.next_u64() as u128) * (n as u128);
        (m >> 64) as u64
    }
}

/// The world a simulation runs over.
///
/// Implementors define how an [`Action`] transforms a [`State`] and what
/// [`Event`] the transition emits.
pub trait World {
    /// The mutable world state.
    type State;
    /// Inputs that drive transitions.
    type Action;
    /// Observations produced by transitions.
    type Event;

    /// Apply `action` to `state`, returning the next state and an event.
    fn step(
        &self,
        state: &Self::State,
        action: &Self::Action,
        rng: &mut SimRng,
    ) -> (Self::State, Self::Event);
}

/// A single recorded transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepRecord<S, A, E> {
    /// The action that was applied.
    pub action: A,
    /// The resulting state.
    pub state: S,
    /// The event emitted.
    pub event: E,
    /// Zero-based step index.
    pub index: u64,
}

/// A full simulation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simulation<W: World> {
    world: W,
    state: W::State,
    rng: SimRng,
    steps: Vec<StepRecord<W::State, W::Action, W::Event>>,
}

impl<W: World> Simulation<W>
where
    W::State: Clone,
    W::Event: Clone,
{
    /// Begin a simulation from an initial state and a seed.
    pub fn new(world: W, initial: W::State, seed: u64) -> Self {
        Simulation {
            world,
            state: initial,
            rng: SimRng::new(seed),
            steps: Vec::new(),
        }
    }

    /// The current state.
    pub fn state(&self) -> &W::State {
        &self.state
    }

    /// The recorded step log.
    pub fn steps(&self) -> &[StepRecord<W::State, W::Action, W::Event>] {
        &self.steps
    }

    /// Apply one action, recording the transition.
    pub fn apply(&mut self, action: W::Action) {
        let index = self.steps.len() as u64;
        let (next, event) = self.world.step(&self.state, &action, &mut self.rng);
        self.steps.push(StepRecord {
            action,
            state: next.clone(),
            event,
            index,
        });
        self.state = next;
    }

    /// Apply a sequence of actions.
    pub fn run(&mut self, actions: impl IntoIterator<Item = W::Action>) {
        for a in actions {
            self.apply(a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter;
    impl World for Counter {
        type State = i64;
        type Action = i64;
        type Event = i64;
        fn step(&self, s: &i64, a: &i64, _rng: &mut SimRng) -> (i64, i64) {
            let next = s + a;
            (next, next)
        }
    }

    #[test]
    fn deterministic_run() {
        let mut sim = Simulation::new(Counter, 0, 123);
        sim.run([1i64, 2, 3, 4]);
        assert_eq!(*sim.state(), 10);
        assert_eq!(sim.steps().len(), 4);
        assert_eq!(sim.steps()[3].state, 10);

        // Same seed/inputs reproduce exactly.
        let mut sim2 = Simulation::new(Counter, 0, 123);
        sim2.run([1i64, 2, 3, 4]);
        assert_eq!(sim.steps(), sim2.steps());
    }
}
