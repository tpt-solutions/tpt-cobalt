//! Example: a deterministic, replayable bounded-queue simulation.
//!
//! `Simulation` advances a `World` one step at a time from a seeded PRNG,
//! recording every transition so a run can be replayed and inspected. Because the
//! RNG is deterministic, identical `(world, actions, seed)` inputs always produce
//! identical logs — essential for reproducible fault analysis.
use tpt_for_deterministic_sim::{SimRng, Simulation, StepRecord, World};

/// A bounded queue: arrivals add to `waiting`; anything past `capacity` is
/// dropped; each tick the RNG decides how many waiting items get serviced.
#[derive(Clone, Copy)]
struct Queue {
    capacity: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct QState {
    waiting: u32,
    dropped: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct Arrivals(u8);

#[derive(Clone, Debug, PartialEq)]
struct Service {
    processed: u32,
    dropped_this_tick: u32,
}

impl World for Queue {
    type State = QState;
    type Action = Arrivals;
    type Event = Service;

    fn step(&self, s: &QState, a: &Arrivals, rng: &mut SimRng) -> (QState, Service) {
        let candidate = s.waiting.saturating_add(a.0 as u32);
        let (waiting, dropped_this_tick) = if candidate > self.capacity {
            (self.capacity, candidate - self.capacity)
        } else {
            (candidate, 0)
        };
        // The RNG picks how many of the waiting items are serviced this tick.
        let processed = rng.gen_range((waiting as u64) + 1) as u32;
        let next = QState {
            waiting: waiting - processed,
            dropped: s.dropped.saturating_add(dropped_this_tick),
        };
        (
            next,
            Service {
                processed,
                dropped_this_tick,
            },
        )
    }
}

fn main() {
    let world = Queue { capacity: 10 };

    // --- run 1: same (world, actions, seed) is deterministic --------------
    let mut sim = Simulation::new(
        world,
        QState {
            waiting: 0,
            dropped: 0,
        },
        42,
    );
    sim.run([Arrivals(5), Arrivals(20), Arrivals(3), Arrivals(10)]);

    println!("run #1 (seed 42):");
    for StepRecord {
        index,
        action,
        state,
        event,
    } in sim.steps()
    {
        println!(
            "  tick {index}: +{} arrivals → served {}, dropped {}, now waiting {}",
            action.0, event.processed, event.dropped_this_tick, state.waiting
        );
    }
    println!("  total dropped after run #1 = {}", sim.state().dropped);

    // --- replay: identical seed reproduces the log byte-for-byte ----------
    let mut replay = Simulation::new(
        world,
        QState {
            waiting: 0,
            dropped: 0,
        },
        42,
    );
    replay.run([Arrivals(5), Arrivals(20), Arrivals(3), Arrivals(10)]);
    assert_eq!(sim.steps(), replay.steps());
    println!(
        "replay (seed 42) matches run #1 exactly: {}",
        sim.steps() == replay.steps()
    );

    // --- a different seed yields a *different* service schedule ------------
    let mut other = Simulation::new(
        world,
        QState {
            waiting: 0,
            dropped: 0,
        },
        7,
    );
    other.run([Arrivals(5), Arrivals(20), Arrivals(3), Arrivals(10)]);
    println!(
        "run with seed 7 produced a different schedule? {}",
        sim.steps() != other.steps()
    );
}
