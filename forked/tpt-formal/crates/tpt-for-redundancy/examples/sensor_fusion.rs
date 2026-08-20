//! Example: fault-tolerant sensor fusion via redundant voting.
//!
//! A flight controller reads several independent copies of a value and votes on
//! the correct result so a single faulty replica cannot corrupt the output. We
//! show Triple Modular Redundancy, N-copy majority voting, a straight majority
//! vote (which can tie), and a single-bit parity check for a transmitted payload.
use tpt_for_redundancy::{majority_vote, Parity, Redundant, Tmr};

fn main() {
    // Triple Modular Redundancy: one of three throttle sensors is faulty (70).
    let tmr = Tmr::new(50u8, 50, 70);
    match tmr.vote() {
        Some(v) => println!("TMR: trusted throttle = {v} (faulty 70 masked out)"),
        None => println!("TMR: all three disagree — no safe value"),
    }
    // If all three differ, no value is agreed upon.
    println!("TMR all-differ vote = {:?}", Tmr::new(1u8, 2, 3).vote());

    // N-copy redundancy: 5 replicas of an armed-mode flag (1 = armed).
    let r: Redundant<u8, 5> = Redundant::new([1, 1, 1, 0, 0]);
    println!(
        "5-copy majority = {:?}, replicas = {}",
        r.majority(),
        r.len()
    );
    // Deref exposes the backing array; N >= 1 so it is never empty.
    println!("first replica = {}, empty? {}", r[0], r.is_empty());

    // A plain majority vote over arbitrary copies; an even count can tie.
    println!(
        "majority_vote([1,1,2,2]) = {:?} (tie → None)",
        majority_vote(&[1u8, 1, 2, 2])
    );
    println!(
        "majority_vote([1,1,2])   = {:?}",
        majority_vote(&[1u8, 1, 2])
    );

    // Parity: detect a single flipped bit in a transmitted config payload.
    let sent = [0b1010_1010u8, 0b0101_0101];
    let parity = Parity::compute(&sent);
    let mut corrupted = sent;
    corrupted[0] ^= 0b0000_1000; // flip one bit
    let corrupted_parity = Parity::compute(&corrupted);
    println!(
        "parity bit sent = {}, after corruption = {}",
        parity.bit(),
        corrupted_parity.bit()
    );
    println!(
        "parity still matches? {}",
        parity.matches(&corrupted_parity)
    );
}
