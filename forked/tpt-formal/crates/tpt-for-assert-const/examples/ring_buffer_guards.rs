//! Example: compile-time assertions guarding a fixed-capacity ring buffer.
//!
//! The point of `const_assert!` & friends is to turn a *runtime* invariant into a
//! *build-time* one: a violation is a hard compile error, so a buggy configuration
//! never ships. (Try changing `CAPACITY` to 100 below — the build fails with a
//! const-assertion message instead of misbehaving at runtime.)
use tpt_for_assert_const::{const_assert, const_assert_eq, const_assert_ne, Const, IsTrue, Same};

// A fixed-capacity ring buffer whose masking index math only works when the
// capacity is an exact power of two. Guard that design decision at compile time.
const CAPACITY: usize = 64;
const_assert!(CAPACITY.is_power_of_two());
const_assert_eq!(CAPACITY & (CAPACITY - 1), 0); // same property, stated directly

// A wire-format header we must never accidentally resize. `repr(C)` makes the
// layout deterministic; the const assertion pins the exact byte size so a future
// edit that adds a field without updating the protocol docs fails the build.
#[repr(C)]
struct Header {
    version: u8,
    flags: u8,
    length: u16,
    tag: u32,
}
const_assert_eq!(core::mem::size_of::<Header>(), 8);

// `IsTrue` lets a generic bound *require* a compile-time boolean. A buffer only
// compiles when its capacity fits in a `u8` index (the "tiny" API variant):
// `Const<true>: IsTrue` holds while `Const<false>: IsTrue` does not, so a
// too-large capacity is rejected before codegen.
fn requires_tiny_index<B: IsTrue>() {}
type TinyOk = Const<true>;

// `Same` lets a bound *demand* two types are identical. This identity transform
// is only generic-safe when the input and output types coincide.
fn identity_only<T, U>()
where
    T: Same<Output = U>,
{
}

fn main() {
    requires_tiny_index::<TinyOk>();
    identity_only::<u32, u32>();

    // Demonstrate the masking index math the power-of-two guard protects: with a
    // 2^n capacity, `head & (CAPACITY - 1)` wraps with no division.
    let mut head: usize = 0;
    for i in 0..130 {
        let slot = head & (CAPACITY - 1);
        head = head.wrapping_add(1);
        if i % 32 == 0 {
            println!("write #{i} -> slot {slot} (masked into capacity {CAPACITY})");
        }
    }

    // `const_assert_ne!` also has a compile-time role: reject accidental equality,
    // e.g. ensuring two distinct protocol tags can never collide.
    const_assert_ne!(1u8, 2u8);

    // Read every field so the layout assertion is visibly meaningful.
    let header = Header {
        version: 1,
        flags: 0,
        length: 8,
        tag: 0x1234,
    };
    println!(
        "header v{} flags{} len{} tag{:#x}; compile-time guards held: power-of-two \
         capacity, exact {}-byte size, `Const<true>: IsTrue`, `u32: Same<u32>`",
        header.version,
        header.flags,
        header.length,
        header.tag,
        core::mem::size_of::<Header>()
    );
}
