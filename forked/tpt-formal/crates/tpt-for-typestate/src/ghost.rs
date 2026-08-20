//! Ghost types and typestate tokens.
//!
//! A *ghost* value carries a logical predicate `P` about a wrapped value `T`
//! in the type system. The predicate has no runtime cost — it is erased to a
//! phantom — but it lets the compiler distinguish values that satisfy
//! different invariants.

use crate::phantom::Contravariant;
use core::marker::PhantomData;

/// Marker trait for typestate *states*.
///
/// A state is a zero-sized type that classifies a [`Stateful`] value. Sealing
/// is intentionally *not* applied here so downstream crates (e.g.
/// `tpt-for-witness`) can define their own states; for crate-local sealing
/// use [`crate::phantom::Sealed`].
pub trait State {}

/// A zero-sized typestate token of state `S`.
///
/// Tokens are useful as proof objects: a function that requires a value to be
/// in state `S` takes a `Token<S>` (obtained from a transition), and the
/// compiler enforces that the caller performed the transition.
pub struct Token<S: State>(PhantomData<S>);

impl<S: State> Token<S> {
    /// Construct a typestate token. This is safe because tokens carry no
    /// information — they only attest to a type-level fact.
    #[inline]
    pub const fn new() -> Self {
        Token(PhantomData)
    }
}

impl<S: State> Default for Token<S> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> Clone for Token<S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: State> Copy for Token<S> {}

/// A value `T` annotated with typestate `S`.
///
/// The wrapped value is unchanged; only the phantom state is tracked.
pub struct Stateful<S: State, T> {
    _state: PhantomData<S>,
    value: T,
}

impl<S: State, T> Stateful<S, T> {
    /// Wrap `value` as being in state `S`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` actually satisfies the
    /// invariant expressed by `S`.
    #[inline]
    pub unsafe fn new_unchecked(value: T) -> Self {
        Stateful {
            _state: PhantomData,
            value,
        }
    }

    /// Consume the state annotation and recover the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<S: State, T> AsRef<T> for Stateful<S, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<S: State, T: Clone> Stateful<S, T> {
    /// Clone the inner value, keeping the same state.
    #[inline]
    pub fn cloned(&self) -> T {
        self.value.clone()
    }
}

/// A ghost value: `T` annotated with a logical predicate `P`.
///
/// `P` is stored as a contravariant phantom so that ghost wrappers do not
/// accidentally become covariant in their predicate. Construct with
/// [`Ghost::new`] once you have established that `value` satisfies `P`.
pub struct Ghost<P, T> {
    _pred: Contravariant<P>,
    value: T,
}

impl<P, T> Ghost<P, T> {
    /// Wrap `value` as satisfying predicate `P`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` satisfies the predicate `P`.
    #[inline]
    pub unsafe fn new(value: T) -> Self {
        Ghost {
            _pred: Contravariant::new(),
            value,
        }
    }

    /// Recover the inner value, discarding the ghost predicate.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<P, T> AsRef<T> for Ghost<P, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<P, T> Ghost<P, T> {
    /// Map the inner value, preserving the predicate `P`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the transformed value still satisfies `P`.
    #[inline]
    pub unsafe fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Ghost<P, U> {
        Ghost {
            _pred: Contravariant::new(),
            value: f(self.value),
        }
    }
}

impl<P, T: Clone> Clone for Ghost<P, T> {
    #[inline]
    fn clone(&self) -> Self {
        Ghost {
            _pred: Contravariant::new(),
            value: self.value.clone(),
        }
    }
}

impl<P, T: Copy> Copy for Ghost<P, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    struct Locked;
    struct Unlocked;
    impl State for Locked {}
    impl State for Unlocked {}

    #[test]
    fn stateful_round_trips() {
        let s = unsafe { Stateful::<Locked, u32>::new_unchecked(7) };
        assert_eq!(*s.as_ref(), 7);
        assert_eq!(s.into_inner(), 7);
    }

    #[test]
    fn ghost_preserves_value() {
        let g = unsafe { Ghost::<(), u16>::new(42) };
        assert_eq!(g.into_inner(), 42);
    }

    #[test]
    fn token_is_constructible() {
        let _tok: Token<Unlocked> = Token::new();
    }
}
