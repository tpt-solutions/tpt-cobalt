//! Fine-grained, single-threaded reactivity (`Signal` + `Effect`).
//!
//! A thread-local "current observer" is installed while an effect body runs.
//! Any [`Signal::get`] performed inside that scope subscribes the effect to the
//! signal, so a later [`Signal::set`] re-runs exactly the effects that read it.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::{Rc, Weak};

struct EffectNode {
    f: RefCell<Box<dyn FnMut()>>,
    running: Cell<bool>,
    alive: Cell<bool>,
}

thread_local! {
    /// Effect currently executing (dependency-tracking scope).
    static OBSERVER: RefCell<Option<Rc<EffectNode>>> = const { RefCell::new(None) };
    /// Owns every live effect so signals can hold weak references.
    static REGISTRY: RefCell<Vec<Rc<EffectNode>>> = const { RefCell::new(Vec::new()) };
}

fn run_effect(node: &Rc<EffectNode>) {
    if !node.alive.get() || node.running.get() {
        return; // disposed, or re-entrant write from inside the effect itself
    }
    node.running.set(true);
    let prev = OBSERVER.with(|o| o.borrow_mut().replace(node.clone()));
    if let Ok(mut f) = node.f.try_borrow_mut() {
        f();
    }
    OBSERVER.with(|o| *o.borrow_mut() = prev);
    node.running.set(false);
}

/// Handle to a running effect. The effect stays alive until [`Effect::dispose`].
#[derive(Clone)]
pub struct Effect(Rc<EffectNode>);

impl Effect {
    /// Stop the effect; it will no longer react to signal updates.
    pub fn dispose(&self) {
        self.0.alive.set(false);
        REGISTRY.with(|r| r.borrow_mut().retain(|e| !Rc::ptr_eq(e, &self.0)));
    }

    /// Run the effect body again immediately (re-tracking dependencies).
    pub fn run(&self) {
        run_effect(&self.0);
    }

    /// `true` while the effect is still subscribed.
    pub fn is_alive(&self) -> bool {
        self.0.alive.get()
    }
}

impl fmt::Debug for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Effect")
            .field("alive", &self.is_alive())
            .finish()
    }
}

/// Register a reactive effect. It runs once immediately and again whenever any
/// signal read during its latest run is updated.
pub fn create_effect(f: impl FnMut() + 'static) -> Effect {
    let node = Rc::new(EffectNode {
        f: RefCell::new(Box::new(f)),
        running: Cell::new(false),
        alive: Cell::new(true),
    });
    REGISTRY.with(|r| r.borrow_mut().push(node.clone()));
    run_effect(&node);
    Effect(node)
}

/// Run `f` without subscribing the current effect to any signal it reads.
pub fn untrack<R>(f: impl FnOnce() -> R) -> R {
    let prev = OBSERVER.with(|o| o.borrow_mut().take());
    let out = f();
    OBSERVER.with(|o| *o.borrow_mut() = prev);
    out
}

struct SignalInner<T> {
    value: RefCell<T>,
    subs: RefCell<Vec<Weak<EffectNode>>>,
}

/// A reactive cell. Cloning a `Signal` shares the same underlying state.
pub struct Signal<T> {
    inner: Rc<SignalInner<T>>,
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Signal {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T> Signal<T> {
    /// Create a signal holding `initial`.
    pub fn new(initial: T) -> Self {
        Signal {
            inner: Rc::new(SignalInner {
                value: RefCell::new(initial),
                subs: RefCell::new(Vec::new()),
            }),
        }
    }

    /// Read the value by reference, subscribing the current effect.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        self.track();
        let v = self.inner.value.borrow();
        f(&v)
    }

    /// Read the value by reference without subscribing.
    pub fn with_untracked<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.inner.value.borrow())
    }

    /// Replace the value and notify subscribers.
    pub fn set(&self, value: T) {
        *self.inner.value.borrow_mut() = value;
        self.notify();
    }

    /// Mutate the value in place and notify subscribers.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        f(&mut self.inner.value.borrow_mut());
        self.notify();
    }

    /// Number of live subscribers (useful in tests).
    pub fn subscriber_count(&self) -> usize {
        self.inner
            .subs
            .borrow()
            .iter()
            .filter(|w| w.strong_count() > 0)
            .count()
    }

    fn track(&self) {
        if let Some(obs) = OBSERVER.with(|o| o.borrow().clone()) {
            let mut subs = self.inner.subs.borrow_mut();
            let known = subs
                .iter()
                .any(|w| w.upgrade().is_some_and(|e| Rc::ptr_eq(&e, &obs)));
            if !known {
                subs.push(Rc::downgrade(&obs));
            }
        }
    }

    fn notify(&self) {
        let live: Vec<Rc<EffectNode>> = {
            let mut subs = self.inner.subs.borrow_mut();
            subs.retain(|w| w.upgrade().is_some_and(|e| e.alive.get()));
            subs.iter().filter_map(|w| w.upgrade()).collect()
        };
        for e in live {
            run_effect(&e);
        }
    }
}

impl<T: Clone> Signal<T> {
    /// Clone the current value, subscribing the current effect.
    pub fn get(&self) -> T {
        self.with(|v| v.clone())
    }

    /// Clone the current value without subscribing.
    pub fn get_untracked(&self) -> T {
        self.with_untracked(|v| v.clone())
    }
}

impl<T: Default> Default for Signal<T> {
    fn default() -> Self {
        Signal::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with_untracked(|v| f.debug_tuple("Signal").field(v).finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_reruns_on_set() {
        let sig = Signal::new(1i32);
        let seen = Rc::new(RefCell::new(Vec::new()));
        let (s, o) = (sig.clone(), seen.clone());
        let _e = create_effect(move || o.borrow_mut().push(s.get()));
        assert_eq!(&*seen.borrow(), &[1]);
        sig.set(7);
        assert_eq!(&*seen.borrow(), &[1, 7]);
        assert_eq!(sig.subscriber_count(), 1);
    }

    #[test]
    fn untracked_reads_do_not_subscribe() {
        let a = Signal::new(0i32);
        let runs = Rc::new(Cell::new(0));
        let (s, r) = (a.clone(), runs.clone());
        let _e = create_effect(move || {
            untrack(|| s.get());
            r.set(r.get() + 1);
        });
        a.set(5);
        assert_eq!(runs.get(), 1);
        assert_eq!(a.subscriber_count(), 0);
    }

    #[test]
    fn dispose_stops_updates() {
        let a = Signal::new(0i32);
        let runs = Rc::new(Cell::new(0));
        let (s, r) = (a.clone(), runs.clone());
        let e = create_effect(move || {
            s.get();
            r.set(r.get() + 1);
        });
        a.set(1);
        e.dispose();
        a.set(2);
        assert_eq!(runs.get(), 2);
        assert!(!e.is_alive());
    }

    #[test]
    fn update_and_self_write_do_not_loop() {
        let a = Signal::new(0i32);
        let (s, r) = (a.clone(), Rc::new(Cell::new(0)));
        let runs = r.clone();
        let _e = create_effect(move || {
            let v = s.get();
            r.set(r.get() + 1);
            if v == 0 {
                s.set(1); // re-entrant write is ignored, no infinite loop
            }
        });
        a.update(|v| *v += 10);
        assert_eq!(a.get_untracked(), 11);
        assert_eq!(runs.get(), 2);
    }
}
