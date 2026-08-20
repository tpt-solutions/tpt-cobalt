//! # tpt-eng-schedule
//!
//! Project-scheduling primitives for the TPT engineering toolkit.
//!
//! All durations and time values are plain [`f64`] expressed in a single
//! consistent, caller-chosen unit of time (e.g. days). Do **not** mix units
//! within one [`Schedule`]; the crate performs no unit conversion or checking.
//!
//! The crate covers four coherent, real sub-domains of project scheduling:
//!
//! * **CPM/PERT activity-on-node networks** — [`Schedule`] builds an AON
//!   network from [`Activity`] definitions and runs the forward and backward
//!   passes to obtain early/late start and finish, total float, and the
//!   critical path.
//! * **PERT estimation** — [`pert_expected`] and [`pert_variance`] convert
//!   optimistic/most-likely/pessimistic estimates into a single expected
//!   duration and its variance.
//! * **Resource leveling** — [`leveling_peak`] reports the maximum daily
//!   demand, and [`leveling_smooth_variance_reduction`] quantifies how much a
//!   simple centered moving-average smoothing reduces day-to-day demand
//!   variance.
//! * **Earned Value Management** — [`cpi`], [`spi`], and [`eac`] compute the
//!   cost/schedule performance indices and the estimate-at-completion.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_schedule::{Activity, Schedule};
//!
//! let sched = Schedule::new(vec![
//!     Activity { id: "A".into(), duration: 2.0, predecessors: vec![] },
//!     Activity { id: "B".into(), duration: 3.0, predecessors: vec!["A".into()] },
//! ])?;
//! assert_eq!(sched.early_start("B")?, 2.0);
//! assert_eq!(sched.total_float("A")?, 0.0);
//! assert!(sched.is_critical("A")?);
//! assert_eq!(sched.critical_path(), vec!["A".to_string(), "B".to_string()]);
//! # Ok::<(), tpt_eng_schedule::ScheduleError>(())
//! ```

use std::collections::HashMap;
use std::fmt;

/// An activity in an activity-on-node (AON) project network.
///
/// `duration` is in the caller's chosen unit of time (the crate does not
/// enforce any particular unit — see the crate-level documentation).
/// `predecessors` lists the activity `id`s that must finish before this
/// activity can start.
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    /// Unique activity identifier.
    pub id: String,
    /// Planned duration of the activity in the project's time unit.
    pub duration: f64,
    /// Identifiers of activities that must complete before this one starts.
    pub predecessors: Vec<String>,
}

/// Errors produced while constructing or querying a [`Schedule`].
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    /// Two or more activities share the same `id`.
    DuplicateActivityId(String),
    /// An activity references a predecessor `id` that is not present in the
    /// network.
    MissingPredecessor {
        /// The dependent activity.
        activity: String,
        /// The absent predecessor.
        predecessor: String,
    },
    /// The predecessor graph contains a cycle, so no valid topological order
    /// (and therefore no forward/backward pass) exists.
    CycleDetected,
    /// A query referenced an `id` that is not part of the network.
    UnknownActivity(String),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::DuplicateActivityId(id) => {
                write!(f, "duplicate activity id: {id}")
            }
            ScheduleError::MissingPredecessor {
                activity,
                predecessor,
            } => write!(
                f,
                "activity {activity} references unknown predecessor {predecessor}"
            ),
            ScheduleError::CycleDetected => {
                write!(f, "activity network contains a cycle")
            }
            ScheduleError::UnknownActivity(id) => {
                write!(f, "unknown activity id: {id}")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

/// A resolved, fully-computed CPM schedule.
///
/// Construct with [`Schedule::new`], which validates the network and runs the
/// forward and backward passes eagerly. All query methods are then O(1) lookups.
#[derive(Debug, Clone)]
pub struct Schedule {
    activities: Vec<Activity>,
    index: HashMap<String, usize>,
    order: Vec<usize>,
    es: Vec<f64>,
    ef: Vec<f64>,
    ls: Vec<f64>,
    lf: Vec<f64>,
    project_duration: f64,
}

impl Schedule {
    /// Build a [`Schedule`] from a set of [`Activity`]s and compute its
    /// early/late times via the forward and backward passes.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::DuplicateActivityId`] if two activities share
    /// an `id`, [`ScheduleError::MissingPredecessor`] if an activity names a
    /// predecessor that is not in the set, and [`ScheduleError::CycleDetected`]
    /// if the predecessor graph contains a cycle.
    pub fn new(activities: Vec<Activity>) -> Result<Self, ScheduleError> {
        let mut index = HashMap::with_capacity(activities.len());
        for (i, a) in activities.iter().enumerate() {
            if index.contains_key(&a.id) {
                return Err(ScheduleError::DuplicateActivityId(a.id.clone()));
            }
            index.insert(a.id.clone(), i);
        }

        // Precompute predecessor indices once. Validation of every predecessor
        // id happens here, bubbling up a `MissingPredecessor` error rather than
        // panicking later.
        let mut pred_idx: Vec<Vec<usize>> = Vec::with_capacity(activities.len());
        for a in &activities {
            let mut preds = Vec::with_capacity(a.predecessors.len());
            for p in &a.predecessors {
                match index.get(p) {
                    Some(&pi) => preds.push(pi),
                    None => {
                        return Err(ScheduleError::MissingPredecessor {
                            activity: a.id.clone(),
                            predecessor: p.clone(),
                        });
                    }
                }
            }
            pred_idx.push(preds);
        }

        let order = topological_order(&pred_idx)?;

        // Forward pass: ES = max(EF of predecessors), or 0 if none.
        // EF = ES + duration.
        let n = activities.len();
        let mut es = vec![0.0; n];
        let mut ef = vec![0.0; n];
        for &i in &order {
            let dur = activities[i].duration;
            let mut start = 0.0;
            for &pi in &pred_idx[i] {
                start = f64::max(start, ef[pi]);
            }
            es[i] = start;
            ef[i] = start + dur;
        }

        let project_duration = ef.iter().copied().fold(0.0_f64, f64::max);

        // Backward pass: LF = min(LS of successors), or project_duration if it
        // is a sink. LS = LF - duration.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, preds) in pred_idx.iter().enumerate() {
            for &pi in preds {
                successors[pi].push(i);
            }
        }

        let mut lf = vec![0.0; n];
        let mut ls = vec![0.0; n];
        for &i in order.iter().rev() {
            let dur = activities[i].duration;
            let mut finish = project_duration;
            if !successors[i].is_empty() {
                finish = f64::INFINITY;
                for &s in &successors[i] {
                    finish = f64::min(finish, ls[s]);
                }
            }
            lf[i] = finish;
            ls[i] = finish - dur;
        }

        Ok(Self {
            activities,
            index,
            order,
            es,
            ef,
            ls,
            lf,
            project_duration,
        })
    }

    /// The total project duration: the maximum early finish across all
    /// activities (i.e. the length of the critical path).
    #[must_use]
    pub fn project_duration(&self) -> f64 {
        self.project_duration
    }

    /// Early start of `id` — the earliest time the activity can begin given
    /// its predecessors' early finishes.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn early_start(&self, id: &str) -> Result<f64, ScheduleError> {
        self.idx(id).map(|i| self.es[i])
    }

    /// Early finish of `id` — `early_start(id) + duration(id)`.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn early_finish(&self, id: &str) -> Result<f64, ScheduleError> {
        self.idx(id).map(|i| self.ef[i])
    }

    /// Late start of `id` — the latest time the activity can begin without
    /// delaying the project (or any successor's late start).
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn late_start(&self, id: &str) -> Result<f64, ScheduleError> {
        self.idx(id).map(|i| self.ls[i])
    }

    /// Late finish of `id` — `late_start(id) + duration(id)`.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn late_finish(&self, id: &str) -> Result<f64, ScheduleError> {
        self.idx(id).map(|i| self.lf[i])
    }

    /// Total float of `id` — `late_start(id) − early_start(id)`. It is the
    /// amount of time the activity can slip without delaying the project.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn total_float(&self, id: &str) -> Result<f64, ScheduleError> {
        let i = self.idx(id)?;
        Ok(self.ls[i] - self.es[i])
    }

    /// Whether `id` lies on the critical path (total float ≈ 0). Activities
    /// within a small tolerance of zero float are treated as critical.
    ///
    /// # Errors
    ///
    /// Returns [`ScheduleError::UnknownActivity`] if `id` is not in the
    /// network.
    pub fn is_critical(&self, id: &str) -> Result<bool, ScheduleError> {
        Ok(self.total_float(id)? <= CRITICAL_EPSILON)
    }

    /// The critical path as a sequence of activity `id`s in topological order:
    /// every activity with zero total float, listed start-to-finish.
    #[must_use]
    pub fn critical_path(&self) -> Vec<String> {
        self.order
            .iter()
            .filter(|&&i| self.ls[i] - self.es[i] <= CRITICAL_EPSILON)
            .map(|&i| self.activities[i].id.clone())
            .collect()
    }

    fn idx(&self, id: &str) -> Result<usize, ScheduleError> {
        self.index
            .get(id)
            .copied()
            .ok_or_else(|| ScheduleError::UnknownActivity(id.to_string()))
    }
}

/// Tolerance (in the project's time unit) below which total float is treated
/// as zero when deciding criticality. Guards against floating-point noise on
/// round durations.
const CRITICAL_EPSILON: f64 = 1e-9;

/// Compute a deterministic topological order via Kahn's algorithm.
///
/// `pred_idx[i]` is the list of predecessor indices of activity `i`. Returns
/// the activity indices in an order where every predecessor precedes its
/// successors. Activities with no predecessors come first; ties are broken by
/// their original declaration order for reproducibility.
fn topological_order(pred_idx: &[Vec<usize>]) -> Result<Vec<usize>, ScheduleError> {
    let n = pred_idx.len();
    let mut indeg = vec![0usize; n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, preds) in pred_idx.iter().enumerate() {
        indeg[i] = preds.len();
        for &pi in preds {
            succ[pi].push(i);
        }
    }

    // Stable priority queue: process ready nodes in original-index order.
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.iter().copied().min() {
        ready.retain(|&x| x != i);
        order.push(i);
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }

    if order.len() != n {
        return Err(ScheduleError::CycleDetected);
    }
    Ok(order)
}

/// PERT expected duration from three-point estimates.
///
/// `te = (o + 4·m + p) / 6`, where `o`, `m`, `p` are the optimistic,
/// most-likely, and pessimistic durations in the project's time unit.
#[must_use]
pub fn pert_expected(o: f64, m: f64, p: f64) -> f64 {
    (o + 4.0 * m + p) / 6.0
}

/// PERT variance of a duration estimate from three-point estimates.
///
/// `σ² = ((p − o) / 6)²`, where `o` and `p` are the optimistic and
/// pessimistic durations. A narrow spread (small `p − o`) yields low variance.
#[must_use]
pub fn pert_variance(o: f64, p: f64) -> f64 {
    let spread = (p - o) / 6.0;
    spread * spread
}

/// The peak daily resource demand across the schedule horizon.
///
/// Returns the maximum element of `demands`, or `0.0` when `demands` is empty.
/// `demands[k]` is the total resource demand (e.g. worker-days) on day `k`.
#[must_use]
pub fn leveling_peak(demands: &[f64]) -> f64 {
    demands.iter().copied().fold(0.0_f64, f64::max)
}

/// Variance of a slice of daily resource demands.
///
/// Population variance `Σ(x − μ)² / n` over `demands`; `0.0` when empty.
#[must_use]
pub fn resource_variance(demands: &[f64]) -> f64 {
    if demands.is_empty() {
        return 0.0;
    }
    let mean = demands.iter().copied().sum::<f64>() / demands.len() as f64;
    let ss = demands.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
    ss / demands.len() as f64
}

/// Smooth daily resource demands with a centered moving average of `window`
/// points, returning a new demand series of the same length.
///
/// Near the ends where a full window is unavailable, the average shrinks to
/// the values actually present (so endpoints keep their immediate neighbours).
/// `window` is clamped to at least 1, and `window == 1` returns a copy of the
/// input (no smoothing).
#[must_use]
pub fn smooth_demands(demands: &[f64], window: usize) -> Vec<f64> {
    let w = window.max(1);
    if w == 1 || demands.is_empty() {
        return demands.to_vec();
    }
    let half = w / 2;
    let mut out = Vec::with_capacity(demands.len());
    for i in 0..demands.len() {
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(demands.len() - 1);
        let sum: f64 = demands[lo..=hi].iter().copied().sum();
        out.push(sum / (hi - lo + 1) as f64);
    }
    out
}

/// Variance reduction achieved by smoothing `demands` with a centered
/// moving-average window of `window` points.
///
/// Returns `resource_variance(original) − resource_variance(smoothed)`. A
/// positive result means the smoothing flattened the demand profile (the usual
/// goal of resource leveling); a negative result means it increased variance.
#[must_use]
pub fn leveling_smooth_variance_reduction(demands: &[f64], window: usize) -> f64 {
    let original = resource_variance(demands);
    let smoothed = resource_variance(&smooth_demands(demands, window));
    original - smoothed
}

/// Cost Performance Index: `cpi = ev / ac`.
///
/// Values below 1.0 indicate the work performed cost more than planned. For
/// `ac == 0.0` this returns `f64::INFINITY` (IEEE-754 division by zero), so
/// callers should guard against a zero actual cost before interpreting the
/// index.
#[must_use]
pub fn cpi(ev: f64, ac: f64) -> f64 {
    ev / ac
}

/// Schedule Performance Index: `spi = ev / pv`.
///
/// Values below 1.0 indicate the work performed is behind the planned
/// schedule. For `pv == 0.0` this returns `f64::INFINITY` (IEEE-754 division
/// by zero); callers should guard against a zero planned value.
#[must_use]
pub fn spi(ev: f64, pv: f64) -> f64 {
    ev / pv
}

/// Estimate At Completion from budget, earned value, and actual cost:
/// `eac = bac · ac / ev` (equivalently `bac / cpi`).
///
/// Projects total cost at completion assuming current cost efficiency
/// persists. For `ev == 0.0` this returns `f64::INFINITY` (IEEE-754 division
/// by zero); callers should guard against a zero earned value.
#[must_use]
pub fn eac(bac: f64, ev: f64, ac: f64) -> f64 {
    bac * ac / ev
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> Schedule {
        Schedule::new(vec![
            Activity {
                id: "A".into(),
                duration: 2.0,
                predecessors: vec![],
            },
            Activity {
                id: "B".into(),
                duration: 3.0,
                predecessors: vec!["A".into()],
            },
        ])
        .unwrap()
    }

    fn parallel() -> Schedule {
        Schedule::new(vec![
            Activity {
                id: "A".into(),
                duration: 2.0,
                predecessors: vec![],
            },
            Activity {
                id: "B".into(),
                duration: 1.0,
                predecessors: vec![],
            },
            Activity {
                id: "C".into(),
                duration: 1.0,
                predecessors: vec!["A".into(), "B".into()],
            },
        ])
        .unwrap()
    }

    #[test]
    fn chain_b_early_start_is_a_finish() {
        let s = chain();
        assert_eq!(s.early_start("A").unwrap(), 0.0);
        assert_eq!(s.early_finish("A").unwrap(), 2.0);
        assert_eq!(s.early_start("B").unwrap(), 2.0);
        assert_eq!(s.early_finish("B").unwrap(), 5.0);
        assert_eq!(s.project_duration(), 5.0);
    }

    #[test]
    fn chain_is_all_critical() {
        let s = chain();
        assert_eq!(s.total_float("A").unwrap(), 0.0);
        assert_eq!(s.total_float("B").unwrap(), 0.0);
        assert!(s.is_critical("A").unwrap());
        assert!(s.is_critical("B").unwrap());
        assert_eq!(s.critical_path(), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn parallel_c_and_a_critical_b_has_float() {
        let s = parallel();
        assert_eq!(s.early_start("C").unwrap(), 2.0);
        assert_eq!(s.total_float("A").unwrap(), 0.0);
        assert_eq!(s.total_float("B").unwrap(), 1.0);
        assert!(s.is_critical("C").unwrap());
        assert!(s.is_critical("A").unwrap());
        assert!(!s.is_critical("B").unwrap());
        // C is the sink and critical; A feeds it with zero float.
        assert_eq!(s.critical_path(), vec!["A".to_string(), "C".to_string()]);
    }

    #[test]
    fn backward_pass_late_times() {
        let s = parallel();
        // Project duration = 3 (A:0-2, B:0-1, C:2-3). Sink C late = 2..3.
        assert_eq!(s.late_start("C").unwrap(), 2.0);
        assert_eq!(s.late_finish("C").unwrap(), 3.0);
        // B can slip a day: late start 1, early start 0 -> float 1.
        assert_eq!(s.late_start("B").unwrap(), 1.0);
        // A is critical: late start 0.
        assert_eq!(s.late_start("A").unwrap(), 0.0);
    }

    #[test]
    fn construction_errors() {
        assert_eq!(
            Schedule::new(vec![
                Activity {
                    id: "X".into(),
                    duration: 1.0,
                    predecessors: vec![]
                },
                Activity {
                    id: "X".into(),
                    duration: 1.0,
                    predecessors: vec![]
                },
            ])
            .unwrap_err(),
            ScheduleError::DuplicateActivityId("X".into())
        );
        assert_eq!(
            Schedule::new(vec![Activity {
                id: "Y".into(),
                duration: 1.0,
                predecessors: vec!["Z".into()],
            }])
            .unwrap_err(),
            ScheduleError::MissingPredecessor {
                activity: "Y".into(),
                predecessor: "Z".into()
            }
        );
        assert_eq!(
            Schedule::new(vec![
                Activity {
                    id: "P".into(),
                    duration: 1.0,
                    predecessors: vec!["Q".into()]
                },
                Activity {
                    id: "Q".into(),
                    duration: 1.0,
                    predecessors: vec!["P".into()]
                },
            ])
            .unwrap_err(),
            ScheduleError::CycleDetected
        );
        assert_eq!(
            chain().early_start("nope").unwrap_err(),
            ScheduleError::UnknownActivity("nope".into())
        );
    }

    #[test]
    fn pert_formulae() {
        // te = (1 + 4*2 + 3)/6 = 2.0
        assert_eq!(pert_expected(1.0, 2.0, 3.0), 2.0);
        // variance = ((3-1)/6)^2 = (1/3)^2 = 1/9
        assert!((pert_variance(1.0, 3.0) - 1.0 / 9.0).abs() < 1e-12);
        // symmetric around m for balanced spread
        assert_eq!(pert_expected(0.0, 10.0, 20.0), 10.0);
    }

    #[test]
    fn resource_peak_and_variance() {
        let d = [3.0, 7.0, 2.0, 5.0];
        assert_eq!(leveling_peak(&d), 7.0);
        assert_eq!(leveling_peak(&[]), 0.0);
        // mean = 17/4 = 4.25; variance = ((3-4.25)^2+(7-4.25)^2+(2-4.25)^2+(5-4.25)^2)/4
        let expected = (((3.0 - 4.25) * (3.0 - 4.25))
            + ((7.0 - 4.25) * (7.0 - 4.25))
            + ((2.0 - 4.25) * (2.0 - 4.25))
            + ((5.0 - 4.25) * (5.0 - 4.25)))
            / 4.0;
        assert!((resource_variance(&d) - expected).abs() < 1e-12);
    }

    #[test]
    fn smoothing_reduces_variance_for_spiky_profile() {
        let d = [1.0, 9.0, 1.0, 9.0, 1.0];
        let reduction = leveling_smooth_variance_reduction(&d, 3);
        assert!(
            reduction > 0.0,
            "smoothing should flatten a spiky profile, got {reduction}"
        );
        // window 1 is identity -> zero reduction.
        assert_eq!(leveling_smooth_variance_reduction(&d, 1), 0.0);
    }

    #[test]
    fn evm_indices() {
        // BAC 100, EV 50, AC 40 -> CPI 1.25 (under budget), SPI needs PV.
        assert!((cpi(50.0, 40.0) - 1.25).abs() < 1e-12);
        assert!((spi(50.0, 100.0) - 0.5).abs() < 1e-12);
        // EAC = 100 * 40 / 50 = 80.
        assert!((eac(100.0, 50.0, 40.0) - 80.0).abs() < 1e-12);
        // EAC == BAC / CPI consistency.
        assert!((eac(100.0, 50.0, 40.0) - 100.0 / cpi(50.0, 40.0)).abs() < 1e-12);
    }

    #[test]
    fn zero_denominator_returns_infinity() {
        // f64 division by zero is defined as ±inf (not a panic); callers must
        // guard before interpreting the index.
        assert!(cpi(10.0, 0.0).is_infinite());
        assert!(spi(10.0, 0.0).is_infinite());
        assert!(eac(100.0, 0.0, 40.0).is_infinite());
    }
}
