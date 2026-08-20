# tpt-eng-schedule

CPM/PERT project-scheduling networks, resource leveling, and Earned Value
Management primitives.

All durations and time values are plain `f64` expressed in a single
consistent, caller-chosen unit of time (e.g. days); the crate performs no
unit conversion or checking within one `Schedule`. It covers four coherent
sub-domains: CPM/PERT activity-on-node networks (`Schedule` builds an AON
network from `Activity` definitions and runs the forward/backward passes for
early/late start and finish, total float, and the critical path); PERT
three-point estimation; resource leveling (peak demand and moving-average
smoothing variance reduction); and Earned Value Management indices (CPI, SPI,
EAC).

## Features

- **[`Activity`] / [`Schedule`]** — AON network construction with eager
  forward/backward pass: `early_start`/`early_finish`,
  `late_start`/`late_finish`, `total_float`, `is_critical`, `critical_path`,
  `project_duration`.
- **[`ScheduleError`]** — duplicate ids, missing predecessors, cycle
  detection, and unknown-activity query errors.
- **[`pert_expected`] / [`pert_variance`]** — three-point (optimistic /
  most-likely / pessimistic) PERT duration estimation.
- **[`leveling_peak`] / [`resource_variance`] / [`smooth_demands`] /
  [`leveling_smooth_variance_reduction`]** — resource-leveling analysis via
  centered moving-average smoothing.
- **[`cpi`] / [`spi`] / [`eac`]** — Earned Value Management cost/schedule
  performance indices and estimate-at-completion.

## Installation

```toml
[dependencies]
tpt-eng-schedule = "0.1"
```

## Quick start

```rust
use tpt_eng_schedule::{Activity, Schedule};

let sched = Schedule::new(vec![
    Activity { id: "A".into(), duration: 2.0, predecessors: vec![] },
    Activity { id: "B".into(), duration: 3.0, predecessors: vec!["A".into()] },
])?;
assert_eq!(sched.early_start("B")?, 2.0);
assert_eq!(sched.total_float("A")?, 0.0);
assert!(sched.is_critical("A")?);
assert_eq!(sched.critical_path(), vec!["A".to_string(), "B".to_string()]);
# Ok::<(), tpt_eng_schedule::ScheduleError>(())
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Activity` / `Schedule` | CPM/PERT activity-on-node network and forward/backward pass. |
| `ScheduleError` | Construction/query error type. |
| `pert_expected` / `pert_variance` | Three-point PERT duration estimation. |
| `leveling_peak` / `smooth_demands` / `leveling_smooth_variance_reduction` | Resource leveling analysis. |
| `cpi` / `spi` / `eac` | Earned Value Management indices. |

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
