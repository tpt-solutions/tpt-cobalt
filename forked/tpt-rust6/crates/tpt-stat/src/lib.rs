//! # tpt-stat — Unified Statistical Modeling
//!
//! Distributions, hypothesis tests, regression, Bayesian inference, and time
//! series under one typed API. Every result renders through `Display` (and the
//! `Summary` wrapper) so it can be embedded directly in `tpt-lab`.

pub mod bayes;
pub mod dist;
pub mod error;
pub mod regression;
pub mod special;
pub mod tests;
pub mod time_series;

use std::fmt;

pub use error::StatError;

/// A renderable model/test summary. This is the type `tpt-lab` consumes for
/// rich inline rendering.
#[derive(Debug, Clone)]
pub enum Summary {
    Test(tests::TestResult),
    OLS(regression::OLSResult),
    ARIMA(time_series::ARIMAResult),
    GARCH(time_series::GARCHResult),
    Bayesian(bayes::Posterior),
    Distribution(String),
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Summary::Test(t) => write!(f, "{}", t),
            Summary::OLS(r) => write!(f, "{}", r),
            Summary::ARIMA(r) => write!(f, "{}", r),
            Summary::GARCH(r) => write!(f, "{}", r),
            Summary::Bayesian(p) => {
                writeln!(
                    f,
                    "Posterior ({} chains, {} params)",
                    p.chains.len(),
                    p.dim()
                )?;
                for name in &p.names {
                    let (lo, hi) = p.credible_interval(name, 0.95);
                    let rhat = match p.rhat(name) {
                        Ok(r) => format!("{r:.3}"),
                        Err(_) => "n/a".to_string(),
                    };
                    writeln!(
                        f,
                        "  {:<6} mean = {:.4}  95% CI = [{:.4}, {:.4}]  R-hat = {}",
                        name,
                        p.mean(name),
                        lo,
                        hi,
                        rhat
                    )?;
                }
                Ok(())
            }
            Summary::Distribution(s) => write!(f, "{}", s),
        }
    }
}

/// Construct a [`Summary`] from any result type.
pub trait IntoSummary {
    fn into_summary(self) -> Summary;
}

impl IntoSummary for tests::TestResult {
    fn into_summary(self) -> Summary {
        Summary::Test(self)
    }
}
impl IntoSummary for regression::OLSResult {
    fn into_summary(self) -> Summary {
        Summary::OLS(self)
    }
}
impl IntoSummary for time_series::ARIMAResult {
    fn into_summary(self) -> Summary {
        Summary::ARIMA(self)
    }
}
impl IntoSummary for time_series::GARCHResult {
    fn into_summary(self) -> Summary {
        Summary::GARCH(self)
    }
}
impl IntoSummary for bayes::Posterior {
    fn into_summary(self) -> Summary {
        Summary::Bayesian(self)
    }
}

pub mod prelude {
    pub use crate::bayes;
    pub use crate::dist;
    pub use crate::regression;
    pub use crate::tests;
    pub use crate::time_series;
    pub use crate::StatError;
    pub use crate::{IntoSummary, Summary};
}
