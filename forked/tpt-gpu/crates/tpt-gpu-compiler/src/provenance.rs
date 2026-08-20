//! Provenance metadata for emitted TPTIR kernels.
//!
//! Every `.mlir` file produced by the `tpt-gpu-kernelgen` tool carries a
//! `# TPTIR Provenance:` comment as its first line, embedding the date,
//! host/model identifier, GFLOPS score, and hardware triple of the run
//! that produced it. This lets CI and developers trace a kernel back to
//! the exact environment in which it was generated.
//!
//! Example:
//! ```text
//! # TPTIR Provenance: {"date":"2026-06-28T13:37:00","model":"tptc-rs/x86_64-pc-windows-msvc/vector_add","score_gflops":"0.0000","host":"x86_64-pc-windows-msvc"}
//! module {
//!   ...
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Recorded provenance for one generated kernel artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// ISO-8601 date+time of generation (seconds precision, UTC).
    pub date: String,
    /// Free-form model identifier. By default
    /// `tptc-rs/<host_triple>/<kernel>`; override with `TPT_MODEL`.
    pub model: String,
    /// GFLOPS score from the Quick benchmark, formatted to 4 decimals.
    /// `0.0000` when no benchmark is available (e.g. `--no-bench`).
    pub score_gflops: String,
    /// Host triple of the generator (`<arch>-<os>-<env>`), e.g.
    /// `x86_64-pc-windows-msvc`.
    pub host: String,
}

impl Provenance {
    /// Build a provenance record from the running host environment and an
    /// optional benchmark score.
    pub fn new(kernel: &str, score_gflops: f64) -> Self {
        Provenance {
            date: Self::now_iso8601(),
            model: Self::derive_model(kernel),
            score_gflops: format!("{score_gflops:.4}"),
            host: Self::host_triple(),
        }
    }

    /// Render a one-line comment suitable for prepending to `.mlir`.
    pub fn to_mlir_comment(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"date":"unknown","model":"unknown","score_gflops":"0.0000","host":"unknown"}"#
                .to_string()
        });
        format!("# TPTIR Provenance: {json}\n")
    }

    #[cfg(target_os = "windows")]
    fn now_iso8601() -> String {
        use std::mem::MaybeUninit;
        use std::os::raw::c_ushort;
        #[repr(C)]
        struct SystemTime {
            w_year: c_ushort,
            w_month: c_ushort,
            w_day_of_week: c_ushort,
            w_day: c_ushort,
            w_hour: c_ushort,
            w_minute: c_ushort,
            w_second: c_ushort,
            w_milliseconds: c_ushort,
        }
        extern "system" {
            fn GetLocalTime(lp: *mut SystemTime);
        }
        unsafe {
            let mut st = MaybeUninit::<SystemTime>::uninit();
            GetLocalTime(st.as_mut_ptr());
            let st = st.assume_init();
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                st.w_year, st.w_month, st.w_day, st.w_hour, st.w_minute, st.w_second,
            )
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn now_iso8601() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        if let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) {
            let s = dur.as_secs();
            let year = 1970 + (s / 86400) / 365;
            let day_of_year = (s / 86400) % 365;
            let hour = (s % 86400) / 3600;
            let minute = (s % 3600) / 60;
            let second = s % 60;
            format!("{year:04}-{day_of_year:03}T{hour:02}:{minute:02}:{second:02}",)
        } else {
            "unknown".to_string()
        }
    }

    fn derive_model(kernel: &str) -> String {
        if let Ok(m) = std::env::var("TPT_MODEL") {
            return format!("tptc-rs/{m}/{kernel}");
        }
        format!("tptc-rs/{}/{kernel}", Self::host_triple())
    }

    fn host_triple() -> String {
        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        let env = std::env::consts::FAMILY;
        format!("{arch}-{os}-{env}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_comment_contains_required_fields() {
        let p = Provenance::new("vector_add", 0.0);
        let c = p.to_mlir_comment();
        assert!(c.starts_with("# TPTIR Provenance:"));
        assert!(c.contains("\"date\""));
        assert!(c.contains("\"model\""));
        assert!(c.contains("\"score_gflops\""));
        assert!(c.contains("\"host\""));
    }

    #[test]
    fn test_provenance_score_formatting() {
        let p = Provenance::new("matmul", 12.3456);
        assert_eq!(p.score_gflops, "12.3456");
    }

    #[test]
    fn test_provenance_kernel_in_model() {
        let p = Provenance::new("conv3d", 0.0);
        assert!(p.model.contains("conv3d"));
    }
}
