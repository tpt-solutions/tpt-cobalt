//! Certification report generation

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::tests::TestResults;
use crate::VendorProfile;

/// Certification report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationReport {
    pub vendor: String,
    pub gpu_model: String,
    pub certification_tier: u32,
    pub certification_date: String,
    pub report_date: String,
    pub passed: bool,
    pub compatibility: TestSummary,
    pub correctness: TestSummary,
    pub performance: Option<TestSummary>,
    pub overall_score: f64,
    pub recommendations: Vec<String>,
}

/// Test summary for a category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub passed: usize,
    pub total: usize,
    pub success_rate: f64,
    pub failures: Vec<String>,
}

impl From<&TestResults> for TestSummary {
    fn from(results: &TestResults) -> Self {
        TestSummary {
            passed: results.passed,
            total: results.total,
            success_rate: results.success_rate(),
            failures: results.failures.clone(),
        }
    }
}

/// Generate a certification report
pub fn generate_report(
    profile: &VendorProfile,
    compatibility: &TestResults,
    correctness: &TestResults,
    performance: Option<&TestResults>,
) -> Result<CertificationReport> {
    let compat_summary: TestSummary = compatibility.into();
    let correct_summary: TestSummary = correctness.into();
    let perf_summary: Option<TestSummary> = performance.map(|p| p.into());

    // Calculate overall score (weighted average)
    let mut total_weight = 0.0;
    let mut weighted_score = 0.0;

    // Compatibility: 30% weight
    total_weight += 0.3;
    weighted_score += compat_summary.success_rate * 0.3;

    // Correctness: 40% weight
    total_weight += 0.4;
    weighted_score += correct_summary.success_rate * 0.4;

    // Performance: 30% weight (if available)
    if let Some(ref perf) = perf_summary {
        total_weight += 0.3;
        weighted_score += perf.success_rate * 0.3;
    }

    let overall_score = if total_weight > 0.0 {
        weighted_score / total_weight
    } else {
        0.0
    };

    // Determine pass/fail based on tier
    let passed = match profile.certification_tier {
        1 => {
            // Tier 1: 80% compatibility, 90% correctness
            compat_summary.success_rate >= 0.8 && correct_summary.success_rate >= 0.9
        }
        2 => {
            // Tier 2: 90% compatibility, 95% correctness, 80% performance
            compat_summary.success_rate >= 0.9
                && correct_summary.success_rate >= 0.95
                && perf_summary.as_ref().is_some_and(|p| p.success_rate >= 0.8)
        }
        3 => {
            // Tier 3: 95% compatibility, 99% correctness, 90% performance
            compat_summary.success_rate >= 0.95
                && correct_summary.success_rate >= 0.99
                && perf_summary.as_ref().is_some_and(|p| p.success_rate >= 0.9)
        }
        _ => false,
    };

    // Generate recommendations
    let mut recommendations = Vec::new();

    if compat_summary.success_rate < 1.0 {
        recommendations.push("Address compatibility test failures before proceeding".to_string());
    }

    if correct_summary.success_rate < 1.0 {
        recommendations.push("Fix correctness issues to ensure numerical accuracy".to_string());
    }

    if let Some(ref perf) = perf_summary {
        if perf.success_rate < 0.9 {
            recommendations
                .push("Consider performance optimization for better efficiency".to_string());
        }
    }

    if passed {
        recommendations.push("Certification requirements met - ready for submission".to_string());
    } else {
        recommendations
            .push("Certification requirements not yet met - address issues above".to_string());
    }

    Ok(CertificationReport {
        vendor: profile.vendor.clone(),
        gpu_model: profile.gpu_model.clone(),
        certification_tier: profile.certification_tier,
        certification_date: profile.certification_date.clone().unwrap_or_default(),
        report_date: Utc::now().to_rfc3339(),
        passed,
        compatibility: compat_summary,
        correctness: correct_summary,
        performance: perf_summary,
        overall_score,
        recommendations,
    })
}
