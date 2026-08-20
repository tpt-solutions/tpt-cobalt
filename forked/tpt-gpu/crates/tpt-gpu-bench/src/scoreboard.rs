//! Community scoreboard — aggregates `results/<gpu>-<ts>.json` files and
//! renders a summary section for injection into BENCHMARKS.md.
//!
//! The rendered block is bracketed by HTML comment markers so the CI workflow
//! can safely replace it on every push without touching the rest of the file:
//!
//! ```text
//! <!-- SCOREBOARD:START -->
//! …generated markdown…
//! <!-- SCOREBOARD:END -->
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

use crate::report::BenchReport;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KindStats {
    pub avg_gflops: f64,
    pub peak_gflops: f64,
    pub avg_efficiency_pct: Option<f64>,
    #[allow(dead_code)]
    pub case_count: usize,
}

#[derive(Debug, Clone)]
pub struct GpuScorecard {
    pub gpu_model: String,
    pub run_count: usize,
    pub latest_timestamp: String,
    pub total_cases: usize,
    pub valid_cases: usize,
    /// Sorted by kernel kind name.
    pub by_kind: Vec<(String, KindStats)>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Read every `*.json` file in `dir`, parse as [`BenchReport`], skip bad files.
pub fn load_results(dir: &Path) -> Result<Vec<BenchReport>> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("opening results directory: {}", dir.display()))?;

    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort(); // lexicographic == chronological for <gpu>-<YYYYMMDDTHHmmSS>.json

    let mut reports = Vec::with_capacity(paths.len());
    for path in &paths {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        match serde_json::from_str::<BenchReport>(&raw) {
            Ok(r) => reports.push(r),
            Err(e) => eprintln!("scoreboard: skipping {} ({})", path.display(), e),
        }
    }
    Ok(reports)
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Group reports by GPU model and compute per-kind aggregate stats.
///
/// Returned scorecards are sorted: best peak GEMM GFLOPS first, ties broken
/// alphabetically by GPU model.
pub fn build_scorecards(reports: &[BenchReport]) -> Vec<GpuScorecard> {
    // (gflops_sum, peak_gflops, eff_sum, run_count, eff_count)
    type KindAccum = (f64, f64, f64, usize, usize);
    type GpuAccum = (usize, String, usize, usize, HashMap<String, KindAccum>);

    let mut by_gpu: HashMap<String, GpuAccum> = HashMap::new();

    for report in reports {
        let entry = by_gpu
            .entry(report.gpu_model.clone())
            .or_insert_with(|| (0, String::new(), 0, 0, HashMap::new()));
        entry.0 += 1; // run_count

        // Keep the latest timestamp
        if report.timestamp > entry.1 {
            entry.1 = report.timestamp.clone();
        }
        entry.2 += report.summary.total;
        entry.3 += report.summary.valid;

        for run in &report.runs {
            let k: &mut KindAccum = entry.4.entry(run.kind.clone()).or_default();
            k.0 += run.gflops;
            if run.gflops > k.1 {
                k.1 = run.gflops;
            }
            if let Some(eff) = run.efficiency_pct {
                k.2 += eff;
                k.4 += 1;
            }
            k.3 += 1;
        }
    }

    let mut scorecards: Vec<GpuScorecard> = by_gpu
        .into_iter()
        .map(|(gpu_model, (run_count, latest_ts, total, valid, kinds))| {
            let mut by_kind: Vec<(String, KindStats)> = kinds
                .into_iter()
                .map(|(kind, (gsum, peak, esum, cnt, ecnt))| {
                    (
                        kind,
                        KindStats {
                            avg_gflops: if cnt > 0 { gsum / cnt as f64 } else { 0.0 },
                            peak_gflops: peak,
                            avg_efficiency_pct: if ecnt > 0 {
                                Some(esum / ecnt as f64)
                            } else {
                                None
                            },
                            case_count: cnt,
                        },
                    )
                })
                .collect();
            by_kind.sort_by(|a, b| a.0.cmp(&b.0));
            GpuScorecard {
                gpu_model,
                run_count,
                latest_timestamp: latest_ts,
                total_cases: total,
                valid_cases: valid,
                by_kind,
            }
        })
        .collect();

    scorecards.sort_by(|a, b| {
        let ag = peak_kind(a, "gemm");
        let bg = peak_kind(b, "gemm");
        bg.partial_cmp(&ag)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.gpu_model.cmp(&b.gpu_model))
    });

    scorecards
}

fn peak_kind(c: &GpuScorecard, kind: &str) -> f64 {
    c.by_kind
        .iter()
        .find(|(k, _)| k == kind)
        .map(|(_, s)| s.peak_gflops)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the community scoreboard as a Markdown string (without the markers).
pub fn render_scoreboard(scorecards: &[GpuScorecard], updated_at: &str) -> String {
    let mut out = String::new();

    out.push_str("## Community GPU Scoreboard\n\n");
    out.push_str(&format!("_Last updated: {}_  \n", updated_at));
    out.push_str(&format!("_GPUs reported: {}_\n\n", scorecards.len()));

    if scorecards.is_empty() {
        out.push_str(
            "No results submitted yet.  \n\
             Run `cargo run -p tpt-gpu-bench -- --contribute` on your machine to be the first!\n",
        );
        return out;
    }

    // --- GEMM leaderboard ---
    let gemm_cards: Vec<_> = scorecards
        .iter()
        .filter(|c| c.by_kind.iter().any(|(k, _)| k == "gemm"))
        .collect();
    if !gemm_cards.is_empty() {
        out.push_str("### GEMM Leaderboard\n\n");
        out.push_str("| GPU | Avg GFLOPS | Peak GFLOPS | Avg vs cuBLAS | Submissions |\n");
        out.push_str("|-----|:----------:|:-----------:|:-------------:|:-----------:|\n");
        for c in &gemm_cards {
            if let Some((_, s)) = c.by_kind.iter().find(|(k, _)| k == "gemm") {
                let eff = fmt_eff(s.avg_efficiency_pct);
                out.push_str(&format!(
                    "| {} | {:.0} | {:.0} | {} | {} |\n",
                    c.gpu_model, s.avg_gflops, s.peak_gflops, eff, c.run_count
                ));
            }
        }
        out.push('\n');
    }

    // --- Attention leaderboard ---
    let mut attn_cards: Vec<_> = scorecards
        .iter()
        .filter(|c| c.by_kind.iter().any(|(k, _)| k == "attention"))
        .collect();
    if !attn_cards.is_empty() {
        attn_cards.sort_by(|a, b| {
            peak_kind(b, "attention")
                .partial_cmp(&peak_kind(a, "attention"))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.push_str("### Attention Leaderboard\n\n");
        out.push_str("| GPU | Avg GFLOPS | Peak GFLOPS | Avg vs FlashAttn v2 | Submissions |\n");
        out.push_str("|-----|:----------:|:-----------:|:-------------------:|:-----------:|\n");
        for c in &attn_cards {
            if let Some((_, s)) = c.by_kind.iter().find(|(k, _)| k == "attention") {
                let eff = fmt_eff(s.avg_efficiency_pct);
                out.push_str(&format!(
                    "| {} | {:.0} | {:.0} | {} | {} |\n",
                    c.gpu_model, s.avg_gflops, s.peak_gflops, eff, c.run_count
                ));
            }
        }
        out.push('\n');
    }

    // --- All submissions summary ---
    out.push_str("### All Submissions\n\n");
    out.push_str("| GPU | Kernel types | Cases | Valid | Last submitted |\n");
    out.push_str("|-----|-------------|:-----:|:-----:|:--------------:|\n");
    for c in scorecards {
        let kinds: Vec<_> = c.by_kind.iter().map(|(k, _)| k.as_str()).collect();
        let last = short_date(&c.latest_timestamp);
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            c.gpu_model,
            kinds.join(", "),
            c.total_cases,
            c.valid_cases,
            last
        ));
    }
    out.push('\n');

    out.push_str(
        "> **Contribute your GPU:** run `cargo run -p tpt-gpu-bench -- --contribute` then open a PR \
         adding `results/<gpu>-<ts>.json` — CI updates this table automatically.\n",
    );

    out
}

fn fmt_eff(e: Option<f64>) -> String {
    e.map(|v| format!("{:.1}%", v))
        .unwrap_or_else(|| "—".into())
}

fn short_date(ts: &str) -> &str {
    if ts.len() >= 10 {
        &ts[..10]
    } else {
        ts
    }
}

// ---------------------------------------------------------------------------
// BENCHMARKS.md injection
// ---------------------------------------------------------------------------

const MARKER_START: &str = "<!-- SCOREBOARD:START -->";
const MARKER_END: &str = "<!-- SCOREBOARD:END -->";

/// Inject or replace the community scoreboard block in `path`.
///
/// * If the file already contains the `SCOREBOARD:START/END` markers, the
///   content between them is replaced in-place.
/// * If the markers are absent, the block is inserted before the first
///   `## Contributing` heading, or appended if that heading is not found.
///
/// Returns `true` when the file was actually changed.
pub fn update_benchmarks_md(path: &Path, scorecards: &[GpuScorecard]) -> Result<bool> {
    let updated_at = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let body = render_scoreboard(scorecards, &updated_at);
    let block = format!("{}\n{}{}\n", MARKER_START, body, MARKER_END);

    let existing = if path.exists() {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        String::new()
    };

    let updated =
        if let (Some(si), Some(ei)) = (existing.find(MARKER_START), existing.find(MARKER_END)) {
            // Replace between (and including) the markers.
            let after = ei + MARKER_END.len();
            format!("{}{}{}", &existing[..si], block, &existing[after..])
        } else {
            // Insert before "## Contributing" or append.
            let insert_at = existing.find("\n## Contributing").unwrap_or(existing.len());
            format!(
                "{}\n{}{}",
                &existing[..insert_at],
                block,
                &existing[insert_at..]
            )
        };

    if updated == existing {
        return Ok(false);
    }

    std::fs::write(path, &updated).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}
