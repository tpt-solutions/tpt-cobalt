//! `tpt-bench-report`: prints the fixed kernel suite as Markdown + JSON
//! (the Cobalt half of the PyTorch comparison in benches/PYTORCH_PROTOCOL.md).

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let ms = tpt_bench::run_report(iters);
    println!("```markdown\n{}```", tpt_bench::report_markdown(&ms));
    println!("{}", tpt_bench::report_json(&ms));
}
