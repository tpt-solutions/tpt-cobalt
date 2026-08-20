// `tpt-gpu-script` binary — thin wrapper around the shared CLI entry point.
//
// The `tpt` binary in `src/bin/tpt.rs` is an alias that calls the exact same
// `run` so both names work after `cargo install tpt-gpu-script-cli`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(tpt_gpu_script_cli::run(&args));
}
