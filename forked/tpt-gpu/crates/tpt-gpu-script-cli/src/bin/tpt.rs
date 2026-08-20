// `tpt` binary — a short alias for `tpt-gpu-script`.
//
// Both binaries share the same `run` implementation in `lib.rs`, so `tpt` and
// `tpt-gpu-script` are interchangeable. The shorter name matches the command
// used throughout the documentation and tutorials.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(tpt_gpu_script_cli::run(&args));
}
