//! TPT-GPU Doctor — environment health check.
//!
//! Mirrors the checks CI runs so a contributor can verify the same things
//! locally before opening a pull request:
//!
//! * Rust toolchain present and recent enough (>= 1.75)
//! * `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`
//!   (the exact gates CI enforces)
//! * Optional vendor SDK presence (CUDA, ROCm, Metal) — informational only
//! * Python + `tptr` importability — informational only
//!
//! Exit code is non-zero if any *required* check fails, so the command can be
//! used directly as a pre-commit hook (`--pre-commit` runs only fmt + clippy).

use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

const MIN_RUST: (u32, u32, u32) = (1, 75, 0);

struct Check {
    name: &'static str,
    status: Status,
    detail: String,
}

#[derive(PartialEq, Clone, Copy)]
enum Status {
    Pass,
    Fail,
    Warn,
    Skip,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let pre_commit = args.iter().any(|a| a == "--pre-commit");
    let fast = args.iter().any(|a| a == "--fast");

    println!("TPT-GPU Doctor");
    println!("{}", "─".repeat(64));

    let mut checks: Vec<Check> = Vec::new();

    checks.push(toolchain());

    if !fast {
        checks.push(run_gate(
            "cargo fmt --check",
            &["cargo", "fmt", "--all", "--", "--check"],
        ));
        checks.push(run_gate(
            "cargo clippy (-D warnings)",
            &["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
        ));
    }

    if !pre_commit && !fast {
        checks.push(vendor_cuda());
        checks.push(vendor_rocm());
        checks.push(vendor_metal());
        checks.push(python());
    }

    println!();
    let mut failed = 0;
    for c in &checks {
        let (glyph, label) = match c.status {
            Status::Pass => ("[PASS]", "ok"),
            Status::Fail => ("[FAIL]", "error"),
            Status::Warn => ("[WARN]", "warn"),
            Status::Skip => ("[SKIP]", "skip"),
        };
        println!("  {glyph} {:<32}", c.name);
        if !c.detail.is_empty() {
            println!("       {label}: {}", c.detail);
        }
        if c.status == Status::Fail {
            failed += 1;
        }
    }

    println!();
    if failed == 0 {
        println!("All required checks passed.");
        ExitCode::SUCCESS
    } else {
        println!("{failed} required check(s) failed.");
        ExitCode::FAILURE
    }
}

fn toolchain() -> Check {
    if run_capture(&["cargo", "--version"]).is_err() {
        return Check {
            name: "cargo present",
            status: Status::Fail,
            detail: "cargo not found on PATH. Install Rust from https://rustup.rs/".into(),
        };
    }
    let Ok(out) = Command::new("rustc").arg("--version").output() else {
        return Check {
            name: "cargo present",
            status: Status::Fail,
            detail: "rustc not found on PATH (cargo exists but rustc does not)".into(),
        };
    };
    let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parsed = parse_rustc_version(&version);
    let Some((major, minor, patch)) = parsed else {
        return Check {
            name: "Rust toolchain",
            status: Status::Warn,
            detail: format!("could not parse rustc version from: {version}"),
        };
    };
    let ok = (major, minor, patch) >= MIN_RUST;
    let required = format!("{}.{}.{}", MIN_RUST.0, MIN_RUST.1, MIN_RUST.2);
    Check {
        name: "Rust toolchain",
        status: if ok { Status::Pass } else { Status::Fail },
        detail: format!("found rustc {major}.{minor}.{patch} ({version}); {required}+ required"),
    }
}

/// `rustc --version` → `rustc 1.75.0 (82e1608df 2023-12-21)`.
fn parse_rustc_version(v: &str) -> Option<(u32, u32, u32)> {
    let field = v.split_whitespace().nth(1)?;
    let mut parts = field.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

fn run_gate(name: &'static str, args: &[&str]) -> Check {
    let sub = args.get(1).copied().unwrap_or(args[0]);
    match run_capture(args) {
        Ok((true, _, _)) => Check {
            name,
            status: Status::Pass,
            detail: String::new(),
        },
        Ok((false, stdout, stderr)) => Check {
            name,
            status: Status::Fail,
            detail: format!(
                "`cargo {sub}` failed — first lines of output:\n{}",
                summarize(&[&stdout, &stderr])
            ),
        },
        Err(e) => Check {
            name,
            status: Status::Warn,
            detail: format!("could not run `cargo {sub}`: {e}"),
        },
    }
}

/// First ~6 non-empty lines of the given outputs, indented.
fn summarize(parts: &[&str]) -> String {
    parts
        .iter()
        .flat_map(|s| s.lines())
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .take(6)
        .map(|l| format!("           {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn vendor_cuda() -> Check {
    let nvcc = Command::new("nvcc").arg("--version").output().is_ok();
    let env_hint = env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .is_some();
    let gpu = Command::new("nvidia-smi").arg("-L").output().is_ok();
    let detail = if nvcc || env_hint {
        if gpu {
            "CUDA toolkit found and NVIDIA GPU detected (nvidia-smi)".to_string()
        } else {
            "CUDA toolkit found; no NVIDIA GPU detected via nvidia-smi".to_string()
        }
    } else {
        "no nvcc / CUDA_PATH / CUDA_HOME found — CUDA backend will not activate".to_string()
    };
    Check {
        name: "CUDA SDK",
        status: if nvcc || env_hint {
            Status::Pass
        } else {
            Status::Warn
        },
        detail,
    }
}

fn vendor_rocm() -> Check {
    let rocm_dir = Path::new("/opt/rocm").exists();
    let hipcc = Command::new("hipcc").arg("--version").output().is_ok();
    let env_hint = env::var_os("ROCM_PATH").is_some();
    let detail = if rocm_dir {
        "ROCm found at /opt/rocm".to_string()
    } else if hipcc {
        "hipcc found on PATH".to_string()
    } else if env_hint {
        "ROCM_PATH is set".to_string()
    } else {
        "no /opt/rocm, hipcc, or ROCM_PATH found — ROCm backend will not activate".to_string()
    };
    Check {
        name: "ROCm SDK",
        status: if rocm_dir || hipcc || env_hint {
            Status::Pass
        } else {
            Status::Warn
        },
        detail,
    }
}

fn vendor_metal() -> Check {
    if !cfg!(target_os = "macos") {
        return Check {
            name: "Metal SDK",
            status: Status::Skip,
            detail: "Metal is macOS-only; skipped on this platform".to_string(),
        };
    }
    let ok = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
        .is_ok();
    Check {
        name: "Metal SDK",
        status: if ok { Status::Pass } else { Status::Warn },
        detail: if ok {
            "system_profiler reports display hardware (Metal expected)".to_string()
        } else {
            "could not query display hardware via system_profiler".to_string()
        },
    }
}

fn python() -> Check {
    let Some(python) = ["python3", "python"].iter().find(|p| {
        run_capture(&[p, "--version"])
            .map(|(s, _, _)| s)
            .unwrap_or(false)
    }) else {
        return Check {
            name: "Python / tptr",
            status: Status::Warn,
            detail: "no python3/python found on PATH".to_string(),
        };
    };
    let version = Command::new(python).arg("--version").output().ok();
    let version_str = version
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{python} --version produced no output"));
    let importable = Command::new(python)
        .args(["-c", "import tptr; print(tptr.__file__)"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    Check {
        name: "Python / tptr",
        status: Status::Warn,
        detail: format!(
            "{}; tptr {}importable (simulation fallback may be in use)",
            version_str,
            if importable { "" } else { "NOT " }
        ),
    }
}

/// Run a program capturing stdout/stderr. Returns `Ok((success, stdout, stderr))`
/// on a successful spawn, `Err` if the binary could not be spawned.
fn run_capture(args: &[&str]) -> std::io::Result<(bool, String, String)> {
    let out = Command::new(args[0]).args(&args[1..]).output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}
