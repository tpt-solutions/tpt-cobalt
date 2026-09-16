//! `tpt` — the TPT Script command-line runner.
//!
//! ```text
//! tpt run script.tpt      execute a script file
//! tpt check script.tpt    static checks only (units + shapes + syntax)
//! tpt repl                interactive session (same as `tpt-repl`)
//! ```
//!
//! Exit codes: 0 success, 1 script/check error, 2 usage error.

use std::io::{BufRead, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") if args.len() == 2 => run_script(&args[1]),
        Some("check") if args.len() == 2 => check_script(&args[1]),
        Some("repl") => {
            repl();
            ExitCode::SUCCESS
        }
        Some(_) | None => {
            eprintln!(
                "usage:\n  tpt run <script.tpt>    execute a script file\n  \
                 tpt check <script.tpt>  static checks (units, shapes, syntax)\n  \
                 tpt repl                interactive session"
            );
            ExitCode::from(2)
        }
    }
}

type ReadRes = Result<String, ExitCode>;

fn read_file(path: &str) -> ReadRes {
    std::fs::read_to_string(path).map_err(|e| {
        eprintln!("tpt: cannot read {path}: {e}");
        ExitCode::from(2)
    })
}

fn run_script(path: &str) -> ExitCode {
    let src = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let mut interp = tpt_lang::Interpreter::new();
    match interp.run(&src) {
        Ok(Some(v)) => {
            // echo the value of the last expression, like the REPL does
            println!("= {v}");
            print_output(interp.output());
            ExitCode::SUCCESS
        }
        Ok(None) => {
            print_output(interp.output());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{}: {} [{path}]", e.kind, e.message);
            ExitCode::FAILURE
        }
    }
}

fn check_script(path: &str) -> ExitCode {
    let src = match read_file(path) {
        Ok(s) => s,
        Err(code) => return code,
    };
    match tpt_lang::check::check_program(&src) {
        Ok(()) => {
            println!("{path}: ok (units, shapes, syntax)");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{path}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn repl() {
    println!("TPT Script REPL — eager mode. Type :help for commands.");
    let mut repl = tpt_lang::Repl::new();
    let stdin = std::io::stdin();
    loop {
        print!("{}", repl.prompt());
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            break;
        }
        match repl.feed(line.trim_end()) {
            tpt_lang::ReplOutcome::NeedMore => {}
            tpt_lang::ReplOutcome::Done { output, value: _ } => {
                if !output.is_empty() {
                    print!("{output}");
                }
            }
            tpt_lang::ReplOutcome::Magic(out) => println!("{out}"),
            tpt_lang::ReplOutcome::Error(e) => eprintln!("{}", e),
            tpt_lang::ReplOutcome::Quit => {
                println!("bye");
                break;
            }
        }
    }
}

fn print_output(stdout: &str) {
    if !stdout.is_empty() {
        print!("{stdout}");
    }
}
