//! The TPT Script REPL (`tpt-repl` binary).
/*!
Eager, stateful, tensor-aware: expressions echo their value, `print` output
is captured per entry, multi-line entries accumulate until brackets balance,
and magic commands start with a colon. Stdin line-editing is plain buffered
reading (no external line-editor dependency).
*/

use std::io::{BufRead, Write};

use tpt_lang::{Repl, ReplOutcome};

fn main() {
    println!("TPT Script REPL — eager mode. Type :help for commands.");
    let mut repl = Repl::new();
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
            ReplOutcome::NeedMore => {}
            ReplOutcome::Done { output, value: _ } => {
                if !output.is_empty() {
                    print!("{output}");
                }
            }
            ReplOutcome::Magic(out) => println!("{out}"),
            ReplOutcome::Error(e) => eprintln!("{}", e),
            ReplOutcome::Quit => {
                println!("bye");
                break;
            }
        }
    }
}
