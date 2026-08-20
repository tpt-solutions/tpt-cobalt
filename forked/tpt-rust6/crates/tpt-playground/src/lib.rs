//! Wasm runtime for the browser-based **Try TPT** playground.
//!
//! `tpt_script::run_script` is compiled to `wasm32-unknown-unknown` and exposed
//! through a tiny C-ABI surface so a hand-written JS host can run `.tpt` scripts
//! pasted into the editor without any backend:
//!
//! * [`tpt_alloc`] — reserve `len` bytes of linear memory (a bump allocation).
//! * [`tpt_run`] — run the script whose bytes live at `(ptr, len)`, capturing
//!   stdout and any rendered error into an internal buffer; returns its length.
//! * [`tpt_output_ptr`] / [`tpt_output_len`] — expose that buffer's address and
//!   length so JS can copy the result back out.
//!
//! The runtime is dependency-free at the ABI level (no `wasm-bindgen` glue):
//! the host only needs the three exports above plus the exported `memory`.

use std::alloc::{alloc, Layout};
use std::cell::RefCell;

thread_local! {
    /// The most recent run's captured output (stdout + error traceback).
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Reserve `len` bytes of linear memory and return the pointer.
///
/// Backed by the Rust global allocator; the host writes the script bytes here
/// before calling [`tpt_run`].
#[no_mangle]
pub extern "C" fn tpt_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let layout = Layout::from_size_align(len, 1).expect("valid layout");
        alloc(layout)
    }
}

/// Run `tpt_script::run_script` over the `(code_ptr, code_len)` bytes.
///
/// `print` statements and any error traceback are captured into an internal
/// buffer; the byte length of that buffer is returned. Read it back with
/// [`tpt_output_ptr`] / [`tpt_output_len`].
///
/// # Safety
///
/// `code_ptr` must point to `code_len` valid, initialized bytes that remain
/// valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn tpt_run(code_ptr: *const u8, code_len: usize) -> usize {
    let captured = match unsafe { std::str::from_utf8(std::slice::from_raw_parts(code_ptr, code_len)) }
    {
        Ok(code) => {
            // `tpt_script::capture` routes `print` output into a string that we
            // can return across the wasm ABI (std::io::set_print is unavailable
            // on wasm32-unknown-unknown).
            let (result, printed) = tpt_script::capture(|| tpt_script::run_script(code));
            let mut out = printed;
            if let Err(traceback) = result {
                out.push('\n');
                out.push_str(&traceback);
            }
            out
        }
        Err(_) => "error: input is not valid UTF-8\n".to_string(),
    };

    OUTPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        *buf = captured.into_bytes();
        buf.len()
    })
}

/// Pointer to the captured output buffer (valid until the next [`tpt_run`]).
#[no_mangle]
pub extern "C" fn tpt_output_ptr() -> *const u8 {
    OUTPUT.with(|cell| cell.borrow().as_ptr())
}

/// Length of the captured output buffer.
#[no_mangle]
pub extern "C" fn tpt_output_len() -> usize {
    OUTPUT.with(|cell| cell.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_simple_script_and_captures_print() {
        let src = "let x = 2 + 3\nprint x\n";
        let len = unsafe { tpt_run(src.as_ptr(), src.len()) };
        let out = OUTPUT.with(|cell| cell.borrow().clone());
        assert_eq!(len, out.len());
        let text = String::from_utf8(out).expect("utf-8 output");
        assert!(text.contains('5'), "expected printed value, got {text:?}");
    }

    #[test]
    fn captures_error_traceback() {
        let src = "assert 1 == 2\n";
        let _ = unsafe { tpt_run(src.as_ptr(), src.len()) };
        let out = OUTPUT.with(|cell| cell.borrow().clone());
        let text = String::from_utf8(out).expect("utf-8 output");
        assert!(!text.is_empty(), "expected an error traceback, got empty");
        assert!(
            text.to_lowercase().contains("error") || text.to_lowercase().contains("assert"),
            "got {text:?}"
        );
    }
}
