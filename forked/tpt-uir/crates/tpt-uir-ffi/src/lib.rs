#![doc = include_str!("../README.md")]

//! C ABI for read-only inspection of TPT-UIR regions.
//!
//! The API is opaque-handle based: call [`tpt_uir_load`] to deserialize a
//! postcard-encoded region into a heap-allocated handle, then query it with the
//! accessor functions. Call [`tpt_uir_free`] when done. All functions return
//! `0` on success and a non-zero code on error.

use libc::{c_char, c_int, c_void, size_t};
use std::ffi::CStr;

use tpt_uir_core::ir::Region;
use tpt_uir_core::validate_region;

/// Opaque handle owning a deserialized [`Region`].
///
/// `strings` caches NUL-terminated `CString`s handed out by the accessors so the
/// returned pointers stay valid until [`tpt_uir_free`] (the documented lifetime
/// contract), without requiring the underlying `String`s to be NUL-terminated.
struct Handle {
    region: Region,
    strings: std::cell::RefCell<Vec<std::ffi::CString>>,
}

/// Deserialize a postcard-encoded region from `data[0..len]` into a new handle.
///
/// On success, `*out_handle` receives the opaque pointer (free with
/// [`tpt_uir_free`]). Returns `0` on success, `1` on a bad pointer, or `2` on a
/// decode error.
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_load(
    data: *const u8,
    len: size_t,
    out_handle: *mut *mut c_void,
) -> c_int {
    if data.is_null() || out_handle.is_null() {
        return 1;
    }
    let slice = std::slice::from_raw_parts(data, len);
    match tpt_uir_serde::deserialize_region(slice) {
        Ok(region) => {
            let handle = Box::new(Handle {
                region,
                strings: std::cell::RefCell::new(Vec::new()),
            });
            *out_handle = Box::into_raw(handle) as *mut c_void;
            0
        }
        Err(_) => 2,
    }
}

/// Free a handle produced by [`tpt_uir_load`].
///
/// # Safety
/// `handle` must be a pointer returned by [`tpt_uir_load`] and not previously
/// freed. Passing a null pointer is a no-op.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut Handle));
    }
}

/// Write the number of blocks in the region to `*out`.
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_block_count(handle: *mut c_void, out: *mut size_t) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    *out = h.region.blocks.len();
    0
}

/// Write the number of operations in block `block_index` to `*out`.
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_op_count(
    handle: *mut c_void,
    block_index: size_t,
    out: *mut size_t,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match h.region.blocks.get(block_index) {
        Some(b) => {
            *out = b.operations.len();
            0
        }
        None => 1,
    }
}

/// Write the dialect portion of operation `(block_index, op_index)` to `*out`.
///
/// The returned pointer references storage owned by the handle and is valid
/// until [`tpt_uir_free`].
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_op_dialect(
    handle: *mut c_void,
    block_index: size_t,
    op_index: size_t,
    out: *mut *const c_char,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match h
        .region
        .blocks
        .get(block_index)
        .and_then(|b| b.operations.get(op_index))
    {
        Some(op) => {
            let cs = std::ffi::CString::new(op.op_name.dialect.as_bytes())
                .expect("op dialect must not contain NUL");
            let ptr = cs.as_ptr();
            h.strings.borrow_mut().push(cs);
            *out = ptr;
            0
        }
        None => 1,
    }
}

/// Write the op portion of operation `(block_index, op_index)` to `*out`.
///
/// See [`tpt_uir_op_dialect`] for lifetime semantics.
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_op_op(
    handle: *mut c_void,
    block_index: size_t,
    op_index: size_t,
    out: *mut *const c_char,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match h
        .region
        .blocks
        .get(block_index)
        .and_then(|b| b.operations.get(op_index))
    {
        Some(op) => {
            let cs =
                std::ffi::CString::new(op.op_name.op.as_bytes()).expect("op must not contain NUL");
            let ptr = cs.as_ptr();
            h.strings.borrow_mut().push(cs);
            *out = ptr;
            0
        }
        None => 1,
    }
}

/// Write the number of operands of operation `(block_index, op_index)` to `*out`.
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_op_operand_count(
    handle: *mut c_void,
    block_index: size_t,
    op_index: size_t,
    out: *mut size_t,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match h
        .region
        .blocks
        .get(block_index)
        .and_then(|b| b.operations.get(op_index))
    {
        Some(op) => {
            *out = op.operands.len();
            0
        }
        None => 1,
    }
}

/// Write operand `operand_index` of operation `(block_index, op_index)` to `*out`.
///
/// # Safety
/// `handle` must be a valid handle; `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_op_operand(
    handle: *mut c_void,
    block_index: size_t,
    op_index: size_t,
    operand_index: size_t,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match h
        .region
        .blocks
        .get(block_index)
        .and_then(|b| b.operations.get(op_index))
    {
        Some(op) => match op.operands.get(operand_index) {
            Some(v) => {
                *out = *v;
                0
            }
            None => 1,
        },
        None => 1,
    }
}

/// Validate SSA well-formedness. Returns `0` if valid, `1` if invalid.
///
/// # Safety
/// `handle` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_validate_ssa(handle: *mut c_void) -> c_int {
    if handle.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    match validate_region(&h.region) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Validate that every op's dialect starts with `prefix`. Returns `0` if all
/// match, `1` otherwise (or on a bad `prefix` pointer).
///
/// # Safety
/// `handle` must be a valid handle; `prefix` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tpt_uir_validate_dialect(
    handle: *mut c_void,
    prefix: *const c_char,
) -> c_int {
    if handle.is_null() || prefix.is_null() {
        return 1;
    }
    let h = &*(handle as *const Handle);
    let prefix = match CStr::from_ptr(prefix).to_str() {
        Ok(p) => p,
        Err(_) => return 1,
    };
    for block in &h.region.blocks {
        for op in &block.operations {
            if !op.op_name.dialect.starts_with(prefix) {
                return 1;
            }
        }
    }
    0
}
