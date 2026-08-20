use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
#[repr(C)]
pub enum TptirStatus {
    Ok = 0,
    ErrorGeneric = -1,
    ErrorParse = -2,
    ErrorNullPointer = -7,
}
#[repr(C)]
pub struct TptirString {
    pub data: *mut c_char,
    pub size: usize,
}
#[repr(C)]
pub struct TptirVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}
extern "C" {
    #[allow(dead_code)]
    fn tptir_get_version() -> TptirVersion;
    fn tptir_compile(
        source: *const c_char,
        len: usize,
        target: c_int,
        out: *mut TptirString,
        err: *mut TptirString,
    ) -> c_int;
    fn tptir_string_free(s: *mut TptirString);
}
pub fn compile_via_ffi(source: &str, target: &str) -> Result<String, String> {
    let tid = match target {
        "tptisa" => 0i32,
        "llvmir" => 1i32,
        "text" => 2i32,
        _ => return Err(format!("Unknown target: {}", target)),
    };
    let csrc = CString::new(source).map_err(|e| format!("Invalid source: {}", e))?;
    unsafe {
        let mut out = TptirString {
            data: std::ptr::null_mut(),
            size: 0,
        };
        let mut err = TptirString {
            data: std::ptr::null_mut(),
            size: 0,
        };
        let status = tptir_compile(csrc.as_ptr(), source.len(), tid, &mut out, &mut err);
        if status == 0 && !out.data.is_null() {
            let result = CStr::from_ptr(out.data).to_string_lossy().into_owned();
            tptir_string_free(&mut out);
            Ok(result)
        } else {
            let msg = if !err.data.is_null() {
                CStr::from_ptr(err.data).to_string_lossy().into_owned()
            } else {
                "FFI error".into()
            };
            tptir_string_free(&mut err);
            Err(msg)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ffi_version() {
        unsafe {
            let v = tptir_get_version();
            assert!(v.major == 0);
        }
    }
}
