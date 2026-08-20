//! TPTIR C API FFI Bindings — Rust interface to the TPTIR compiler.
//!
//! When compiled with the `tptir-backend` feature the real C API is used.
//! Without it (default / `sim` mode) a stub implementation is used so the
//! crate links and tests run without a built tptc C library.

// ---------------------------------------------------------------------------
// Real FFI — only compiled when the tptc C library is available
// ---------------------------------------------------------------------------

#[cfg(feature = "tptir-backend")]
mod real {
    use crate::error::{TptpError, TptpResult};
    use libc::{c_char, c_int, c_void, size_t};
    use std::ffi::CStr;
    use std::ptr;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TptirStatus {
        Ok = 0,
        ErrorGeneric = -1,
        ErrorParse = -2,
        ErrorType = -3,
        ErrorCodegen = -5,
        ErrorNullPointer = -7,
    }

    impl TptirStatus {
        pub fn is_ok(self) -> bool {
            self == TptirStatus::Ok
        }
        pub fn to_result(self, msg: &str) -> TptpResult<()> {
            if self.is_ok() {
                Ok(())
            } else {
                Err(TptpError::FfiError {
                    message: msg.to_string(),
                    status_code: self as i32,
                })
            }
        }
    }

    #[repr(C)]
    pub struct TptirContext(c_void);
    #[repr(C)]
    pub struct TptirModule(c_void);
    #[repr(C)]
    pub struct TptirString {
        pub data: *mut c_char,
        pub size: size_t,
    }
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct TptirVersion {
        pub major: u32,
        pub minor: u32,
        pub patch: u32,
    }

    extern "C" {
        pub fn tptir_init(context: *mut *mut TptirContext) -> TptirStatus;
        pub fn tptir_shutdown(context: *mut TptirContext) -> TptirStatus;
        pub fn tptir_get_version() -> TptirVersion;
        pub fn tptir_status_string(status: TptirStatus) -> *const c_char;
        pub fn tptir_module_create(
            context: *mut TptirContext,
            module: *mut *mut TptirModule,
        ) -> TptirStatus;
        pub fn tptir_module_destroy(module: *mut TptirModule) -> TptirStatus;
        pub fn tptir_module_parse(
            module: *mut TptirModule,
            source: *const c_char,
            source_size: size_t,
            error_msg: *mut TptirString,
        ) -> TptirStatus;
        pub fn tptir_compile(
            source: *const c_char,
            source_size: size_t,
            target: i32,
            output: *mut TptirString,
            error_msg: *mut TptirString,
        ) -> TptirStatus;
        pub fn tptir_string_free(string: *mut TptirString);
    }

    pub struct TptirCompilerHandle {
        context: *mut TptirContext,
    }
    unsafe impl Send for TptirCompilerHandle {}
    unsafe impl Sync for TptirCompilerHandle {}

    impl TptirCompilerHandle {
        pub fn new() -> TptpResult<Self> {
            let mut context = ptr::null_mut();
            let status = unsafe { tptir_init(&mut context) };
            status.to_result("failed to initialize TPTIR context")?;
            if context.is_null() {
                return Err(TptpError::internal("tptir_init returned null context"));
            }
            Ok(TptirCompilerHandle { context })
        }
        pub fn version() -> (u32, u32, u32) {
            let v = unsafe { tptir_get_version() };
            (v.major, v.minor, v.patch)
        }
        pub fn status_string(status: TptirStatus) -> &'static str {
            let ptr = unsafe { tptir_status_string(status) };
            if ptr.is_null() {
                "unknown error"
            } else {
                unsafe { CStr::from_ptr(ptr) }
                    .to_str()
                    .unwrap_or("invalid UTF-8")
            }
        }
        pub fn context_ptr(&self) -> *mut TptirContext {
            self.context
        }
    }

    impl Drop for TptirCompilerHandle {
        fn drop(&mut self) {
            if !self.context.is_null() {
                unsafe { tptir_shutdown(self.context) };
            }
        }
    }

    pub struct TptirModuleWrapper {
        module: *mut TptirModule,
    }

    impl TptirModuleWrapper {
        pub fn new(context: &TptirCompilerHandle) -> TptpResult<Self> {
            let mut module = ptr::null_mut();
            let status = unsafe { tptir_module_create(context.context_ptr(), &mut module) };
            status.to_result("failed to create TPTIR module")?;
            if module.is_null() {
                return Err(TptpError::internal("tptir_module_create returned null"));
            }
            Ok(TptirModuleWrapper { module })
        }
        pub fn parse(&mut self, source: &str) -> TptpResult<()> {
            let mut error = TptirString {
                data: ptr::null_mut(),
                size: 0,
            };
            let status = unsafe {
                tptir_module_parse(
                    self.module,
                    source.as_ptr() as *const c_char,
                    source.len(),
                    &mut error,
                )
            };
            let result = status.to_result("failed to parse TPTIR source");
            if !error.data.is_null() {
                unsafe { tptir_string_free(&mut error) };
            }
            result
        }
        pub fn module_ptr(&self) -> *mut TptirModule {
            self.module
        }
    }

    impl Drop for TptirModuleWrapper {
        fn drop(&mut self) {
            if !self.module.is_null() {
                unsafe { tptir_module_destroy(self.module) };
            }
        }
    }

    pub fn compile_kernel(source: &str, target: i32) -> TptpResult<String> {
        let mut output = TptirString {
            data: ptr::null_mut(),
            size: 0,
        };
        let mut error = TptirString {
            data: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe {
            tptir_compile(
                source.as_ptr() as *const c_char,
                source.len(),
                target,
                &mut output,
                &mut error,
            )
        };
        if !error.data.is_null() {
            unsafe { tptir_string_free(&mut error) };
        }
        if !status.is_ok() {
            return Err(TptpError::compilation(format!(
                "compilation failed: {}",
                TptirCompilerHandle::status_string(status)
            )));
        }
        if output.data.is_null() {
            return Err(TptpError::internal("compilation returned null output"));
        }
        let result = unsafe {
            let slice = std::slice::from_raw_parts(output.data as *const u8, output.size);
            let string = String::from_utf8_lossy(slice).to_string();
            tptir_string_free(&mut output);
            string
        };
        Ok(result)
    }

    pub fn get_tptir_version() -> String {
        let (major, minor, patch) = TptirCompilerHandle::version();
        format!("{}.{}.{}", major, minor, patch)
    }
}

// ---------------------------------------------------------------------------
// Stub implementation — used when tptir-backend is not enabled (default / sim)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "tptir-backend"))]
mod stub {
    use crate::error::TptpResult;

    pub struct TptirCompilerHandle;

    impl TptirCompilerHandle {
        pub fn new() -> TptpResult<Self> {
            Ok(TptirCompilerHandle)
        }
        pub fn version() -> (u32, u32, u32) {
            (0, 1, 0)
        }
        pub fn status_string(_status: i32) -> &'static str {
            "OK"
        }
    }

    pub struct TptirModuleWrapper;

    impl TptirModuleWrapper {
        pub fn new(_context: &TptirCompilerHandle) -> TptpResult<Self> {
            Ok(TptirModuleWrapper)
        }
        pub fn parse(&mut self, _source: &str) -> TptpResult<()> {
            Ok(())
        }
    }

    /// In stub mode, compilation is a no-op: the source is returned as-is.
    /// This allows the parameterized substitution logic to be exercised in
    /// tests without a real compiler.
    pub fn compile_kernel(source: &str, _target: i32) -> TptpResult<String> {
        Ok(source.to_owned())
    }

    pub fn get_tptir_version() -> String {
        "0.1.0-stub".to_string()
    }
}

// ---------------------------------------------------------------------------
// Re-export the active implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "tptir-backend")]
pub use real::*;

#[cfg(not(feature = "tptir-backend"))]
pub use stub::*;

// ---------------------------------------------------------------------------
// Tests (run against whichever implementation is active)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_tptir_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_handle_creation() {
        let handle = TptirCompilerHandle::new();
        assert!(handle.is_ok());
    }

    #[test]
    fn test_compile_stub_roundtrip() {
        let source = "module { }";
        let result = compile_kernel(source, 0);
        assert!(result.is_ok());
    }
}
