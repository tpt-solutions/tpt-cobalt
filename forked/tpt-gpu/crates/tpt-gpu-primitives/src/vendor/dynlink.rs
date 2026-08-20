//! Cross-platform dynamic library loading used by the vendor backends.
//!
//! Wraps `libloading` so vendor runtimes (CUDA, ROCm, Metal) are resolved at
//! runtime via `dlopen`/`LoadLibrary` rather than link time. When a vendor
//! library is absent the backend degrades gracefully (see `VendorBackend::detect`).
use crate::error::{TptpError, TptpResult};
use std::ffi::OsStr;
use std::os::raw::c_void;

/// A loaded shared library with symbol resolution.
pub struct Library {
    inner: libloading::Library,
}

impl Library {
    /// Load a shared library by file name (resolved against the system search
    /// path, e.g. `cublas64_12.dll` on Windows, `libcublas.so` on Linux).
    pub fn open<P: AsRef<OsStr>>(name: P) -> TptpResult<Self> {
        let name_ref = name.as_ref();
        let name_str = name_ref.to_string_lossy().into_owned();
        let inner = unsafe { libloading::Library::new(name_ref) }.map_err(|e| {
            TptpError::vendor_unavailable(format!("failed to load {}: {}", name_str, e))
        })?;
        Ok(Library { inner })
    }

    /// Resolve a symbol, returning a typed function pointer.
    ///
    /// # Safety
    /// The caller must ensure the resolved symbol has the declared signature
    /// and is called with a compatible ABI.
    pub unsafe fn get<T>(&self, symbol: &[u8]) -> TptpResult<libloading::Symbol<'_, T>> {
        self.inner.get(symbol).map_err(|e| {
            TptpError::vendor_unavailable(format!(
                "missing symbol {}: {}",
                String::from_utf8_lossy(symbol),
                e
            ))
        })
    }
}

/// A resolved C function pointer (opaque, must be cast before use).
pub type RawSymbol = *mut c_void;

/// Helper to build a symbol name as a `CStr`-compatible byte slice.
#[macro_export]
macro_rules! sym {
    ($name:expr) => {
        concat!($name, "\0").as_bytes()
    };
}
