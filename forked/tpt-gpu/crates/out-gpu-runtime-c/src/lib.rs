//! TPT Runtime - C ABI
//!
//! Stable C interface over `tpt-gpu-runtime`. Generated header:
//! `include/tptr/tptr_capi.h` (see `cbindgen.toml`).
#![allow(non_camel_case_types)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
use std::sync::Mutex;
use tpt_gpu_runtime::device::DeviceProperties;
use tpt_gpu_runtime::{
    Device, ErrorCode, Kernel, KernelConfig, KernelHandle, MemAccess, MemType, MemoryAllocation,
    MemoryRegion, TptrError,
};

/// Opaque device handle (owns a `Mutex<Device>`).
pub struct tptr_device_t {
    inner: Mutex<Device>,
}

/// Opaque memory allocation handle.
pub struct tptr_memory_t {
    inner: MemoryAllocation,
}

/// Opaque kernel handle.
pub struct tptr_kernel_t {
    inner: Kernel,
}

/// Opaque kernel-launch configuration handle.
pub struct tptr_kernel_config_t {
    inner: KernelConfig,
}

/// Opaque kernel-launch result handle.
pub struct tptr_kernel_handle_t {
    inner: KernelHandle,
}

/// Status codes returned by the C API.
#[repr(i32)]
pub enum tptr_status_t {
    TptrOk = 0,
    ErrorGeneric = -1,
    ErrorOutOfMemory = -2,
    ErrorInvalidAddress = -3,
    ErrorArgumentMismatch = -4,
    ErrorConfiguration = -5,
    ErrorInvalidKernel = -6,
    ErrorNullPointer = -7,
}

fn map_err(err: &TptrError) -> tptr_status_t {
    match err.code {
        ErrorCode::OutOfMemory => tptr_status_t::ErrorOutOfMemory,
        ErrorCode::InvalidAddress => tptr_status_t::ErrorInvalidAddress,
        ErrorCode::ArgumentMismatch => tptr_status_t::ErrorArgumentMismatch,
        ErrorCode::ConfigurationError => tptr_status_t::ErrorConfiguration,
        ErrorCode::InvalidKernel => tptr_status_t::ErrorInvalidKernel,
        _ => tptr_status_t::ErrorGeneric,
    }
}

thread_local! {
    /// Last error message storage (thread-local for safety).
    static LAST_ERROR: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
}

fn store_err(err: &TptrError) -> tptr_status_t {
    LAST_ERROR.with(|e| e.borrow_mut().clone_from(&err.to_string()));
    map_err(err)
}

#[no_mangle]
pub extern "C" fn tptr_device_create(
    index: u32,
    total_memory_bytes: u64,
    out: *mut *mut tptr_device_t,
) -> tptr_status_t {
    if out.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let props = DeviceProperties::simulated(&format!("TPT Device {}", index), total_memory_bytes);
    let dev = Device::new_simulated(index as u64, props);
    unsafe {
        *out = Box::into_raw(Box::new(tptr_device_t {
            inner: Mutex::new(dev),
        }))
    };
    tptr_status_t::TptrOk
}

/// Open the best available device backend. When the `cuda` feature is enabled
/// and an NVIDIA GPU is present this returns a real CUDA-backed device;
/// otherwise it falls back to the simulated device. Mirrors the Rust
/// `Device::open()` entry point used by external integrations.
#[no_mangle]
pub extern "C" fn tptr_device_open(_index: u32, out: *mut *mut tptr_device_t) -> tptr_status_t {
    if out.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    match Device::open() {
        Ok(dev) => {
            unsafe {
                *out = Box::into_raw(Box::new(tptr_device_t {
                    inner: Mutex::new(dev),
                }))
            };
            tptr_status_t::TptrOk
        }
        Err(e) => store_err(&e),
    }
}

/// Returns non-zero if the device is backed by real hardware (e.g. CUDA).
#[no_mangle]
pub extern "C" fn tptr_device_is_real(dev: *mut tptr_device_t) -> u32 {
    if dev.is_null() {
        return 0;
    }
    unsafe { (*dev).inner.lock().unwrap().is_real() as u32 }
}

#[no_mangle]
pub extern "C" fn tptr_device_destroy(dev: *mut tptr_device_t) {
    if !dev.is_null() {
        unsafe { drop(Box::from_raw(dev)) };
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_allocate(
    dev: *mut tptr_device_t,
    size: u64,
    mem_type: u32,
    access: u32,
    out: *mut *mut tptr_memory_t,
) -> tptr_status_t {
    if dev.is_null() || out.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let mt = match mem_type {
        1 => MemType::HostPinned,
        2 => MemType::Managed,
        _ => MemType::Device,
    };
    let acc = match access {
        1 => MemAccess::ReadOnly,
        2 => MemAccess::WriteOnly,
        _ => MemAccess::ReadWrite,
    };
    let result = unsafe {
        (*dev)
            .inner
            .lock()
            .unwrap()
            .allocate(size, MemoryRegion::Global, mt, acc)
    };
    match result {
        Ok(alloc) => {
            unsafe { *out = Box::into_raw(Box::new(tptr_memory_t { inner: alloc })) };
            tptr_status_t::TptrOk
        }
        Err(e) => store_err(&e),
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_free(
    dev: *mut tptr_device_t,
    mem: *mut tptr_memory_t,
) -> tptr_status_t {
    if dev.is_null() || mem.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let alloc = unsafe { (*mem).inner.clone() };
    let result = unsafe { (*dev).inner.lock().unwrap().free(&alloc) };
    unsafe { drop(Box::from_raw(mem)) };
    match result {
        Ok(()) => tptr_status_t::TptrOk,
        Err(e) => store_err(&e),
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_memcpy_htod(
    dev: *mut tptr_device_t,
    dst: *mut tptr_memory_t,
    src: *const u8,
    size: u64,
    dst_offset: u64,
) -> tptr_status_t {
    if dev.is_null() || dst.is_null() || src.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let slice = unsafe { std::slice::from_raw_parts(src, size as usize) };
    let alloc = unsafe { (*dst).inner.clone() };
    let result = unsafe {
        (*dev)
            .inner
            .lock()
            .unwrap()
            .memcpy_htod(&alloc, slice, size, dst_offset)
    };
    match result {
        Ok(()) => tptr_status_t::TptrOk,
        Err(e) => store_err(&e),
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_memcpy_dtoh(
    dev: *mut tptr_device_t,
    src: *mut tptr_memory_t,
    dst: *mut u8,
    size: u64,
    src_offset: u64,
) -> tptr_status_t {
    if dev.is_null() || src.is_null() || dst.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(dst, size as usize) };
    let alloc = unsafe { (*src).inner.clone() };
    let result = unsafe {
        (*dev)
            .inner
            .lock()
            .unwrap()
            .memcpy_dtoh(slice, &alloc, size, src_offset)
    };
    match result {
        Ok(()) => tptr_status_t::TptrOk,
        Err(e) => store_err(&e),
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_create_kernel(
    dev: *mut tptr_device_t,
    name: *const std::os::raw::c_char,
    out: *mut *mut tptr_kernel_t,
) -> tptr_status_t {
    if dev.is_null() || out.is_null() || name.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let kernel = unsafe { (*dev).inner.lock().unwrap().create_kernel(&name) };
    unsafe { *out = Box::into_raw(Box::new(tptr_kernel_t { inner: kernel })) };
    tptr_status_t::TptrOk
}

#[no_mangle]
pub extern "C" fn tptr_device_load_module(
    dev: *mut tptr_device_t,
    tptir_text: *const std::os::raw::c_char,
    out: *mut *mut tptr_kernel_t,
) -> tptr_status_t {
    if dev.is_null() || out.is_null() || tptir_text.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(tptir_text) }
        .to_string_lossy()
        .into_owned();
    match unsafe { (*dev).inner.lock().unwrap().load_module(&text) } {
        Ok(kernel) => {
            unsafe { *out = Box::into_raw(Box::new(tptr_kernel_t { inner: kernel })) };
            tptr_status_t::TptrOk
        }
        Err(e) => store_err(&e),
    }
}

#[no_mangle]
pub extern "C" fn tptr_kernel_destroy(kernel: *mut tptr_kernel_t) {
    if !kernel.is_null() {
        unsafe { drop(Box::from_raw(kernel)) };
    }
}

#[no_mangle]
pub extern "C" fn tptr_kernel_config_create(
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    out: *mut *mut tptr_kernel_config_t,
) -> tptr_status_t {
    if out.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let cfg = KernelConfig::new((grid_x, grid_y, grid_z), (block_x, block_y, block_z));
    unsafe { *out = Box::into_raw(Box::new(tptr_kernel_config_t { inner: cfg })) };
    tptr_status_t::TptrOk
}

#[no_mangle]
pub extern "C" fn tptr_kernel_config_destroy(cfg: *mut tptr_kernel_config_t) {
    if !cfg.is_null() {
        unsafe { drop(Box::from_raw(cfg)) };
    }
}

#[no_mangle]
pub extern "C" fn tptr_device_launch_kernel(
    dev: *mut tptr_device_t,
    kernel: *mut tptr_kernel_t,
    cfg: *mut tptr_kernel_config_t,
    out: *mut *mut tptr_kernel_handle_t,
) -> tptr_status_t {
    if dev.is_null() || kernel.is_null() || cfg.is_null() || out.is_null() {
        return tptr_status_t::ErrorNullPointer;
    }
    let k = unsafe { (*kernel).inner.clone() };
    let c = unsafe { (*cfg).inner.clone() };
    let handle = unsafe { (*dev).inner.lock().unwrap().launch_kernel(&k, &c, &[]) };
    unsafe { *out = Box::into_raw(Box::new(tptr_kernel_handle_t { inner: handle })) };
    tptr_status_t::TptrOk
}

#[no_mangle]
pub extern "C" fn tptr_kernel_handle_destroy(handle: *mut tptr_kernel_handle_t) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[no_mangle]
pub extern "C" fn tptr_kernel_handle_is_complete(handle: *mut tptr_kernel_handle_t) -> bool {
    if handle.is_null() {
        return false;
    }
    unsafe { (*handle).inner.is_complete() }
}

#[no_mangle]
pub extern "C" fn tptr_device_synchronize(dev: *mut tptr_device_t) {
    if !dev.is_null() {
        unsafe { (*dev).inner.lock().unwrap().synchronize() };
    }
}

/// Returns the last error message for the calling thread, or NULL if none.
#[no_mangle]
pub extern "C" fn tptr_last_error() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|e| {
        let s = e.borrow();
        if s.is_empty() {
            std::ptr::null()
        } else {
            // Leak the CString for the caller to read; it is tiny and bounded.
            let c = std::ffi::CString::new(s.clone()).unwrap_or_default();
            c.into_raw()
        }
    })
}

/// Frees a string previously returned by `tptr_last_error`.
#[no_mangle]
pub extern "C" fn tptr_string_free(s: *mut std::os::raw::c_char) {
    if !s.is_null() {
        unsafe { drop(std::ffi::CString::from_raw(s)) };
    }
}

/// Returns the major version of the runtime.
#[no_mangle]
pub extern "C" fn tptr_version_major() -> u32 {
    env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0)
}

/// Returns the minor version of the runtime.
#[no_mangle]
pub extern "C" fn tptr_version_minor() -> u32 {
    env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0)
}

/// Returns the patch version of the runtime.
#[no_mangle]
pub extern "C" fn tptr_version_patch() -> u32 {
    env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0)
}
