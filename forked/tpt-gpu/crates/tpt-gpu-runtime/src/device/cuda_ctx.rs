//! Real CUDA driver context for `tpt-gpu-runtime`.
//!
//! When the `cuda` feature is enabled this module resolves the CUDA driver
//! (`nvcuda.dll`/`libcuda.so`) and cuBLAS (`cublas64_*.dll`/`libcublas.so`)
//! at runtime via `libloading`, initializes the driver, creates a context, and
//! exposes the small surface `Device` needs to perform real device memory
//! allocation and host↔device copies. When the feature is off (default) this
//! module compiles to an empty stub so the simulated path remains the default
//! and `cargo test` on hardware-less CI keeps working.

use crate::error::{ErrorCode, TptrError, TptrResult};

#[cfg(feature = "cuda")]
mod imp {
    use super::*;
    use std::sync::Arc;
    use tpt_gpu_primitives::sym;
    use tpt_gpu_primitives::vendor::dynlink::Library;

    type CUresult = i32;
    type CUdevice = i32;
    type CUcontext = *mut std::ffi::c_void;
    type CUdeviceptr = u64;

    const CUDA_SUCCESS: CUresult = 0;

    fn map(e: impl std::fmt::Display) -> TptrError {
        TptrError::new(ErrorCode::DeviceNotFound, e.to_string())
    }

    /// Resolved CUDA driver + cuBLAS symbols held for the lifetime of the context.
    pub struct CudaContext {
        #[allow(dead_code)]
        lib_cuda: Arc<Library>,
        #[allow(dead_code)]
        lib_cublas: Arc<Library>,
        #[allow(dead_code)]
        ctx: CUcontext,
        cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
        cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
        cu_memcpy_htod:
            unsafe extern "C" fn(CUdeviceptr, *const std::ffi::c_void, usize) -> CUresult,
        cu_memcpy_dtoh: unsafe extern "C" fn(*mut std::ffi::c_void, CUdeviceptr, usize) -> CUresult,
        cu_ctx_sync: unsafe extern "C" fn() -> CUresult,
    }

    unsafe impl Send for CudaContext {}
    unsafe impl Sync for CudaContext {}

    impl CudaContext {
        /// Initialize the CUDA driver and create a primary context on the
        /// given device ordinal.
        pub(super) fn new(ordinal: i32) -> TptrResult<Self> {
            let lib_cublas = Arc::new(
                Library::open("cublas64_12.dll")
                    .or_else(|_| Library::open("cublas64_11.dll"))
                    .or_else(|_| Library::open("libcublas.so.12"))
                    .or_else(|_| Library::open("libcublas.so.11"))
                    .or_else(|_| Library::open("libcublas.so"))
                    .map_err(map)?,
            );
            let lib_cuda = Arc::new(
                Library::open("nvcuda.dll")
                    .or_else(|_| Library::open("libcuda.so"))
                    .map_err(map)?,
            );

            let cu_init: unsafe extern "C" fn(u32) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuInit")).map_err(map)? };
            let cu_device_get: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuDeviceGet")).map_err(map)? };
            let cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuCtxCreate")).map_err(map)? };
            let cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuMemAlloc_v2")).map_err(map)? };
            let cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuMemFree_v2")).map_err(map)? };
            let cu_memcpy_htod: unsafe extern "C" fn(
                CUdeviceptr,
                *const std::ffi::c_void,
                usize,
            ) -> CUresult = unsafe { *lib_cuda.get(sym!("cuMemcpyHtoD_v2")).map_err(map)? };
            let cu_memcpy_dtoh: unsafe extern "C" fn(
                *mut std::ffi::c_void,
                CUdeviceptr,
                usize,
            ) -> CUresult = unsafe { *lib_cuda.get(sym!("cuMemcpyDtoH_v2")).map_err(map)? };
            let cu_ctx_sync: unsafe extern "C" fn() -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuCtxSynchronize")).map_err(map)? };
            let cu_ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult =
                unsafe { *lib_cuda.get(sym!("cuCtxSetCurrent")).map_err(map)? };
            let cu_device_primary_ctx_retain: unsafe extern "C" fn(
                *mut CUcontext,
                CUdevice,
            ) -> CUresult = unsafe {
                *lib_cuda
                    .get(sym!("cuDevicePrimaryCtxRetain"))
                    .map_err(map)?
            };

            if unsafe { cu_init(0) } != CUDA_SUCCESS {
                return Err(TptrError::new(
                    ErrorCode::DeviceNotFound,
                    "CUDA: cuInit failed",
                ));
            }
            let mut device: CUdevice = 0;
            if unsafe { cu_device_get(&mut device, ordinal) } != CUDA_SUCCESS {
                return Err(TptrError::new(
                    ErrorCode::DeviceNotFound,
                    "CUDA: cuDeviceGet failed",
                ));
            }
            // Use the driver-managed *primary* context instead of an explicit
            // `cuCtxCreate`. The primary context is shared with the display and
            // other libraries (cuBLAS), and is the supported way to obtain a
            // compute context on a GPU that is also driving a display (e.g. a
            // laptop RTX 3050).
            let mut ctx: CUcontext = std::ptr::null_mut();
            let retain_res = unsafe { cu_device_primary_ctx_retain(&mut ctx, device) };
            if retain_res != CUDA_SUCCESS || ctx.is_null() {
                return Err(TptrError::new(
                    ErrorCode::DeviceNotFound,
                    format!("CUDA: cuDevicePrimaryCtxRetain failed (res={})", retain_res),
                ));
            }
            if unsafe { cu_ctx_set_current(ctx) } != CUDA_SUCCESS {
                return Err(TptrError::new(
                    ErrorCode::DeviceNotFound,
                    "CUDA: cuCtxSetCurrent failed",
                ));
            }

            Ok(Self {
                lib_cuda,
                lib_cublas,
                ctx,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_ctx_sync,
            })
        }

        /// Allocate `size` bytes of device memory, returning a real device pointer.
        pub(super) fn alloc(&self, size: u64) -> TptrResult<u64> {
            let mut ptr: CUdeviceptr = 0;
            if unsafe { (self.cu_mem_alloc)(&mut ptr, size as usize) } != CUDA_SUCCESS || ptr == 0 {
                return Err(TptrError::new(
                    ErrorCode::OutOfMemory,
                    "CUDA: cuMemAlloc failed",
                ));
            }
            Ok(ptr)
        }

        pub(super) fn free(&self, ptr: u64) -> TptrResult<()> {
            if ptr == 0 {
                return Ok(());
            }
            if unsafe { (self.cu_mem_free)(ptr) } != CUDA_SUCCESS {
                return Err(TptrError::new(
                    ErrorCode::InvalidAddress,
                    "CUDA: cuMemFree failed",
                ));
            }
            Ok(())
        }

        pub(super) fn upload(&self, dst: u64, src: &[u8]) -> TptrResult<()> {
            if unsafe {
                (self.cu_memcpy_htod)(dst, src.as_ptr() as *const std::ffi::c_void, src.len())
            } != CUDA_SUCCESS
            {
                return Err(TptrError::new(
                    ErrorCode::DeviceLost,
                    "CUDA: cuMemcpyHtoD failed",
                ));
            }
            Ok(())
        }

        pub(super) fn download(&self, src: u64, dst: &mut [u8]) -> TptrResult<()> {
            if unsafe {
                (self.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut std::ffi::c_void, src, dst.len())
            } != CUDA_SUCCESS
            {
                return Err(TptrError::new(
                    ErrorCode::DeviceLost,
                    "CUDA: cuMemcpyDtoH failed",
                ));
            }
            Ok(())
        }

        pub(super) fn synchronize(&self) -> TptrResult<()> {
            if unsafe { (self.cu_ctx_sync)() } != CUDA_SUCCESS {
                return Err(TptrError::new(
                    ErrorCode::SynchronizationError,
                    "CUDA: cuCtxSynchronize failed",
                ));
            }
            Ok(())
        }
    }

    impl Drop for CudaContext {
        fn drop(&mut self) {
            // The CUDA driver releases the primary context automatically when the
            // process detaches, so we only need to ensure in-flight work completes.
            unsafe {
                (self.cu_ctx_sync)();
            }
        }
    }
}

/// A real device backend handle. When the `cuda` feature is enabled and a real
/// GPU is present this holds a live CUDA context; otherwise it is a no-op stub
/// and `Device` falls back to the simulated (host-backed) arena.
pub enum DeviceBackend {
    #[cfg(feature = "cuda")]
    Cuda(imp::CudaContext),
    #[allow(dead_code)]
    None,
}

impl DeviceBackend {
    /// Try to open a real CUDA context on the given device ordinal. Returns
    /// `None` when the `cuda` feature is disabled or no GPU/driver is
    /// available at that ordinal, allowing the caller to transparently fall
    /// back to the simulated backend.
    #[allow(dead_code)]
    pub(super) fn try_cuda(ordinal: i32) -> Option<Self> {
        #[cfg(feature = "cuda")]
        {
            imp::CudaContext::new(ordinal).ok().map(DeviceBackend::Cuda)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            None
        }
    }

    #[allow(unused_variables)]
    pub(super) fn alloc(&self, size: u64) -> TptrResult<u64> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceBackend::Cuda(ctx) => ctx.alloc(size),
            DeviceBackend::None => Err(TptrError::new(
                ErrorCode::DeviceNotFound,
                "no real device backend",
            )),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn free(&self, ptr: u64) -> TptrResult<()> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceBackend::Cuda(ctx) => ctx.free(ptr),
            DeviceBackend::None => Ok(()),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn upload(&self, dst: u64, src: &[u8]) -> TptrResult<()> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceBackend::Cuda(ctx) => ctx.upload(dst, src),
            DeviceBackend::None => Err(TptrError::new(
                ErrorCode::DeviceNotFound,
                "no real device backend",
            )),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn download(&self, src: u64, dst: &mut [u8]) -> TptrResult<()> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceBackend::Cuda(ctx) => ctx.download(src, dst),
            DeviceBackend::None => Err(TptrError::new(
                ErrorCode::DeviceNotFound,
                "no real device backend",
            )),
        }
    }

    pub(super) fn synchronize(&self) -> TptrResult<()> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceBackend::Cuda(ctx) => ctx.synchronize(),
            DeviceBackend::None => Ok(()),
        }
    }

    pub(super) fn is_real(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, DeviceBackend::Cuda(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}
