//! CUDA/cuBLAS Backend
//!
//! NVIDIA GPU support via the CUDA driver API (`nvcuda.dll`/`libcuda.so`) and
//! cuBLAS (`cublas64_*.dll`/`libcublas.so`), resolved at runtime with
//! `libloading`. When CUDA is not available the backend constructor returns an
//! error so `VendorBackend::detect` falls back to the simulated path.

use super::VendorLibrary;
use crate::error::{TptpError, TptpResult};
use crate::memory::GpuBuffer;
use crate::vendor::dynlink::Library;
#[cfg(feature = "cuda")]
use std::sync::Arc;

// Opaque CUDA / cuBLAS handles.
type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut std::ffi::c_void;
type CUdeviceptr = u64;
#[allow(non_camel_case_types)]
type cublasHandle_t = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type cublasStatus_t = i32;
#[allow(non_camel_case_types)]
type cublasOperation_t = i32;

#[allow(dead_code)]
const CUBLAS_OP_N: cublasOperation_t = 0;
#[allow(dead_code)]
const CUBLAS_OP_T: cublasOperation_t = 1;
#[allow(dead_code)]
const CUDA_SUCCESS: CUresult = 0;
#[allow(dead_code)]
const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

/// Resolved CUDA + cuBLAS entry points held for the lifetime of the backend.
#[derive(Clone)]
#[allow(dead_code)]
struct CudaSymbols {
    lib_cuda: std::sync::Arc<Library>,
    lib_cublas: std::sync::Arc<Library>,
    cu_init: unsafe extern "C" fn(u32) -> CUresult,
    cu_device_get: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult,
    cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult,
    cu_ctx_sync: unsafe extern "C" fn() -> CUresult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const std::ffi::c_void, usize) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut std::ffi::c_void, CUdeviceptr, usize) -> CUresult,
    cublas_create: unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t,
    cublas_sgemm: unsafe extern "C" fn(
        cublasHandle_t,
        cublasOperation_t,
        cublasOperation_t,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
    ) -> cublasStatus_t,
    cublas_destroy: unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t,
}

/// CUDA backend handle.
#[derive(Clone)]
pub struct CudaBackend {
    device_id: i32,
    device_name: String,
    ctx: CUcontext,
    handle: cublasHandle_t,
    syms: Box<CudaSymbols>,
}

unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .field("device_name", &self.device_name)
            .finish()
    }
}

impl CudaBackend {
    /// Create a new CUDA backend on device ordinal 0 by dynamically loading
    /// the CUDA driver and cuBLAS, initializing the runtime, and creating a
    /// cuBLAS handle.
    pub fn new() -> TptpResult<Self> {
        Self::new_on(0)
    }

    /// Create a new CUDA backend bound to a specific device ordinal (as
    /// returned by `cuDeviceGet`). Used for multi-GPU data-parallel dispatch,
    /// where each replica binds to a different physical GPU.
    pub fn new_on(ordinal: i32) -> TptpResult<Self> {
        #[cfg(feature = "cuda")]
        {
            Self::new_inner(ordinal)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(TptpError::vendor_unavailable(
                "CUDA support not compiled in (build with --features cuda)",
            ))
        }
    }

    /// Number of CUDA-visible devices (0 if CUDA is unavailable/not compiled in).
    pub fn device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            Self::device_count_inner().unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    #[cfg(feature = "cuda")]
    fn device_count_inner() -> TptpResult<usize> {
        let lib_cuda = Library::open("nvcuda.dll").or_else(|_| Library::open("libcuda.so"))?;
        let cu_init: unsafe extern "C" fn(u32) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuInit"))? };
        let cu_device_get_count: unsafe extern "C" fn(*mut i32) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuDeviceGetCount"))? };
        if unsafe { cu_init(0) } != CUDA_SUCCESS {
            return Err(TptpError::vendor_unavailable("cuInit failed"));
        }
        let mut count: i32 = 0;
        if unsafe { cu_device_get_count(&mut count) } != CUDA_SUCCESS {
            return Err(TptpError::vendor_unavailable("cuDeviceGetCount failed"));
        }
        Ok(count.max(0) as usize)
    }

    #[cfg(feature = "cuda")]
    fn new_inner(ordinal: i32) -> TptpResult<Self> {
        // Prefer the versioned cuBLAS DLL name; fall back to unversioned.
        let lib_cublas = Arc::new(
            Library::open("cublas64_12.dll")
                .or_else(|_| Library::open("cublas64_11.dll"))
                .or_else(|_| Library::open("libcublas.so.12"))
                .or_else(|_| Library::open("libcublas.so.11"))
                .or_else(|_| Library::open("libcublas.so"))?,
        );

        let lib_cuda =
            Arc::new(Library::open("nvcuda.dll").or_else(|_| Library::open("libcuda.so"))?);

        let cu_init: unsafe extern "C" fn(u32) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuInit"))? };
        let cu_device_get: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuDeviceGet"))? };
        let cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuCtxCreate"))? };
        let cu_ctx_sync: unsafe extern "C" fn() -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuCtxSynchronize"))? };
        let cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuMemAlloc_v2"))? };
        let cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult =
            unsafe { *lib_cuda.get(sym!("cuMemFree_v2"))? };
        let cu_memcpy_htod: unsafe extern "C" fn(
            CUdeviceptr,
            *const std::ffi::c_void,
            usize,
        ) -> CUresult = unsafe { *lib_cuda.get(sym!("cuMemcpyHtoD_v2"))? };
        let cu_memcpy_dtoh: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            CUdeviceptr,
            usize,
        ) -> CUresult = unsafe { *lib_cuda.get(sym!("cuMemcpyDtoH_v2"))? };

        let cublas_create: unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t =
            unsafe { *lib_cublas.get(sym!("cublasCreate_v2"))? };
        let cublas_sgemm: unsafe extern "C" fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            i32,
            i32,
            i32,
            *const f32,
            *const f32,
            i32,
            *const f32,
            i32,
            *const f32,
            *mut f32,
            i32,
        ) -> cublasStatus_t = unsafe { *lib_cublas.get(sym!("cublasSgemm_v2"))? };
        let cublas_destroy: unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t =
            unsafe { *lib_cublas.get(sym!("cublasDestroy_v2"))? };

        // Initialize the CUDA driver.
        if unsafe { cu_init(0) } != CUDA_SUCCESS {
            return Err(TptpError::vendor_unavailable("cuInit failed"));
        }
        let mut device: CUdevice = 0;
        if unsafe { cu_device_get(&mut device, ordinal) } != CUDA_SUCCESS {
            return Err(TptpError::vendor_unavailable("cuDeviceGet failed"));
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        if unsafe { cu_ctx_create(&mut ctx, 0, device) } != CUDA_SUCCESS {
            return Err(TptpError::vendor_unavailable("cuCtxCreate failed"));
        }

        // Query device name.
        let mut name_buf = [0i8; 256];
        if let Ok(get_name) = unsafe {
            lib_cuda.get::<unsafe extern "C" fn(*mut i8, i32, CUdevice) -> CUresult>(sym!(
                "cuDeviceGetName"
            ))
        } {
            unsafe {
                get_name(name_buf.as_mut_ptr(), name_buf.len() as i32, device);
            }
        }
        let device_name = {
            let end = name_buf
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(name_buf.len());
            String::from_utf8_lossy(&name_buf[..end].iter().map(|&c| c as u8).collect::<Vec<_>>())
                .into_owned()
        };

        // Create a cuBLAS handle.
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        if unsafe { cublas_create(&mut handle) } != CUBLAS_STATUS_SUCCESS {
            return Err(TptpError::vendor_unavailable("cublasCreate failed"));
        }

        let syms = Box::new(CudaSymbols {
            lib_cuda,
            lib_cublas,
            cu_init,
            cu_device_get,
            cu_ctx_create,
            cu_ctx_sync,
            cu_mem_alloc,
            cu_mem_free,
            cu_memcpy_htod,
            cu_memcpy_dtoh,
            cublas_create,
            cublas_sgemm,
            cublas_destroy,
        });

        Ok(CudaBackend {
            device_id: device,
            device_name,
            ctx,
            handle,
            syms,
        })
    }

    /// Get the CUDA device ID.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get the CUDA device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[allow(dead_code)]
    fn stage_in(&self, data: &[f32], ptr: &mut CUdeviceptr) -> TptpResult<()> {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        if unsafe { (self.syms.cu_mem_alloc)(ptr, bytes.len()) } != CUDA_SUCCESS {
            return Err(TptpError::device_error("cuMemAlloc failed"));
        }
        if unsafe {
            (self.syms.cu_memcpy_htod)(*ptr, bytes.as_ptr() as *const std::ffi::c_void, bytes.len())
        } != CUDA_SUCCESS
        {
            return Err(TptpError::device_error("cuMemcpyHtoD failed"));
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn stage_out(&self, ptr: CUdeviceptr, data: &mut [f32]) -> TptpResult<()> {
        let bytes = bytemuck::cast_slice_mut::<f32, u8>(data);
        if unsafe {
            (self.syms.cu_memcpy_dtoh)(
                bytes.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                bytes.len(),
            )
        } != CUDA_SUCCESS
        {
            return Err(TptpError::device_error("cuMemcpyDtoH failed"));
        }
        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                (self.syms.cublas_destroy)(self.handle);
            }
            if !self.ctx.is_null() {
                // cuCtxDestroy not resolved; CUDA driver releases the primary
                // context automatically when the process detaches.
                let _ = self.syms.cu_ctx_sync;
            }
        }
    }
}

/// Host-side staging helper for `GpuBuffer<f32>`.
#[allow(dead_code)]
fn read_f32(buf: &GpuBuffer<f32>) -> TptpResult<Vec<f32>> {
    let mut data = vec![0f32; buf.num_elements()];
    buf.copy_to_host(&mut data)?;
    Ok(data)
}

impl VendorLibrary for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        cfg!(feature = "cuda")
    }

    fn gemm(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        alpha: f32,
        beta: f32,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> TptpResult<()> {
        #[cfg(feature = "cuda")]
        {
            let m = a
                .dim(0)
                .ok_or_else(|| TptpError::shape_error("A missing row dim"))?;
            let k = a
                .dim(1)
                .ok_or_else(|| TptpError::shape_error("A missing col dim"))?;
            let n = b
                .dim(1)
                .ok_or_else(|| TptpError::shape_error("B missing col dim"))?;
            if b.dim(0) != Some(k) {
                return Err(TptpError::shape_error("B row dim != K"));
            }
            if c.dim(0) != Some(m) || c.dim(1) != Some(n) {
                return Err(TptpError::shape_error("C dims != (M,N)"));
            }

            let host_a = read_f32(a)?;
            let host_b = read_f32(b)?;
            let mut host_c = vec![0f32; m * n];

            let mut d_a = 0u64;
            let mut d_b = 0u64;
            let mut d_c = 0u64;
            self.stage_in(&host_a, &mut d_a)?;
            self.stage_in(&host_b, &mut d_b)?;
            if unsafe { (self.syms.cu_mem_alloc)(&mut d_c, host_c.len() * 4) } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuMemAlloc(C) failed"));
            }

            let lda = k as i32;
            let ldb = n as i32;
            let ldc = n as i32;
            let status = unsafe {
                (self.syms.cublas_sgemm)(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    host_b.as_ptr(),
                    ldb,
                    host_a.as_ptr(),
                    lda,
                    &beta,
                    host_c.as_mut_ptr(),
                    ldc,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.cu_mem_free)(d_a);
                    (self.syms.cu_mem_free)(d_b);
                    (self.syms.cu_mem_free)(d_c);
                }
                return Err(TptpError::device_error(format!(
                    "cublasSgemm failed (status {})",
                    status
                )));
            }
            self.stage_out(d_c, &mut host_c)?;
            if unsafe { (self.syms.cu_ctx_sync)() } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuCtxSynchronize failed"));
            }
            unsafe {
                (self.syms.cu_mem_free)(d_a);
                (self.syms.cu_mem_free)(d_b);
                (self.syms.cu_mem_free)(d_c);
            }

            c.copy_from_host(&host_c)?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (a, b, c, alpha, beta);
            Err(TptpError::vendor_unavailable(
                "CUDA support not compiled in",
            ))
        }
    }

    fn attention(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scale: f32,
        seq_len: usize,
        d_k: usize,
    ) -> TptpResult<()> {
        #[cfg(feature = "cuda")]
        {
            // Standard scaled-dot-product attention:
            //   scores = scale * (Q @ K^T)         [seq_len x seq_len]  (cuBLAS GEMM)
            //   attn   = softmax(scores, axis=-1)  (host reduction)
            //   out    = attn @ V                  [seq_len x d_k]       (cuBLAS GEMM)
            // The dominant matmul work runs on the GPU; softmax is applied on
            // the host staging buffer (a cuDNN/custom-kernel path is a future
            // optimization and not required for correctness).
            let qd = q
                .dim(0)
                .ok_or_else(|| TptpError::shape_error("Q missing dim0"))?;
            let dk = q
                .dim(1)
                .ok_or_else(|| TptpError::shape_error("Q missing dim1"))?;
            if qd != seq_len || dk != d_k {
                return Err(TptpError::shape_error(
                    "Q shape mismatch (expected [seq_len, d_k])",
                ));
            }
            if k.dim(0) != Some(seq_len) || k.dim(1) != Some(d_k) {
                return Err(TptpError::shape_error("K shape mismatch"));
            }
            if v.dim(0) != Some(seq_len) || v.dim(1) != Some(d_k) {
                return Err(TptpError::shape_error("V shape mismatch"));
            }
            if output.dim(0) != Some(seq_len) || output.dim(1) != Some(d_k) {
                return Err(TptpError::shape_error("output shape mismatch"));
            }

            let hq = read_f32(q)?;
            let hk = read_f32(k)?;
            let hv = read_f32(v)?;
            let mut scores = vec![0f32; seq_len * seq_len];

            // scores = scale * Q @ K^T  (GEMM on GPU).
            let mut d_q = 0u64;
            let mut d_kptr = 0u64;
            let mut d_s = 0u64;
            self.stage_in(&hq, &mut d_q)?;
            self.stage_in(&hk, &mut d_kptr)?;
            if unsafe { (self.syms.cu_mem_alloc)(&mut d_s, scores.len() * 4) } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuMemAlloc(scores) failed"));
            }
            let status = unsafe {
                (self.syms.cublas_sgemm)(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    seq_len as i32,
                    seq_len as i32,
                    d_k as i32,
                    &scale,
                    hq.as_ptr(),
                    d_k as i32,
                    hk.as_ptr(),
                    d_k as i32,
                    &0.0f32,
                    scores.as_mut_ptr(),
                    seq_len as i32,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.cu_mem_free)(d_q);
                    (self.syms.cu_mem_free)(d_kptr);
                    (self.syms.cu_mem_free)(d_s);
                }
                return Err(TptpError::device_error("attention QK^T GEMM failed"));
            }
            self.stage_out(d_s, &mut scores)?;
            unsafe {
                (self.syms.cu_mem_free)(d_q);
                (self.syms.cu_mem_free)(d_kptr);
                (self.syms.cu_mem_free)(d_s);
            }

            // softmax over the last axis (host).
            for row in 0..seq_len {
                let start = row * seq_len;
                let slice = &mut scores[start..start + seq_len];
                let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in slice.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in slice.iter_mut() {
                    *s /= sum;
                }
            }

            // out = attn @ V  (GEMM on GPU).
            let mut attn = scores;
            let mut hv2 = hv;
            let mut out = vec![0f32; seq_len * d_k];
            let mut d_a = 0u64;
            let mut d_v = 0u64;
            let mut d_o = 0u64;
            self.stage_in(&attn, &mut d_a)?;
            self.stage_in(&hv2, &mut d_v)?;
            if unsafe { (self.syms.cu_mem_alloc)(&mut d_o, out.len() * 4) } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuMemAlloc(out) failed"));
            }
            let status = unsafe {
                (self.syms.cublas_sgemm)(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    d_k as i32,
                    seq_len as i32,
                    seq_len as i32,
                    &1.0f32,
                    hv2.as_ptr(),
                    d_k as i32,
                    attn.as_ptr(),
                    seq_len as i32,
                    &0.0f32,
                    out.as_mut_ptr(),
                    d_k as i32,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.cu_mem_free)(d_a);
                    (self.syms.cu_mem_free)(d_v);
                    (self.syms.cu_mem_free)(d_o);
                }
                return Err(TptpError::device_error("attention AV GEMM failed"));
            }
            self.stage_out(d_o, &mut out)?;
            unsafe {
                (self.syms.cu_mem_free)(d_a);
                (self.syms.cu_mem_free)(d_v);
                (self.syms.cu_mem_free)(d_o);
            }

            output.copy_from_host(&out)?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (q, k, v, output, scale, seq_len, d_k);
            Err(TptpError::vendor_unavailable(
                "CUDA support not compiled in",
            ))
        }
    }

    fn conv2d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
    ) -> TptpResult<()> {
        #[cfg(feature = "cuda")]
        {
            // im2col + GEMM via cuBLAS. Shapes are [N, C, H, W] / [K, C, R, S].
            let in_shape = input.shape().clone();
            let f_shape = filter.shape().clone();
            if in_shape.ndim() != 4 || f_shape.ndim() != 4 {
                return Err(TptpError::shape_error("conv2d expects 4D NCHW tensors"));
            }
            let (n, c, h, w) = (
                in_shape.dim(0).unwrap(),
                in_shape.dim(1).unwrap(),
                in_shape.dim(2).unwrap(),
                in_shape.dim(3).unwrap(),
            );
            let (k, _fc, r, s) = (
                f_shape.dim(0).unwrap(),
                f_shape.dim(1).unwrap(),
                f_shape.dim(2).unwrap(),
                f_shape.dim(3).unwrap(),
            );
            if _fc != c {
                return Err(TptpError::shape_error(
                    "filter in-channels != input channels",
                ));
            }
            let sh = strides[0] as usize;
            let sw = strides[1] as usize;
            let ph = padding[0] as usize;
            let pw = padding[1] as usize;
            let oh = (h + 2 * ph - r) / sh + 1;
            let ow = (w + 2 * pw - s) / sw + 1;
            if oh == 0 || ow == 0 {
                return Err(TptpError::shape_error(
                    "output spatial dims are non-positive",
                ));
            }

            let hin = read_f32(input)?;
            let hfil = read_f32(filter)?;

            // Build im2col columns: [N * oh * ow, c * r * s].
            let col_cols = c * r * s;
            let out_pixels = n * oh * ow;
            let mut cols = vec![0f32; out_pixels * col_cols];
            for nn in 0..n {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let in_y = oy as isize * sh as isize - ph as isize;
                        let in_x = ox as isize * sw as isize - pw as isize;
                        let mut ci = 0usize;
                        for cc in 0..c {
                            for ky in 0..r {
                                for kx in 0..s {
                                    let yy = in_y + ky as isize;
                                    let xx = in_x + kx as isize;
                                    let val =
                                        if yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize
                                        {
                                            hin[((nn * c + cc) * h + yy as usize) * w + xx as usize]
                                        } else {
                                            0.0
                                        };
                                    cols[((nn * oh + oy) * ow + ox) * col_cols + ci] = val;
                                    ci += 1;
                                }
                            }
                        }
                    }
                }
            }

            // Flatten filter to [K, c*r*s].
            let mut fmat = vec![0f32; k * col_cols];
            for kk in 0..k {
                for cc in 0..c {
                    for ky in 0..r {
                        for kx in 0..s {
                            fmat[(kk * c + cc) * r * s + (ky * s + kx)] =
                                hfil[((kk * c + cc) * r + ky) * s + kx];
                        }
                    }
                }
            }

            let mut out_flat = vec![0f32; k * out_pixels];
            let mut d_cols = 0u64;
            let mut d_f = 0u64;
            let mut d_o = 0u64;
            self.stage_in(&cols, &mut d_cols)?;
            self.stage_in(&fmat, &mut d_f)?;
            if unsafe { (self.syms.cu_mem_alloc)(&mut d_o, out_flat.len() * 4) } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuMemAlloc(conv out) failed"));
            }
            let status = unsafe {
                (self.syms.cublas_sgemm)(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    out_pixels as i32,
                    k as i32,
                    col_cols as i32,
                    &1.0f32,
                    cols.as_ptr(),
                    out_pixels as i32,
                    fmat.as_ptr(),
                    col_cols as i32,
                    &0.0f32,
                    out_flat.as_mut_ptr(),
                    out_pixels as i32,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.cu_mem_free)(d_cols);
                    (self.syms.cu_mem_free)(d_f);
                    (self.syms.cu_mem_free)(d_o);
                }
                return Err(TptpError::device_error("conv2d GEMM failed"));
            }
            self.stage_out(d_o, &mut out_flat)?;
            unsafe {
                (self.syms.cu_mem_free)(d_cols);
                (self.syms.cu_mem_free)(d_f);
                (self.syms.cu_mem_free)(d_o);
            }

            // Reshape [K, N*oh*ow] -> [N, K, oh, ow].
            let mut out_nchw = vec![0f32; n * k * oh * ow];
            for kk in 0..k {
                for p in 0..out_pixels {
                    let nn = p / (oh * ow);
                    let rem = p % (oh * ow);
                    let oy = rem / ow;
                    let ox = rem % ow;
                    out_nchw[(((nn * k + kk) * oh) + oy) * ow + ox] = out_flat[kk * out_pixels + p];
                }
            }
            output.copy_from_host(&out_nchw)?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable(
                "CUDA support not compiled in",
            ))
        }
    }

    fn conv3d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
    ) -> TptpResult<()> {
        #[cfg(feature = "cuda")]
        {
            let in_shape = input.shape().clone();
            let f_shape = filter.shape().clone();
            if in_shape.ndim() != 5 || f_shape.ndim() != 5 {
                return Err(TptpError::shape_error("conv3d expects 5D NCDHW tensors"));
            }
            let (n, c, d, h, w) = (
                in_shape.dim(0).unwrap(),
                in_shape.dim(1).unwrap(),
                in_shape.dim(2).unwrap(),
                in_shape.dim(3).unwrap(),
                in_shape.dim(4).unwrap(),
            );
            let (k, _fc, kt, r, s) = (
                f_shape.dim(0).unwrap(),
                f_shape.dim(1).unwrap(),
                f_shape.dim(2).unwrap(),
                f_shape.dim(3).unwrap(),
                f_shape.dim(4).unwrap(),
            );
            if _fc != c {
                return Err(TptpError::shape_error(
                    "filter in-channels != input channels",
                ));
            }
            let sd = strides[0] as usize;
            let sh = strides[1] as usize;
            let sw = strides[2] as usize;
            let pd = padding[0] as usize;
            let ph = padding[1] as usize;
            let pw = padding[2] as usize;
            let od = (d + 2 * pd - kt) / sd + 1;
            let oh = (h + 2 * ph - r) / sh + 1;
            let ow = (w + 2 * pw - s) / sw + 1;
            if od == 0 || oh == 0 || ow == 0 {
                return Err(TptpError::shape_error("output spatial dims non-positive"));
            }

            let hin = read_f32(input)?;
            let hfil = read_f32(filter)?;
            let col_cols = c * kt * r * s;
            let out_voxels = n * od * oh * ow;
            let mut cols = vec![0f32; out_voxels * col_cols];
            for nn in 0..n {
                for oz in 0..od {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let zz = oz as isize * sd as isize - pd as isize;
                            let yy = oy as isize * sh as isize - ph as isize;
                            let xx = ox as isize * sw as isize - pw as isize;
                            let mut ci = 0usize;
                            for cc in 0..c {
                                for kz in 0..kt {
                                    for ky in 0..r {
                                        for kx in 0..s {
                                            let vz = zz + kz as isize;
                                            let vy = yy + ky as isize;
                                            let vx = xx + kx as isize;
                                            let val = if vz >= 0
                                                && vz < d as isize
                                                && vy >= 0
                                                && vy < h as isize
                                                && vx >= 0
                                                && vx < w as isize
                                            {
                                                hin[(((nn * c + cc) * d + vz as usize) * h
                                                    + vy as usize)
                                                    * w
                                                    + vx as usize]
                                            } else {
                                                0.0
                                            };
                                            cols[(((nn * od + oz) * oh + oy) * ow + ox)
                                                * col_cols
                                                + ci] = val;
                                            ci += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut fmat = vec![0f32; k * col_cols];
            for kk in 0..k {
                for cc in 0..c {
                    for kz in 0..kt {
                        for ky in 0..r {
                            for kx in 0..s {
                                fmat[(kk * c + cc) * kt * r * s + (kz * r + ky) * s + kx] =
                                    hfil[(((kk * c + cc) * kt + kz) * r + ky) * s + kx];
                            }
                        }
                    }
                }
            }
            let mut out_flat = vec![0f32; k * out_voxels];
            let mut d_cols = 0u64;
            let mut d_f = 0u64;
            let mut d_o = 0u64;
            self.stage_in(&cols, &mut d_cols)?;
            self.stage_in(&fmat, &mut d_f)?;
            if unsafe { (self.syms.cu_mem_alloc)(&mut d_o, out_flat.len() * 4) } != CUDA_SUCCESS {
                return Err(TptpError::device_error("cuMemAlloc(conv3d out) failed"));
            }
            let status = unsafe {
                (self.syms.cublas_sgemm)(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    out_voxels as i32,
                    k as i32,
                    col_cols as i32,
                    &1.0f32,
                    cols.as_ptr(),
                    out_voxels as i32,
                    fmat.as_ptr(),
                    col_cols as i32,
                    &0.0f32,
                    out_flat.as_mut_ptr(),
                    out_voxels as i32,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.cu_mem_free)(d_cols);
                    (self.syms.cu_mem_free)(d_f);
                    (self.syms.cu_mem_free)(d_o);
                }
                return Err(TptpError::device_error("conv3d GEMM failed"));
            }
            self.stage_out(d_o, &mut out_flat)?;
            unsafe {
                (self.syms.cu_mem_free)(d_cols);
                (self.syms.cu_mem_free)(d_f);
                (self.syms.cu_mem_free)(d_o);
            }

            let mut out_ncdhw = vec![0f32; n * k * od * oh * ow];
            for kk in 0..k {
                for p in 0..out_voxels {
                    let nn = p / (od * oh * ow);
                    let rem = p % (od * oh * ow);
                    let oz = rem / (oh * ow);
                    let rem2 = rem % (oh * ow);
                    let oy = rem2 / ow;
                    let ox = rem2 % ow;
                    out_ncdhw[(((nn * k + kk) * od + oz) * oh + oy) * ow + ox] =
                        out_flat[kk * out_voxels + p];
                }
            }
            output.copy_from_host(&out_ncdhw)?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable(
                "CUDA support not compiled in",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};

    #[cfg(feature = "cuda")]
    fn make(shape: &[usize]) -> GpuBuffer<f32> {
        GpuBuffer::new(Shape::new(shape), DType::F32, BufferFlags::empty()).unwrap()
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_gemm_roundtrip() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("CUDA unavailable, skipping");
                return;
            }
        };
        let m = 64;
        let k = 48;
        let n = 32;
        let mut a = make(&[m, k]);
        let mut b = make(&[k, n]);
        let mut c = make(&[m, n]);
        let ha: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let hb: Vec<f32> = (0..k * n).map(|i| (i as f32).cos()).collect();
        a.copy_from_host(&ha).unwrap();
        b.copy_from_host(&hb).unwrap();

        backend
            .gemm(&a, &b, &mut c, 1.0, 0.0, m, n, k)
            .expect("gemm");

        let mut hc = vec![0f32; m * n];
        c.copy_to_host(&mut hc).unwrap();

        // CPU reference.
        let mut refc = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for p in 0..k {
                    s += ha[i * k + p] * hb[p * n + j];
                }
                refc[i * n + j] = s;
            }
        }
        let mut max_err = 0f32;
        for i in 0..m * n {
            max_err = max_err.max((hc[i] - refc[i]).abs());
        }
        assert!(max_err < 1e-2, "max GEMM error {} too large", max_err);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_attention_roundtrip() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("CUDA unavailable, skipping");
                return;
            }
        };
        let seq = 16;
        let d = 8;
        let mut q = make(&[seq, d]);
        let mut k = make(&[seq, d]);
        let mut v = make(&[seq, d]);
        let mut out = make(&[seq, d]);
        let h: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.1).sin()).collect();
        q.copy_from_host(&h).unwrap();
        k.copy_from_host(&h).unwrap();
        v.copy_from_host(&h).unwrap();

        backend
            .attention(&q, &k, &v, &mut out, (d as f32).sqrt().recip(), seq, d)
            .expect("attention");

        let mut ho = vec![0f32; seq * d];
        out.copy_to_host(&mut ho).unwrap();
        // Sanity: outputs are finite and non-NaN.
        assert!(
            ho.iter().all(|x| x.is_finite()),
            "attention produced non-finite output"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_conv2d_roundtrip() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("CUDA unavailable, skipping");
                return;
            }
        };
        let n = 1;
        let c = 1;
        let hh = 5;
        let w = 5;
        let kf = 1;
        let r = 3;
        let s = 3;
        let mut input = make(&[n, c, hh, w]);
        let mut filt = make(&[kf, c, r, s]);
        let mut out = make(&[n, kf, hh, w]);
        let hi: Vec<f32> = (0..n * c * hh * w).map(|i| (i % 5) as f32).collect();
        let hf: Vec<f32> = (0..kf * c * r * s).map(|i| 1.0).collect();
        input.copy_from_host(&hi).unwrap();
        filt.copy_from_host(&hf).unwrap();
        backend
            .conv2d(&input, &filt, &mut out, [1, 1], [1, 1])
            .expect("conv2d");
        let mut ho = vec![0f32; n * kf * hh * w];
        out.copy_to_host(&mut ho).unwrap();
        assert!(ho.iter().all(|x| x.is_finite()));
    }
}
