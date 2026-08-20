//! ROCm/rocBLAS Backend
//!
//! AMD GPU support via the HIP runtime (`amdhip64.dll`/`libamdhip64.so`) and
//! rocBLAS (`rocblas.dll`/`librocblas.so`), resolved at runtime with
//! `libloading`. When ROCm is not available the backend constructor returns an
//! error so `VendorBackend::detect` falls back to the simulated path.

use super::VendorLibrary;
use crate::error::{TptpError, TptpResult};
use crate::memory::GpuBuffer;
use crate::vendor::dynlink::Library;
use std::sync::Arc;

#[allow(non_camel_case_types)]
type hipError_t = i32;
#[allow(non_camel_case_types)]
type hipDevice_t = i32;
#[allow(non_camel_case_types)]
#[allow(dead_code)]
type hipStream_t = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type hipDeviceptr_t = u64;
#[allow(non_camel_case_types)]
type rocblas_handle = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
type rocblas_status = i32;
#[allow(non_camel_case_types)]
type rocblas_operation = i32;

#[allow(dead_code)]
const HIP_SUCCESS: hipError_t = 0;
#[allow(dead_code)]
const ROCBLAS_STATUS_SUCCESS: rocblas_status = 0;
#[allow(dead_code)]
const ROCBLAS_OPERATION_NONE: rocblas_operation = 0;

#[derive(Clone)]
#[allow(dead_code)]
struct RocmSymbols {
    lib_hip: Arc<Library>,
    lib_rocblas: Arc<Library>,
    hip_init: unsafe extern "C" fn(u32) -> hipError_t,
    hip_device_get: unsafe extern "C" fn(*mut hipDevice_t, i32) -> hipError_t,
    hip_malloc: unsafe extern "C" fn(*mut hipDeviceptr_t, usize) -> hipError_t,
    hip_free: unsafe extern "C" fn(hipDeviceptr_t) -> hipError_t,
    hip_memcpy_htod:
        unsafe extern "C" fn(hipDeviceptr_t, *const std::ffi::c_void, usize) -> hipError_t,
    hip_memcpy_dtoh:
        unsafe extern "C" fn(*mut std::ffi::c_void, hipDeviceptr_t, usize) -> hipError_t,
    hip_device_synchronize: unsafe extern "C" fn() -> hipError_t,
    rocblas_create: unsafe extern "C" fn(*mut rocblas_handle) -> rocblas_status,
    rocblas_sgemm: unsafe extern "C" fn(
        rocblas_handle,
        rocblas_operation,
        rocblas_operation,
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
    ) -> rocblas_status,
    rocblas_destroy: unsafe extern "C" fn(rocblas_handle) -> rocblas_status,
}

/// ROCm backend handle.
#[derive(Clone)]
pub struct RocmBackend {
    device_id: i32,
    device_name: String,
    handle: rocblas_handle,
    syms: Box<RocmSymbols>,
}

impl std::fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend")
            .field("device_id", &self.device_id)
            .field("device_name", &self.device_name)
            .finish()
    }
}

unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Create a new ROCm backend by dynamically loading HIP and rocBLAS.
    pub fn new() -> TptpResult<Self> {
        #[cfg(feature = "rocm")]
        {
            Self::new_inner()
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(TptpError::vendor_unavailable(
                "ROCm support not compiled in (build with --features rocm)",
            ))
        }
    }

    #[cfg(feature = "rocm")]
    fn new_inner() -> TptpResult<Self> {
        let lib_rocblas =
            Arc::new(Library::open("rocblas.dll").or_else(|_| Library::open("librocblas.so"))?);
        let lib_hip =
            Arc::new(Library::open("amdhip64.dll").or_else(|_| Library::open("libamdhip64.so"))?);

        let hip_init: unsafe extern "C" fn(u32) -> hipError_t =
            unsafe { *lib_hip.get(sym!("hipInit"))? };
        let hip_device_get: unsafe extern "C" fn(*mut hipDevice_t, i32) -> hipError_t =
            unsafe { *lib_hip.get(sym!("hipDeviceGet"))? };
        let hip_malloc: unsafe extern "C" fn(*mut hipDeviceptr_t, usize) -> hipError_t =
            unsafe { *lib_hip.get(sym!("hipMalloc"))? };
        let hip_free: unsafe extern "C" fn(hipDeviceptr_t) -> hipError_t =
            unsafe { *lib_hip.get(sym!("hipFree"))? };
        let hip_memcpy_htod: unsafe extern "C" fn(
            hipDeviceptr_t,
            *const std::ffi::c_void,
            usize,
        ) -> hipError_t = unsafe { *lib_hip.get(sym!("hipMemcpyHtoD"))? };
        let hip_memcpy_dtoh: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            hipDeviceptr_t,
            usize,
        ) -> hipError_t = unsafe { *lib_hip.get(sym!("hipMemcpyDtoH"))? };
        let hip_device_synchronize: unsafe extern "C" fn() -> hipError_t =
            unsafe { *lib_hip.get(sym!("hipDeviceSynchronize"))? };

        let rocblas_create: unsafe extern "C" fn(*mut rocblas_handle) -> rocblas_status =
            unsafe { *lib_rocblas.get(sym!("rocblas_create_handle"))? };
        let rocblas_sgemm: unsafe extern "C" fn(
            rocblas_handle,
            rocblas_operation,
            rocblas_operation,
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
        ) -> rocblas_status = unsafe { *lib_rocblas.get(sym!("rocblas_sgemm"))? };
        let rocblas_destroy: unsafe extern "C" fn(rocblas_handle) -> rocblas_status =
            unsafe { *lib_rocblas.get(sym!("rocblas_destroy_handle"))? };

        if unsafe { hip_init(0) } != HIP_SUCCESS {
            return Err(TptpError::vendor_unavailable("hipInit failed"));
        }
        let mut device: hipDevice_t = 0;
        if unsafe { hip_device_get(&mut device, 0) } != HIP_SUCCESS {
            return Err(TptpError::vendor_unavailable("hipDeviceGet failed"));
        }
        let device_name = {
            let mut name_buf = [0i8; 256];
            if let Ok(get_name) = unsafe {
                lib_hip.get::<unsafe extern "C" fn(*mut i8, i32, hipDevice_t) -> hipError_t>(sym!(
                    "hipDeviceGetName"
                ))
            } {
                unsafe {
                    get_name(name_buf.as_mut_ptr(), name_buf.len() as i32, device);
                }
            }
            let end = name_buf
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(name_buf.len());
            String::from_utf8_lossy(&name_buf[..end].iter().map(|&c| c as u8).collect::<Vec<_>>())
                .into_owned()
        };

        let mut handle: rocblas_handle = std::ptr::null_mut();
        if unsafe { rocblas_create(&mut handle) } != ROCBLAS_STATUS_SUCCESS {
            return Err(TptpError::vendor_unavailable(
                "rocblas_create_handle failed",
            ));
        }

        let syms = Box::new(RocmSymbols {
            lib_hip,
            lib_rocblas,
            hip_init,
            hip_device_get,
            hip_malloc,
            hip_free,
            hip_memcpy_htod,
            hip_memcpy_dtoh,
            hip_device_synchronize,
            rocblas_create,
            rocblas_sgemm,
            rocblas_destroy,
        });
        Ok(RocmBackend {
            device_id: 0,
            device_name,
            handle,
            syms,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[allow(dead_code)]
    fn stage_in(&self, data: &[f32], ptr: &mut hipDeviceptr_t) -> TptpResult<()> {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        if unsafe { (self.syms.hip_malloc)(ptr, bytes.len()) } != HIP_SUCCESS {
            return Err(TptpError::device_error("hipMalloc failed"));
        }
        if unsafe {
            (self.syms.hip_memcpy_htod)(
                *ptr,
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len(),
            )
        } != HIP_SUCCESS
        {
            return Err(TptpError::device_error("hipMemcpyHtoD failed"));
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn stage_out(&self, ptr: hipDeviceptr_t, data: &mut [f32]) -> TptpResult<()> {
        let bytes = bytemuck::cast_slice_mut::<f32, u8>(data);
        if unsafe {
            (self.syms.hip_memcpy_dtoh)(
                bytes.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                bytes.len(),
            )
        } != HIP_SUCCESS
        {
            return Err(TptpError::device_error("hipMemcpyDtoH failed"));
        }
        Ok(())
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                (self.syms.rocblas_destroy)(self.handle);
            }
            let _ = self.syms.hip_device_synchronize;
        }
    }
}

#[allow(dead_code)]
fn read_f32(buf: &GpuBuffer<f32>) -> TptpResult<Vec<f32>> {
    let mut data = vec![0f32; buf.num_elements()];
    buf.copy_to_host(&mut data)?;
    Ok(data)
}

impl VendorLibrary for RocmBackend {
    fn name(&self) -> &str {
        "ROCm"
    }
    fn is_available(&self) -> bool {
        cfg!(feature = "rocm")
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
        #[cfg(feature = "rocm")]
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

            let ha = read_f32(a)?;
            let hb = read_f32(b)?;
            let mut hc = vec![0f32; m * n];
            let mut d_a = 0u64;
            let mut d_b = 0u64;
            let mut d_c = 0u64;
            self.stage_in(&ha, &mut d_a)?;
            self.stage_in(&hb, &mut d_b)?;
            if unsafe { (self.syms.hip_malloc)(&mut d_c, hc.len() * 4) } != HIP_SUCCESS {
                return Err(TptpError::device_error("hipMalloc(C) failed"));
            }
            let status = unsafe {
                (self.syms.rocblas_sgemm)(
                    self.handle,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    hb.as_ptr(),
                    n as i32,
                    ha.as_ptr(),
                    k as i32,
                    &beta,
                    hc.as_mut_ptr(),
                    n as i32,
                )
            };
            if status != ROCBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.hip_free)(d_a);
                    (self.syms.hip_free)(d_b);
                    (self.syms.hip_free)(d_c);
                }
                return Err(TptpError::device_error(format!(
                    "rocblas_sgemm failed (status {})",
                    status
                )));
            }
            self.stage_out(d_c, &mut hc)?;
            unsafe {
                (self.syms.hip_device_synchronize)();
                (self.syms.hip_free)(d_a);
                (self.syms.hip_free)(d_b);
                (self.syms.hip_free)(d_c);
            }
            c.copy_from_host(&hc)?;
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (a, b, c, alpha, beta);
            Err(TptpError::vendor_unavailable(
                "ROCm support not compiled in",
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
        #[cfg(feature = "rocm")]
        {
            let qd = q
                .dim(0)
                .ok_or_else(|| TptpError::shape_error("Q missing dim0"))?;
            let dk = q
                .dim(1)
                .ok_or_else(|| TptpError::shape_error("Q missing dim1"))?;
            if qd != seq_len || dk != d_k {
                return Err(TptpError::shape_error("Q shape mismatch"));
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
            let mut d_q = 0u64;
            let mut d_kptr = 0u64;
            let mut d_s = 0u64;
            self.stage_in(&hq, &mut d_q)?;
            self.stage_in(&hk, &mut d_kptr)?;
            if unsafe { (self.syms.hip_malloc)(&mut d_s, scores.len() * 4) } != HIP_SUCCESS {
                return Err(TptpError::device_error("hipMalloc(scores) failed"));
            }
            let status = unsafe {
                (self.syms.rocblas_sgemm)(
                    self.handle,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
                    seq_len as i32,
                    seq_len as i32,
                    d_k as i32,
                    &scale,
                    hk.as_ptr(),
                    d_k as i32,
                    hq.as_ptr(),
                    d_k as i32,
                    &0.0f32,
                    scores.as_mut_ptr(),
                    seq_len as i32,
                )
            };
            if status != ROCBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.hip_free)(d_q);
                    (self.syms.hip_free)(d_kptr);
                    (self.syms.hip_free)(d_s);
                }
                return Err(TptpError::device_error("attention QK^T GEMM failed"));
            }
            self.stage_out(d_s, &mut scores)?;
            unsafe {
                (self.syms.hip_free)(d_q);
                (self.syms.hip_free)(d_kptr);
                (self.syms.hip_free)(d_s);
            }
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
            let mut attn = scores;
            let mut hv2 = hv;
            let mut out = vec![0f32; seq_len * d_k];
            let mut d_a = 0u64;
            let mut d_v = 0u64;
            let mut d_o = 0u64;
            self.stage_in(&attn, &mut d_a)?;
            self.stage_in(&hv2, &mut d_v)?;
            if unsafe { (self.syms.hip_malloc)(&mut d_o, out.len() * 4) } != HIP_SUCCESS {
                return Err(TptpError::device_error("hipMalloc(out) failed"));
            }
            let status = unsafe {
                (self.syms.rocblas_sgemm)(
                    self.handle,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
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
            if status != ROCBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.hip_free)(d_a);
                    (self.syms.hip_free)(d_v);
                    (self.syms.hip_free)(d_o);
                }
                return Err(TptpError::device_error("attention AV GEMM failed"));
            }
            self.stage_out(d_o, &mut out)?;
            unsafe {
                (self.syms.hip_free)(d_a);
                (self.syms.hip_free)(d_v);
                (self.syms.hip_free)(d_o);
            }
            output.copy_from_host(&out)?;
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (q, k, v, output, scale, seq_len, d_k);
            Err(TptpError::vendor_unavailable(
                "ROCm support not compiled in",
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
        #[cfg(feature = "rocm")]
        {
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
                return Err(TptpError::shape_error("output spatial dims non-positive"));
            }
            let hin = read_f32(input)?;
            let hfil = read_f32(filter)?;
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
            if unsafe { (self.syms.hip_malloc)(&mut d_o, out_flat.len() * 4) } != HIP_SUCCESS {
                return Err(TptpError::device_error("hipMalloc(conv out) failed"));
            }
            let status = unsafe {
                (self.syms.rocblas_sgemm)(
                    self.handle,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
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
            if status != ROCBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.hip_free)(d_cols);
                    (self.syms.hip_free)(d_f);
                    (self.syms.hip_free)(d_o);
                }
                return Err(TptpError::device_error("conv2d GEMM failed"));
            }
            self.stage_out(d_o, &mut out_flat)?;
            unsafe {
                (self.syms.hip_free)(d_cols);
                (self.syms.hip_free)(d_f);
                (self.syms.hip_free)(d_o);
            }
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
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable(
                "ROCm support not compiled in",
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
        #[cfg(feature = "rocm")]
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
            if unsafe { (self.syms.hip_malloc)(&mut d_o, out_flat.len() * 4) } != HIP_SUCCESS {
                return Err(TptpError::device_error("hipMalloc(conv3d out) failed"));
            }
            let status = unsafe {
                (self.syms.rocblas_sgemm)(
                    self.handle,
                    ROCBLAS_OPERATION_NONE,
                    ROCBLAS_OPERATION_NONE,
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
            if status != ROCBLAS_STATUS_SUCCESS {
                unsafe {
                    (self.syms.hip_free)(d_cols);
                    (self.syms.hip_free)(d_f);
                    (self.syms.hip_free)(d_o);
                }
                return Err(TptpError::device_error("conv3d GEMM failed"));
            }
            self.stage_out(d_o, &mut out_flat)?;
            unsafe {
                (self.syms.hip_free)(d_cols);
                (self.syms.hip_free)(d_f);
                (self.syms.hip_free)(d_o);
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
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable(
                "ROCm support not compiled in",
            ))
        }
    }
}
