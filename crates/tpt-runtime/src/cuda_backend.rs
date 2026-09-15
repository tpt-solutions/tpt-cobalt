//! # CUDA backend (Phase 5) — native Driver-API kernels via `cust`
//!
//! The second non-CPU backend (the WGPU backend is the first), following the
//! same `upload → dispatch → readback` seam: [`CudaContext::try_new`]
//! returns `Ok(None)` when no device/driver is present so callers degrade
//! gracefully to the CPU path, and [`CudaContext::tape_add`] records the GPU
//! op on the `tpt-autograd` tape so gradients flow through a CUDA node back
//! to host leaves — the same cross-device story as `wgpu_backend`.
//!
//! Kernels are the embedded PTX in `kernels/cuda_kernels.ptx` (f32; a CUDA
//! Driver-JIT compiles it at context creation, so *running* this module
//! needs only the driver, not nvcc — regenerating the PTX needs the
//! toolkit, see `kernels/cuda_kernels.cu`). f32-only is documented like the
//! WGPU path: f64 has no portable CUDA-side story at this layer yet.
//!
//! Feature-gated (`cuda`) so the default workspace build stays green on
//! machines without `cuda.lib` on the link path.

use cust::context::Context;
use cust::device::Device;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use tpt_autograd::custom_vjp;
use tpt_tensor::{DType, Tensor};

const PTX: &str = include_str!("kernels/cuda_kernels.ptx");

/// Errors from the CUDA backend (driver, JIT, launch, or copy).
#[derive(Debug)]
pub enum CudaError {
    Cuda(cust::error::CudaError),
    NoDevice,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Cuda(e) => write!(f, "cuda error: {e}"),
            CudaError::NoDevice => write!(f, "no CUDA device available"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<cust::error::CudaError> for CudaError {
    fn from(e: cust::error::CudaError) -> Self {
        CudaError::Cuda(e)
    }
}

/// A live CUDA context with the tpt-runtime kernels JIT-loaded.
pub struct CudaContext {
    /// Owns the primary context (kept alive for the module/stream).
    _ctx: Context,
    module: Module,
    stream: Stream,
    device_name: String,
}

impl CudaContext {
    /// Initialize device 0 and JIT-load the kernels. Returns `Ok(None)` when
    /// no CUDA device/driver is available (mirrors `WgpuContext::try_new`).
    pub fn try_new() -> Result<Option<Self>, CudaError> {
        // cuInit is idempotent at the driver level, so a plain call is fine
        if cust::init(CudaFlags::empty()).is_err() {
            return Ok(None);
        }
        let Ok(device) = Device::get_device(0) else {
            return Ok(None);
        };
        let device_name = device.name()?;
        let ctx = Context::new(device)?;
        ctx.set_flags(ContextFlags::SCHED_AUTO)?;
        let module = Module::from_ptx(PTX, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Some(CudaContext {
            _ctx: ctx,
            module,
            stream,
            device_name,
        }))
    }

    /// GPU name (for diagnostics / reports).
    pub fn name(&self) -> &str {
        &self.device_name
    }

    /// Upload onto the backend stream (stream-ordered with the kernels,
    /// which avoids a copy/kernel race across streams).
    fn upload(&self, data: &[f32]) -> Result<DeviceBuffer<f32>, CudaError> {
        // unsafe: contained, same sanction as the shared-memory map calls —
        // the async copy is issued on the backend's own stream
        Ok(unsafe { DeviceBuffer::from_slice_async(data, &self.stream) }?)
    }

    fn f32_vec(t: &Tensor, what: &str) -> Vec<f32> {
        assert!(t.dtype() == DType::F32, "{what} must be f32 for the cuda backend");
        t.to_vec::<f32>().unwrap()
    }

    fn tensor_from(data: Vec<f32>) -> Tensor {
        Tensor::from_typed(data)
    }

    /// Element-wise add of two same-shaped F32 tensors on the GPU.
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, CudaError> {
        let av = Self::f32_vec(a, "add lhs");
        let bv = Self::f32_vec(b, "add rhs");
        assert_eq!(av.len(), bv.len(), "add shapes differ");
        let n = av.len() as i32;
        let dev_a = self.upload(&av)?;
        let dev_b = self.upload(&bv)?;
        let dev_out = unsafe { DeviceBuffer::<f32>::uninitialized(av.len())? };
        {
            let func = self.module.get_function("add_f32")?;
            let stream = &self.stream; // the launch! chevron needs a bare ident
            let blocks = (av.len() as u32).div_ceil(64);
            unsafe {
                launch!(
                    func<<<(blocks, 1, 1), (64, 1, 1), 0, stream>>>(
                        dev_a.as_device_ptr(),
                        dev_b.as_device_ptr(),
                        dev_out.as_device_ptr(),
                        n
                    )
                )?;
            }
        }
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; av.len()];
        dev_out.copy_to(&mut out)?;
        Ok(Self::tensor_from(out))
    }

    /// Naive tiled-naught matmul `C[M,N] = A[M,K]·B[K,N]` on the GPU
    /// (correctness-shaped reference kernel, 16×16 blocks).
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, CudaError> {
        assert_eq!(a.ndim(), 2, "matmul lhs must be 2-D");
        assert_eq!(b.ndim(), 2, "matmul rhs must be 2-D");
        let av = Self::f32_vec(a, "matmul lhs");
        let bv = Self::f32_vec(b, "matmul rhs");
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);
        assert_eq!(k, k2, "matmul inner dims differ");
        let dev_a = self.upload(&av)?;
        let dev_b = self.upload(&bv)?;
        let dev_out = unsafe { DeviceBuffer::<f32>::uninitialized(m * n)? };
        {
            let func = self.module.get_function("matmul_f32")?;
            let stream = &self.stream;
            let grid_x = (n as u32).div_ceil(16);
            let grid_y = (m as u32).div_ceil(16);
            unsafe {
                launch!(
                    func<<<(grid_x, grid_y, 1), (16, 16, 1), 0, stream>>>(
                        dev_a.as_device_ptr(),
                        dev_b.as_device_ptr(),
                        dev_out.as_device_ptr(),
                        m as i32,
                        n as i32,
                        k as i32
                    )
                )?;
            }
        }
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; m * n];
        dev_out.copy_to(&mut out)?;
        Ok(Tensor::from_typed(out).reshape(&[m, n]).unwrap())
    }

    /// Tape-connected GPU add: like `WgpuContext::tape_add`, the op is
    /// recorded with a `custom_vjp` so `backward` propagates gradients
    /// through the CUDA node to the host leaves unchanged (add is linear).
    pub fn tape_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, CudaError> {
        let out = self.add(a, b)?;
        let parents: Vec<std::sync::Arc<tpt_tensor::AutogradNode>> =
            a.node().into_iter().chain(b.node()).collect();
        if parents.is_empty() {
            return Ok(out);
        }
        let closure_parents = parents.clone();
        Ok(custom_vjp(out, parents, move |grad: &Tensor| {
            for p in &closure_parents {
                p.accumulate_grad(grad);
            }
        }))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use tpt_autograd::backward;

    fn ctx() -> CudaContext {
        CudaContext::try_new()
            .expect("cuda probe should not error")
            .expect("a CUDA device is required for these tests")
    }

    fn f32_tensor(data: Vec<f32>) -> Tensor {
        Tensor::from_typed(data)
    }

    #[test]
    fn add_matches_cpu() {
        let ctx = ctx();
        assert!(!ctx.name().is_empty());
        let a = f32_tensor(vec![1.0, 2.0, 3.0, -4.0]);
        let b = f32_tensor(vec![10.0, 20.0, 30.0, 40.0]);
        let out = ctx.add(&a, &b).unwrap();
        let v = out.to_vec::<f32>().unwrap();
        assert_eq!(v, vec![11.0, 22.0, 33.0, 36.0]);
        // larger than one workgroup
        let n = 10_000;
        let a = f32_tensor(vec![0.5; n]);
        let b = f32_tensor(vec![0.25; n]);
        let out = ctx.add(&a, &b).unwrap();
        assert!(out.to_vec::<f32>().unwrap().iter().all(|&x| x == 0.75));
    }

    #[test]
    fn matmul_matches_reference() {
        let ctx = ctx();
        // 2×2 exact
        let a = f32_tensor(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
        let b = f32_tensor(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]).unwrap();
        let c = ctx.matmul(&a, &b).unwrap();
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
        // 3×5 @ 5×2 vs a manual reference
        let a = f32_tensor((0..15).map(|i| i as f32).collect())
            .reshape(&[3, 5])
            .unwrap();
        let b = f32_tensor((0..10).map(|i| (i as f32) * 0.5 - 1.0).collect())
            .reshape(&[5, 2])
            .unwrap();
        let c = ctx.matmul(&a, &b).unwrap().to_vec::<f32>().unwrap();
        for r in 0..3 {
            for col in 0..2 {
                let mut acc = 0.0f32;
                for p in 0..5 {
                    acc += a.to_vec::<f32>().unwrap()[r * 5 + p]
                        * b.to_vec::<f32>().unwrap()[p * 2 + col];
                }
                assert!(
                    (c[r * 2 + col] - acc).abs() < 1e-3,
                    "c[{r}][{col}] {} vs {}",
                    c[r * 2 + col],
                    acc
                );
            }
        }
    }

    #[test]
    fn gradients_flow_through_the_cuda_node() {
        let ctx = ctx();
        let a = f32_tensor(vec![1.0, 2.0]).with_autograd();
        let b = f32_tensor(vec![10.0, 20.0]).with_autograd();
        let out = ctx.tape_add(&a, &b).unwrap();
        // seed f32 (Tensor::ones is f64 — same pattern as the WGPU test)
        let seed = f32_tensor(vec![1.0, 1.0]);
        tpt_autograd::backward_seeded(&out, &seed);
        // add is linear: both leaves receive the upstream gradient (ones)
        assert_eq!(
            a.grad().unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            b.grad().unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 1.0]
        );
    }
}

