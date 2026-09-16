//! WGPU compute backend (Phase 4: cross-device execution on a non-CPU device).
//!
//! A minimal but *real* GPU path end to end: adapter/device/queue init, WGSL
//! compute kernels for element-wise add and naive matmul, buffer upload,
//! dispatch, and staging-buffer readback into [`tpt_tensor::Tensor`]s.
//!
//! Design notes:
//! - Kernels operate on **f32** (WGSL has no portable f64); inputs must be
//!   `DType::F32` and results come back as F32 tensors.
//! - `WgpuContext::try_new()` returns `Ok(None)` when no adapter is available
//!   (headless machines), so callers degrade gracefully to the CPU path.
//! - This is the seam the CUDA/ROCm/Metal backends (Phase 5) will hang off;
//!   the API shape (`upload → dispatch → readback`) is what they will reuse.

use tpt_tensor::{DType, Tensor};

const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) {
        out[i] = a[i] + b[i];
    }
}
"#;

const MATMUL_WGSL: &str = r#"
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row < dims.m && col < dims.n) {
        var acc = 0.0;
        for (var i = 0u; i < dims.k; i = i + 1u) {
            acc = acc + a[row * dims.k + i] * b[i * dims.n + col];
        }
        out[row * dims.n + col] = acc;
    }
}
"#;

/// Errors raised by the WGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    /// Logical device creation failed.
    #[error("device request failed")]
    Device(#[from] wgpu::RequestDeviceError),
    /// Buffer mapping failed.
    #[error("buffer map failed")]
    Map(#[from] wgpu::BufferAsyncError),
    /// The map-result channel died (should never happen).
    #[error("map result channel closed")]
    MapChannel,
}

/// An initialized wgpu device + queue pair with prebuilt pipelines.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    add_pipeline: wgpu::ComputePipeline,
    matmul_pipeline: wgpu::ComputePipeline,
}

impl WgpuContext {
    /// Request the first available adapter and create the context.
    ///
    /// Returns `Ok(None)` when no adapter is present (headless CI, no
    /// drivers), letting callers fall back to the CPU path.
    pub fn try_new() -> Result<Option<Self>, WgpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        else {
            return Ok(None);
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tpt-runtime-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;
        let add_pipeline = Self::pipeline(&device, ADD_WGSL);
        let matmul_pipeline = Self::pipeline(&device, MATMUL_WGSL);
        Ok(Some(WgpuContext {
            device,
            queue,
            add_pipeline,
            matmul_pipeline,
        }))
    }

    fn pipeline(device: &wgpu::Device, src: &str) -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        })
    }

    fn upload(&self, bytes: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn staging(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Read `len` bytes back from a GPU buffer into host memory (blocking).
    fn readback(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        len: usize,
    ) -> Result<Vec<u8>, WgpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, len as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = dst.slice(0..len as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| WgpuError::MapChannel)?
            .map_err(WgpuError::Map)?;
        let data = slice.get_mapped_range().to_vec();
        dst.unmap();
        Ok(data)
    }
}

impl WgpuContext {
    fn f32_checked(t: &Tensor, what: &str) {
        assert!(
            t.dtype() == DType::F32,
            "{what} must be f32 for the wgpu backend"
        );
    }

    /// Tape-connected GPU add: runs the kernel on the device and records the
    /// op on the `tpt-autograd` tape (add is linear, so both parents receive
    /// the upstream gradient unchanged). This is the cross-device deliverable:
    /// `backward` flows gradients *through* a GPU node back to host leaves.
    pub fn tape_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, WgpuError> {
        let out = self.add(a, b)?;
        let parents: Vec<std::sync::Arc<tpt_tensor::AutogradNode>> =
            a.node().into_iter().chain(b.node()).collect();
        if parents.is_empty() {
            return Ok(out);
        }
        let closure_parents = parents.clone();
        Ok(tpt_autograd::custom_vjp(
            out,
            parents,
            move |grad: &Tensor| {
                for p in &closure_parents {
                    p.accumulate_grad(grad);
                }
            },
        ))
    }

    /// Element-wise add of two same-shaped F32 tensors on the GPU.
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, WgpuError> {
        Self::f32_checked(a, "add lhs");
        Self::f32_checked(b, "add rhs");
        assert_eq!(a.shape(), b.shape(), "add shapes differ");
        let bytes = a.as_bytes();
        assert_eq!(bytes.len(), b.as_bytes().len());
        let buf_a = self.upload(bytes);
        let buf_b = self.upload(b.as_bytes());
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.staging(bytes.len());
        let bind = {
            let layout = self.add_pipeline.get_bind_group_layout(0);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out.as_entire_binding(),
                    },
                ],
            })
        };
        let n = (bytes.len() / 4) as u32;
        let wg = n.div_ceil(64);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let data = self.readback(&out, &stage, bytes.len())?;
        Ok(Tensor::from_le_bytes(data, DType::F32, a.shape()))
    }

    /// Naive workgroup-tiled matmul of two F32 tensors ([m,k] @ [k,n]).
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, WgpuError> {
        Self::f32_checked(a, "matmul lhs");
        Self::f32_checked(b, "matmul rhs");
        assert!(a.ndim() == 2 && b.ndim() == 2, "matmul requires 2-D");
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let (k2, n) = (b.shape()[0] as u32, b.shape()[1] as u32);
        assert_eq!(k, k2, "matmul inner dims differ");
        let mut dims_bytes = [0u8; 16];
        dims_bytes[0..4].copy_from_slice(&m.to_le_bytes());
        dims_bytes[4..8].copy_from_slice(&n.to_le_bytes());
        dims_bytes[8..12].copy_from_slice(&k.to_le_bytes());
        use wgpu::util::DeviceExt;
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &dims_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let buf_a = self.upload(a.as_bytes());
        let buf_b = self.upload(b.as_bytes());
        let out_len = (m * n) as usize * 4;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = self.staging(out_len);
        let bind = {
            let layout = self.matmul_pipeline.get_bind_group_layout(0);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                ],
            })
        };
        let wg_x = n.div_ceil(8);
        let wg_y = m.div_ceil(8);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let data = self.readback(&out, &stage, out_len)?;
        Ok(Tensor::from_le_bytes(
            data,
            DType::F32,
            &[m as usize, n as usize],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(vals: &[f32]) -> Tensor {
        Tensor::from_typed(vals.to_vec())
    }

    /// Skip gracefully when the machine has no usable adapter.
    fn context() -> Option<WgpuContext> {
        WgpuContext::try_new().ok().flatten()
    }

    #[test]
    fn wgpu_add_matches_cpu() {
        let Some(ctx) = context() else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        let a = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32_tensor(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let c = ctx.add(&a, &b).expect("gpu add");
        assert_eq!(c.shape(), a.shape());
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![9.0; 8]);
    }

    #[test]
    fn wgpu_tape_add_backprop_crosses_device() {
        let Some(ctx) = context() else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        // leaves on the host tape -> GPU add node -> backward with a scaled
        // seed. Gradients must flow through the GPU node back to both host
        // leaves (tape ops are f64-only, so the scale lives in the seed).
        let a = f32_tensor(&[1.0, 2.0, 3.0, 4.0]).with_autograd();
        let b = f32_tensor(&[10.0, 20.0, 30.0, 40.0]).with_autograd();
        let s = ctx.tape_add(&a, &b).expect("tape gpu add");
        assert!(s.requires_grad(), "gpu result must carry a tape node");
        let seed = Tensor::from_typed(vec![2.0_f32; 4])
            .reshape(s.shape())
            .unwrap();
        tpt_autograd::backward_seeded(&s, &seed);
        // d(2*(a+b))/da = 2 ; d(...)/db = 2
        assert_eq!(a.grad().unwrap().to_vec::<f32>().unwrap(), vec![2.0; 4]);
        assert_eq!(b.grad().unwrap().to_vec::<f32>().unwrap(), vec![2.0; 4]);
    }

    #[test]
    fn wgpu_matmul_matches_cpu() {
        let Some(ctx) = context() else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = f32_tensor(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
        let b = f32_tensor(&[5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]).unwrap();
        let c = ctx.matmul(&a, &b).expect("gpu matmul");
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
        // non-square, non-multiple-of-workgroup sizes
        let a = f32_tensor(&(0..15).map(|i| i as f32).collect::<Vec<_>>())
            .reshape(&[3, 5])
            .unwrap();
        let b = f32_tensor(&(0..10).map(|i| (i % 3) as f32).collect::<Vec<_>>())
            .reshape(&[5, 2])
            .unwrap();
        let c = ctx.matmul(&a, &b).expect("gpu matmul 3x5@5x2");
        // manual f32 reference (raw Tensor::matmul is f64-only)
        let av = a.to_vec::<f32>().unwrap();
        let bv = b.to_vec::<f32>().unwrap();
        let mut want = vec![0.0_f32; 6];
        for r in 0..3 {
            for c in 0..2 {
                let mut s = 0.0;
                for k in 0..5 {
                    s += av[r * 5 + k] * bv[k * 2 + c];
                }
                want[r * 2 + c] = s;
            }
        }
        let got = c.to_vec::<f32>().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "gpu {g} vs cpu {w}");
        }
    }
}
