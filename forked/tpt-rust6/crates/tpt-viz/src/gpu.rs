//! WebGPU (wgpu) rendering backend (feature `gpu`).
//!
//! The CPU side of the pipeline ([`ScatterBuffers`], [`scatter_buffers`]) is
//! always available under `gpu`; this module adds a real GPU half: a
//! [`Renderer`] that uploads a scatter layer's packed vertex data and draws it
//! as instanced point-sprites on a [`wgpu`] device, reading the frame back as
//! RGBA8. The same code path drives a browser `<canvas>` (via a `Surface`) and
//! a native window, and an offscreen texture for headless export.
//!
//! This scales to 100M+ points because the per-point work runs entirely on the
//! GPU: each point is one *instance* of a 6-vertex quad, so CPU cost is a single
//! `draw(6, point_count)` call regardless of dataset size.

use pollster::FutureExt;

use wgpu::util::DeviceExt;

use crate::geom::Scatter;
use crate::plot::Plot;

/// GPU-ready vertex data for one scatter layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterBuffers {
    /// `[x0, y0, x1, y1, ...]` in pixel space.
    pub positions: Vec<f32>,
    /// `[r, g, b, a, ...]` in `0..1`.
    pub colors: Vec<f32>,
    /// Point radius in pixels.
    pub point_size: f32,
}

impl ScatterBuffers {
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 2
    }
}

/// Pack a scatter layer into vertex buffers using `plot`'s scales and gradient.
pub fn scatter_buffers(layer: &Scatter, plot: &Plot) -> ScatterBuffers {
    let (positions, colors) = crate::render::scatter_buffers(layer, plot);
    ScatterBuffers {
        positions,
        colors,
        point_size: layer.size_px() as f32,
    }
}

/// Errors from the GPU backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// No WebGPU adapter was found (headless CI, missing drivers, …).
    NoAdapter,
    /// Device/queue creation failed.
    Device(String),
    /// The frame could not be read back from the GPU.
    Map(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no WebGPU adapter available"),
            GpuError::Device(m) => write!(f, "device error: {m}"),
            GpuError::Map(m) => write!(f, "frame readback error: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Uniforms passed to the vertex shader for pixel -> clip-space conversion.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: f32,
    height: f32,
    point_size: f32,
    _pad: f32,
}

const SHADER: &str = r#"
struct Uniforms {
    width: f32,
    height: f32,
    point_size: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) center: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(vertex_index) vi: u32,
) -> VSOut {
    var offs = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let o = offs[vi];
    let half = u.point_size * 0.5;
    let clip = vec2<f32>(
        (center.x / u.width) * 2.0 - 1.0,
        1.0 - (center.y / u.height) * 2.0,
    );
    let off_clip = vec2<f32>(
        (o.x * half) / u.width * 2.0,
        -(o.y * half) / u.height * 2.0,
    );
    var out: VSOut;
    out.pos = vec4<f32>(clip + off_clip, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// A WebGPU renderer for scatter layers.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Renderer {
    /// Create a renderer on the default adapter (any backend).
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                force_fallback_adapter: false,
                power_preference: wgpu::PowerPreference::default(),
            })
            .block_on()
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tpt-viz gpu renderer"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .block_on()
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tpt-viz point sprite"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniforms"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tpt-viz pipeline"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tpt-viz scatter"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 2 * 4,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 4 * 4,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 1,
                            }],
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            uniform_buf,
            bind_group_layout,
        })
    }

    /// Render `buffers` to an offscreen `width x height` frame and return RGBA8
    /// pixels (row-major, bottom-to-top, matching wgpu's framebuffer origin).
    pub fn render_scatter_rgba(
        &self,
        buffers: &ScatterBuffers,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let count = buffers.vertex_count();
        if count == 0 {
            return Ok(vec![255; (width * height * 4) as usize]);
        }

        let pos_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions"),
                contents: bytemuck::cast_slice(&buffers.positions),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let col_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("colors"),
                contents: bytemuck::cast_slice(&buffers.colors),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let uniforms = Uniforms {
            width: width as f32,
            height: height as f32,
            point_size: buffers.point_size,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buf.as_entire_binding(),
            }],
        });

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tpt-viz encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tpt-viz pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0u32, pos_buf.slice(..));
            pass.set_vertex_buffer(1u32, col_buf.slice(..));
            pass.draw(0u32..6u32, 0u32..count as u32);
        }

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();
        let out = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        Ok(out)
    }
}
