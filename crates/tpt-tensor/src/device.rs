/// The device a tensor's storage lives on.
///
/// Only `Cpu` is backed by a `Storage` implementation today (see
/// [`crate::CpuStorage`]); the remaining variants are reserved so that the
/// `Tensor` handle is stable once the runtime (Phase 4) and exotic backends
/// (Phase 5) land their own `Storage` impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Rocm(usize),
    Metal(usize),
    Wgpu,
    Fpga(usize),
    Mcu(usize),
}

impl Device {
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Stable string key used by the runtime dispatcher and serialization.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Rocm(_) => "rocm",
            Device::Metal(_) => "metal",
            Device::Wgpu => "wgpu",
            Device::Fpga(_) => "fpga",
            Device::Mcu(_) => "mcu",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
            Device::Rocm(i) => write!(f, "rocm:{i}"),
            Device::Metal(i) => write!(f, "metal:{i}"),
            Device::Wgpu => write!(f, "wgpu"),
            Device::Fpga(i) => write!(f, "fpga:{i}"),
            Device::Mcu(i) => write!(f, "mcu:{i}"),
        }
    }
}
