//! CPU kernel dispatch. The full runtime dispatches to CPU/WGPU/CUDA/ROCm/FPGA/
//! MCU; here we implement the host path over `tpt-tensor`'s CPU ops so the
//! dispatch surface is real and testable before the GPU backends land.

use tpt_tensor::Tensor;

/// Target device for a dispatch. Only `Cpu` is backed today (Phase 4 adds the
/// rest); mirrors `tpt_tensor::Device` by name so the dispatch API is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }
}

/// Dispatch an element-wise add on `device` (host path).
pub fn dispatch_add(a: &Tensor, b: &Tensor, device: Device) -> Tensor {
    assert!(
        device.is_cpu(),
        "only CPU dispatch is implemented; got {device:?}"
    );
    a.add(b)
}

/// Dispatch a 2-D matrix multiply on `device` (host path).
pub fn dispatch_matmul(a: &Tensor, b: &Tensor, device: Device) -> Tensor {
    assert!(
        device.is_cpu(),
        "only CPU dispatch is implemented; got {device:?}"
    );
    a.matmul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_runs_on_cpu() {
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.0_f64, 1.0, 1.0, 0.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = dispatch_matmul(&a, &b, Device::Cpu);
        assert_eq!(c.to_vec::<f64>().unwrap(), vec![2.0, 1.0, 4.0, 3.0]);
    }
}
