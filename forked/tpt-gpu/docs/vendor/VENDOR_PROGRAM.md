# TPT-GPU Third-Party Hardware Vendor Support Program

## Overview

The TPT-GPU Third-Party Hardware Vendor Support Program provides a structured framework for hardware vendors to integrate their GPUs with the TPT-GPU platform. This program ensures compatibility, performance, and reliability across diverse hardware ecosystems while maintaining the platform's hardware-agnostic design principles.

## Program Benefits

### For Hardware Vendors
- **Market Access**: Integration with TPT-GPU's growing ecosystem of ML frameworks (PyTorch; JAX planned)
- **Performance Optimization**: Access to TPT-GPU's kernel optimization tools and tuning infrastructure
- **Certification Badge**: Official "TPT-GPU Certified" designation for qualifying hardware
- **Technical Support**: Direct engineering support from TPT-GPU maintainers
- **Co-Marketing**: Joint marketing opportunities and featured vendor listings

### For the TPT-GPU Ecosystem
- **Hardware Diversity**: Support for NVIDIA, AMD, Intel, and custom silicon
- **User Choice**: End users can select optimal hardware for their workloads
- **Innovation**: Vendor-specific optimizations benefit the entire community
- **Standardization**: Consistent API surface across all supported hardware

## Certification Tiers

### Tier 1: Basic Compatibility
**Requirements:**
- TPTIR compilation succeeds for all core operations
- Memory allocation and deallocation work correctly
- Basic kernel launch functionality
- Passes automated compatibility test suite

**Badge:** "TPT-GPU Compatible"

### Tier 2: Performance Optimized
**Requirements:**
- All Tier 1 requirements
- Vendor backend implementation for at least one primitive (GEMM, Attention, or Conv2D)
- Performance within 80% of vendor's native library (cuBLAS, rocBLAS, etc.)
- Passes performance regression tests
- Submitted tuning profile to `tuning/vendor/` directory

**Badge:** "TPT-GPU Optimized"

### Tier 3: Fully Certified
**Requirements:**
- All Tier 2 requirements
- Vendor backend implementation for all core primitives (GEMM, Attention, Conv2D, Conv3D)
- Performance within 90% of vendor's native library
- Comprehensive documentation and examples
- Dedicated vendor support channel
- Quarterly performance reviews

**Badge:** "TPT-GPU Certified"

## Vendor Integration Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TPT-GPU Applications                      │
├─────────────────────────────────────────────────────────────┤
│              TPT Script / PyTorch (JAX planned)               │
├─────────────────────────────────────────────────────────────┤
│                    TPTIR Compiler (Layer 3)                  │
├─────────────────────────────────────────────────────────────┤
│                    TPT Runtime (Layer 4)                     │
├─────────────────────────────────────────────────────────────┤
│              Vendor Backend Interface (Layer 5)              │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│   NVIDIA    │    AMD      │   Intel     │  Custom Silicon   │
│   Backend   │   Backend   │   Backend   │    Backend        │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│   cuBLAS    │   rocBLAS   │   oneDNN    │   Vendor SDK      │
│   cuDNN     │   MIOpen    │   oneMKL    │   Driver          │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

### Vendor Backend Interface

All vendor backends must implement the `VendorLibrary` trait:

```rust
pub trait VendorLibrary: Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn gemm(&self, a: &GpuBuffer<f32>, b: &GpuBuffer<f32>, c: &mut GpuBuffer<f32>,
            alpha: f32, beta: f32, m: usize, n: usize, k: usize) -> TptpResult<()>;
    fn attention(&self, q: &GpuBuffer<f32>, k: &GpuBuffer<f32>, v: &GpuBuffer<f32>,
                 output: &mut GpuBuffer<f32>, scale: f32, seq_len: usize, d_k: usize) -> TptpResult<()>;
    fn conv2d(&self, input: &GpuBuffer<f32>, filter: &GpuBuffer<f32>, output: &mut GpuBuffer<f32>,
              strides: [u32; 2], padding: [u32; 2]) -> TptpResult<()>;
    fn conv3d(&self, input: &GpuBuffer<f32>, filter: &GpuBuffer<f32>, output: &mut GpuBuffer<f32>,
              strides: [u32; 3], padding: [u32; 3]) -> TptpResult<()>;
}
```

## Getting Started

### Step 1: Vendor Registration

1. Submit a vendor registration issue to the TPT-GPU GitHub repository using the template:
   - Vendor name and contact information
   - Hardware specifications (GPU model, memory, compute capability)
   - Target certification tier
   - Expected timeline

2. Receive vendor ID and access to vendor support resources

### Step 2: Development Environment Setup

```bash
# Clone TPT-GPU repository
git clone https://github.com/tpt-solutions/tpt-gpu.git
cd tpt-gpu

# Build the primitives crate (simulation mode requires no hardware)
cd crates/tpt-gpu-primitives
cargo build --features sim

# Run primitives tests
cargo test --features sim
```

### Step 3: Implement Vendor Backend

Create a new backend in `crates/tpt-gpu-primitives/src/vendor/<vendor_name>.rs`:

```rust
//! <Vendor Name> Backend
//!
//! <Vendor> GPU support via <Vendor SDK>.

use crate::error::{TptpError, TptpResult};
use crate::memory::GpuBuffer;
use super::VendorLibrary;

pub struct VendorBackend {
    device_id: i32,
    // Vendor-specific handles
}

impl VendorBackend {
    pub fn new() -> TptpResult<Self> {
        // Initialize vendor SDK
        Ok(VendorBackend {
            device_id: 0,
        })
    }
}

impl VendorLibrary for VendorBackend {
    fn name(&self) -> &str {
        "<Vendor>"
    }

    fn is_available(&self) -> bool {
        // Check if vendor SDK is available
        true
    }

    // Implement required operations...
}
```

### Step 4: Register Backend in Vendor Dispatch

Update `crates/tpt-gpu-primitives/src/vendor/mod.rs`:

```rust
pub mod <vendor_name>;

pub enum VendorBackend {
    // ... existing variants
    VendorName(<vendor_name>::VendorBackend),
}

impl VendorBackend {
    pub fn detect() -> Self {
        // ... existing detection
        if let Ok(backend) = <vendor_name>::VendorBackend::new() {
            return VendorBackend::VendorName(backend);
        }
        // ... fallback
    }
}
```

### Step 5: Submit for Certification

1. Run the certification test suite:
   ```bash
   cargo run -p tpt-gpu-vendor-cert -- certify --vendor <vendor_name> --tier 1
   ```

2. Submit results via pull request to `tuning/vendor/<vendor_name>.json`

3. TPT-GPU maintainers will review and provide feedback

## Certification Test Suite

All tests live in a single module, `crates/tpt-gpu-vendor-cert/src/tests.rs`, grouped into the three
entry points used by the CLI:

### Compatibility Tests

`run_compatibility_tests` — backend detection for the requested vendor (via
`tpt-gpu-primitives::vendor::detect`) plus checks that each primitive (GEMM, attention, Conv2D, Conv3D)
is supported by the detected backend.

### Performance Tests

`run_performance_tests` — for each supported primitive, actually launches the kernel on the detected
backend and asserts a real elapsed-time measurement was taken. Thresholds are intentionally lenient;
pass/fail is determined by "the operation ran and produced output", not by an analytical cost model.

### Correctness Tests

`run_correctness_tests` — for each supported primitive, runs the backend operation against a small
CPU reference implementation (in `tests.rs`) and compares outputs within a tolerance. The CPU
references are included so a backend cannot pass by doing nothing: a backend that fabricates success
fails the numerical comparison.

> Note: certificates awarded from this suite reflect the backend's actual behavior on the
> submitting vendor's hardware. No backend is certified by default — the tests only pass when a real
> `tpt-gpu-primitives` vendor backend is detected (`--features cuda` / `rocm` / `metal` / `sim` are
> passed through to the primitives crate).

## Vendor Profile Format

Submit vendor profiles to `tuning/vendor/<vendor_name>.json`:

```json
{
  "vendor": "Example Corp",
  "gpu_model": "Example GPU X100",
  "driver_version": "1.0.0",
  "certification_tier": 2,
  "certification_date": "2026-06-29",
  "hardware_specs": {
    "memory_gb": 32,
    "memory_bandwidth_gbps": 900,
    "compute_tflops_fp32": 40,
    "compute_tflops_fp16": 80,
    "tensor_cores": true
  },
  "supported_operations": {
    "gemm": true,
    "attention": true,
    "conv2d": true,
    "conv3d": false
  },
  "performance_baselines": {
    "gemm_4096x4096_ms": 2.5,
    "attention_1024x1024_ms": 1.8,
    "conv2d_224x224_ms": 3.2
  },
  "tuning_parameters": {
    "gemm": {
      "tile_m": 128,
      "tile_n": 128,
      "tile_k": 32,
      "vec_width": 4,
      "unroll": 4
    },
    "attention": {
      "tile_seq": 64,
      "tile_head": 64
    },
    "conv2d": {
      "tile_oh": 14,
      "tile_ow": 14,
      "tile_ic": 64
    }
  },
  "contact": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com",
    "github": "janedoe"
  }
}
```

## Support Channels

### Vendor Developer Mailing List
- Join: `vendors-join@tpt-gpu.org`
- Archives: https://lists.tpt-gpu.org/vendors

### GitHub Discussions
- Category: "Vendor Support"
- Tag: `vendor-support`

### Monthly Vendor Sync
- First Tuesday of each month
- 10:00 AM Pacific Time
- Video call link provided to registered vendors

### Emergency Support
- Email: `vendor-emergency@tpt-gpu.org`
- Response time: 24 hours for Tier 3 vendors

## Compliance Requirements

### Code Quality
- All vendor backends must pass `cargo clippy` with no warnings
- Code coverage must be ≥ 80% for vendor-specific code
- Documentation must be complete for all public APIs

### Performance Standards
- Performance regressions > 5% block certification renewal
- Quarterly performance reviews required for Tier 3 certification
- Tuning profiles must be updated when performance characteristics change

### Security
- Vendor backends must not introduce memory safety vulnerabilities
- All FFI calls must be properly documented and tested
- Security vulnerabilities must be reported within 24 hours

## Certification Renewal

### Annual Review
- All certifications expire after 12 months
- Vendors must submit updated performance profiles
- Compatibility tests must be re-run against latest TPT-GPU version

### Continuous Integration
- Vendor backends are tested in TPT-GPU CI pipeline
- Performance regressions trigger automatic notifications
- Failed tests must be addressed within 30 days

## Example Vendor Integration

See the following reference implementations:

- **NVIDIA Backend**: `crates/tpt-gpu-primitives/src/vendor/cuda.rs`
- **AMD Backend**: `crates/tpt-gpu-primitives/src/vendor/rocm.rs`
- **Apple Metal Backend**: `crates/tpt-gpu-primitives/src/vendor/metal.rs`

## Contact Information

- **Program Inquiries**: `vendor-program@tpt-gpu.org`
- **Technical Support**: `vendor-tech@tpt-gpu.org`
- **Certification**: `vendor-cert@tpt-gpu.org`
- **GitHub**: https://github.com/tpt-solutions/tpt-gpu

## License

The TPT-GPU Vendor Program is part of the TPT-GPU project and is licensed under Apache 2.0 with Express Patent Grant.