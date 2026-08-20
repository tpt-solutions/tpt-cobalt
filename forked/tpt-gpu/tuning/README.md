# Hardware-Profit Tuning Database

This directory contains community-contributed GPU kernel tuning profiles for TPT-GPU.

## Directory Contents

```
tuning/
+-- README.md              # This file - community guidelines
+-- RTX_3090.json          # NVIDIA Ampere (pre-existing)
+-- RTX_4090.json          # NVIDIA Ada Lovelace (community)
+-- RTX_4080.json          # NVIDIA Ada Lovelace (community)
+-- A100.json              # NVIDIA Ampere Datacenter (community)
+-- RX_7900_XTX.json       # AMD RDNA 3 (community)
+-- A770.json              # Intel Arc Alchemist (community)
+-- dispatch_table.json    # Cross-GPU dispatch table (auto-generated)
```

## GPU Models Covered

| Model | Architecture | Memory | Contributor | Kernels |
|-------|-------------|--------|-------------|----------|
| RTX 3090 | Ampere | 24 GB GDDR6X | community | 5 kernels |
| RTX 4090 | Ada Lovelace | 24 GB GDDR6X | community | 6 kernels |
| RTX 4080 | Ada Lovelace | 16 GB GDDR6X | community | 5 kernels |
| A100 | Ampere | 80 GB HBM2e | community | 5 kernels |
| RX 7900 XTX | RDNA 3 | 24 GB GDDR6 | community | 5 kernels |
| A770 | Alchemist | 16 GB GDDR6 | community | 5 kernels |

## Kernel Coverage

### Compute Kernels
- `vector_add_f32_1024` - Vector addition (1024 elements)
- `matmul_f32_1024x1024` - Matrix multiply (1024x1024)
- `matmul_f32_4096x4096` - Large matrix multiply (4096x4096)
- `softmax_f32_1024` - Softmax normalization
- `layer_norm_f32_1024x1024` - Layer normalization
- `group_norm_f32_32x32x256` - Group normalization

### AI Kernels
- `flash_attention_f32_1024x1024` - Flash Attention (seq=1024, dim=1024)
- `flash_attention_f32_4096x4096` - Flash Attention (seq=4096, dim=4096)
- `gemm_f16_4096x4096x4096` - GEMM (FP16)
- `gemm_bf16_4096x4096x4096` - GEMM (BF16)

### Vision Kernels
- `conv2d_f32_224x224_c3_k64` - Convolution (224x224 input)
- `conv2d_f32_112x112_c64_k128` - Convolution (112x112 input)
- `conv_bn_relu_f32_32x32` - Fused Conv+BN+ReLU

## Profile Schema

```json
{
  "gpu_model": "ModelName",
  "contributor": "community|anonymous|username",
  "timestamp": "ISO-8601 timestamp",
  "hardware_specs": {
    "architecture": "Architecture Name",
    "compute_capability": "X.Y",
    "sm_count": 0,
    "cuda_cores": 0,
    "tensor_cores": 0,
    "base_clock_mhz": 0,
    "boost_clock_mhz": 0,
    "memory_gb": 0,
    "memory_type": "GDDR6|GDDR6X|HBM2e",
    "memory_bandwidth_gbps": 0,
    "l2_cache_mb": 0
  },
  "kernel_configs": {
    "kernel_name_shape": {
      "block_size": 0,
      "grid_size": 0,
      "shared_mem_bytes": 0,
      "execution_time_ms": 0.0
    }
  }
}
```

## How to Contribute

### 1. Run Benchmarks

```bash
# Build and run the kernel optimizer
cargo build --release -p tpt-gpu-kernel-optimizer
./target/release/tpt-gpu-kernel-optimizer --gpu <model> --benchmark
```

### 2. Collect Results

Run each kernel 10 times and report the median execution time:

```bash
for kernel in vector_add_f32_1024 matmul_f32_1024x1024 flash_attention_f32_1024x1024; do
  ./bench --kernel $kernel --iterations 10
done
```

### 3. Create Profile

Copy an existing profile and update with your results:

```bash
cp tuning/RTX_4090.json tuning/YOUR_GPU.json
# Edit with your benchmark results
```

### 4. Submit Pull Request

- Include benchmark logs
- Note driver version and OS
- Mention any overclocking or power limit changes

## Hardware-Specific Notes

### NVIDIA GPUs
- Enable persistence mode: `sudo nvidia-smi -pm 1`
- Lock GPU clocks for consistent results: `sudo nvidia-smi -lgc <clock>`
- Disable ECC on A100 for baseline benchmarks: `sudo nvidia-smi -e 0`

### AMD GPUs
- Use ROCm 5.5+ for best performance
- Set HSA_ENABLE_SDMA=0 for accurate kernel timing
- Use `rocm-smi` to monitor power and clocks

### Intel GPUs
- Use Intel Graphics Compute Runtime 23.x+
- Enable performance mode: `echo performance > /sys/class/drm/card0/power_dpm_force_performance_level`

## Validation

All profiles are validated against:
- =90% cuBLAS efficiency (GEMM kernels)
- =90% FlashAttention v2 efficiency (Attention kernels)
- Cross-referenced with vendor library benchmarks

Run validation:

```bash
cargo test -p tpt-gpu-compiler --test tuning_validation
```

## Using Tuning Profiles

```rust
use tptc::tuning::GpuProfile;

// Load all profiles
let profiles = GpuProfile::load_profiles_from_dir("tuning/")?;

// Get config for specific kernel
if let Some(config) = profile.get_kernel_config("matmul", "f32", &[1024, 1024]) {
    println!("block_size: {}, grid_size: {}", 
        config.block_size, config.grid_size);
}
```

## License

All community contributions are under Apache License 2.0.

See individual profiles for contributor attribution.
