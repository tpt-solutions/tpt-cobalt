# TPT-GPU Vendor Certification Tool

The `tpt-gpu-vendor-cert` tool provides automated certification testing for third-party hardware vendor backends in the TPT-GPU ecosystem.

## Overview

This tool runs a comprehensive test suite to verify that vendor backends meet the requirements for TPT-GPU certification at three tiers:

- **Tier 1: Basic Compatibility** - Core functionality works correctly
- **Tier 2: Performance Optimized** - Meets performance targets with vendor optimizations
- **Tier 3: Fully Certified** - Complete feature support with excellent performance

## Installation

```bash
cd crates/tpt-gpu-vendor-cert
cargo build --release
```

## Usage

### Run Certification Tests

```bash
# Run Tier 1 certification for a vendor
cargo run -- certify --vendor "MyVendor" --tier 1

# Run Tier 2 certification with a vendor profile
cargo run -- certify --vendor "MyVendor" --tier 2 --profile tuning/vendor/myvendor.json

# Specify custom output directory
cargo run -- certify --vendor "MyVendor" --tier 1 --output /path/to/results
```

### List Registered Vendors

```bash
# List all registered vendors
cargo run -- list-vendors

# List vendors from a specific directory
cargo run -- list-vendors --dir /path/to/vendor/profiles
```

### Validate Vendor Profile

```bash
# Validate a vendor profile JSON file
cargo run -- validate-profile tuning/vendor/myvendor.json
```

### Generate Vendor Template

```bash
# Generate a template vendor profile
cargo run -- generate-template --vendor "MyVendor" --output myvendor_template.json
```

### Compare Vendors

```bash
# Compare multiple vendors
cargo run -- compare --vendor Vendor1 Vendor2 Vendor3
```

## Test Categories

### Compatibility Tests

- Backend detection
- Memory allocation/deallocation
- Host-to-device data transfer
- Device-to-host data transfer
- Kernel launch functionality
- TPTIR compilation (Tier 2+)
- Vendor library loading (Tier 2+)

### Correctness Tests

- GEMM numerical correctness
- Elementwise operation correctness
- Attention correctness (Tier 2+)
- Conv2D correctness (Tier 2+)
- Conv3D correctness (Tier 3)
- Mixed precision correctness (Tier 3)

### Performance Tests

- GEMM performance benchmark
- Memory bandwidth benchmark
- Attention performance (Tier 2+)
- Conv2D performance (Tier 2+)
- Conv3D performance (Tier 3)
- Sustained performance (Tier 3)

## Certification Criteria

### Tier 1: Basic Compatibility
- ≥80% compatibility tests pass
- ≥90% correctness tests pass

### Tier 2: Performance Optimized
- ≥90% compatibility tests pass
- ≥95% correctness tests pass
- ≥80% performance tests pass

### Tier 3: Fully Certified
- ≥95% compatibility tests pass
- ≥99% correctness tests pass
- ≥90% performance tests pass

## Output

The tool generates a detailed JSON report including:
- Test results for each category
- Overall pass/fail status
- Performance metrics
- Recommendations for improvement

## Integration with CI

The vendor certification tool integrates with TPT-GPU's CI pipeline to automatically test vendor backends on every pull request.

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.