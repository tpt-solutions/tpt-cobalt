# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Tensor: single device-agnostic tensor handle over swappable Storage backends.
- CPU storage backend (CpuStorage) with element-wise, reduction, matmul, transpose, and broadcast ops.
- DType dispatch (F64, F32, I64, I32, I16, I8, U8, Bool) with typed 	o_vec::<T>() access and DTypeError diagnostics.
- Metadata model: Shape, Strides, Layout, TensorMeta.
- Device enum with Cpu plus future backend variants (Cuda(n)).
- AutogradNode slot per tensor with with_autograd() / grad() / ccumulate_grad() for 	pt-autograd.
- Byte-level access (s_bytes, rom_le_bytes) for serialization interop.
- Two runnable examples (asic_ops, dtypes_and_roundtrip) and a comprehensive README.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-tensor-v0.1.0
