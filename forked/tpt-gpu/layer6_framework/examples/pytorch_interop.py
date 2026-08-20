#!/usr/bin/env python3
"""
PyTorch interop example for tptr framework backends.

Demonstrates:
- Registering TPT as a PyTorch backend
- Tensor conversion between PyTorch and TPT
- Autograd-compatible operations
- Stream management
- Hugging Face integration (if available)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tptr.pytorch


def main():
    print("=" * 60)
    print("TPT Framework Backends - PyTorch Interop Example")
    print("=" * 60)

    # 1. Check availability
    print("\n1. Checking TPT availability")
    available = tptr.pytorch.is_available()
    print(f"   TPT available: {available}")

    # 2. Register backend
    print("\n2. Registering TPT backend")
    success = tptr.pytorch.register_backend()
    print(f"   Registration: {'success' if success else 'failed'}")

    # 3. List supported ops
    print("\n3. Supported PyTorch operations")
    supported = tptr.pytorch.ops.get_supported_ops()
    for op in supported[:5]:
        tpt_op = tptr.pytorch.ops.get_tpt_op_name(op)
        print(f"   {op} -> {tpt_op}")
    print(f"   ... and {len(supported) - 5} more")

    # 4. Check specific ops
    print("\n4. Checking specific operations")
    test_ops = ["aten.add.Tensor", "aten.relu.default", "aten.mm.default", "aten.foo.bar"]
    for op in test_ops:
        supported = tptr.pytorch.ops.is_supported(op)
        print(f"   {op}: {'supported' if supported else 'not supported'}")

    # 5. TPT device info
    print("\n5. TPT Device Information")
    from tptr.pytorch import TptrTorchDevice
    dev = TptrTorchDevice(0)
    print(f"   Device: {dev}")
    print(f"   Name: {dev.name}")
    print(f"   Memory: {dev.total_memory / (1024**3):.1f} GB")

    # 6. Tensor conversion
    print("\n6. Tensor Conversion")
    try:
        import torch
        from tptr.pytorch.tensor import from_torch, to_torch

        x_torch = torch.randn(3, 4)
        x_tpt = from_torch(x_torch)
        print(f"   PyTorch tensor: {x_torch.shape}")
        print(f"   TPT tensor: {x_tpt}")

        x_back = to_torch(x_tpt)
        print(f"   Back to PyTorch: {x_back.shape}")
    except ImportError:
        print("   PyTorch not installed, skipping")

    # 7. Autograd operations
    print("\n7. Autograd Operations")
    try:
        import torch
        from tptr.pytorch.autograd import tpt_add, tpt_matmul, tpt_mul, tpt_relu

        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        print(f"   tpt_add: {tpt_add(a, b).shape}")
        print(f"   tpt_mul: {tpt_mul(a, 2.0).shape}")
        print(f"   tpt_matmul: {tpt_matmul(a, torch.randn(4, 5)).shape}")
        print(f"   tpt_relu: {tpt_relu(a).shape}")
    except ImportError:
        print("   PyTorch not installed, skipping")

    # 8. Stream management
    print("\n8. Stream Management")
    from tptr.pytorch.stream import get_stream

    stream = get_stream(0, "normal")
    print(f"   Stream: {stream}")
    stream.submit("test_command")
    stream.synchronize()
    print("   Command submitted and synchronized")

    # 9. Hugging Face integration
    print("\n9. Hugging Face Integration")
    from tptr.pytorch.hf_bridge import is_hf_available
    print(f"   HF available: {is_hf_available()}")

    print("\n" + "=" * 60)
    print("PyTorch interop example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()