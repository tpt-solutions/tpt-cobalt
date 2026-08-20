#!/usr/bin/env python3
"""
Basic usage example for tptr framework backends.

Demonstrates:
- Device creation and memory allocation
- Kernel configuration and launch
- High-level context management
- Tensor operations
"""
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tptr
from tptr.core import TptrContext, TptrDevice, device_context
from tptr.tensor import TptrTensor, float32, zeros


def main():
    print("=" * 60)
    print("TPT Framework Backends - Basic Usage Example")
    print("=" * 60)

    # 1. Device creation
    print("\n1. Device Creation")
    device = TptrDevice(0)
    print(f"   Device: {device}")
    print(f"   Name: {device.name}")
    print(f"   Total Memory: {device.total_memory / (1024**3):.1f} GB")

    # 2. Memory allocation
    print("\n2. Memory Allocation")
    mem = device.allocate(4096)
    print(f"   Allocated: {mem}")
    print(f"   Handle: {mem.handle}")
    print(f"   Size: {mem.size} bytes")
    print(f"   Device Ptr: 0x{mem.device_ptr:x}")

    # 3. Kernel creation
    print("\n3. Kernel Creation")
    kernel = device.create_kernel("matmul")
    print(f"   Kernel: {kernel}")

    # 4. Kernel configuration
    print("\n4. Kernel Configuration")
    config = tptr.KernelConfig(grid=(16, 1, 1), block=(256, 1, 1))
    print(f"   Config: {config}")
    print(f"   Grid: {config.grid_size}")
    print(f"   Block: {config.block_size}")

    # 5. Stream creation
    print("\n5. Stream Creation")
    stream = device.create_stream("normal")
    print(f"   Stream: {stream}")

    # 6. Context manager
    print("\n6. Context Manager")
    with TptrContext(0) as ctx:
        ctx_mem = ctx.device.allocate(1024)
        print(f"   Context device: {ctx.device}")
        print(f"   Context stream: {ctx.stream}")
        print(f"   Memory: {ctx_mem}")

    # 7. Tensor operations
    print("\n7. Tensor Operations")
    a = zeros((32, 64), float32)
    b = zeros((32, 64), float32)
    print(f"   Tensor a: {a}")
    print(f"   Tensor b: {b}")
    c = a + b
    print(f"   a + b = {c}")

    # 8. Device context
    print("\n8. Device Context Manager")
    with device_context(0):
        t = TptrTensor((10, 10), float32)
        print(f"   Tensor in context: {t}")

    # 9. Synchronization
    print("\n9. Synchronization")
    device.synchronize()
    print("   Device synchronized successfully")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

