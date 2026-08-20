# Tutorial 9: Python API

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 1, Python basics

---

## Introduction

The Python API provides access to TPT GPU through PyO3 bindings, enabling rapid prototyping and integration with the Python ML ecosystem.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application                            │
├─────────────────────────────────────────────────────────────────┤
│  import tptr                                                     │
│  device = tptr.Device(0)                                        │
├─────────────────────────────────────────────────────────────────┤
│                    PyO3 Bindings                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Rust Core (tpt-gpu-runtime)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
cargo build -p out-gpu-runtime-py

cd layer6_framework
pip install -e ".[dev]"
```

Verify:
```python
import tptr
print(tptr.__version__)
```

---

## Device Management

```python
import tptr

# Create device
device = tptr.Device(0)

# Query device info
info = device.info()
print(f"Device: {info['name']}")
print(f"Memory: {info['total_memory'] / (1024**3):.1f} GB")
```

---

## Memory Allocation

```python
import tptr

device = tptr.Device(0)

# Allocate GPU memory
mem = device.allocate(4096)
print(f"Handle: {mem.handle}")
print(f"Size: {mem.size}")
print(f"Device Ptr: 0x{mem.device_ptr:x}")

# Allocate with a memory type ("device" | "host_pinned" | "managed")
# and access ("read_write" | "read" | "write")
mem = device.allocate(4096, "device", "read_write")

# Memory freed automatically when mem goes out of scope
```

---

## Kernel Operations

```python
import tptr

device = tptr.Device(0)

# Create kernel
kernel = device.create_kernel("matmul")

# Configure kernel
config = tptr.KernelConfig(
    grid=(64, 1, 1),
    block=(256, 1, 1)
)

# Allocate arguments
a = device.allocate(1024 * 1024 * 4)  # 4 MB
b = device.allocate(1024 * 1024 * 4)
c = device.allocate(1024 * 1024 * 4)

# Launch kernel. The device is passed so the kernel runs on the
# simulated/real backend. `args` is an optional list of raw byte buffers
# (typically little-endian `u64` device pointers for buffer operands).
handle = kernel.launch(device, config, [a, b, c])

# Wait for completion
handle.wait()

# Synchronize the whole device
device.synchronize()
```

---

## Command Queues (Streams)

```python
import tptr

device = tptr.Device(0)

# Create a command queue (stream) on the device. Priority is one of
# "high" | "normal" | "low".
queue = device.create_queue("normal")
print(f"Queue handle: {queue.handle}")

# Submit work through the device, then synchronize to drain the queue.
device.synchronize()
```

---

## Loading a TPTIR Module

You can compile and run a hand-written TPTIR module directly through the
runtime. This is the external-integration entry point: pass TPTIR assembly
(e.g. emitted by `tpt-archon`) and receive a runnable `Kernel`.

```python
import tptr
import struct

device = tptr.Device(0)

module = """
module {
  func.func @reduce_max(%in: memref<*xf32>, %out: memref<*xf32>) attributes {tptir.kernel} {
    ^entry:
      %v = tptir.load(%in)
      %m = tptir.max(%v)
      tptir.store(%m, %out)
      tptir.return
  }
}
"""

kernel = device.load_module(module)

# Allocate input/output buffers and upload data.
data = [1.0, -3.0, 7.5, 2.0, 0.0, 4.0, -1.0, 9.25]
d_in = device.allocate(len(data) * 4)
d_out = device.allocate(4)
device.memcpy_htod(d_in, struct.pack(f"{len(data)}f", *data), len(data) * 4, 0)

# Args are little-endian u64 device pointers.
config = tptr.KernelConfig(grid=(1, 1, 1), block=(1, 1, 1))
kernel.launch(device, config, [
    struct.pack("<Q", d_in.device_ptr),
    struct.pack("<Q", d_out.device_ptr),
]).wait()

out = device.memcpy_dtoh(d_out, 4, 0)
print(f"max = {struct.unpack('<f', out)[0]:.2f}")  # 9.25
```

---

## Error Handling

```python
import tptr

try:
    device = tptr.Device(0)
    mem = device.allocate(1024**4)  # 1 TB - will fail
except tptr.TptrError as e:
    print(f"Error: {e.code} - {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Example: Vector Addition

```python
import tptr
import numpy as np

def vector_add():
    device = tptr.Device(0)
    
    # Create data
    n = 1024 * 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    # Allocate GPU memory
    d_a = device.allocate(a.nbytes)
    d_b = device.allocate(b.nbytes)
    d_c = device.allocate(c.nbytes)
    
    # Copy to GPU
    device.memcpy_htod(d_a, a.tobytes(), a.nbytes, 0)
    device.memcpy_htod(d_b, b.tobytes(), b.nbytes, 0)

    # Launch kernel
    kernel = device.create_kernel("vector_add")
    config = tptr.KernelConfig(grid=(n // 256, 1, 1), block=(256, 1, 1))
    # Args are raw little-endian u64 device pointers for each buffer operand.
    kernel.launch(device, config, [
        d_a.device_ptr.to_bytes(8, "little"),
        d_b.device_ptr.to_bytes(8, "little"),
        d_c.device_ptr.to_bytes(8, "little"),
    ]).wait()

    # Copy result back
    out = device.memcpy_dtoh(d_c, c.nbytes, 0)
    c = np.frombuffer(out, dtype=np.float32)

    return c

result = vector_add()
print(f"Result: {result[:5]}")
```

---

## Exercises

1. **Device Info**: Write a script that prints all device properties
2. **Memory Benchmark**: Measure allocation and copy performance
3. **Kernel Launch**: Implement a simple element-wise operation

---

## Summary

- ✅ Device creation and management
- ✅ Memory allocation with automatic cleanup
- ✅ Kernel creation and launch
- ✅ Stream-based command submission
- ✅ Error handling with Python exceptions

**Next:** [Tutorial 10: PyTorch Integration](10_pytorch_integration.md)
