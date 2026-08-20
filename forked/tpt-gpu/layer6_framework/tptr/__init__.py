"""
TPT Framework Backends (tptr) - Python thin wrapper over Rust runtime.

Provides a Pythonic API over the TPT GPU runtime (tpt-gpu-runtime) via PyO3 bindings.
Includes PyTorch and JAX integration for seamless ML framework interop.
"""

__version__ = "1.0.0"
__license__ = "Apache-2.0"

# Re-export core types from the Rust-backed tptr module
from ._ffi import (
    CommandQueue,
    Device,
    Kernel,
    KernelConfig,
    KernelHandle,
    MemoryAllocation,
    Queue,
    TptrError,
)

# Re-export high-level wrappers
from .core import (
    TptrContext,
    TptrDevice,
    TptrKernel,
    TptrMemory,
    TptrStream,
    get_context,
    get_device,
    synchronize,
)

# Re-export dispatch utilities
from .dispatch import (
    DispatchRegistry,
    get_dispatch_table,
    register_op,
)

# Re-export tensor utilities
from .tensor import (
    TptrDType,
    TptrTensor,
    dtype,
    empty,
    full,
    ones,
    zeros,
)

__all__ = [
    "CommandQueue",
    "Device",
    "DispatchRegistry",
    "Kernel",
    "KernelConfig",
    "KernelHandle",
    "MemoryAllocation",
    "Queue",
    "TptrContext",
    "TptrDType",
    "TptrDevice",
    "TptrError",
    "TptrKernel",
    "TptrMemory",
    "TptrStream",
    "TptrTensor",
    "dtype",
    "empty",
    "full",
    "get_context",
    "get_device",
    "get_dispatch_table",
    "ones",
    "register_op",
    "synchronize",
    "zeros",
]