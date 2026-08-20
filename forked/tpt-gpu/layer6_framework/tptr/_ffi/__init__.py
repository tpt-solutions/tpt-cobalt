"""
FFI bridge to the Rust tptr library.

This module provides direct access to the PyO3-backed tptr extension.
When the native extension is not available, a simulation fallback is provided.
"""

import importlib.util
import os
import warnings

# Try to import the native Rust extension (not the local tptr package)
_native_ext = None

# Strategy 1: load the compiled extension directly out of the Cargo target
# directory. The native extension is built by out-gpu-runtime-py
# (`cargo build -p out-gpu-runtime-py` from the repo root).
def _find_native_extension():
    """Find and load the native tptr extension module."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates", "tpt-gpu-runtime", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates", "tpt-gpu-runtime", "target", "debug"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates", "out-gpu-runtime-py", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates", "out-gpu-runtime-py", "target", "debug"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "debug"),
    ]

    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue

        for filename in ["tptr.pyd", "tptr.so", "tptr.cpython-312-x86_64-linux-gnu.so"]:
            ext_path = os.path.join(base_path, filename)
            if os.path.exists(ext_path):
                try:
                    spec = importlib.util.spec_from_file_location("tptr_native", ext_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
                except (ImportError, AttributeError, OSError):
                    pass

    return None

_native_ext = _find_native_extension()

# Strategy 2: fall back to an installed top-level `tptr` package (e.g. a
# `pip install`-ed wheel), taking care not to re-import ourselves.
if _native_ext is None:
    try:
        _spec = importlib.util.find_spec("tptr")
        if _spec is not None and _spec.origin and "_ffi" not in _spec.origin:
            import tptr as _native_ext  # type: ignore
    except (ImportError, AttributeError):
        pass

if _native_ext is not None and hasattr(_native_ext, "Device"):
    # Re-export all native types from the native extension
    Device = _native_ext.Device
    MemoryAllocation = _native_ext.MemoryAllocation
    CommandQueue = _native_ext.CommandQueue
    Kernel = _native_ext.Kernel
    KernelConfig = _native_ext.KernelConfig
    KernelHandle = _native_ext.KernelHandle
    TptrError = _native_ext.TptrError
    Queue = CommandQueue
else:
    # Simulation fallback for development/testing without native extension
    # Use a simple warning without stacklevel to avoid issues
    warnings.warn(
        "Native tptr extension not found. Using simulation fallback. "
        "Build with: cargo build -p out-gpu-runtime-py",
        RuntimeWarning,
    )
    from ._sim import (
        CommandQueue,
        Device,
        Kernel,
        KernelConfig,
        KernelHandle,
        MemoryAllocation,
        TptrError,
    )
    Queue = CommandQueue

__all__ = [
    "CommandQueue",
    "Device",
    "Kernel",
    "KernelConfig",
    "KernelHandle",
    "MemoryAllocation",
    "Queue",
    "TptrError",
]