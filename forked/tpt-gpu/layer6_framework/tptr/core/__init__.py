"""
High-level Python wrappers over the tptr runtime.
Provides a Pythonic, context-managed API over the low-level Rust bindings.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from .._ffi import (
    CommandQueue as _NativeCommandQueue,
)
from .._ffi import (
    Device as _NativeDevice,
)
from .._ffi import (
    KernelConfig as _NativeKernelConfig,
)
from .._ffi import (
    KernelHandle as _NativeKernelHandle,
)
from .._ffi import (
    MemoryAllocation as _NativeMemoryAllocation,
)


class TptrDevice:
    """High-level device wrapper with context management."""
    def __init__(self, index: int = 0):
        self._device = _NativeDevice(index)
        self._streams: list = []
        self._kernels: dict = {}

    @property
    def index(self) -> int: return self._device._index
    @property
    def info(self) -> dict[str, Any]: return self._device.info()
    @property
    def name(self) -> str: return self.info.get("name", "Unknown")
    @property
    def total_memory(self) -> int: return int(self.info.get("total_memory", 0))

    def allocate(self, size: int, name: str = "device", access: str = "read_write") -> TptrMemory:
        alloc = self._device.allocate(size, name, access)
        return TptrMemory(alloc)

    def free(self, memory: TptrMemory) -> None:
        self._device.free(memory._alloc)

    def memcpy_htod(self, dst: TptrMemory, src: bytes, size: int | None = None, dst_offset: int = 0) -> None:
        size = size or len(src)
        self._device.memcpy_htod(dst._alloc, src, size, dst_offset)

    def memcpy_dtoh(self, src: TptrMemory, size: int, src_offset: int = 0) -> bytes:
        return self._device.memcpy_dtoh(src._alloc, size, src_offset)

    def create_stream(self, priority: str = "normal") -> TptrStream:
        queue = self._device.create_queue(priority)
        stream = TptrStream(queue, self)
        self._streams.append(stream)
        return stream

    def create_kernel(self, name: str) -> TptrKernel:
        kernel = TptrKernel(name, self)
        self._kernels[name] = kernel
        return kernel

    def synchronize(self) -> None: self._device.synchronize()
    def __enter__(self) -> TptrDevice: return self  # noqa: PYI034 (Self requires typing_extensions on py<3.11, not a declared dep)
    def __exit__(self, *args) -> None: self.synchronize()
    def __repr__(self) -> str: return f"TptrDevice(index={self.index}, name='{self.name}')"


class TptrMemory:
    """High-level memory allocation wrapper."""
    def __init__(self, alloc: _NativeMemoryAllocation):
        self._alloc = alloc
    @property
    def handle(self) -> int: return self._alloc.handle
    @property
    def size(self) -> int: return self._alloc.size
    @property
    def device_ptr(self) -> int: return self._alloc.device_ptr
    @property
    def is_freed(self) -> bool: return self._alloc.is_freed()
    def __repr__(self) -> str: return f"TptrMemory(handle={self.handle}, size={self.size})"


class TptrStream:
    """High-level command queue wrapper."""
    def __init__(self, queue: _NativeCommandQueue, device: TptrDevice):
        self._queue = queue
        self._device = device
    @property
    def handle(self) -> int: return self._queue.handle
    @property
    def priority(self) -> str: return self._queue._priority
    def submit(self, command: str, **kwargs) -> int: return self._queue.submit(command, **kwargs)
    def synchronize(self) -> None: self._queue.synchronize()
    def __repr__(self) -> str: return f"TptrStream(handle={self.handle}, priority='{self.priority}')"


class TptrKernel:
    """High-level kernel wrapper."""
    def __init__(self, name: str, device: TptrDevice):
        self._name = name
        self._device = device
        self._native = device._device.create_kernel(name)
    @property
    def name(self) -> str: return self._name
    def launch(self, config: _NativeKernelConfig, args: list | None = None) -> _NativeKernelHandle:
        return _NativeKernelHandle()
    def __repr__(self) -> str: return f"TptrKernel(name='{self._name}')"


class TptrKernelConfig:
    """High-level kernel configuration wrapper."""
    def __init__(self, grid: tuple[int, int, int] = (1, 1, 1),
                 block: tuple[int, int, int] = (1, 1, 1), shared_mem: int = 0):
        self._config = _NativeKernelConfig(grid, block, shared_mem)
    @property
    def grid_size(self) -> tuple[int, int, int]: return self._config.grid_size
    @property
    def block_size(self) -> tuple[int, int, int]: return self._config.block_size
    @property
    def shared_mem_bytes(self) -> int: return self._config.shared_mem_bytes
    def __repr__(self) -> str: return f"TptrKernelConfig(grid={self.grid_size}, block={self.block_size})"


class TptrContext:
    """Context manager for device + stream lifecycle."""
    def __init__(self, device_index: int = 0, stream_priority: str = "normal"):
        self._device = TptrDevice(device_index)
        self._stream = self._device.create_stream(stream_priority)
    @property
    def device(self) -> TptrDevice: return self._device
    @property
    def stream(self) -> TptrStream: return self._stream
    def __enter__(self) -> TptrContext: return self  # noqa: PYI034 (Self requires typing_extensions on py<3.11, not a declared dep)
    def __exit__(self, *args) -> None:
        self._stream.synchronize()
        self._device.synchronize()


_default_device: TptrDevice | None = None


def get_device(index: int = 0) -> TptrDevice:
    global _default_device
    if _default_device is None:
        _default_device = TptrDevice(index)
    return _default_device


def get_context(device_index: int = 0) -> TptrContext:
    return TptrContext(device_index)


def synchronize() -> None:
    dev = get_device()
    dev.synchronize()


@contextmanager
def device_context(index: int = 0):
    dev = TptrDevice(index)
    try:
        yield dev
    finally:
        dev.synchronize()
