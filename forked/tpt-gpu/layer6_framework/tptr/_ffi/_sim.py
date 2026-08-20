"""
Simulation fallback for the tptr native extension.
Provides Python-based simulation of the TPT runtime for development
and testing when the native Rust extension is not available.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, ClassVar


class TptrError(Exception):
    """Simulated TPT runtime error."""
    def __init__(self, code: str, message: str, source: str = "", context: list | None = None):
        self.code = code
        self.message = message
        self.source = source
        self.context = context or []
        super().__init__(f"[{code}] {message}")


@dataclass
class DeviceInfo:
    name: str = "TPT Simulated Device"
    total_memory: int = 16 * (1 << 30)
    compute_capability: tuple[int, int] = (1, 0)
    backend: str = "Simulated"
    max_threads_per_block: int = 1024
    warp_size: int = 32
    num_compute_units: int = 16


class Device:
    """Simulated TPT GPU device."""
    _instances: ClassVar[dict[int, Device]] = {}
    _lock = threading.Lock()

    def __init__(self, index: int = 0):
        self._index = index
        self._info = DeviceInfo(name=f"TPT Device {index} (Simulated)", total_memory=16 << 30)
        self._allocations: dict[int, MemoryAllocation] = {}
        self._next_handle = 1
        self._next_kernel_id = 1
        self._next_queue_id = 1
        self._queues: dict[int, CommandQueue] = {}
        self._kernels: dict[str, Kernel] = {}

    @classmethod
    def get_default(cls) -> Device:
        return cls(0)

    @classmethod
    def enumerate(cls):
        return [f"TPT Device {i} (Simulated)" for i in range(1)]

    def allocate(self, size: int, mem_type: str = "device", access: str = "read_write") -> MemoryAllocation:
        handle = self._next_handle
        self._next_handle += 1
        alloc = MemoryAllocation(handle, size, 0x1000_0000 + handle * 256, device=self)
        self._allocations[handle] = alloc
        return alloc

    def free(self, alloc: MemoryAllocation) -> None:
        alloc._freed = True
        self._allocations.pop(alloc._handle, None)

    def memcpy_htod(self, dst: MemoryAllocation, src: bytes, size: int, dst_offset: int = 0) -> None:
        if dst.is_freed():
            raise TptrError("E0003", "Invalid address: destination is freed")

    def memcpy_dtoh(self, src: MemoryAllocation, size: int, src_offset: int = 0) -> bytes:
        if src.is_freed():
            raise TptrError("E0003", "Invalid address: source is freed")
        return b'\x00' * size

    def create_queue(self, priority: str = "normal") -> CommandQueue:
        handle = self._next_queue_id
        self._next_queue_id += 1
        queue = CommandQueue(handle, priority, device=self)
        self._queues[handle] = queue
        return queue

    def create_kernel(self, name: str) -> Kernel:
        kernel = Kernel(name)
        self._kernels[name] = kernel
        return kernel

    def info(self) -> dict[str, Any]:
        return {"name": self._info.name, "total_memory": str(self._info.total_memory),
                "backend": self._info.backend, "warp_size": str(self._info.warp_size)}

    def synchronize(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Device(index={self._index}, name='{self._info.name}')"


@dataclass
class MemoryAllocation:
    """Simulated GPU memory allocation handle."""
    _handle: int
    _size: int
    _device_ptr: int
    device: Device | None = None
    _freed: bool = False

    @property
    def handle(self) -> int: return self._handle
    @property
    def size(self) -> int: return self._size
    @property
    def device_ptr(self) -> int: return self._device_ptr
    def is_freed(self) -> bool: return self._freed
    def __repr__(self) -> str:
        return f"MemoryAllocation(handle={self._handle}, size={self._size}, ptr=0x{self._device_ptr:x})"


class CommandQueue:
    """Simulated command queue."""
    def __init__(self, handle: int, priority: str = "normal", device: Device | None = None):
        self._handle = handle
        self._priority = priority
        self._device = device
        self._commands: list = []
    @property
    def handle(self) -> int: return self._handle
    def submit(self, command: str, **kwargs) -> int:
        cmd_id = len(self._commands) + 1
        self._commands.append({"id": cmd_id, "command": command, **kwargs})
        return cmd_id
    def synchronize(self) -> None: self._commands.clear()
    def __repr__(self) -> str: return f"CommandQueue(handle={self._handle}, priority='{self._priority}')"


@dataclass
class Kernel:
    """Simulated GPU kernel."""
    _name: str
    @property
    def name(self) -> str: return self._name
    def __repr__(self) -> str: return f"Kernel(name='{self._name}')"


class KernelConfig:
    """Simulated kernel launch configuration."""
    def __init__(self, grid: tuple[int, int, int] = (1, 1, 1),
                 block: tuple[int, int, int] = (1, 1, 1), shared_mem: int = 0):
        self._grid = grid
        self._block = block
        self._shared_mem = shared_mem
    @property
    def grid_size(self) -> tuple[int, int, int]: return self._grid
    @property
    def block_size(self) -> tuple[int, int, int]: return self._block
    @property
    def shared_mem_bytes(self) -> int: return self._shared_mem
    def __repr__(self) -> str: return f"KernelConfig(grid={self._grid}, block={self._block})"


class KernelHandle:
    """Simulated kernel execution handle."""
    _next_id = 0
    def __init__(self):
        KernelHandle._next_id += 1
        self._id = KernelHandle._next_id
        self._complete = True
    @property
    def id(self) -> int: return self._id
    def is_complete(self) -> bool: return self._complete
    def wait(self) -> None: self._complete = True
    def __repr__(self) -> str: return f"KernelHandle(id={self._id}, complete={self._complete})"
