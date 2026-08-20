# TPT PyTorch Stream Management
from __future__ import annotations

import threading
from collections.abc import Callable

from tptr import _ffi


class TptStream:
    """
    TPT stream for asynchronous operations.
    Wraps a native TPT command queue and provides
    a PyTorch-compatible stream interface.
    """

    def __init__(self, device_index: int = 0, priority: str = "normal"):
        self._device_index = device_index
        self._priority = priority
        device = _ffi.Device(device_index)
        self._queue = device.create_queue(priority)
        self._callbacks = []
        self._lock = threading.Lock()

    @property
    def handle(self) -> int:
        return self._queue.handle

    @property
    def priority(self) -> str:
        return self._priority

    @property
    def device_index(self) -> int:
        return self._device_index

    def submit(self, command: str, **kwargs) -> int:
        with self._lock:
            return self._queue.submit(command, **kwargs)

    def synchronize(self) -> None:
        self._queue.synchronize()

    def add_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def wait_stream(self, other: TptStream) -> None:
        self.submit("wait_stream", other_handle=other.handle)

    def __repr__(self):
        return f"TptStream(device={self._device_index}, priority={self._priority})"


class TptEvent:
    """Synchronization event for TPT streams."""

    def __init__(self, stream: TptStream | None = None):
        self._stream = stream
        self._completed = False

    def record(self, stream: TptStream | None = None) -> None:
        self._stream = stream or self._stream
        self._completed = False

    def synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._completed = True

    def is_complete(self) -> bool:
        return self._completed

    def wait(self, stream: TptStream | None = None) -> None:
        if stream is not None:
            stream.synchronize()
        self._completed = True


class StreamContext:
    """Context manager for stream operations."""

    def __init__(self, stream: TptStream):
        self._stream = stream

    def __enter__(self) -> TptStream:
        return self._stream

    def __exit__(self, *args) -> None:
        self._stream.synchronize()


def get_stream(device_index: int = 0, priority: str = "normal") -> TptStream:
    """Get or create a stream for the given device."""
    return TptStream(device_index, priority)


def default_stream(device_index: int = 0) -> TptStream:
    """Get the default stream for a device."""
    return TptStream(device_index, "normal")