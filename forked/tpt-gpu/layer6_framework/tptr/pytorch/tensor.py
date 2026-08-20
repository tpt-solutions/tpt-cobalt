# TPT PyTorch Tensor Wrapper
from __future__ import annotations

from tptr import _ffi


class TptrTorchTensor:
    def __init__(self, shape, dtype="float32", device_index=0, data=None, native_alloc=None):
        self._shape = tuple(shape)
        self._device_index = device_index
        self._nelem = 1
        for s in shape:
            self._nelem *= s
        self._dtype = _resolve_dtype(dtype)
        self._itemsize = self._dtype.itemsize if hasattr(self._dtype, "itemsize") else 4
        self._nbytes = self._nelem * self._itemsize
        if native_alloc is not None:
            self._native_alloc = native_alloc
        else:
            device = _ffi.Device(device_index)
            self._native_alloc = device.allocate(self._nbytes)
        if data is not None:
            device = _ffi.Device(device_index)
            device.memcpy_htod(self._native_alloc, data, len(data))

    @staticmethod
    def from_torch_tensor(torch_tensor):
        shape = tuple(torch_tensor.shape)
        dtype_str = str(torch_tensor.dtype).split(".")[-1]
        data = torch_tensor.detach().contiguous().cpu().numpy().tobytes()
        return TptrTorchTensor(shape, dtype_str, device_index=0, data=data)

    @staticmethod
    def from_numpy(ndarray, device_index=0):
        shape = tuple(ndarray.shape)
        dtype_str = str(ndarray.dtype)
        data = ndarray.tobytes()
        return TptrTorchTensor(shape, dtype_str, device_index=device_index, data=data)

    def to_torch(self):
        import torch
        device = _ffi.Device(self._device_index)
        raw_data = device.memcpy_dtoh(self._native_alloc, self._nbytes)
        import numpy as np
        np_dtype = _dtype_to_numpy(self._dtype)
        arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(self._shape).copy()
        return torch.from_numpy(arr)

    def to_numpy(self):
        import numpy as np
        device = _ffi.Device(self._device_index)
        raw_data = device.memcpy_dtoh(self._native_alloc, self._nbytes)
        np_dtype = _dtype_to_numpy(self._dtype)
        return np.frombuffer(raw_data, dtype=np_dtype).reshape(self._shape).copy()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._nelem

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def device_index(self):
        return self._device_index

    @property
    def alloc(self):
        return self._native_alloc

    @property
    def is_valid(self):
        return not self._native_alloc.is_freed()

    def copy_to_host(self, size=None):
        size = size or self._nbytes
        device = _ffi.Device(self._device_index)
        return device.memcpy_dtoh(self._native_alloc, size)

    def copy_from_host(self, data, size=None):
        size = size or len(data)
        device = _ffi.Device(self._device_index)
        device.memcpy_htod(self._native_alloc, data, size)

    def zero_(self):
        device = _ffi.Device(self._device_index)
        device.memcpy_htod(self._native_alloc, b"\x00" * self._nbytes, self._nbytes)
        return self

    def clone(self):
        device = _ffi.Device(self._device_index)
        new_alloc = device.allocate(self._nbytes)
        data = device.memcpy_dtoh(self._native_alloc, self._nbytes)
        device.memcpy_htod(new_alloc, data, self._nbytes)
        return TptrTorchTensor(self._shape, self._dtype, self._device_index, native_alloc=new_alloc)

    def reshape(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        new_nelem = 1
        for s in new_shape:
            new_nelem *= s
        if new_nelem != self._nelem:
            raise ValueError(f"Cannot reshape tensor of size {self._nelem} to size {new_nelem}")
        result = TptrTorchTensor.__new__(TptrTorchTensor)
        result._shape = tuple(new_shape)
        result._dtype = self._dtype
        result._device_index = self._device_index
        result._nelem = self._nelem
        result._itemsize = self._itemsize
        result._nbytes = self._nbytes
        result._native_alloc = self._native_alloc
        return result

    def __repr__(self):
        dtype_name = self._dtype.name if hasattr(self._dtype, "name") else self._dtype
        return f"TptrTorchTensor(shape={self._shape}, dtype={dtype_name}, device={self._device_index})"

    def __del__(self):
        # Broad + silent: exceptions in __del__ can't propagate usefully, and
        # interpreter shutdown may have already torn down module globals.
        try:
            if hasattr(self, "_native_alloc") and not self._native_alloc.is_freed():
                _ffi.Device(self._device_index).free(self._native_alloc)
        except Exception:  # noqa: BLE001, S110
            pass


def _resolve_dtype(dtype):
    from tptr.tensor import TptrDType
    if isinstance(dtype, TptrDType):
        return dtype
    dtype_map = {
        "float16": TptrDType.FLOAT16, "float32": TptrDType.FLOAT32,
        "float64": TptrDType.FLOAT64, "int8": TptrDType.INT8,
        "int16": TptrDType.INT16, "int32": TptrDType.INT32,
        "int64": TptrDType.INT64, "uint8": TptrDType.UINT8,
        "uint16": TptrDType.UINT16, "uint32": TptrDType.UINT32,
        "bool": TptrDType.BOOL,
    }
    if isinstance(dtype, str):
        return dtype_map.get(dtype, TptrDType.FLOAT32)
    if isinstance(dtype, int):
        try:
            return TptrDType(dtype)
        except ValueError:
            return TptrDType.FLOAT32
    return TptrDType.FLOAT32


def _dtype_to_numpy(tptr_dtype):
    from tptr.tensor import TptrDType
    mapping = {
        TptrDType.FLOAT16: "float16", TptrDType.FLOAT32: "float32",
        TptrDType.FLOAT64: "float64", TptrDType.INT8: "int8",
        TptrDType.INT16: "int16", TptrDType.INT32: "int32",
        TptrDType.INT64: "int64", TptrDType.UINT8: "uint8",
        TptrDType.UINT16: "uint16", TptrDType.UINT32: "uint32",
        TptrDType.BOOL: "bool",
    }
    return mapping.get(tptr_dtype, "float32")


def from_torch(torch_tensor):
    return TptrTorchTensor.from_torch_tensor(torch_tensor)


def to_torch(tptr_tensor):
    return tptr_tensor.to_torch()