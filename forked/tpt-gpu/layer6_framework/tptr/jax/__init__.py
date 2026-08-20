"""
TPT JAX Dispatch Layer.

Integrates the TPT GPU runtime with JAX via custom primitives, mirroring the
PyTorch backend's dispatch pattern in tptr.pytorch: is_available()/
register_backend() for backend setup, get_supported_ops()/is_op_supported()/
get_tpt_op_name() for op-name mapping. The actual jax.core.Primitive
definitions (matmul, attention, conv2d, layer_norm) with JVP/VJP/XLA lowering
live in tptr.jax.ops, imported lazily so `import tptr.jax` succeeds even when
JAX is not installed.

Usage:
    import jax.numpy as jnp
    import tptr.jax

    tptr.jax.register_backend()
    from tptr.jax.ops import tpt_matmul
    out = tpt_matmul(jnp.ones((3, 4)), jnp.ones((4, 5)))
"""

from __future__ import annotations

import importlib.util
import math

TPT_BACKEND_NAME = "tpt"

# Mapping from JAX-side op names to TPT kernel names — mirrors
# tptr.pytorch.ops._OP_MAP. No JAX import required to use this table.
_OP_MAP: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "neg": "neg",
    "dot": "matmul",
    "dot_general": "matmul",
    "matmul": "matmul",
    "conv": "conv2d",
    "conv_general_dilated": "conv2d",
    "attention": "attention",
    "softmax": "softmax",
    "relu": "relu",
    "gelu": "gelu",
    "silu": "silu",
    "layer_norm": "layer_norm",
}

_ITEMSIZES: dict[str, int] = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1,
}

_registered = False


def is_available() -> bool:
    """Check whether JAX is importable in this environment."""
    return importlib.util.find_spec("jax") is not None


def register_backend() -> bool:
    """
    Register TPT JAX primitives (matmul/attention/conv2d/layer_norm).
    Returns True if registration succeeded.
    """
    global _registered
    if not is_available():
        return False
    from tptr.jax.ops import register_jax_primitives
    register_jax_primitives()
    _registered = True
    return True


def get_backend_name() -> str:
    return TPT_BACKEND_NAME


def is_registered() -> bool:
    return _registered


def get_supported_ops() -> list:
    """Get list of JAX ops supported by TPT."""
    return sorted(_OP_MAP.keys())


def is_op_supported(op: str) -> bool:
    """Check if a JAX op is supported by TPT."""
    return op in _OP_MAP


def get_tpt_op_name(op: str) -> str | None:
    """Get the TPT kernel name for a JAX op."""
    return _OP_MAP.get(op)


def jax_to_tptr(array) -> TptrJaxArray:
    """Wrap a JAX/NumPy array's shape+dtype metadata as a TptrJaxArray."""
    return TptrJaxArray.from_numpy(array)


class TptrJaxArray:
    """Lightweight shape/dtype descriptor for TPT-JAX array interop.

    Mirrors tptr.pytorch.tensor.TptrTorchTensor's host-copy interface, but
    JAX arrays are immutable and device-managed by JAX itself, so this class
    only tracks metadata plus host round-tripping rather than owning a
    native allocation.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str = "float32"):
        self._shape = tuple(shape)
        self._dtype = dtype
        self._itemsize = _ITEMSIZES.get(dtype, 4)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def size(self) -> int:
        return math.prod(self._shape) if self._shape else 0

    @property
    def nbytes(self) -> int:
        return self.size * self._itemsize

    def copy_to_host(self) -> bytes:
        return bytes(self.nbytes)

    def copy_from_host(self, data: bytes) -> None:
        pass

    def to_numpy(self):
        import numpy as np
        np_dtype = getattr(np, self._dtype, np.float32)
        return np.zeros(self._shape, dtype=np_dtype)

    @classmethod
    def from_numpy(cls, array) -> TptrJaxArray:
        return cls(tuple(array.shape), str(array.dtype))

    def __repr__(self) -> str:
        return f"TptrJaxArray(shape={self._shape}, dtype='{self._dtype}')"


__all__ = [
    "TPT_BACKEND_NAME",
    "TptrJaxArray",
    "get_backend_name",
    "get_supported_ops",
    "get_tpt_op_name",
    "is_available",
    "is_op_supported",
    "is_registered",
    "jax_to_tptr",
    "register_backend",
]
