"""
TPT Dispatch - Operation dispatch registry for framework integration.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class OpType(Enum):
    ELEMENTWISE_BINARY = "elementwise_binary"
    ELEMENTWISE_UNARY = "elementwise_unary"
    REDUCTION = "reduction"
    MATMUL = "matmul"
    CONVOLUTION = "convolution"
    SOFTMAX = "softmax"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"
    MEMORY = "memory"
    CUSTOM = "custom"


@dataclass
class OpMetadata:
    name: str
    op_type: OpType
    input_count: int
    output_count: int
    supports_inplace: bool = False
    kernel_name: str | None = None
    description: str = ""


@dataclass
class OpEntry:
    metadata: OpMetadata
    factory: Callable
    backend: str = "tptr"


def _default_factory(name: str) -> Callable:
    """Create a default factory that returns a no-op function."""
    def factory():
        return lambda *args, **kwargs: None
    return factory


class DispatchRegistry:
    """Registry for tensor operations mapping high-level ops to TPT kernels."""
    def __init__(self):
        self._ops: dict[str, OpEntry] = {}

    def register(self, name, op_type=OpType.CUSTOM, input_count=2, output_count=1,
                 supports_inplace=False, kernel_name=None, description="",
                 factory=None, backend="tptr"):
        metadata = OpMetadata(name=name, op_type=op_type, input_count=input_count,
                              output_count=output_count, supports_inplace=supports_inplace,
                              kernel_name=kernel_name, description=description)
        if factory is None:
            factory = _default_factory(name)
        self._ops[name] = OpEntry(metadata=metadata, factory=factory, backend=backend)

    def register_custom(self, name, factory, op_type=OpType.CUSTOM, **kwargs):
        self.register(name, op_type, factory=factory, **kwargs)

    def lookup(self, name):
        return self._ops.get(name)

    def dispatch(self, name, *args, **kwargs):
        entry = self._ops.get(name)
        if entry is None:
            raise KeyError(f"Operation '{name}' not registered")
        impl = entry.factory()
        return impl(*args, **kwargs)

    def list_ops(self):
        return list(self._ops.keys())

    def has(self, name):
        return name in self._ops

    def remove(self, name):
        if name in self._ops:
            del self._ops[name]
            return True
        return False

    def clear(self) -> None:
        self._ops.clear()

    def __len__(self) -> int:
        return len(self._ops)

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def list_by_type(self, op_type: OpType) -> list[str]:
        return [name for name, entry in self._ops.items() if entry.metadata.op_type == op_type]


# Module-level default registry
_default_registry = DispatchRegistry()


def register_op(name, op_type=OpType.CUSTOM, **kwargs):
    _default_registry.register(name, op_type, **kwargs)


def get_dispatch_table():
    return _default_registry


def dispatch_op(name, *args, **kwargs):
    return _default_registry.dispatch(name, *args, **kwargs)


_default_registry.register("add", OpType.ELEMENTWISE_BINARY, input_count=2, output_count=1,
                          supports_inplace=True, kernel_name="tptr_add",
                          description="Element-wise addition")
_default_registry.register("mul", OpType.ELEMENTWISE_BINARY, input_count=2, output_count=1,
                          kernel_name="tptr_mul", description="Element-wise multiplication")
_default_registry.register("sub", OpType.ELEMENTWISE_BINARY, input_count=2, output_count=1,
                          kernel_name="tptr_sub", description="Element-wise subtraction")
_default_registry.register("div", OpType.ELEMENTWISE_BINARY, input_count=2, output_count=1,
                          kernel_name="tptr_div", description="Element-wise division")
_default_registry.register("neg", OpType.ELEMENTWISE_UNARY, input_count=1, output_count=1,
                          kernel_name="tptr_neg", description="Negation")
_default_registry.register("relu", OpType.ACTIVATION, input_count=1, output_count=1,
                          supports_inplace=True, kernel_name="tptr_relu", description="ReLU activation")
_default_registry.register("gelu", OpType.ACTIVATION, input_count=1, output_count=1,
                          kernel_name="tptr_gelu", description="GELU activation")
_default_registry.register("silu", OpType.ACTIVATION, input_count=1, output_count=1,
                          kernel_name="tptr_silu", description="SiLU activation")
_default_registry.register("softmax", OpType.SOFTMAX, input_count=1, output_count=1,
                          kernel_name="tptr_softmax", description="Softmax")
_default_registry.register("sum", OpType.REDUCTION, input_count=1, output_count=1,
                          kernel_name="tptr_sum", description="Sum reduction")
_default_registry.register("mean", OpType.REDUCTION, input_count=1, output_count=1,
                          kernel_name="tptr_mean", description="Mean reduction")
_default_registry.register("matmul", OpType.MATMUL, input_count=2, output_count=1,
                          kernel_name="tptr_matmul", description="Matrix multiplication")
_default_registry.register("layer_norm", OpType.LAYER_NORM, input_count=1, output_count=1,
                          kernel_name="tptr_layer_norm", description="Layer normalization")
