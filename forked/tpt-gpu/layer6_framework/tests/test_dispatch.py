"""Tests for tptr dispatch module."""
import pytest
from tptr.dispatch import (
    DispatchRegistry,
    OpType,
    get_dispatch_table,
    register_op,
)


class TestOpType:
    def test_op_type_names(self):
        assert OpType.ELEMENTWISE_BINARY.value == "elementwise_binary"
        assert OpType.MATMUL.value == "matmul"
        assert OpType.ACTIVATION.value == "activation"
        assert OpType.CUSTOM.value == "custom"


class TestDispatchRegistry:
    def test_registry_creation(self):
        reg = DispatchRegistry()
        assert len(reg) == 0
        assert not reg.has("add")

    def test_register_op(self):
        reg = DispatchRegistry()
        reg.register("my_op", OpType.CUSTOM, input_count=2, output_count=1)
        assert reg.has("my_op")
        assert len(reg) == 1

    def test_lookup(self):
        reg = DispatchRegistry()
        reg.register("test", OpType.ELEMENTWISE_BINARY, input_count=2, output_count=1)
        entry = reg.lookup("test")
        assert entry is not None
        assert entry.metadata.name == "test"
        assert entry.metadata.op_type == OpType.ELEMENTWISE_BINARY

    def test_lookup_missing(self):
        reg = DispatchRegistry()
        assert reg.lookup("nonexistent") is None

    def test_dispatch(self):
        reg = DispatchRegistry()
        called = []

        def my_impl(*args, **kwargs):
            called.append(True)
            return "result"

        reg.register("my_op", factory=lambda: my_impl)
        result = reg.dispatch("my_op")
        assert result == "result"
        assert len(called) == 1

    def test_dispatch_missing(self):
        reg = DispatchRegistry()
        with pytest.raises(KeyError):
            reg.dispatch("nonexistent")

    def test_list_ops(self):
        reg = DispatchRegistry()
        reg.register("add", OpType.ELEMENTWISE_BINARY)
        reg.register("mul", OpType.ELEMENTWISE_BINARY)
        ops = reg.list_ops()
        assert "add" in ops
        assert "mul" in ops

    def test_list_by_type(self):
        reg = DispatchRegistry()
        reg.register("add", OpType.ELEMENTWISE_BINARY)
        reg.register("relu", OpType.ACTIVATION)
        reg.register("mul", OpType.ELEMENTWISE_BINARY)
        binary_ops = reg.list_by_type(OpType.ELEMENTWISE_BINARY)
        assert "add" in binary_ops
        assert "mul" in binary_ops
        assert "relu" not in binary_ops

    def test_remove(self):
        reg = DispatchRegistry()
        reg.register("test")
        assert reg.remove("test")
        assert not reg.has("test")

    def test_clear(self):
        reg = DispatchRegistry()
        reg.register("a")
        reg.register("b")
        reg.clear()
        assert len(reg) == 0

    def test_contains(self):
        reg = DispatchRegistry()
        reg.register("test")
        assert "test" in reg
        assert "missing" not in reg


class TestDefaultRegistry:
    def test_default_ops_registered(self):
        reg = get_dispatch_table()
        assert reg.has("add")
        assert reg.has("matmul")
        assert reg.has("relu")

    def test_register_op_function(self):
        register_op("custom_test_op", OpType.CUSTOM)
        assert get_dispatch_table().has("custom_test_op")

    def test_standard_ops(self):
        reg = get_dispatch_table()
        # Verify standard ops are registered
        standard = ["add", "mul", "sub", "div", "neg", "relu", "gelu",
                     "silu", "softmax", "sum", "mean", "matmul", "layer_norm"]
        for op in standard:
            assert reg.has(op), f"Standard op '{op}' not registered"


class TestDispatchRegistryAdvanced:
    """Advanced tests for DispatchRegistry."""

    def test_register_custom_with_factory(self):
        reg = DispatchRegistry()
        call_count = [0]

        def my_impl(*args, **kwargs):
            call_count[0] += 1
            return "result"

        reg.register_custom("custom_op", factory=lambda: my_impl, op_type=OpType.CUSTOM)
        result = reg.dispatch("custom_op")
        assert result == "result"
        assert call_count[0] == 1

    def test_list_by_type(self):
        reg = DispatchRegistry()
        reg.register("add", OpType.ELEMENTWISE_BINARY)
        reg.register("relu", OpType.ACTIVATION)
        reg.register("mul", OpType.ELEMENTWISE_BINARY)
        binary_ops = reg.list_by_type(OpType.ELEMENTWISE_BINARY)
        assert "add" in binary_ops
        assert "mul" in binary_ops
        assert "relu" not in binary_ops

    def test_remove_op(self):
        reg = DispatchRegistry()
        reg.register("test_op", OpType.CUSTOM)
        assert reg.has("test_op")
        assert reg.remove("test_op")
        assert not reg.has("test_op")

    def test_clear_registry(self):
        reg = DispatchRegistry()
        reg.register("a", OpType.CUSTOM)
        reg.register("b", OpType.CUSTOM)
        reg.clear()
        assert len(reg) == 0

    def test_contains(self):
        reg = DispatchRegistry()
        reg.register("test", OpType.CUSTOM)
        assert "test" in reg
        assert "missing" not in reg

