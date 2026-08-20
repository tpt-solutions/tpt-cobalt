"""Tests for tptr PyTorch integration."""


class TestPyTorchOps:
    """Tests for PyTorch operation dispatch."""

    def test_get_supported_ops(self):
        from tptr.pytorch.ops import get_supported_ops
        ops = get_supported_ops()
        assert "aten.add.Tensor" in ops
        assert "aten.relu.default" in ops
        assert "aten.mm.default" in ops

    def test_is_supported(self):
        from tptr.pytorch.ops import is_supported
        assert is_supported("aten.add.Tensor")
        assert is_supported("aten.relu.default")
        assert not is_supported("aten.foo.bar")

    def test_get_tpt_op_name(self):
        from tptr.pytorch.ops import get_tpt_op_name
        assert get_tpt_op_name("aten.add.Tensor") == "add"
        assert get_tpt_op_name("aten.relu.default") == "relu"
        assert get_tpt_op_name("aten.mm.default") == "matmul"

    def test_register_custom_op(self):
        from tptr.pytorch.ops import get_tpt_op_name, is_supported, register_custom_op
        register_custom_op("aten.custom.op", "custom_kernel")
        assert is_supported("aten.custom.op")
        assert get_tpt_op_name("aten.custom.op") == "custom_kernel"


class TestPyTorchBackend:
    """Tests for PyTorch backend registration."""

    def test_is_available(self):
        from tptr.pytorch import is_available
        # Should return True/False without error
        result = is_available()
        assert isinstance(result, bool)

    def test_register_backend(self):
        from tptr.pytorch import register_backend
        # Should return True/False without error
        result = register_backend()
        assert isinstance(result, bool)

    def test_get_tpt_device(self):
        from tptr.pytorch import get_tpt_device
        dev = get_tpt_device("tpt:0")
        assert dev.index == 0

    def test_get_tpt_device_default(self):
        from tptr.pytorch import get_tpt_device
        dev = get_tpt_device("tpt")
        assert dev.index == 0


class TestTptrTorchDevice:
    """Tests for TptrTorchDevice class."""

    def test_device_creation(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        assert dev.index == 0
        assert "Simulated" in dev.name

    def test_device_repr(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        assert repr(dev) == "tptr:0"

    def test_device_allocate(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        tensor = dev.allocate(4096)
        assert tensor.size == 4096
        assert tensor.handle > 0


class TestTptrNativeTensor:
    """Tests for TptrNativeTensor class."""

    def test_tensor_creation(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        tensor = dev.allocate(1024)
        assert tensor.size == 1024

    def test_tensor_copy_to_host(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        tensor = dev.allocate(256)
        data = tensor.copy_to_host(256)
        assert len(data) == 256

    def test_tensor_copy_from_host(self):
        from tptr.pytorch import TptrTorchDevice
        dev = TptrTorchDevice(0)
        tensor = dev.allocate(256)
        data = b'\x00' * 256
        tensor.copy_from_host(data, 256)  # Should not raise

