"""Tests for tptr core module."""
from tptr.core import (
    TptrContext,
    TptrDevice,
    TptrKernelConfig,
    device_context,
    get_context,
    get_device,
    synchronize,
)


class TestTptrDevice:
    def test_device_creation(self):
        dev = TptrDevice(0)
        assert dev.index == 0
        assert "Simulated" in dev.name

    def test_device_info(self):
        dev = TptrDevice(0)
        info = dev.info
        assert "name" in info
        assert "total_memory" in info
        assert "backend" in info

    def test_device_allocate(self):
        dev = TptrDevice(0)
        mem = dev.allocate(4096)
        assert mem.size == 4096
        assert not mem.is_freed

    def test_device_free(self):
        dev = TptrDevice(0)
        mem = dev.allocate(4096)
        dev.free(mem)
        assert mem.is_freed

    def test_device_synchronize(self):
        dev = TptrDevice(0)
        dev.synchronize()  # Should not raise

    def test_device_context_manager(self):
        with TptrDevice(0) as dev:
            mem = dev.allocate(1024)
            assert not mem.is_freed

    def test_device_repr(self):
        dev = TptrDevice(0)
        r = repr(dev)
        assert "TptrDevice" in r


class TestTptrMemory:
    def test_memory_properties(self):
        dev = TptrDevice(0)
        mem = dev.allocate(8192)
        assert mem.size == 8192
        assert mem.handle > 0
        assert mem.device_ptr > 0

    def test_memory_repr(self):
        dev = TptrDevice(0)
        mem = dev.allocate(4096)
        r = repr(mem)
        assert "TptrMemory" in r


class TestTptrStream:
    def test_stream_creation(self):
        dev = TptrDevice(0)
        stream = dev.create_stream("normal")
        assert stream.handle > 0
        assert stream.priority == "normal"

    def test_stream_submit(self):
        dev = TptrDevice(0)
        stream = dev.create_stream()
        cmd_id = stream.submit("barrier")
        assert cmd_id > 0

    def test_stream_synchronize(self):
        dev = TptrDevice(0)
        stream = dev.create_stream()
        stream.synchronize()  # Should not raise


class TestTptrKernel:
    def test_kernel_creation(self):
        dev = TptrDevice(0)
        kernel = dev.create_kernel("test_kernel")
        assert kernel.name == "test_kernel"

    def test_kernel_repr(self):
        dev = TptrDevice(0)
        kernel = dev.create_kernel("my_kernel")
        r = repr(kernel)
        assert "my_kernel" in r


class TestTptrKernelConfig:
    def test_config_creation(self):
        config = TptrKernelConfig(grid=(16, 1, 1), block=(256, 1, 1))
        assert config.grid_size == (16, 1, 1)
        assert config.block_size == (256, 1, 1)
        assert config.shared_mem_bytes == 0

    def test_config_with_shared_mem(self):
        config = TptrKernelConfig(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=1024)
        assert config.shared_mem_bytes == 1024


class TestTptrContext:
    def test_context_creation(self):
        ctx = TptrContext(0)
        assert ctx.device.index == 0
        assert ctx.stream is not None

    def test_context_manager(self):
        with TptrContext(0) as ctx:
            mem = ctx.device.allocate(1024)
            assert not mem.is_freed


class TestModuleFunctions:
    def test_get_device(self):
        dev = get_device(0)
        assert isinstance(dev, TptrDevice)

    def test_get_context(self):
        ctx = get_context(0)
        assert isinstance(ctx, TptrContext)

    def test_synchronize(self):
        synchronize()  # Should not raise

    def test_device_context(self):
        with device_context(0) as dev:
            assert isinstance(dev, TptrDevice)

