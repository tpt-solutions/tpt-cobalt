"""Tests for tptr JAX integration."""
import pytest


class TestJaxBackend:
    """Tests for JAX backend registration."""

    def test_is_available(self):
        from tptr.jax import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_register_backend(self):
        from tptr.jax import register_backend
        result = register_backend()
        assert isinstance(result, bool)

    def test_get_backend_name(self):
        from tptr.jax import get_backend_name
        assert get_backend_name() == "tpt"

    def test_is_registered(self):
        pytest.importorskip("jax")
        from tptr.jax import is_registered, register_backend
        # After registration (requires JAX; without it register_backend()
        # returns False and is_registered() stays False)
        register_backend()
        assert is_registered()


class TestJaxOps:
    """Tests for JAX operation mapping."""

    def test_get_supported_ops(self):
        from tptr.jax import get_supported_ops
        ops = get_supported_ops()
        assert "add" in ops
        assert "matmul" in ops
        assert "relu" in ops

    def test_is_op_supported(self):
        from tptr.jax import is_op_supported
        assert is_op_supported("add")
        assert is_op_supported("relu")
        assert not is_op_supported("foo")

    def test_get_tpt_op_name(self):
        from tptr.jax import get_tpt_op_name
        assert get_tpt_op_name("add") == "add"
        assert get_tpt_op_name("dot") == "matmul"
        assert get_tpt_op_name("relu") == "relu"


class TestTptrJaxArray:
    """Tests for TptrJaxArray class."""

    def test_array_creation(self):
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((32, 64), "float32")
        assert arr.shape == (32, 64)
        assert arr.dtype == "float32"
        assert arr.size == 32 * 64

    def test_array_nbytes(self):
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((32, 64), "float32")
        assert arr.nbytes == 32 * 64 * 4

    def test_array_repr(self):
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((10, 10), "float32")
        r = repr(arr)
        assert "TptrJaxArray" in r
        assert "float32" in r

    def test_array_copy_to_host(self):
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((10,), "float32")
        data = arr.copy_to_host()
        assert len(data) == 10 * 4

    def test_array_copy_from_host(self):
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((10,), "float32")
        data = b'\x00' * 40
        arr.copy_from_host(data)  # Should not raise

    def test_array_to_numpy(self):
        pytest.importorskip("numpy")
        from tptr.jax import TptrJaxArray
        arr = TptrJaxArray((5, 5), "float32")
        np_array = arr.to_numpy()
        assert np_array.shape == (5, 5)

    def test_array_from_numpy(self):
        np = pytest.importorskip("numpy")
        from tptr.jax import TptrJaxArray
        np_array = np.ones((3, 3), dtype=np.float32)
        arr = TptrJaxArray.from_numpy(np_array)
        assert arr.shape == (3, 3)
        assert arr.dtype == "float32"


class TestJaxConversion:
    """Tests for JAX array conversion functions."""

    def test_jax_to_tptr(self):
        np = pytest.importorskip("numpy")
        from tptr.jax import TptrJaxArray, jax_to_tptr
        np_array = np.ones((4, 4), dtype=np.float32)
        tptr_arr = jax_to_tptr(np_array)
        assert isinstance(tptr_arr, TptrJaxArray)
        assert tptr_arr.shape == (4, 4)

