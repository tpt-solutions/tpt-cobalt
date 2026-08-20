"""Tests for tptr tensor module."""
from tptr.tensor import (
    TptrDType,
    TptrTensor,
    empty,
    float32,
    float64,
    full,
    int32,
    ones,
    zeros,
)


class TestTptrDType:
    def test_dtype_names(self):
        assert float32.name == "float32"
        assert float64.name == "float64"
        assert int32.name == "int32"

    def test_dtype_itemsize(self):
        assert float32.itemsize == 4
        assert float64.itemsize == 8
        assert int32.itemsize == 4

    def test_dtype_enum(self):
        assert isinstance(float32, TptrDType)
        assert TptrDType.FLOAT32.value == 2


class TestTptrTensor:
    def test_tensor_creation(self):
        t = TptrTensor((32, 64), float32)
        assert t.shape == (32, 64)
        assert t.ndim == 2
        assert t.dtype == float32
        assert t.size == 32 * 64
        assert t.nbytes == 32 * 64 * 4

    def test_tensor_1d(self):
        t = TptrTensor((100,), float32)
        assert t.shape == (100,)
        assert t.ndim == 1

    def test_tensor_3d(self):
        t = TptrTensor((2, 3, 4), float32)
        assert t.shape == (2, 3, 4)
        assert t.ndim == 3

    def test_tensor_is_valid(self):
        t = TptrTensor((10, 10), float32)
        assert t.is_valid

    def test_tensor_repr(self):
        t = TptrTensor((5, 5), float32)
        r = repr(t)
        assert "TptrTensor" in r
        assert "float32" in r

    def test_tensor_add(self):
        a = TptrTensor((3, 3), float32)
        b = TptrTensor((3, 3), float32)
        c = a + b
        assert isinstance(c, TptrTensor)
        assert c.shape == (3, 3)

    def test_tensor_mul(self):
        a = TptrTensor((3, 3), float32)
        b = TptrTensor((3, 3), float32)
        c = a * b
        assert isinstance(c, TptrTensor)

    def test_tensor_sub(self):
        a = TptrTensor((3, 3), float32)
        b = TptrTensor((3, 3), float32)
        c = a - b
        assert isinstance(c, TptrTensor)


class TestFactoryFunctions:
    def test_zeros(self):
        t = zeros((10, 20), float32)
        assert isinstance(t, TptrTensor)
        assert t.shape == (10, 20)

    def test_ones(self):
        t = ones((5, 5), float32)
        assert isinstance(t, TptrTensor)
        assert t.shape == (5, 5)

    def test_empty(self):
        t = empty((3, 4), float64)
        assert isinstance(t, TptrTensor)
        assert t.dtype == float64

    def test_full(self):
        t = full((2, 3), 1.0, float32)
        assert isinstance(t, TptrTensor)
        assert t.shape == (2, 3)

    def test_int_shape(self):
        t = zeros(10, float32)
        assert t.shape == (10,)


class TestBroadcast:
    def test_broadcast_same_shape(self):
        a = TptrTensor((3, 3), float32)
        b = TptrTensor((3, 3), float32)
        c = a + b
        assert c.shape == (3, 3)

    def test_broadcast_scalar(self):
        a = TptrTensor((3, 3), float32)
        c = a + 1.0
        assert c.shape == (3, 3)

