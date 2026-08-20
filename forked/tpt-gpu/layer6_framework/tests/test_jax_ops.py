"""Tests for the TPT JAX primitives (tptr.jax.ops)."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import numpy as np
from tptr.jax.ops import (
    tpt_attention,
    tpt_attention_jax,
    tpt_conv2d,
    tpt_conv2d_jax,
    tpt_gemm_jax,
    tpt_layer_norm_jax,
    tpt_matmul,
    tpt_matmul_jax,
)

# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

class TestTptMatmul:
    def test_basic(self):
        a = jnp.ones((3, 4))
        b = jnp.ones((4, 5))
        out = tpt_matmul_jax(a, b)
        assert out.shape == (3, 5)
        np.testing.assert_allclose(out, np.full((3, 5), 4.0), atol=1e-5)

    def test_identity(self):
        a = jnp.eye(4)
        b = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        out = tpt_matmul_jax(a, b)
        np.testing.assert_allclose(out, b, atol=1e-5)

    def test_shapes_preserved(self):
        a = jnp.zeros((7, 3))
        b = jnp.zeros((3, 11))
        assert tpt_matmul_jax(a, b).shape == (7, 11)

    def test_abstract_eval_shape_mismatch(self):
        # Eager calls run tpt_matmul_p's impl directly (jnp.matmul -> TypeError);
        # traced/jitted calls go through _tpt_matmul_abstract's ValueError instead.
        with pytest.raises((ValueError, TypeError)):
            tpt_matmul_jax(jnp.ones((3, 4)), jnp.ones((5, 6)))

    def test_jit_compatible(self):
        f = jax.jit(tpt_matmul_jax)
        a = jnp.ones((2, 3))
        b = jnp.ones((3, 2))
        out = f(a, b)
        assert out.shape == (2, 2)

    def test_grad_wrt_a(self):
        def f(a):
            return tpt_matmul(a, jnp.eye(3)).sum()
        a = jnp.ones((3, 3))
        g = jax.grad(f)(a)
        assert g.shape == (3, 3)
        np.testing.assert_allclose(g, jnp.ones((3, 3)), atol=1e-5)

    def test_grad_wrt_b(self):
        def f(b):
            return tpt_matmul(jnp.eye(3), b).sum()
        b = jnp.ones((3, 3))
        g = jax.grad(f)(b)
        assert g.shape == (3, 3)


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------

class TestTptGemm:
    def test_alpha_only(self):
        a = jnp.eye(3)
        b = jnp.full((3, 3), 2.0)
        out = tpt_gemm_jax(a, b, alpha=3.0)
        np.testing.assert_allclose(out, jnp.full((3, 3), 6.0), atol=1e-5)

    def test_alpha_beta(self):
        a = jnp.eye(2)
        b = jnp.eye(2)
        c = jnp.full((2, 2), 10.0)
        out = tpt_gemm_jax(a, b, alpha=1.0, beta=0.5, c=c)
        # 1*I + 0.5*[[10,10],[10,10]] = [[6,5],[5,6]]
        np.testing.assert_allclose(out[0, 0], 6.0, atol=1e-5)
        np.testing.assert_allclose(out[0, 1], 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestTptAttention:
    def _qkv(self, batch=1, heads=2, seq=4, d=8):
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (batch, heads, seq, d))
        k = jax.random.normal(key, (batch, heads, seq, d))
        v = jax.random.normal(key, (batch, heads, seq, d))
        return q, k, v

    def test_output_shape(self):
        q, k, v = self._qkv()
        out = tpt_attention_jax(q, k, v)
        assert out.shape == q.shape

    def test_scale_default(self):
        q, k, v = self._qkv(d=16)
        out = tpt_attention_jax(q, k, v)
        assert out.shape == q.shape

    def test_jit_compatible(self):
        q, k, v = self._qkv()
        f = jax.jit(tpt_attention_jax)
        out = f(q, k, v)
        assert out.shape == q.shape

    def test_grad_wrt_q(self):
        q, k, v = self._qkv(seq=2, d=4)
        def f(q):
            return tpt_attention(q, k, v).sum()
        g = jax.grad(f)(q)
        assert g.shape == q.shape

    def test_grad_wrt_v(self):
        q, k, v = self._qkv(seq=2, d=4)
        def f(v):
            return tpt_attention(q, k, v).sum()
        g = jax.grad(f)(v)
        assert g.shape == v.shape


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class TestTptConv2d:
    def test_output_shape_no_padding(self):
        x = jnp.ones((2, 3, 8, 8))
        w = jnp.ones((16, 3, 3, 3))
        out = tpt_conv2d_jax(x, w, stride=1, padding=0)
        assert out.shape == (2, 16, 6, 6)

    def test_output_shape_with_padding(self):
        x = jnp.ones((1, 1, 4, 4))
        w = jnp.ones((1, 1, 3, 3))
        out = tpt_conv2d_jax(x, w, stride=1, padding=1)
        assert out.shape == (1, 1, 4, 4)

    def test_identity_conv(self):
        x = jnp.arange(9, dtype=jnp.float32).reshape(1, 1, 3, 3)
        w = jnp.zeros((1, 1, 3, 3)).at[0, 0, 1, 1].set(1.0)
        out = tpt_conv2d_jax(x, w, stride=1, padding=1)
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_jit_compatible(self):
        x = jnp.ones((1, 3, 4, 4))
        w = jnp.ones((8, 3, 3, 3))
        # stride/padding are static op-config, not traced data — must be
        # marked static under jit like any other non-array primitive kwarg.
        f = jax.jit(tpt_conv2d_jax, static_argnames=("stride", "padding"))
        out = f(x, w, stride=1, padding=1)
        assert out.shape == (1, 8, 4, 4)

    def test_grad_wrt_x(self):
        x = jnp.ones((1, 1, 4, 4))
        w = jnp.ones((1, 1, 3, 3))
        def f(x):
            return tpt_conv2d(x, w, stride=1, padding=1).sum()
        g = jax.grad(f)(x)
        assert g.shape == x.shape


# ---------------------------------------------------------------------------
# Layer norm
# ---------------------------------------------------------------------------

class TestTptLayerNorm:
    def test_zero_mean_unit_var(self):
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        out = tpt_layer_norm_jax(x, (4,))
        means = out.mean(axis=-1)
        vars_ = out.var(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(vars_, 1.0, atol=1e-4)

    def test_affine(self):
        x = jnp.ones((2, 4))
        w = jnp.full((4,), 2.0)
        b = jnp.full((4,), 1.0)
        out = tpt_layer_norm_jax(x, (4,), weight=w, bias=b)
        # After layer norm of all-ones, xn=0, weight=2, bias=1 -> output=1
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_grad(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 8))
        def f(x):
            return tpt_layer_norm_jax(x, (8,)).sum()
        g = jax.grad(f)(x)
        assert g.shape == x.shape
