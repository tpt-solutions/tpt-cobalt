"""JAX primitive dispatch for TPT GPU operations.

Registers JAX primitives for the core TPT operations:
  - tpt_matmul    (GEMM / matrix multiply)
  - tpt_attention (scaled dot-product attention)
  - tpt_conv2d    (2-D convolution)

Each primitive:
  1. Defines a ``jax.core.Primitive`` with an ``impl`` rule (actual execution).
  2. Registers an ``abstract_eval`` rule so JAX can trace through it.
  3. Registers JVP (forward-mode) and VJP (reverse-mode) differentiation rules.
  4. Registers an XLA/MLIR lowering rule for compilation via ``jax.jit``.

Layer norm is also provided as a ``jax.custom_vjp`` function.

Execution is XLA/jnp-based simulation for now, mirroring the software-fallback
convention used by tptr.pytorch.ops and the layer5 kernel fallbacks — wiring
these up to real TPT hardware dispatch (paralleling tptr._ffi) is future work.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import mlir

try:
    # JAX >= 0.4.29 moved Primitive out of the (now deprecated) jax.core.
    from jax.extend.core import Primitive
except ImportError:
    from jax.core import Primitive

# ---------------------------------------------------------------------------
# Primitive: tpt_matmul  (A @ B)
# ---------------------------------------------------------------------------

tpt_matmul_p = Primitive("tpt_matmul")
tpt_matmul_p.multiple_results = False


def tpt_matmul_jax(a: jax.Array, b: jax.Array) -> jax.Array:
    """Matrix multiply routed through the TPT primitive."""
    return tpt_matmul_p.bind(a, b)


@tpt_matmul_p.def_impl
def _tpt_matmul_impl(a, b):
    return jnp.matmul(a, b)


@tpt_matmul_p.def_abstract_eval
def _tpt_matmul_abstract(a, b):
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"tpt_matmul requires ≥2-D arrays; got {a.ndim}D and {b.ndim}D")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"tpt_matmul: inner dims mismatch — A has shape {a.shape}, B has shape {b.shape}"
        )
    out_shape = (*a.shape[:-1], b.shape[-1])
    return core.ShapedArray(out_shape, a.dtype)


# JVP: d(A @ B) = dA @ B + A @ dB
def _tpt_matmul_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    primal_out = tpt_matmul_jax(a, b)
    tangent_out = jnp.zeros_like(primal_out)
    if not isinstance(da, jax.interpreters.ad.Zero):
        tangent_out = tangent_out + tpt_matmul_jax(da, b)
    if not isinstance(db, jax.interpreters.ad.Zero):
        tangent_out = tangent_out + tpt_matmul_jax(a, db)
    return primal_out, tangent_out


jax.interpreters.ad.primitive_jvps[tpt_matmul_p] = _tpt_matmul_jvp


# VJP registered via custom_vjp on the public-facing wrapper so it composes
# correctly with jax.grad / jax.value_and_grad.
@jax.custom_vjp
def tpt_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return tpt_matmul_jax(a, b)


def _tpt_matmul_fwd(a, b):
    return tpt_matmul(a, b), (a, b)


def _tpt_matmul_bwd(res, g):
    a, b = res
    # dL/dA = g @ B^T,  dL/dB = A^T @ g
    da = tpt_matmul_jax(g, b.swapaxes(-1, -2))
    db = tpt_matmul_jax(a.swapaxes(-1, -2), g)
    return da, db


tpt_matmul.defvjp(_tpt_matmul_fwd, _tpt_matmul_bwd)


# XLA lowering — falls through to jnp.matmul for compiler portability
def _tpt_matmul_lowering(ctx, a, b):
    return mlir.lower_fun(jnp.matmul, multiple_results=False)(ctx, a, b)


mlir.register_lowering(tpt_matmul_p, _tpt_matmul_lowering)


# ---------------------------------------------------------------------------
# GEMM with alpha/beta scalars
# ---------------------------------------------------------------------------

def tpt_gemm_jax(a, b, alpha: float = 1.0, beta: float = 0.0, c=None):
    """alpha * A @ B + beta * C."""
    result = alpha * tpt_matmul_jax(a, b)
    if c is not None:
        result = result + beta * c
    return result


# ---------------------------------------------------------------------------
# Primitive: tpt_attention  (scaled dot-product attention)
# ---------------------------------------------------------------------------

tpt_attention_p = Primitive("tpt_attention")
tpt_attention_p.multiple_results = False


def tpt_attention_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
    scale: float | None = None,
) -> jax.Array:
    """Scaled dot-product attention routed through the TPT primitive."""
    if mask is None:
        mask = jnp.zeros((), dtype=q.dtype)  # sentinel: no mask
    s = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    return tpt_attention_p.bind(q, k, v, mask, scale=s)


@tpt_attention_p.def_impl
def _tpt_attention_impl(q, k, v, mask, *, scale):
    scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
    if mask.ndim > 0:
        scores = scores + mask
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)


@tpt_attention_p.def_abstract_eval
def _tpt_attention_abstract(q, k, v, mask, *, scale):
    # Output shape matches Q (same batch/head dims, seq_q x d_v)
    out_shape = (*q.shape[:-1], v.shape[-1])
    return core.ShapedArray(out_shape, q.dtype)


# VJP via custom_vjp on the public wrapper
@jax.custom_vjp
def tpt_attention(q, k, v, mask=None, scale=None):
    return tpt_attention_jax(q, k, v, mask, scale)


def _tpt_attention_fwd(q, k, v, mask, scale):
    s = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    scores = tpt_matmul_jax(q, k.swapaxes(-1, -2)) * s
    if mask is not None:
        scores = scores + mask
    weights = jax.nn.softmax(scores, axis=-1)
    out = tpt_matmul_jax(weights, v)
    return out, (q, k, v, weights, s)


def _tpt_attention_bwd(res, g):
    q, k, v, weights, s = res
    # dL/dV = weights^T @ g
    dv = tpt_matmul_jax(weights.swapaxes(-1, -2), g)
    # dL/d(weights) = g @ V^T
    dw = tpt_matmul_jax(g, v.swapaxes(-1, -2))
    # Softmax VJP
    dscores = weights * (dw - (dw * weights).sum(axis=-1, keepdims=True))
    dscores = dscores * s
    # dL/dQ = dscores @ K,  dL/dK = dscores^T @ Q
    dq = tpt_matmul_jax(dscores, k)
    dk = tpt_matmul_jax(dscores.swapaxes(-1, -2), q)
    return dq, dk, dv, None, None  # no grad for mask or scale


tpt_attention.defvjp(_tpt_attention_fwd, _tpt_attention_bwd)


# XLA lowering
def _tpt_attention_lowering(ctx, q, k, v, mask, *, scale):
    def _ref(q, k, v, mask):
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        if mask.ndim > 0:
            scores = scores + mask
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(weights, v)
    return mlir.lower_fun(_ref, multiple_results=False)(ctx, q, k, v, mask)


mlir.register_lowering(tpt_attention_p, _tpt_attention_lowering)


# ---------------------------------------------------------------------------
# Primitive: tpt_conv2d
# ---------------------------------------------------------------------------

tpt_conv2d_p = Primitive("tpt_conv2d")
tpt_conv2d_p.multiple_results = False


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


def tpt_conv2d_jax(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2-D convolution routed through the TPT primitive."""
    b_arr = bias if bias is not None else jnp.zeros((), dtype=x.dtype)
    return tpt_conv2d_p.bind(
        x, weight, b_arr,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        groups=int(groups),
        has_bias=bias is not None,
    )


@tpt_conv2d_p.def_impl
def _tpt_conv2d_impl(x, weight, bias, *, stride, padding, dilation, groups, has_bias):
    pH, pW = padding
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
    out = jax.lax.conv_general_dilated(
        x_padded,
        weight,
        window_strides=stride,
        padding=((0, 0), (0, 0)),
        rhs_dilation=dilation,
        feature_group_count=groups,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    if has_bias:
        out = out + bias[None, :, None, None]
    return out


@tpt_conv2d_p.def_abstract_eval
def _tpt_conv2d_abstract(x, weight, bias, *, stride, padding, dilation, groups, has_bias):
    N, _C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    pH, pW = padding
    sH, sW = stride
    dH, dW = dilation
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return core.ShapedArray((N, C_out, H_out, W_out), x.dtype)


# VJP via custom_vjp. stride/padding/dilation/groups are static op-config,
# not differentiable data, so they're declared nondiff_argnums — otherwise
# JAX wraps them in tracers too and _tpt_conv2d_impl's `pH, pW = padding`
# unpacking (and tpt_conv2d_jax's int/tuple coercion) breaks under grad/jit.
@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def tpt_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return tpt_conv2d_jax(x, weight, bias, stride, padding, dilation, groups)


def _tpt_conv2d_fwd(x, weight, bias, stride, padding, dilation, groups):
    out = tpt_conv2d_jax(x, weight, bias, stride, padding, dilation, groups)
    return out, (x, weight, bias)


def _tpt_conv2d_bwd(stride, padding, dilation, groups, res, g):
    x, weight, bias = res
    has_bias = bias is not None
    b_arr = bias if has_bias else jnp.zeros((), dtype=x.dtype)
    # Differentiate the raw jnp/lax implementation directly — tpt_conv2d_p
    # has no jvp/transpose rule of its own (unlike tpt_matmul_p), so nesting
    # jax.vjp around a call that binds the primitive would fail.
    def fwd(x, weight, bias):
        return _tpt_conv2d_impl(x, weight, bias,
                                stride=_pair(stride), padding=_pair(padding),
                                dilation=_pair(dilation), groups=groups, has_bias=has_bias)
    _, vjp_fn = jax.vjp(fwd, x, weight, b_arr)
    dx, dw, db = vjp_fn(g)
    return dx, dw, (db if has_bias else None)


tpt_conv2d.defvjp(_tpt_conv2d_fwd, _tpt_conv2d_bwd)


# XLA lowering
def _tpt_conv2d_lowering(ctx, x, weight, bias, *, stride, padding, dilation, groups, has_bias):
    def _ref(x, weight, bias):
        return _tpt_conv2d_impl(x, weight, bias,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, has_bias=has_bias)
    return mlir.lower_fun(_ref, multiple_results=False)(ctx, x, weight, bias)


mlir.register_lowering(tpt_conv2d_p, _tpt_conv2d_lowering)


# ---------------------------------------------------------------------------
# Layer norm (custom_vjp, no separate primitive needed)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def tpt_layer_norm_jax(x, normalized_shape, weight=None, bias=None, eps: float = 1e-5):
    """Layer normalisation with optional affine transform."""
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    xn = (x - mean) / jnp.sqrt(var + eps)
    if weight is not None:
        xn = xn * weight
    if bias is not None:
        xn = xn + bias
    return xn


def _layer_norm_fwd(x, normalized_shape, weight, bias, eps):
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    inv_std = 1.0 / jnp.sqrt(var + eps)
    xn = (x - mean) * inv_std
    out = xn
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out, (x, mean, inv_std, weight, axes)


def _layer_norm_bwd(res, g):
    x, mean, inv_std, weight, axes = res
    xn = (x - mean) * inv_std
    if weight is not None:
        dw = (g * xn).sum(axis=tuple(i for i in range(g.ndim) if i not in axes))
        g = g * weight
    else:
        dw = None
    db = g.sum(axis=tuple(i for i in range(g.ndim) if i not in axes)) if weight is not None else None
    N = math.prod(x.shape[a] for a in axes)
    dx = (1.0 / N) * inv_std * (
        N * g
        - g.sum(axis=axes, keepdims=True)
        - xn * (g * xn).sum(axis=axes, keepdims=True)
    )
    return dx, None, dw, db, None


tpt_layer_norm_jax.defvjp(_layer_norm_fwd, _layer_norm_bwd)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_jax_primitives() -> None:
    """Ensure all TPT JAX primitives are registered (idempotent)."""
    # Primitives are registered at import time above; this function exists
    # as an explicit hook for initialisation order in user code.
