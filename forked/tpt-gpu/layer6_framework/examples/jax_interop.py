#!/usr/bin/env python3
"""
JAX interop example for tptr framework backends.

Demonstrates:
- Registering TPT JAX primitives
- Supported op mapping
- matmul / attention / conv2d / layer_norm via TPT primitives
- Autodiff (jax.grad) through TPT primitives
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tptr.jax


def main():
    print("=" * 60)
    print("TPT Framework Backends - JAX Interop Example")
    print("=" * 60)

    # 1. Check availability
    print("\n1. Checking JAX availability")
    available = tptr.jax.is_available()
    print(f"   JAX available: {available}")

    if not available:
        print(
            "\n   JAX is not installed. Install with:\n"
            "   pip install -e \"layer6_framework[jax]\"\n"
            "   Falling back to op-mapping demo only (no JAX import required)."
        )
    else:
        # 2. Register backend
        print("\n2. Registering TPT JAX primitives")
        success = tptr.jax.register_backend()
        print(f"   Registration: {'success' if success else 'failed'}")

    # 3. List supported ops
    print("\n3. Supported JAX operations")
    supported = tptr.jax.get_supported_ops()
    for op in supported:
        tpt_op = tptr.jax.get_tpt_op_name(op)
        print(f"   {op} -> {tpt_op}")

    # 4. Check specific ops
    print("\n4. Checking specific operations")
    for op in ["dot", "relu", "conv", "foo"]:
        supported = tptr.jax.is_op_supported(op)
        print(f"   {op}: {'supported' if supported else 'not supported'}")

    if not available:
        print("\n" + "=" * 60)
        print("JAX interop example completed (op-mapping only)!")
        print("=" * 60)
        return

    # 5. matmul / attention / conv2d / layer_norm via TPT primitives
    print("\n5. TPT primitives")
    import jax
    import jax.numpy as jnp
    from tptr.jax.ops import tpt_attention, tpt_conv2d, tpt_layer_norm_jax, tpt_matmul

    a = jnp.ones((3, 4))
    b = jnp.ones((4, 5))
    print(f"   tpt_matmul: {tpt_matmul(a, b).shape}")

    q = k = v = jnp.ones((1, 2, 4, 8))
    print(f"   tpt_attention: {tpt_attention(q, k, v).shape}")

    x = jnp.ones((1, 3, 8, 8))
    w = jnp.ones((16, 3, 3, 3))
    print(f"   tpt_conv2d: {tpt_conv2d(x, w, stride=1, padding=1).shape}")

    ln_x = jnp.ones((2, 4))
    print(f"   tpt_layer_norm_jax: {tpt_layer_norm_jax(ln_x, (4,)).shape}")

    # 6. Autodiff through TPT primitives
    print("\n6. Autodiff (jax.grad)")
    grad_fn = jax.grad(lambda a: tpt_matmul(a, jnp.eye(4)).sum())
    print(f"   d(tpt_matmul)/da shape: {grad_fn(a).shape}")

    print("\n" + "=" * 60)
    print("JAX interop example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
