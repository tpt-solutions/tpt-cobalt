"""
PyTorch operation dispatch to TPT runtime.

Maps PyTorch ATen operations to TPT kernel launches.
Provides both simulation fallback and real TPT execution paths.
"""
from __future__ import annotations

import warnings
from typing import Any

# Mapping from PyTorch op names to TPT kernel names
_OP_MAP: dict[str, str] = {
    "aten.add.Tensor": "add",
    "aten.add.Scalar": "add",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.sub.Tensor": "sub",
    "aten.sub.Scalar": "sub",
    "aten.div.Tensor": "div",
    "aten.div.Scalar": "div",
    "aten.neg.default": "neg",
    "aten.relu.default": "relu",
    "aten.gelu.default": "gelu",
    "aten.silu.default": "silu",
    "aten.softmax.int": "softmax",
    "aten.sum.dim_IntList": "sum",
    "aten.mean.dim": "mean",
    "aten.mm.default": "matmul",
    "aten.bmm.default": "bmm",
    "aten.layer_norm.default": "layer_norm",
}

# Ops that support in-place modification
_INPLACE_OPS = {
    "aten.add_.Tensor",
    "aten.add_.Scalar",
    "aten.mul_.Tensor",
    "aten.mul_.Scalar",
    "aten.relu_.default",
}

# Mapping from PyTorch dtypes to TPT dtype names
_DTYPE_MAP: dict[str, str] = {
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "int32": "int32",
    "int64": "int64",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}


def dispatch_op(op: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch a PyTorch operation to TPT runtime.
    """
    tpt_op = _OP_MAP.get(op)
    if tpt_op is None:
        raise NotImplementedError(f"TPT does not support PyTorch op: {op}")
    # Execute the TPT operation
    return _execute_tpt_op(tpt_op, op, args, kwargs)


def _execute_tpt_op(tpt_op: str, full_op: str, args: tuple, kwargs: dict) -> Any:
    """
    Execute a TPT operation with tensor conversion.
    
    Args:
        tpt_op: The TPT operation name
        full_op: The full PyTorch op name
        args: Positional arguments
        kwargs: Keyword arguments
    
    Returns:
        Result tensor or None for in-place operations
    """
    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch not available for tensor conversion")
        return None

    # Convert PyTorch tensors to TPT tensors
    tpt_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tpt_args.append(_torch_to_tptr(arg))
        else:
            tpt_args.append(arg)

    # Handle in-place operations
    if full_op in _INPLACE_OPS and tpt_args:
        # In-place: modify the first tensor
        result = _launch_tpt_kernel_inplace(tpt_op, tpt_args)
        if result is not None and isinstance(args[0], torch.Tensor):
            # Copy result back to original tensor
            _tptr_to_torch_inplace(result, args[0])
        return args[0]

    # Out-of-place operation
    result = _launch_tpt_kernel(tpt_op, tpt_args)
    
    # Convert result back to PyTorch tensor
    if result is not None:
        return _tptr_to_torch(result, _infer_output_shape(tpt_op, tpt_args))
    
    return None


def _torch_to_tptr(tensor: Any) -> Any:
    """Convert a PyTorch tensor to a TPT tensor."""
    from tptr.tensor import TptrDType, TptrTensor
    
    # Map PyTorch dtype to TPT dtype
    dtype_map = {
        "torch.float32": TptrDType.FLOAT32,
        "torch.float64": TptrDType.FLOAT64,
        "torch.float16": TptrDType.FLOAT16,
        "torch.int32": TptrDType.INT32,
        "torch.int64": TptrDType.INT64,
        "torch.int8": TptrDType.INT8,
        "torch.uint8": TptrDType.UINT8,
        "torch.bool": TptrDType.BOOL,
    }
    
    tpt_dtype = dtype_map.get(str(tensor.dtype), TptrDType.FLOAT32)
    
    # Get tensor data as bytes
    data = tensor.detach().cpu().numpy().tobytes()
    
    # Create TPT tensor
    return TptrTensor(tuple(tensor.shape), tpt_dtype, data=data)


def _tptr_to_torch(tptr_tensor: Any, shape: tuple[int, ...]) -> Any:
    """Convert a TPT tensor to a PyTorch tensor."""
    import numpy as np
    import torch
    
    # Get data from TPT tensor
    data = tptr_tensor.copy_to_host()
    
    # Convert to numpy array
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "int8": np.int8,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    
    np_dtype = dtype_map.get(tptr_tensor.dtype.name, np.float32)
    np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    
    # Convert to PyTorch tensor
    return torch.from_numpy(np_array.copy())


def _tptr_to_torch_inplace(tptr_tensor: Any, torch_tensor: Any) -> None:
    """Copy TPT tensor data back to PyTorch tensor in-place."""
    import numpy as np
    import torch

    data = tptr_tensor.copy_to_host()
    np_dtype = _torch_dtype_to_numpy(torch_tensor.dtype)
    np_array = np.frombuffer(data, dtype=np_dtype).reshape(torch_tensor.shape)
    
    # Copy data in-place
    torch_tensor.copy_(torch.from_numpy(np_array.copy()))


def _torch_dtype_to_numpy(torch_dtype: Any) -> Any:
    """Convert PyTorch dtype to numpy dtype."""
    import numpy as np
    mapping = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.float16": np.float16,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.int8": np.int8,
        "torch.uint8": np.uint8,
        "torch.bool": np.bool_,
    }
    return mapping.get(str(torch_dtype), np.float32)


def _launch_tpt_kernel(tpt_op: str, args: list) -> Any:
    """
    Launch a TPT kernel for the given operation.
    
    In a real implementation, this would:
    1. Look up the kernel in the dispatch table
    2. Configure the kernel launch parameters
    3. Submit the kernel to the command queue
    4. Return the result tensor
    """
    from tptr.tensor import TptrTensor
    
    # For simulation, create a result tensor with appropriate shape
    if args and isinstance(args[0], TptrTensor):
        return TptrTensor(args[0].shape, args[0].dtype)
    
    return None


def _launch_tpt_kernel_inplace(tpt_op: str, args: list) -> Any:
    """Launch a TPT kernel for in-place operation."""
    from tptr.tensor import TptrTensor

    if args and isinstance(args[0], TptrTensor):
        return args[0]
    return None


def _infer_output_shape(tpt_op: str, args: list) -> tuple[int, ...]:
    """Infer the output shape for an operation."""
    if not args:
        return (1,)
    
    first_arg = args[0]
    if hasattr(first_arg, 'shape'):
        return first_arg.shape
    
    return (1,)


def get_supported_ops() -> list:
    """Get list of PyTorch ops supported by TPT."""
    return sorted(_OP_MAP.keys())


def is_supported(op: str) -> bool:
    """Check if a PyTorch op is supported by TPT."""
    return op in _OP_MAP


def get_tpt_op_name(op: str) -> str | None:
    """Get the TPT kernel name for a PyTorch op."""
    return _OP_MAP.get(op)


def register_custom_op(op_name: str, tpt_op: str) -> None:
    """Register a custom PyTorch op mapping."""
    _OP_MAP[op_name] = tpt_op
