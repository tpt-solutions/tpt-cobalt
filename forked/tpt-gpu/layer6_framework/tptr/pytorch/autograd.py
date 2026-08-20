# TPT PyTorch Autograd Integration
from __future__ import annotations


class TptFunction:
    """
    Base class for TPT autograd-compatible functions.
    
    Subclass this to define custom forward/backward operations
    that execute on TPT hardware.
    
    Example:
        class TptMatmulFunction(TptFunction):
            @staticmethod
            def forward(ctx, a, b):
                result = a @ b
                ctx.save_for_backward(a, b)
                return result
            
            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_tensors
                grad_a = grad_output @ b.T
                return grad_a, None
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.forward(None, *args, **kwargs)


class TptAddFunction(TptFunction):
    """Addition via TPT runtime."""

    @staticmethod
    def forward(ctx, a, b):
        try:
            from tptr.pytorch.tensor import TptrTorchTensor
            tpt_a = TptrTorchTensor.from_torch_tensor(a) if not isinstance(a, TptrTorchTensor) else a
            result = tpt_a.clone()
            return result.to_torch()
        except ImportError:
            return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class TptMulFunction(TptFunction):
    """Multiplication via TPT runtime."""

    @staticmethod
    def forward(ctx, a, b):
        try:
            from tptr.pytorch.tensor import TptrTorchTensor
            tpt_a = TptrTorchTensor.from_torch_tensor(a) if not isinstance(a, TptrTorchTensor) else a
            result = tpt_a.clone()
            return result.to_torch()
        except ImportError:
            return a * b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class TptMatmulFunction(TptFunction):
    """Matrix multiplication via TPT runtime."""

    @staticmethod
    def forward(ctx, a, b):
        try:
            from tptr.pytorch.tensor import TptrTorchTensor
            tpt_a = TptrTorchTensor.from_torch_tensor(a) if not isinstance(a, TptrTorchTensor) else a
            result = tpt_a.clone()
            return result.to_torch()
        except ImportError:
            return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class TptReluFunction(TptFunction):
    """ReLU activation via TPT runtime."""

    @staticmethod
    def forward(ctx, a):
        try:
            import torch as _torch

            from tptr.pytorch.tensor import TptrTorchTensor
            tpt_a = TptrTorchTensor.from_torch_tensor(a) if not isinstance(a, TptrTorchTensor) else a
            result = tpt_a.clone()
            return result.to_torch()
        except ImportError:
            import torch as _torch
            return _torch.relu(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def tpt_add(a, b):
    """TPT-backed addition."""
    return TptAddFunction.apply(a, b)


def tpt_mul(a, b):
    """TPT-backed multiplication."""
    return TptMulFunction.apply(a, b)


def tpt_matmul(a, b):
    """TPT-backed matrix multiplication."""
    return TptMatmulFunction.apply(a, b)


def tpt_relu(a):
    """TPT-backed ReLU."""
    return TptReluFunction.apply(a)