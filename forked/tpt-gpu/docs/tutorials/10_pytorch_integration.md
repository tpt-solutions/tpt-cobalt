# Tutorial 10: PyTorch Integration

**Estimated Time:** 50 minutes  
**Prerequisites:** Tutorial 9, PyTorch basics

---

## Introduction

TPT GPU integrates with PyTorch through custom autograd functions and device dispatch, enabling seamless GPU acceleration for ML workloads.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch Application                           │
├─────────────────────────────────────────────────────────────────┤
│  import torch                                                    │
│  import tptr.pytorch as tpt                                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TptrTorchDevice                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │  Autograd   │  │   Tensor    │  │   Stream    │      │  │
│  │  │  Functions  │  │   Wrapper   │  │   Manager   │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    TPT Runtime (Layer 4)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
cd layer6_framework
pip install -e ".[dev]"
```

Verify:
```python
import torch
import tptr.pytorch as tpt
print(tpt.is_available())
```

---

## Device Management

```python
import torch
import tptr.pytorch as tpt

# Check availability and register TPT as a PyTorch backend device
if tpt.is_available():
    tpt.register_backend()
    device = tpt.get_tpt_device("tpt:0")  # not tpt.device(0)
    print(f"Device: {device.name}")
    print(f"Memory: {device.total_memory / 1024**3:.1f} GB")

# There is no tpt.device_context() — pass the device string directly
tensor = torch.randn(1024, 1024, device="tpt:0")
```

---

## Tensor Operations

```python
import torch
import tptr.pytorch as tpt

# Create tensors on TPT device
a = torch.randn(1024, 512, device='tpt')
b = torch.randn(512, 768, device='tpt')

# Standard PyTorch operations work
c = torch.matmul(a, b)
d = torch.relu(c)
e = torch.softmax(d, dim=-1)

# Convert to CPU
cpu_tensor = e.cpu()
```

---

## Custom Autograd Functions

`tptr.pytorch` already ships `TptAddFunction`/`TptMulFunction`/`TptMatmulFunction`/`TptReluFunction`
(subclasses of the lightweight `TptFunction` base, not `torch.autograd.Function`) plus their
functional wrappers — there is no `tpt.add`/`tpt.matmul` on the top-level module:

```python
import torch
import tptr.pytorch as tpt

a = torch.randn(1024, 512, requires_grad=True)
b = torch.randn(512, 768, requires_grad=True)

# Functional wrappers re-exported from tptr.pytorch.autograd
c = tpt.tpt_matmul(a, b)
d = tpt.tpt_relu(c)
```

To write your own, subclass `tptr.pytorch.autograd.TptFunction`:

```python
from tptr.pytorch.autograd import TptFunction

class MyCustomFunction(TptFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

result = MyCustomFunction.apply(a, b)
```

Note: `TptFunction.apply()` just calls `forward()` directly (`cls.forward(None, *args, **kwargs)`)
— it does not register with PyTorch's autograd graph the way `torch.autograd.Function.apply` does,
so `backward()` on subclasses is not wired into `.backward()` calls on the resulting tensor today.

---

## Stream Management

```python
from tptr.pytorch.stream import TptStream, TptEvent, StreamContext

# Class is TptStream, constructor args are (device_index, priority)
stream = TptStream(device_index=0, priority="high")

# Use stream for operations
with StreamContext(stream):
    a = torch.randn(1024, 1024, device="tpt:0")
    b = torch.randn(1024, 1024, device="tpt:0")
    c = torch.matmul(a, b)

# Synchronize
stream.synchronize()

# Events for cross-stream sync — record()/wait() take the stream as an argument;
# there is no stream.record_event()/stream.wait_event()
event = TptEvent()
event.record(stream)

other_stream = TptStream(device_index=0)
event.wait(other_stream)

# TptStream also has wait_stream() to block until another stream drains:
other_stream.wait_stream(stream)
```

---

## HuggingFace Integration

```python
from tptr.pytorch.hf_bridge import TptHFModel

# TptHFModel takes a model *name* (it loads the model and tokenizer itself),
# not an already-constructed model instance.
bridge = TptHFModel("bert-base-uncased", device="tpt:0")

# predict()/embed() run tokenization + inference together — there is no
# tpt_model(input_ids, attention_mask) call signature.
result = bridge.predict("Hello world")
print(result["logits"].shape)

embedding = bridge.embed("Hello world")
```

Note: `load_model()` currently maps a `"tpt:*"` device string onto `"cuda:{idx}"` if CUDA is
available, or `"cpu"` otherwise — there is no dedicated TPT `torch.device` backend used here yet.

---

## Example: Training Loop

```python
import torch
import torch.nn as nn
import tptr.pytorch as tpt

class SimpleModel(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

def train():
    device = "tpt:0"  # tpt.device(0) does not exist; use the device string directly
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        # Generate dummy data
        x = torch.randn(32, 128, 512, device=device)
        y = torch.randn(32, 128, 512, device=device)
        
        # Forward
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
```

---

## Exercises

1. **Custom Op**: Implement a custom activation function with autograd support
2. **Mixed Precision**: Add AMP (Automatic Mixed Precision) support
3. **Multi-GPU**: Extend training to multiple TPT devices

---

## Summary

- ✅ Device management with PyTorch integration
- ✅ Tensor operations on TPT device
- ✅ Custom autograd functions for backward pass
- ✅ Stream and event management
- ✅ HuggingFace model integration

**Next:** [Tutorial 11: TPT Script Basics](11_tpt_script_basics.md)
