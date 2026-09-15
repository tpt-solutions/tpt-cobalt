# PyTorch comparison protocol

The Phase 7 "PyTorch benchmark suite" is two halves that must be run the
same way to be comparable. This file fixes the recipe; numbers are only
meaningful when both halves run on the same machine, same build profile, and
same thread budget.

## Cobalt half

```sh
cargo run --release -p tpt-bench --bin tpt-bench-report -- 200   # wall-clock report
cargo bench -p tpt-bench                                          # criterion statistics
```

- Build profile: `--release` (opt-level 3, codegen-units default).
- Threads: leave at default, but **record** `RAYON_NUM_THREADS`/core count.
- Report iters: `200` (the suite's warmup rule is 10 untimed runs, then the
  timed loop — identical to the PyTorch side below).
- Determinism: kernels are seeded with a fixed LCG, so reports are
  reproducible for a given binary.

## PyTorch half

Same machine, `torch.cuda` disabled (Cobalt's suite is CPU-path; GPU
cross-device numbers are a separate WGPU/CUDA comparison). The reference
script (keep it outside this repo — no Python in the workspace):

```python
import torch, time

def bench(name, elements, iters, f):
    for _ in range(10): f()                      # warmup, matches tpt-bench
    t = time.perf_counter()
    for _ in range(iters): f()
    us = (time.perf_counter() - t) * 1e6 / iters
    print(f"| {name} | {elements} | {iters} | {us:.1f} |")

torch.set_num_threads(8)                          # record this

a64 = torch.randn(256, 256, dtype=torch.float64)
b64 = torch.randn(256, 256, dtype=torch.float64)
bench("matmul_256x256_f64", 256*256, 200, lambda: a64 @ b64)

x = torch.randn(128, 256, dtype=torch.float64, requires_grad=True)
w = torch.randn(256, 128, dtype=torch.float64, requires_grad=True)
def lin_fwd_bwd():
    y = x @ w
    y.sum().backward()
    x.grad = None; w.grad = None
bench("linear_128x256x128_fwd_bwd", 128*256, 200, lin_fwd_bwd)

blk = torch.nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    batch_first=True, dtype=torch.float64).eval()
inp = torch.randn(1, 8, 64, dtype=torch.float64)
with torch.no_grad():
    bench("transformer_block_b8_d64_fwd", 8*64, 200, lambda: blk(inp))

net = torch.nn.Sequential(
    torch.nn.Linear(8, 32, dtype=torch.float64), torch.nn.Tanh(),
    torch.nn.Linear(32, 4, dtype=torch.float64))
xs = torch.ones(16, 8, dtype=torch.float64)
ys = torch.ones(16, 4, dtype=torch.float64)
opt = torch.optim.AdamW(net.parameters(), lr=0.01)
def train_step():
    opt.zero_grad()
    loss = ((net(xs) - ys) ** 2).mean()
    loss.backward()
    opt.step()
bench("tpt_script_train_step_b16", 16*8, 200, train_step)
```

Differences that are **by design** and must be quoted when publishing:
- `train_step` on the Cobalt side includes interpreter dispatch overhead
  (the benchmark is the *language* path); the torch side is pure framework.
- Cobalt tensors are f64 host-side; the f16/i8 artifact paths (FPGA) are not
  benchmarked here.
- The transformer comparison is forward-only; torch's layer includes a
  LayerNorm placement difference (post-norm here matches tpt-ml's block).
