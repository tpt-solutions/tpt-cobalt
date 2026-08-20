# Distributed Training Examples

Two distributed training strategies implemented in TPT Script.

---

## FSDP — Fully Sharded Data Parallel (`fsdp_8gpu.tpts`)

**When to use:** largest models, memory-limited single-GPU training.

FSDP shards **parameters + gradients + optimizer states** evenly across all
GPUs.  For an 8-GPU setup this gives ~8× peak VRAM reduction vs. a single
GPU, enabling models that would not otherwise fit.

### Key operations used

| Operation | Description |
|-----------|-------------|
| `tpt.dist.init(strategy="fsdp", ...)` | Initialise NCCL/RCCL process group |
| `tpt.dist.scatter(x, dim=0, ...)` | Split global batch into per-GPU shards |
| `tpt.dist.all_gather(params, layer)` | Pull full weight tensor from all peers |
| `tpt.dist.free_unshard(params)` | Release all-gathered buffer to reclaim VRAM |
| `tpt.dist.reduce_scatter(grads, op="mean", ...)` | Sum + scatter gradient fragments |
| `tpt.dist.parallel_adamw(...)` | Each GPU updates only its own shard |
| `tpt.dist.barrier()` | Global synchronisation point |
| `tpt.dist.gather_model(params, root_rank=0)` | Collect full model for checkpointing |
| `tpt.dist.shutdown()` | Tear down the process group |

### Annotations required

```tpt
@distributed_strategy("fsdp")
@distributed_devices(8)
@requires_gpu(true)   // or false for host orchestration
```

### Model configuration (this example)

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2-style decoder |
| `d_model` | 1 024 |
| `d_ff` | 4 096 |
| `n_heads` | 16 |
| `n_layers` | 8 |
| `seq_len` | 2 048 |
| Micro-batch per GPU | 4 |
| Global batch | 32 (4 × 8) |

---

## Pipeline Parallel — GPipe schedule (`pipeline_parallel.tpts`)

**When to use:** model too tall (many layers) for single-GPU memory, or when
you want different hardware for different layers.

The model is split by **layer depth** across 4 GPUs (stages).  A GPipe
schedule feeds multiple micro-batches through the pipeline to overlap
computation across stages and reduce the "bubble" idle time.

### Layout

```
GPU 0 — embedding + layers 0–1   (stage 0)
GPU 1 — layers 2–3               (stage 1)
GPU 2 — layers 4–5               (stage 2)
GPU 3 — layers 6–7 + LM head     (stage 3)
```

### Key operations used

| Operation | Description |
|-----------|-------------|
| `tpt.dist.init(strategy="pipeline_parallel", ...)` | Initialise P2P process group |
| `tpt.dist.send_recv(tensor, src, dst)` | Send activations between adjacent GPUs |
| `tpt.dist.accumulate_grads(grads, n)` | Accumulate over N micro-batches |
| `tpt.dist.pipeline_adamw(...)` | Per-stage parameter update |
| `tpt.dist.gather_pipeline_model(...)` | Collect full model for checkpointing |

### Annotations required

```tpt
@distributed_strategy("pipeline_parallel")
@distributed_devices(4)
```

### GPipe bubble ratio

With 4 stages and 4 micro-batches the pipeline bubble is 43 %.
Using 8 micro-batches reduces it to 27 %.  A rule of thumb is
`n_microbatches ≥ 2 × n_stages`.

---

## Running locally (simulation mode)

In the beta, distributed annotations are **parsed and type-checked** but
actual multi-GPU dispatch requires the `tptr` runtime to be compiled with
NCCL or RCCL support and real hardware.  You can still type-check:

```bash
tpt check examples/distributed/fsdp_8gpu.tpts
tpt check examples/distributed/pipeline_parallel.tpts
```

## Combining FSDP + pipeline parallelism

You can combine both strategies (e.g., FSDP within each pipeline stage) by
applying both annotations and using `tpt.dist.init` with
`strategy="fsdp_pipeline"`.  See the `todo.md` entry for the planned
`tpt.dist` standard-library module.
