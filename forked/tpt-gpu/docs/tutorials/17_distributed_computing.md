# Tutorial 17: Distributed Computing

**Estimated Time:** 70 minutes  
**Prerequisites:** Tutorial 15

---

## Introduction

This tutorial covers multi-GPU training with FSDP and pipeline parallelism.

---

## Fully Sharded Data Parallel (FSDP)

```tpts
@doc("Train with FSDP across 8 GPUs")
@requires_gpu(true)
@distributed(strategy="fsdp", devices=8)
fn train_fsdp(
    model: Model,
    data: DataLoader,
    lr: f32,
) -> f32 {
    let optimizer = tpt.optim.Adam(lr=lr)
    let mut total_loss = 0.0
    
    for batch in data {
        let logits = model.forward(batch.input)
        let loss = tpt.cross_entropy(logits, batch.labels)
        loss.backward()
        
        optimizer.step(model)
        optimizer.zero_grad(model)
        total_loss = total_loss + loss
    }
    
    return total_loss
}
```

### FSDP Sharding

```rust
pub struct FsdpShard {
    device_id: usize,
    num_devices: usize,
    parameters: Vec<Tensor>,
    gradients: Vec<Tensor>,
}

impl FsdpShard {
    pub fn all_reduce_gradients(&mut self) {
        for grad in &mut self.gradients {
            self.comm.all_reduce(grad, ReduceOp::Sum);
            grad.div_scalar(self.num_devices as f32);
        }
    }
}
```

---

## Pipeline Parallelism

```tpts
@doc("Train with pipeline parallelism")
@requires_gpu(true)
@distributed(strategy="pipeline", stages=4)
fn train_pipeline(
    model: Model,
    data: DataLoader,
    lr: f32,
) -> f32 {
    let micro_batch_size = data.batch_size / 4
    let optimizer = tpt.optim.Adam(lr=lr)
    
    for batch in data {
        let micro_batches = batch.split(micro_batch_size)
        let logits = pipeline_forward(model, micro_batches)
        let loss = tpt.cross_entropy(logits, batch.labels)
        pipeline_backward(model, loss)
        optimizer.step(model)
        optimizer.zero_grad(model)
    }
}
```

---

## Communication Primitives

The module is `tpt.dist` (not `tpt.distributed`), and its operation list
(`crates/tpt-gpu-script-core/src/modules/stdlib.rs`) has `scatter`, not `reduce_scatter`:

```tpts
// All-reduce: Sum across all devices
let result = tpt.dist.all_reduce(tensor, op="sum")

// Broadcast: Send from src to all devices
let result = tpt.dist.broadcast(tensor, src=0)

// Scatter: distribute chunks of a tensor across devices
let result = tpt.dist.scatter(tensor, src=0)

// All-gather: Gather from all devices
let result = tpt.dist.all_gather(tensor)
```

`tpt.dist` also lists `init_process_group`, `destroy_process_group`, `get_rank`,
`get_world_size`, `barrier`, `send`, `recv`, `DistributedDataParallel`, and
`FullyShardedDataParallel`.

---

## Example: 8-GPU Training

```tpts
import tpt

@doc("Train transformer on 8 GPUs with FSDP")
@requires_gpu(true)
@distributed(strategy="fsdp", devices=8)
fn train_8gpu() {
    let model = Model::transformer(
        d_model=1024, heads=16, d_ff=4096, num_layers=24, vocab_size=50000,
    )
    
    let data = DataLoader(path="data/train.bin", batch_size=32, seq_length=512)
    let optimizer = tpt.optim.Adam(lr=1e-4)
    model.set_optimizer(optimizer)
    
    for epoch in 0..10 {
        let mut total_loss = 0.0
        let mut count = 0
        
        for batch in data {
            let logits = model.forward(batch.input)
            let loss = tpt.cross_entropy(logits, batch.labels)
            loss.backward()
            optimizer.step(model)
            optimizer.zero_grad(model)
            total_loss = total_loss + loss
            count = count + 1
        }
        
        let avg_loss = total_loss / tpt.cast(count, dtype=f32)
        print(f"Epoch {epoch}, Loss: {avg_loss}")
    }
}
```

---

## Exercises

1. **FSDP**: Implement manual gradient all-reduce
2. **Pipeline**: Design a custom pipeline schedule
3. **Mixed**: Combine data and model parallelism

---

## Summary

- ✅ FSDP: Fully Sharded Data Parallel training
- ✅ Pipeline parallelism with micro-batching
- ✅ Communication primitives: all-reduce, broadcast, etc.
- ✅ Multi-GPU training example

**End of Tutorial Series**
