# Tutorial 15: Building a Model

**Estimated Time:** 75 minutes  
**Prerequisites:** Tutorial 14

---

## Introduction

This tutorial builds a complete transformer model in TPT Script.

---

## Multi-Head Attention

```tpts
import tpt

type Batch = Tensor[f32, batch, seq, d_model]

@doc("Multi-head self-attention")
@requires_gpu(true)
@requires_tensor_cores(true)
@constraint("d_model % heads == 0", "d_model must be divisible by heads")
@complexity("O(batch * heads * seq^2 * d_k)")
fn multi_head_attention(
    x: Batch,
    w_q: Tensor[f32, d_model, d_model],
    w_k: Tensor[f32, d_model, d_model],
    w_v: Tensor[f32, d_model, d_model],
    w_o: Tensor[f32, d_model, d_model],
    heads: i64,
) -> Tensor[f32, batch, seq, d_model] {
    let d_k = d_model / heads
    
    let q = tpt.matmul(x, tpt.transpose(w_q))
    let k = tpt.matmul(x, tpt.transpose(w_k))
    let v = tpt.matmul(x, tpt.transpose(w_v))
    
    let q = tpt.reshape(q, [batch, seq, heads, d_k])
    let k = tpt.reshape(k, [batch, seq, heads, d_k])
    let v = tpt.reshape(v, [batch, seq, heads, d_k])
    
    let q = tpt.transpose(q, [0, 2, 1, 3])
    let k = tpt.transpose(k, [0, 2, 1, 3])
    let v = tpt.transpose(v, [0, 2, 1, 3])
    
    let scale = tpt.sqrt(tpt.cast(d_k, dtype=f32))
    let logits = tpt.matmul(q, tpt.transpose(k, [0, 1, 3, 2])) * (1.0 / scale)
    let weights = tpt.softmax(logits, dim=-1)
    let attn = tpt.matmul(weights, v)
    
    let attn = tpt.transpose(attn, [0, 2, 1, 3])
    let attn = tpt.reshape(attn, [batch, seq, d_model])
    
    return tpt.matmul(attn, tpt.transpose(w_o))
}
```

---

## Feed-Forward Network

```tpts
@doc("Position-wise feed-forward network")
@requires_gpu(true)
@requires_tensor_cores(true)
@complexity("O(batch * seq * d_model * d_ff)")
fn feed_forward(
    x: Batch,
    w1: Tensor[f32, d_ff, d_model],
    b1: Tensor[f32, d_ff],
    w2: Tensor[f32, d_model, d_ff],
    b2: Tensor[f32, d_model],
) -> Batch {
    let h = tpt.matmul(x, tpt.transpose(w1))
    let h = tpt.broadcast_add(h, b1)
    let h = tpt.gelu(h)
    let h = tpt.matmul(h, tpt.transpose(w2))
    return tpt.broadcast_add(h, b2)
}
```

---

## Transformer Block

```tpts
@doc("Transformer block with pre-norm")
@requires_gpu(true)
@requires_tensor_cores(true)
@complexity("O(batch * seq^2 * d_model)")
fn transformer_block(
    x: Batch,
    attn_weights: Tensor[f32, 4, d_model, d_model],
    ffn_weights: Tensor[f32, 4],
    heads: i64,
) -> Batch {
    let normed = tpt.layer_norm(x)
    let attn_out = multi_head_attention(
        normed, attn_weights[0], attn_weights[1], attn_weights[2], attn_weights[3], heads
    )
    let residual1 = tpt.add(x, attn_out)
    
    let normed = tpt.layer_norm(residual1)
    let ff_out = feed_forward(normed, ffn_weights[0], ffn_weights[1], ffn_weights[2], ffn_weights[3])
    return tpt.add(residual1, ff_out)
}
```

---

## Training Loop

```tpts
@doc("Train transformer for one step")
@requires_gpu(false)
fn train_step(
    model: Model,
    input_ids: Tensor[i64, batch, seq],
    labels: Tensor[i64, batch, seq],
) -> f32 {
    let logits = model.forward(input_ids)
    let loss = tpt.cross_entropy(logits, labels)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss
}
```

---

## Exercises

1. **Attention**: Implement causal masking for autoregressive attention
2. **Optimization**: Add gradient clipping and learning rate scheduling
3. **Evaluation**: Implement perplexity calculation

---

## Summary

- ✅ Multi-head attention with scaled dot-product
- ✅ Position-wise feed-forward network with GELU
- ✅ Transformer block with pre-norm architecture
- ✅ Training loop with backpropagation

**Next:** [Tutorial 16: Performance Tuning](16_performance_tuning.md)
