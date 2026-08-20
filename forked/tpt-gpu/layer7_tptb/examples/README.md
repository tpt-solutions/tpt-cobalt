# TPT Script Examples

Worked examples for the TPT Script beta release.

| File | What it shows |
|------|---------------|
| [`01_hello_tensor.tpts`](01_hello_tensor.tpts) | First program — basic tensor operations |
| [`02_custom_kernel.tpts`](02_custom_kernel.tpts) | Writing your own GPU kernel |
| [`03_transformer_block.tpts`](03_transformer_block.tpts) | Multi-head attention transformer block |
| [`04_training_loop.tpts`](04_training_loop.tpts) | Complete supervised training loop |
| [`distributed/fsdp_8gpu.tpts`](distributed/fsdp_8gpu.tpts) | FSDP across 8 GPUs |
| [`distributed/pipeline_parallel.tpts`](distributed/pipeline_parallel.tpts) | Pipeline parallelism (4 stages) |

## Running an example

```bash
# Type-check
tpt check examples/01_hello_tensor.tpts

# Compile to Rust + TPTIR
tpt compile examples/01_hello_tensor.tpts -o out/

# Inspect the generated code
cat out/01_hello_tensor.rs
cat out/01_hello_tensor.tptir
```
