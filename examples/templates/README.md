# TPT Script templates

Runnable starting points — execute any of them with:

```sh
cargo run --release -p tpt-lang --bin tpt -- run examples/templates/<name>.tpt
```

| template | what it shows |
|---|---|
| `physics-units.tpt` | compile-time unit checking + interpolated printing |
| `ml-training.tpt` | training an MLP from script with `train_step` |
| `modules.tpt` | namespacing with `module`, closures, keyword args |
| `data-pipeline.tpt` | tensors, slicing, for-loops, aggregation |
