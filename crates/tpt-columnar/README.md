# tpt-columnar

The clean-room columnar data engine of the `tpt-cobalt` workspace: typed
arrays, schemas, record batches, compute kernels, display helpers, and a
native self-describing container format ("TPTC") — with **zero
dependencies**.

## Features

- **Typed arrays** — `PrimitiveArray<T>` (`Int32Array`, `Int64Array`,
  `UInt32Array`, `Float32Array`, `Float64Array`), `BooleanArray`,
  `StringArray`, and `BinaryArray` behind a type-erased `Array` trait
  (`ArrayRef = Arc<dyn Array>`) so batches can hold mixed column types.
- **Schemas as data** — `Schema`, `Field`, `DataType` describe every batch;
  `RecordBatch::try_new` validates column count, types, and lengths.
- **Compute kernels** — `filter`, `take`, `and`, `or`, `not`, and
  `concat_batches` operate on `ArrayRef`s (boolean-mask and index-list
  selection, logical combination, concatenation).
- **Native TPTC container** — a self-describing file format
  (`ipc::FileWriter` / `ipc::FileReader`) for persisting batches; the
  format `tpt-hub`'s Arrow-IPC bridge aligns with.
- **Display helpers** — `array_value_to_string` and friends for
  human-readable table rendering (notebook/REPL output).
- **Clean-room, dependency-free** — implemented from the columnar data
  model itself; no code, formats, or specifications were copied from any
  third-party project.

## Usage

```rust
use std::sync::Arc;
use tpt_columnar::prelude::*;

let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::Int32, false),
    Field::new("score", DataType::Float64, false),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])),
    ],
)?;

// select rows where score > 1.0
let mask = ...; // BooleanArray from a compute/comparison kernel
let taken = take(batch.column(0), &mask)?;
```

See the crate docs (`cargo doc -p tpt-columnar --open`) for the full
kernel and IPC surface.

## Testing

```sh
cargo test -p tpt-columnar
```

## Status

Stable at 0.1.x. Serves as the in-workspace columnar foundation for
`tpt-hub`'s serialization/IPC story; a native Rust foundation for the
tabular side of the ML stack.
