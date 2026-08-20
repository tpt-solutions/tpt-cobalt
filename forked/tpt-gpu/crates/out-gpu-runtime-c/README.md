# out-gpu-runtime-c

TPT Runtime — C ABI.

## Overview

`out-gpu-runtime-c` (compiled as the `tptr_c` cdylib / staticlib) exposes a stable C API
over `tpt-gpu-runtime`. It mirrors the TPTIR C API pattern (cbindgen-generated header):
device create, allocate/free, `memcpy_htod`/`memcpy_dtoh`, create kernel / launch kernel,
synchronize, and error accessors.

> This is an internal, non-published crate (`publish = false`).

## Usage

Include the generated header (see `crates/tpt-gpu-runtime`'s C example) and link against
`tptr_c`. The C ABI is the entry point for non-Rust languages that want to drive the TPT
GPU runtime.
