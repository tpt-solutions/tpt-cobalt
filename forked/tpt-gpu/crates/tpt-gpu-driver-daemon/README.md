# tptd-daemon

TPT GPU userspace daemon — context management and VRAM isolation for the TPT GPU stack.

## Overview

`tptd-daemon` is the privileged userspace process that brokers access to TPT GPU hardware. It manages GPU context lifecycles, enforces VRAM isolation between clients, and exposes a Unix socket protocol for the `tpt-gpu-runtime` to connect to.

## Running

```sh
cargo install tpt-gpu-driver-daemon
tptd  # starts the daemon; listens on /var/run/tptd.sock by default
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
