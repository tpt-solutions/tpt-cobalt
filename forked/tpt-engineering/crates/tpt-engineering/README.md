# tpt-engineering

Umbrella meta-crate re-exporting every `tpt-eng-*` crate behind its own
feature flag, so a downstream consumer can depend on the whole toolkit with a
single crate and enable only the domains they need.

Each feature is named after the crate without the `tpt-eng-` prefix (hyphens
become underscores in the module name):

```toml
[dependencies]
tpt-engineering = { version = "0.1", features = ["structural", "props"] }
```

```rust,ignore
use tpt_engineering::structural;
use tpt_engineering::props;
```

This crate is an internal convenience wrapper and is **not published** to
crates.io (its dependencies are released individually).

Dual-licensed under MIT OR Apache-2.0.
