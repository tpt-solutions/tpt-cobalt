# tpt-eng-plot

Plotting and diagram generation for the TPT ecosystem, built on [`plotters`]. Provides engineering-style charts and 2D cross-section / geometry drawings that can be exported to PNG or SVG.

## Features

- Multi-series XY line / scatter charts (`XyPlot`).
- Bar charts of named results (`ResultChart`).
- 2D cross-section / geometry drawings (`SectionDrawing`).
- Export to raster (PNG) or vector (SVG) via the shared `Drawing` trait.

## Installation

```toml
[dependencies]
tpt-eng-plot = "0.1"
```

## Quick start

```rust
use tpt_eng_plot::xy::{XyPlot, XySeries};

let series = XySeries::new(
    "sine",
    (0..100)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, x.sin())
        })
        .collect(),
);
let plot = XyPlot::new("Sine wave")
    .with_x_label("x")
    .with_y_label("sin(x)")
    .with_series(series);

plot.save_png("sine.png").unwrap();
plot.save_svg("sine.svg").unwrap();
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `xy` | XY line / scatter charts (`XyPlot`, `XySeries`). |
| `chart` | Bar charts of named results (`ResultChart`). |
| `section` | 2D cross-section drawings (`SectionDrawing`, `SectionShape`). |
| `error` | Crate `Error` / `Result` types. |
| `font` | Self-contained bitmap font for axis and label rendering. |

## Related crates

- [tpt-eng-report](../tpt-eng-report/) — pair generated plots with calculation reports.
- [tpt-eng-cli](../tpt-eng-cli/) — command-line front-end that emits results charts and diagrams.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
