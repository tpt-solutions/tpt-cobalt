# Contributing to TPT Crucible

Thank you for your interest in contributing to TPT Crucible!

## Getting Started

### Prerequisites

- **Rust** (stable) — `rustup default stable`
- **Python 3.10+** — with `pip`
- **Go 1.22+** — for the Observer backend
- **Node.js 18+** — for the Observer frontend (Phase 4)

### Quick Start

```bash
# Clone the repo
git clone https://github.com/PhillipC05/tpt-crucible.git
cd tpt-crucible

# Build Rust crates
cargo build

# Run Rust tests
cargo test

# Run Python tests
PYTHONPATH=python/tpt_catalyst:python/tpt_alloy:python/tpt_element:python/tpt_fusion:python/tpt_emulator:python/tpt_mosaic:python/tpt_train pytest python/*/tests -q

# Build Go service
cd services/tpt-observer && go build ./...
```

## Project Structure

```
tpt-crucible/
├── crates/                    # Rust crates
│   ├── tpt-catalyst/          # Core IR compiler
│   ├── tpt-alloy/             # Swarm partitioning
│   ├── tpt-catalyst-python/   # Python bindings for Catalyst
│   └── tpt-alloy-python/      # Python bindings for Alloy
├── python/                    # Python packages
│   ├── tpt_catalyst/          # Catalyst Python layer
│   ├── tpt_alloy/             # Alloy Python layer
│   ├── tpt_element/           # Analog compute module
│   ├── tpt_fusion/            # FPGA synthesis module
│   ├── tpt_emulator/          # Software-in-the-Loop emulator
│   ├── tpt_mosaic/            # Hybrid deployment orchestrator
│   └── tpt_train/             # Training hooks for profiles
├── services/
│   └── tpt-observer/          # Go backend service
└── frontend/                  # Next.js Observer dashboard
```

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run all tests: `cargo test` + `pytest`
4. Submit a pull request

## Code Style

- **Rust**: Follow `cargo fmt` defaults, run `cargo clippy` before committing
- **Python**: Follow PEP 8, run `ruff check` if available
- **Go**: Follow `gofmt` defaults

## Testing

- **Rust**: Unit tests live in `#[cfg(test)]` modules within each crate
- **Python**: Tests live in `tests/` directories within each package
- **Go**: Tests live alongside source files (when added)

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
