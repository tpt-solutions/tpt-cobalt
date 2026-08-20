# tpt-eng-cli

Command-line interface for the TPT engineering ecosystem.

## Commands

| Command | Description |
| --- | --- |
| `validate <file>` | Validate an engineering input file (`json`, `csv`, `stl`, `obj`). |
| `report --out <path>` | Generate a calculation report. The output format is chosen by the extension (`.md`, `.html`, `.json`). Use `--chart <png>` to also emit a results chart. |
| `units convert <value> <from> <to>` | Convert between units (length, mass, force, pressure, temperature). |
| `units list` | List supported units. |
| `materials [name]` | Inspect a built-in material, or list available materials. |
| `sections inspect <shape> ...` | Inspect built-in cross-section properties (rectangle / circle / i-beam). |
| `calc beam <span> <load> --material <m> <section> [--report <path>] [--plot <png>]` | Simply supported beam under a uniformly distributed load. |
| `props water <T> <P> [--temp-unit k\|c] [--pressure-unit pa\|kpa\|mpa\|bar\|psi]` | Water/steam (IAPWS-IF97) property lookup. |
| `props air <T> <RH%> [--pressure <P>] [--temp-unit k\|c] [--pressure-unit ...]` | Moist-air (ASHRAE) lookup: humidity ratio, dew point, enthalpy. |
| `props fuel <methane\|hydrogen\|natural-gas\|diesel\|blend> [--h2 <0..1>]` | Fuel heating values, density, air–fuel ratio, CO₂. |
| `pid <kp> <ki> <kd> [--setpoint <sp>] [--tau <s>] [--dt <s>] [--steps <n>] [--limit <±>] [--plot <png>] [--csv <path>]` | PID step-response simulation over a first-order plant. |
| `tolerance stackup -D "name=nominal±tol" ... [--methods worst,rss,monte] [--samples <n>] [--low <L>] [--high <H>] [--seed <s>]` | Worst-case / RSS / Monte-Carlo dimensional stack-up. |

## Examples

```sh
# Validate a CSV file
tpt-eng-cli validate data.csv

# Convert 1 metre to millimetres
tpt-eng-cli units convert 1 m mm

# Inspect structural steel
tpt-eng-cli materials steel

# Properties of a 100 mm x 200 mm rectangle
tpt-eng-cli sections inspect rectangle 0.1 0.2

# Run a simply supported beam calc and emit a report + deflection diagram
tpt-eng-cli calc beam 5 10000 --material steel rectangle 0.1 0.2 \
    --report beam.md --plot beam.png

# Water/steam properties at 300 K, 3 MPa (IAPWS-IF97)
tpt-eng-cli props water 300 3 --temp-unit k --pressure-unit mpa

# Moist-air at 25 °C, 50 % RH, 101.325 kPa
tpt-eng-cli props air 25 50

# Fuel properties and a 30 % hydrogen blend
tpt-eng-cli props fuel methane
tpt-eng-cli props fuel blend --h2 0.3

# PID step-response for a first-order plant (tau = 1 s)
tpt-eng-cli pid 2 1 0 --setpoint 10 --tau 1 --dt 0.01 --steps 1000

# Tolerance stack-up: two dimensions, yield within [29.5, 30.5]
tpt-eng-cli tolerance stackup -D "a=10±0.1" -D "b=20±0.2" --low 29.5 --high 30.5
```

## Notes

- The `calc beam` example uses SI units (metres, newtons). Material properties and section
  second moments of area are looked up from the built-in tables.
- Material / section inspection and beam calculation now use the
  [`tpt-eng-materials`](../tpt-eng-materials/),
  [`tpt-eng-sections`](../tpt-eng-sections/), and
  [`tpt-eng-structural`](../tpt-eng-structural/) crates for evaluation.
- Plotting uses a self-contained bitmap font so that no external (copyleft) font dependencies
  are required.
