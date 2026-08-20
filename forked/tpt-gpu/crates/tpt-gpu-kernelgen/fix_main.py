"""Fix main.rs parse shape code."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\tools\kernel-generator\src\main.rs")

lines = p.read_text(encoding="utf-8").splitlines()

# Find and replace the parse shape code
for i, line in enumerate(lines):
    if "let shape_params: Vec<i64> = shape" in line and i < 70:
        lines[i] = "            let shape_params: Vec<i64> = shape"
        lines[i + 1] = "                .split('x')"
        lines[i + 2] = "                .map(|s| s.parse::<i64>().map_err(|e| anyhow::anyhow!(\"Invalid shape: {}\", e)))"
        lines[i + 3] = "                .collect::<Result<Vec<_>, _>>()?;"
        break

p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(lines)} lines")
