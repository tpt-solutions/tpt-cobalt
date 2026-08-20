"""Truncate ir.rs to 200 lines."""
import pathlib
p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")
lines = p.read_text(encoding="utf-8").splitlines()
p.write_text("\n".join(lines[:200]) + "\n", encoding="utf-8")
print(f"truncated to 200 lines")
