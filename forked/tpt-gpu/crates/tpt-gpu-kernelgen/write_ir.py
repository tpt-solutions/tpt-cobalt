"""Write ir.rs from scratch."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

# Read the existing file to preserve the original content
existing = p.read_text(encoding="utf-8")
lines = existing.splitlines()

# Find where the original content ends (before our additions)
# The original file ends at line 60 with "Ok(region)" and "}"
# We want to keep lines 1-60 and replace the rest
original_end = 60

# Write just the original content
p.write_text("\n".join(lines[:original_end]) + "\n", encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes (original content only)")
