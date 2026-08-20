"""Fix ir.rs syntax error on line 109."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")
lines = p.read_text(encoding="utf-8").splitlines()

# Find and fix line 109 (0-indexed: 108)
for i, line in enumerate(lines):
    if "OpKind::Constant(_)=>TptirOp::Constant,OpKind::Custom(_)=>TptirOp::Custom" in line:
        # Fix: add proper closing brace
        lines[i] = "  OpKind::Constant(_)=>TptirOp::Constant,OpKind::Custom(_)=>TptirOp::Custom"
        lines.insert(i + 1, "}")
        break

p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"fixed ir.rs: {len(lines)} lines")
