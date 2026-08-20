"""Fix 'as' keyword issue in ir.rs - replace all 'as' variable names with 'addr'."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

content = p.read_text(encoding="utf-8")

# Replace all occurrences of 'as' used as a variable name with 'addr'
# We need to be careful to only replace the 'as' that refers to the AddressSpace variable
# and not the 'as' keyword in other contexts (like type casts).

# The pattern is: in the TypeKind::Tensor and TypeKind::MemRef variants,
# the third element is the AddressSpace variable. We need to replace all
# occurrences of 'as' in those contexts with 'addr'.

# Simple approach: replace all standalone 'as' that appear as identifiers
# This is a bit risky but should work for this specific file
import re

# Replace 'as' when it appears as a standalone identifier (not part of another word)
# and not followed by ' ' (which would be a type cast like 'as AddressSpace')
content = re.sub(r'\bas\b(?!:)', 'addr', content)

p.write_text(content, encoding="utf-8")
print(f"fixed: {p.stat().st_size} bytes")
