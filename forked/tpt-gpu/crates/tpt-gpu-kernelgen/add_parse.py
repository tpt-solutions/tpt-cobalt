"""Add parse_assembly function to ir.rs."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

lines = p.read_text(encoding="utf-8").splitlines()

# Find the line after OpKind::Custom(v) => write!(f, "custom {}", v),
# which is line 49 (0-indexed: 48)
# We want to insert parse_assembly after the OpKind display impl

# Find the line with "OpKind::Custom(v) => write!(f, \"custom {}\", v),"
insert_idx = None
for i, line in enumerate(lines):
    if 'OpKind::Custom(v) => write!(f, "custom {}", v),' in line:
        insert_idx = i + 1
        break

if insert_idx is None:
    print("Could not find insertion point")
else:
    # Insert parse_assembly function
    new_lines = [
        "",
        "/// Parse a minimal TPTIR-style assembly into a Region.",
        "pub fn parse_assembly(source:&str)->Result<Region,String>{",
        " let mut region=Region::new();",
        " let mut block=Block::new(\"entry\");",
        " for line in source.lines(){",
        "  let line=line.trim();",
        "  if line.is_empty()||line.starts_with(';')||line.starts_with('#'){continue;}",
        "  if line.starts_with('^'){if!block.operations.is_empty(){region.blocks.push(std::mem::replace(&mut block,Block::new(&line[1..])));}}",
        " }",
        " if!block.operations.is_empty()||region.blocks.is_empty(){region.blocks.push(block);}",
        " Ok(region)",
        "}",
    ]
    
    for j, new_line in enumerate(new_lines):
        lines.insert(insert_idx + j, new_line)
    
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines")
