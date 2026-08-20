"""Helper: write ir.rs since editor has a 6KB payload limit."""
import pathlib

T = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

existing = T.read_text(encoding="utf-8")
marker = "pub fn parse_assembly(source:&str)->Result<Region,String>{"
idx = existing.find(marker)
if idx < 0:
    raise SystemExit("marker not found")
head = existing[:idx]
T.write_text(head, encoding="utf-8")
print(f"wrote {len(head)} bytes (truncated at parse_assembly)")

"""Helper: write ir.rs since editor has a 6KB payload limit."""
import pathlib

T = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

existing = T.read_text(encoding="utf-8")
marker = "pub fn parse_assembly(source:&str)->Result<Region,String>{"
idx = existing.find(marker)
if idx < 0:
    raise SystemExit("marker not found")
head = existing[:idx]

tail_path = pathlib.Path(__file__).with_name("ir_tail.rs")
tail = tail_path.read_text(encoding="utf-8")

T.write_text(head + tail, encoding="utf-8")
print(f"wrote {len(head) + len(tail)} bytes to {T}")