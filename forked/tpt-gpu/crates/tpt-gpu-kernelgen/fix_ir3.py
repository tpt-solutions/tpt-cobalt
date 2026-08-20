"""Fix remaining ir.rs and dispatch.rs issues."""
import pathlib

# Fix ir.rs
p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")
content = p.read_text(encoding="utf-8")

# Fix KERNEL_TEMPLATES type
content = content.replace(
    'pub const KERNEL_TEMPLATES:&[&str]=["vector_add","matmul","softmax","flash_attention","conv_bn_relu"];',
    'pub const KERNEL_TEMPLATES:&[&str]=&["vector_add","matmul","softmax","flash_attention","conv_bn_relu"];'
)

p.write_text(content, encoding="utf-8")
print(f"fixed ir.rs: {p.stat().st_size} bytes")

# Fix dispatch.rs
p2 = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\dispatch.rs")
content2 = p2.read_text(encoding="utf-8")

# Fix KERNEL_TEMPLATES iteration
content2 = content2.replace(
    'for &name in &KERNEL_TEMPLATES {',
    'for &name in KERNEL_TEMPLATES {'
)

p2.write_text(content2, encoding="utf-8")
print(f"fixed dispatch.rs: {p2.stat().st_size} bytes")
