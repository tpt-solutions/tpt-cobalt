"""Fix 'as' keyword issue in ir.rs."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

content = p.read_text(encoding="utf-8")

# Replace 'as' parameter names with 'addr'
# Only replace in specific contexts where 'as' is used as a parameter name
content = content.replace("el:Type,as:AddressSpace", "el:Type,addr:AddressSpace")
content = content.replace("el:Type,as:AddressSpace", "el:Type,addr:AddressSpace")
content = content.replace(",as)=", ",addr)=")

p.write_text(content, encoding="utf-8")
print(f"fixed: {p.stat().st_size} bytes")

"""Fix 'as' keyword issue in ir.rs."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

content = p.read_text(encoding="utf-8")

# Replace all occurrences of 'as' used as a variable name with 'addr'
content = content.replace("TypeKind::Tensor(s,e,a)", "TypeKind::Tensor(s,e,addr)")
content = content.replace("TypeKind::MemRef(s,e,a)", "TypeKind::MemRef(s,e,addr)")
content = content.replace("if*a!=", "if*addr!=")
content = content.replace('write!(f,", {}",a)', 'write!(f,", {}",addr)')

p.write_text(content, encoding="utf-8")
print(f"fixed: {p.stat().st_size} bytes")