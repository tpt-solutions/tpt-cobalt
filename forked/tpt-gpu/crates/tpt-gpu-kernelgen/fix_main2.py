"""Fix main.rs parse shape code."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\tools\kernel-generator\src\main.rs")

content = p.read_text(encoding="utf-8")

# Replace the parse shape code
old_code = """            let shape_params: Vec<i64> = shape
                .split('x')
                .map(|s| s.parse::<i64>().map_err(|e| anyhow::anyhow!("Invalid shape: {}", e)))
                .collect::<Result<Vec<_>, _>>()?;"""

new_code = """            let shape_params: Vec<i64> = shape
                .split('x')
                .map(|s| s.parse::<i64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("Invalid shape: {}", e))?;"""

content = content.replace(old_code, new_code)

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
