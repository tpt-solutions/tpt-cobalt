"""Fix main.rs parse_shape function."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\tools\kernel-generator\src\main.rs")

content = p.read_text(encoding="utf-8")

# Fix the parse_shape function - remove the map_err since anyhow::Error doesn't implement From<anyhow::Error>
old_code = """fn parse_shape(shape: &str) -> anyhow::Result<Vec<i64>> {
    shape.split('x')
        .map(|s| s.parse::<i64>()
            .map_err(|e| anyhow::anyhow!("Invalid shape '{}': {}", s, e)))
        .collect()
}"""

new_code = """fn parse_shape(shape: &str) -> anyhow::Result<Vec<i64>> {
    shape.split('x')
        .map(|s| s.parse::<i64>()
            .map_err(|e| anyhow::anyhow!("Invalid shape '{}': {}", s, e)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Parse error: {}", e))
}"""

content = content.replace(old_code, new_code)

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
