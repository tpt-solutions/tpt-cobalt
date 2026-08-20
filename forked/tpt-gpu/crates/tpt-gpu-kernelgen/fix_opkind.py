"""Add Display impl to OpKind."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\ir.rs")

content = p.read_text(encoding="utf-8")

# Add Display impl after OpKind definition
old = '#[derive(Debug,Clone)]\npub enum OpKind{Addi,Subi,Muli,Addf,Subf,Mulf,And,Or,Xor,CmpEq,CmpLt,Load,Store,Branch,Return,Constant(String),Custom(String)}'

new = '''#[derive(Debug,Clone)]
pub enum OpKind{Addi,Subi,Muli,Addf,Subf,Mulf,And,Or,Xor,CmpEq,CmpLt,Load,Store,Branch,Return,Constant(String),Custom(String)}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Addi => write!(f, "addi"),
            OpKind::Subi => write!(f, "subi"),
            OpKind::Muli => write!(f, "muli"),
            OpKind::Addf => write!(f, "addf"),
            OpKind::Subf => write!(f, "subf"),
            OpKind::Mulf => write!(f, "mulf"),
            OpKind::And => write!(f, "andi"),
            OpKind::Or => write!(f, "ori"),
            OpKind::Xor => write!(f, "xori"),
            OpKind::CmpEq => write!(f, "cmpeq"),
            OpKind::CmpLt => write!(f, "cmplt"),
            OpKind::Load => write!(f, "load"),
            OpKind::Store => write!(f, "store"),
            OpKind::Branch => write!(f, "br"),
            OpKind::Return => write!(f, "return"),
            OpKind::Constant(v) => write!(f, "constant {}", v),
            OpKind::Custom(v) => write!(f, "custom {}", v),
        }
    }
}'''

content = content.replace(old, new)

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
