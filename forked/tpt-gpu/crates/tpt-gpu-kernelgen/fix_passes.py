"""Fix passes.rs by removing duplicate content."""
import pathlib

p = pathlib.Path(r"D:\Programming\1PRODUCTION\Open Source\tpt-gpu\layer3_tptc\rust\src\passes.rs")

content = """use crate::ir::Region;
use crate::validate::ValidatePass;
use crate::fusion::FusionPass;

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, region: &Region) -> usize;
}

pub struct CanonicalizePass;
impl Pass for CanonicalizePass {
    fn name(&self) -> &str { "canonicalize" }
    fn run(&self, _: &Region) -> usize { 0 }
}

pub struct DeadCodeEliminationPass;
impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &str { "dce" }
    fn run(&self, _: &Region) -> usize { 0 }
}

pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl PassPipeline {
    pub fn new() -> Self {
        PassPipeline { passes: vec![] }
    }
    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }
    pub fn run(&self, r: &Region) -> usize {
        let mut t = 0;
        for p in &self.passes {
            t += p.run(r);
        }
        t
    }
}

/// Default pipeline: canonicalize → dce → validate → fusion
pub fn default_pipeline() -> PassPipeline {
    let mut p = PassPipeline::new();
    p.add(Box::new(CanonicalizePass));
    p.add(Box::new(DeadCodeEliminationPass));
    p.add(Box::new(ValidatePass));
    p.add(Box::new(FusionPass));
    p
}
"""

p.write_text(content, encoding="utf-8")
print(f"wrote {p.stat().st_size} bytes")
