// Operation router — maps string op names to Rust dispatch functions.
//
// Used by py_dispatch() for the generic JSON-args call path.

use std::collections::HashMap;

pub struct DispatchTable {
    ops: HashMap<&'static str, &'static str>, // op → description
}

impl Default for DispatchTable {
    fn default() -> Self {
        let mut ops = HashMap::new();
        ops.insert("gemm", "alpha * A @ B + beta * C");
        ops.insert("matmul", "A @ B (no alpha/beta)");
        ops.insert("attention", "softmax(Q @ K^T / sqrt(d_k)) @ V");
        ops.insert("conv2d", "2-D convolution NCHW × OIHW → NCHW");
        Self { ops }
    }
}

impl DispatchTable {
    pub fn ops(&self) -> Vec<String> {
        let mut v: Vec<String> = self.ops.keys().map(|s| s.to_string()).collect();
        v.sort();
        v
    }

    pub fn contains(&self, name: &str) -> bool {
        self.ops.contains_key(name)
    }
}

#[derive(Default)]
pub struct OpRouter {
    table: DispatchTable,
}

impl OpRouter {
    /// Route an op by name; args_json is a JSON object of named parameters.
    /// Returns a JSON string describing the result or an error message.
    pub fn route(&self, op: &str, args_json: &str) -> Result<String, String> {
        if !self.table.contains(op) {
            return Err(format!("unknown op `{op}`"));
        }
        // In a full implementation, args_json would be parsed and dispatch
        // would call the appropriate Rust function.  Here we return a metadata
        // response indicating the op is registered.
        Ok(format!(
            "{{\"op\":\"{op}\",\"status\":\"registered\",\"args\":{args_json}}}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_contains_core_ops() {
        let t = DispatchTable::default();
        assert!(t.contains("gemm"));
        assert!(t.contains("attention"));
        assert!(t.contains("conv2d"));
        assert!(!t.contains("frobnicator"));
    }

    #[test]
    fn test_ops_sorted() {
        let t = DispatchTable::default();
        let v = t.ops();
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(v, sorted);
    }

    #[test]
    fn test_router_known_op() {
        let r = OpRouter::default();
        let result = r.route("gemm", "{}");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("gemm"));
    }

    #[test]
    fn test_router_unknown_op() {
        let r = OpRouter::default();
        assert!(r.route("frobnicator", "{}").is_err());
    }
}
