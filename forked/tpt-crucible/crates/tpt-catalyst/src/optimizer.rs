use crate::ir::TptIr;

pub fn fuse_sequential_ops(ir: &mut TptIr) {
    let fuseable_patterns: &[(&[&str], &str)] = &[
        (&["matmul", "relu"], "fused_matmul_relu"),
        (&["matmul", "gelu"], "fused_matmul_gelu"),
        (&["add", "relu"], "fused_add_relu"),
    ];

    let mut fused_indices = Vec::new();

    for &(ops, fused_type) in fuseable_patterns {
        let mut i = 0;
        while i < ir.graph.edges.len() {
            let edge = &ir.graph.edges[i];
            let (from_id, to_id) = (edge.from, edge.to);

            let from_type = ir.graph.nodes.get(from_id).map(|n| n.op_type.clone());
            let to_type = ir.graph.nodes.get(to_id).map(|n| n.op_type.clone());

            if let (Some(ft), Some(tt)) = (from_type, to_type) {
                if ft == ops[0] && tt == ops[1] {
                    let fused_name = format!(
                        "{}_{}",
                        ir.graph.nodes[from_id].name, ir.graph.nodes[to_id].name
                    );
                    ir.graph.nodes[from_id].op_type = fused_type.to_string();
                    ir.graph.nodes[from_id].name = fused_name;
                    fused_indices.push(to_id);
                }
            }
            i += 1;
        }
    }

    ir.graph.nodes.retain(|n| !fused_indices.contains(&n.id));
    ir.graph
        .edges
        .retain(|e| !fused_indices.contains(&e.from) && !fused_indices.contains(&e.to));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Edge, OpNode, TptIr};

    #[test]
    fn test_fuse_matmul_relu() {
        let mut ir = TptIr::new("test".into(), "pytorch".into());
        ir.graph.nodes.push(OpNode {
            id: 0,
            op_type: "matmul".into(),
            name: "layer1".into(),
            attributes: std::collections::HashMap::new(),
        });
        ir.graph.nodes.push(OpNode {
            id: 1,
            op_type: "relu".into(),
            name: "act1".into(),
            attributes: std::collections::HashMap::new(),
        });
        ir.graph.edges.push(Edge {
            from: 0,
            to: 1,
            tensor_name: "x".into(),
        });

        fuse_sequential_ops(&mut ir);
        assert_eq!(ir.graph.nodes.len(), 1);
        assert_eq!(ir.graph.nodes[0].op_type, "fused_matmul_relu");
    }
}
