//! # tpt-ml — Embedding (Phase 2, spec §5.3)
//!
//! A lookup table mapping integer token indices to dense vectors. `weight` is
//! `[num_embeddings, embedding_dim]`; the forward gathers rows. Gradient flows
//! back into the table via scatter-add (a custom autograd node), since the
//! backend has no gather/scatter primitive.

use std::sync::Arc;

use tpt_tensor::{AutogradNode, Tensor};

use crate::module::Module;

pub struct Embedding {
    pub weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = Tensor::from_typed(vec![0.0_f64; num_embeddings * embedding_dim]).with_autograd();
        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Tensor {
        let idim = indices.shape().to_vec();
        let d = self.embedding_dim;
        let v = self.weight.to_vec::<f64>().unwrap();
        let idx = indices.to_vec::<f64>().unwrap();
        let mut out = vec![0.0f64; idx.len() * d];
        for (k, &i) in idx.iter().enumerate() {
            let vi = i as usize;
            debug_assert!(vi < self.num_embeddings, "Embedding index out of range");
            for j in 0..d {
                out[k * d + j] = v[vi * d + j];
            }
        }
        let mut shape = idim.clone();
        shape.push(d);
        let mut result = Tensor::from_typed(out).reshape(&shape).unwrap();

        if self.weight.requires_grad() {
            if let Some(node) = self.weight.node() {
                let v_node = node.clone();
                let idim2 = idim.clone();
                let d2 = d;
                let nv = self.num_embeddings;
                let shape2 = shape.clone();
                let idx2 = idx.clone();
                let node = AutogradNode::new(
                    vec![node],
                    Box::new(move |grad: &Tensor| {
                        let g = grad.to_vec::<f64>().unwrap();
                        let mut gw = vec![0.0f64; nv * d2];
                        for (k, &i) in idx2.iter().enumerate() {
                            let vi = i as usize;
                            for j in 0..d2 {
                                gw[vi * d2 + j] += g[k * d2 + j];
                            }
                        }
                        v_node.accumulate_grad(&Tensor::from_typed(gw).reshape(&[nv, d2]).unwrap());
                        let _ = &idim2;
                        let _ = &shape2;
                    }),
                );
                result.set_node(Arc::new(node));
            }
        }
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        assert_eq!(params.len(), 1, "Embedding: wrong parameter count");
        self.weight = params[0].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn embedding_lookup() {
        let mut emb = Embedding::new(3, 2);
        emb.weight = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).with_autograd();
        let idx = Tensor::from_typed(vec![0.0_f64, 2.0]);
        let y = emb.forward(&idx);
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn embedding_scatter_grad() {
        let mut emb = Embedding::new(3, 2);
        emb.weight = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).with_autograd();
        // indices [0, 0, 1] -> token 0 seen twice, token 1 once
        let idx = Tensor::from_typed(vec![0.0_f64, 0.0, 1.0]);
        let y = emb.forward(&idx); // shape [3, 2]
        backward(&y); // seed = ones
        let wg = emb.weight.grad().unwrap().to_vec::<f64>().unwrap();
        // token0 grad = 1+1 = 2 per dim; token1 grad = 1; token2 = 0
        assert_eq!(wg, vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
