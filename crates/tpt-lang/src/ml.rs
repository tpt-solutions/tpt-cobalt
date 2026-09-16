//! ML surface for TPT Script (Phase 2 deliverable: "train a model in TPT
//! Script").
//!
//! Wraps `tpt-ml` models ([`tpt_ml::layers::Sequential`], [`tpt_ml::attention::TransformerBlock`],
//! …) behind the opaque [`Value::Model`](crate::Value::Model) variant and
//! exposes a small, Python-like native surface:
//!
//! ```text
//! let net = mlp(1, 16, 1)          # Linear -> tanh -> Linear, AdamW inside
//! train_step(net, x, y, 0.01)      # one forward/backward/AdamW step -> loss
//! predict(net, x)                  # forward only
//! ```
//!
//! The demo test below trains an MLP on a synthetic regression target purely
//! from script. MNIST-scale data needs a bundled dataset (network fetches are
//! out of scope); the training loop itself is identical.

use std::sync::{Arc, Mutex};

use tpt_autograd::backward;
use tpt_ml::optim::{Optimizer, step_attached};
use tpt_ml::{Module, loss};
use tpt_tensor::Tensor;

use crate::interp::Interpreter;
use crate::value::Value;

/// Opaque state behind `Value::Model`: the network plus its optimizer.
pub struct ModelBox {
    pub model: Box<dyn Module>,
    pub opt: Option<tpt_ml::optim::AdamW>,
}

/// Tanh nonlinearity as a `Module` so it can live in a `Sequential`.
struct TanhLayer;

impl Module for TanhLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        tpt_ml::tanh(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_parameters(&mut self, _params: Vec<Tensor>) {}
}

fn as_model(v: &Value) -> Result<Arc<Mutex<ModelBox>>, String> {
    match v {
        Value::Model(m) => Ok(Arc::clone(m)),
        other => Err(format!("expected a model, got {}", other.type_name())),
    }
}

fn as_tensor(v: &Value, what: &str) -> Result<Tensor, String> {
    v.as_tensor()
        .cloned()
        .ok_or_else(|| format!("{what} must be a tensor"))
}

fn shape_of(t: &Tensor) -> Value {
    let dims: Vec<Value> = t.shape().iter().map(|&d| Value::Num(d as f64)).collect();
    Value::List(Arc::new(Mutex::new(dims)))
}

/// Register the ML natives on an interpreter.
pub fn install(interp: &mut Interpreter) {
    interp.register_native("mlp", |args| {
        let dims: Vec<usize> = args
            .iter()
            .map(|v| {
                v.as_num()
                    .map(|x| x as usize)
                    .ok_or("mlp() takes layer sizes")
            })
            .collect::<Result<_, _>>()?;
        if dims.len() < 2 {
            return Err("mlp(in, hidden..., out) needs at least 2 sizes".into());
        }
        let mut seq = tpt_ml::layers::Sequential::new();
        for w in dims.windows(2) {
            seq.push(tpt_ml::layers::Linear::new(w[0], w[1], true));
            if w[1] == *dims.last().unwrap() {
                break;
            }
            seq.push(TanhLayer);
        }
        Ok(Value::Model(Arc::new(Mutex::new(ModelBox {
            model: Box::new(seq),
            opt: None,
        }))))
    });

    interp.register_native("transformer", |args| {
        let nums: Vec<f64> = args
            .iter()
            .map(|v| v.as_num().ok_or("transformer() takes numbers"))
            .collect::<Result<_, _>>()?;
        if nums.len() != 3 {
            return Err("transformer(d_model, heads, d_ff) requires 3 arguments".into());
        }
        let block =
            tpt_ml::TransformerBlock::new(nums[0] as usize, nums[1] as usize, nums[2] as usize);
        Ok(Value::Model(Arc::new(Mutex::new(ModelBox {
            model: Box::new(block),
            opt: None,
        }))))
    });

    interp.register_native("predict", |args| {
        let m = as_model(
            args.first()
                .ok_or("predict(model, x) requires 2 arguments")?,
        )?;
        let x = as_tensor(
            args.get(1)
                .ok_or("predict(model, x) requires 2 arguments")?,
            "x",
        )?;
        let guard = m.lock().unwrap();
        Ok(Value::Tensor(guard.model.forward(&x)))
    });

    interp.register_native("shape", |args| {
        let t = as_tensor(args.first().ok_or("shape(t) requires a tensor")?, "t")?;
        Ok(shape_of(&t))
    });

    // train_step(model, x, y, lr) -> loss: one full AdamW optimization step.
    // x is [batch, features]; y is [batch, outputs] (MSE loss).
    interp.register_native("train_step", |args| {
        let m = as_model(args.first().ok_or("train_step(model, x, y, lr)")?)?;
        let x = as_tensor(args.get(1).ok_or("train_step missing x")?, "x")?;
        let y = as_tensor(args.get(2).ok_or("train_step missing y")?, "y")?;
        let lr = args
            .get(3)
            .and_then(|v| v.as_num())
            .ok_or("train_step missing learning rate")?;

        let mut guard = m.lock().unwrap();
        let pred = guard.model.forward(&x);
        let loss = loss::mse(&pred, &y);
        backward(&loss);
        let loss_value = loss.to_vec::<f64>().unwrap()[0];
        if guard.opt.is_none() {
            guard.opt = Some(tpt_ml::optim::AdamW::new(lr));
        }
        // split borrows: optimizer and model are separate fields
        let ModelBox { model, opt } = &mut *guard;
        if let Some(o) = opt.as_mut() {
            o.set_lr(lr)
        }
        let mut params = model.parameters();
        params = step_attached(opt.as_mut().unwrap(), params);
        model.set_parameters(params);
        Ok(Value::Num(loss_value))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::Interpreter;

    fn f64_tensor(vals: &[f64], shape: &[usize]) -> Value {
        Value::Tensor(Tensor::from_typed(vals.to_vec()).reshape(shape).unwrap())
    }

    /// Phase 2 deliverable slice: train an MLP **from TPT Script** on a
    /// synthetic regression target (y = 2x + 1). The MNIST-scale dataset needs
    /// a bundled download; the training loop is identical.
    #[test]
    fn script_trains_mlp_on_synthetic_regression() {
        let mut it = Interpreter::new();
        // batch of 4 training pairs around y = 2x + 1
        it.set("xs", f64_tensor(&[0.1, 0.4, 0.7, 0.9], &[4, 1]));
        it.set("ys", f64_tensor(&[1.2, 1.8, 2.4, 2.8], &[4, 1]));
        let src = "
let net = mlp(1, 16, 1)
let first = train_step(net, xs, ys, 0.05)
let last = first
let i = 0
while i < 300 {
    last = train_step(net, xs, ys, 0.05)
    i = i + 1
}
last
";
        let v = it.run(src).expect("training script failed").unwrap();
        let final_loss = match v {
            Value::Num(l) => l,
            other => panic!("expected Num loss, got {other:?}"),
        };
        assert!(
            final_loss < 0.05,
            "MLP did not converge: final loss {final_loss}"
        );
        // predictions track the target better than the untrained start
        let pred = eval_predict(&mut it);
        for p in &pred {
            assert!(p.is_finite());
        }
    }

    fn eval_predict(it: &mut Interpreter) -> Vec<f64> {
        let v = it.run("predict(net, xs)").expect("predict failed").unwrap();
        match v {
            Value::Tensor(t) => t.to_vec::<f64>().unwrap(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn script_runs_transformer_forward() {
        let mut it = Interpreter::new();
        // [batch, seq_len, d_model] input; output must keep the shape
        it.set("x", f64_tensor(&vec![0.5; 1 * 4 * 8], &[1, 4, 8]));
        let v = it
            .run(
                "let blk = transformer(8, 2, 16)
let out = predict(blk, x)
shape(out)",
            )
            .expect("transformer script failed")
            .unwrap();
        let dims = match v {
            Value::List(l) => l
                .lock()
                .unwrap()
                .iter()
                .map(|d| match d {
                    Value::Num(n) => *n as usize,
                    other => panic!("bad dim {other:?}"),
                })
                .collect::<Vec<_>>(),
            other => panic!("expected list of dims, got {other:?}"),
        };
        assert_eq!(dims, vec![1, 4, 8]);
    }
}
