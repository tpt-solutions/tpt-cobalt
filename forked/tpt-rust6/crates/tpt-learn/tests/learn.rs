use tpt_learn::prelude::*;
use tpt_omni::{ndarray::ArrayD, Tensor};

fn tensor(rows: usize, cols: usize, data: Vec<f64>) -> Tensor<f64> {
    Tensor::new(ArrayD::from_shape_vec(vec![rows, cols], data).expect("shape"))
}

/// `y = 2x + 1` over [0, 4).
fn line_dataset(n: usize) -> (Tensor<f64>, Tensor<f64>) {
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 4.0 / n as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
    (tensor(n, 1, xs), tensor(n, 1, ys))
}

#[test]
fn adam_fits_two_x_plus_one() {
    let (x, y) = line_dataset(32);
    let model = Trainer::new(Linear::<1, 1>::new())
        .optimizer(Adam::new(0.1))
        .loss(Loss::MeanSquaredError)
        .epochs(200)
        .batch_size(8)
        .lr_schedule(LrSchedule::Cosine {
            t_max: 200,
            min_lr: 1e-3,
        })
        .on_epoch(|_e, _t, _v| {})
        .fit(&x, &y, &x, &y)
        .expect("training must succeed");

    let mse = evaluate(&model, &x, &y, Loss::MeanSquaredError).unwrap();
    assert!(mse < 0.1, "final train MSE too high: {mse}");

    // Predictions track 2x + 1, including outside the training range.
    let probe = tensor(3, 1, vec![-1.0, 0.5, 5.0]);
    let pred = model.forward(&probe).unwrap().to_vec();
    for (p, x) in pred.iter().zip([-1.0, 0.5, 5.0]) {
        let want = 2.0 * x + 1.0;
        assert!((p - want).abs() < 0.5, "pred {p} vs {want}");
    }
    assert!((model.weight[[0, 0]] - 2.0).abs() < 0.3);
    assert!((model.bias[0] - 1.0).abs() < 0.3);
}

#[test]
fn early_stopping_and_logging_hook() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let (x, y) = line_dataset(16);
    let calls = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&calls);
    let _model = Trainer::new(Linear::<1, 1>::new())
        .optimizer(Adam::new(0.2))
        .epochs(500)
        .batch_size(16)
        .early_stopping(3, Metric::ValLoss)
        .on_epoch(move |e, t, v| sink.borrow_mut().push((e, t, v)))
        .fit(&x, &y, &x, &y)
        .unwrap();

    let log = calls.borrow();
    assert!(!log.is_empty(), "logging hook was never called");
    assert!(log.len() < 500, "early stopping never triggered");
    assert!(log.last().unwrap().1 < log[0].1, "loss did not decrease");
}

/// A tiny convex problem: fit `w` in `y = w * x` for `y = 2x` (plus a bias
/// parameter left at 0), driven directly through the `Optimizer` trait.
fn convex_run(opt: &mut dyn Optimizer, steps: usize) -> (f64, f64) {
    let xs = [0.5f64, 1.0, 1.5, 2.0];
    let loss = |p: &[f64]| -> f64 {
        xs.iter()
            .map(|x| {
                let e = p[0] * x + p[1] - 2.0 * x;
                e * e
            })
            .sum::<f64>()
            / xs.len() as f64
    };
    let grad = |p: &[f64]| -> Vec<f64> {
        let n = xs.len() as f64;
        let mut g = vec![0.0, 0.0];
        for x in xs {
            let e = p[0] * x + p[1] - 2.0 * x;
            g[0] += 2.0 * e * x / n;
            g[1] += 2.0 * e / n;
        }
        g
    };

    let mut params = vec![-1.0, 0.5];
    let initial = loss(&params);
    for _ in 0..steps {
        let g = grad(&params);
        opt.step(&mut params, &g);
    }
    (initial, loss(&params))
}

#[test]
fn every_optimizer_reduces_loss() {
    let mut cases: Vec<(&str, Box<dyn Optimizer>, usize)> = vec![
        ("sgd", Box::new(Sgd::new(0.1).momentum(0.9)), 100),
        ("adam", Box::new(Adam::new(0.1)), 100),
        ("adamw", Box::new(AdamW::new(0.1)), 100),
        ("lamb", Box::new(Lamb::new(0.05)), 200),
        ("lion", Box::new(Lion::new(0.02)), 200),
    ];
    for (name, opt, steps) in cases.iter_mut() {
        let (before, after) = convex_run(opt.as_mut(), *steps);
        assert!(
            after < 0.5 * before,
            "{name}: loss {before} -> {after} (needs >50% reduction)"
        );
        assert!(after.is_finite(), "{name}: diverged");
    }
}

#[test]
fn sgd_and_adam_drive_mse_down_on_model() {
    let (x, y) = line_dataset(20);
    for (name, opt) in [
        ("sgd", Box::new(Sgd::new(0.05)) as Box<dyn Optimizer>),
        ("adam", Box::new(Adam::new(0.1))),
    ] {
        let start = Linear::<1, 1>::zeros();
        let before = evaluate(&start, &x, &y, Loss::MeanSquaredError).unwrap();
        let trained = Trainer::new(Linear::<1, 1>::zeros())
            .optimizer(BoxedOpt(opt))
            .epochs(150)
            .batch_size(5)
            .fit(&x, &y, &x, &y)
            .unwrap();
        let after = evaluate(&trained, &x, &y, Loss::MeanSquaredError).unwrap();
        assert!(after < 0.1 * before, "{name}: {before} -> {after}");
    }
}

/// Adapter so a `Box<dyn Optimizer>` can be handed to the `Trainer` builder.
struct BoxedOpt(Box<dyn Optimizer>);
impl Optimizer for BoxedOpt {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        self.0.step(params, grads)
    }
    fn lr(&self) -> f64 {
        self.0.lr()
    }
    fn set_lr(&mut self, lr: f64) {
        self.0.set_lr(lr)
    }
}

#[test]
fn cross_entropy_learns_a_separable_problem() {
    // 2 features -> 3 classes, one cluster per class.
    let x = tensor(
        6,
        2,
        vec![
            2.0, 0.0, 2.2, 0.1, // class 0
            0.0, 2.0, 0.1, 2.2, // class 1
            -2.0, -2.0, -2.1, -1.9, // class 2
        ],
    );
    let y = tensor(6, 1, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let model = Trainer::new(Linear::<2, 3>::new())
        .optimizer(Adam::new(0.1))
        .loss(Loss::CrossEntropy)
        .epochs(200)
        .batch_size(6)
        .fit(&x, &y, &x, &y)
        .unwrap();

    let logits = as_matrix(&model.forward(&x).unwrap()).unwrap();
    for (i, row) in logits.rows().into_iter().enumerate() {
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, i / 2, "row {i} misclassified");
    }
    assert!(evaluate(&model, &x, &y, Loss::CrossEntropy).unwrap() < 0.3);
}

#[test]
fn save_load_roundtrip() {
    let dir = std::env::temp_dir().join("tpt-learn-tests");
    let path = dir.join("linear_roundtrip.bin");
    let model = Linear::<3, 2>::with_seed(42);
    model.save(&path).expect("save");

    let loaded = Linear::<3, 2>::load(&path).expect("load");
    assert_eq!(model.weight, loaded.weight);
    assert_eq!(model.bias, loaded.bias);
    assert_eq!(model.params(), loaded.params());

    // Same predictions after a round-trip.
    let x = tensor(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.25]);
    assert_eq!(
        model.forward(&x).unwrap().to_vec(),
        loaded.forward(&x).unwrap().to_vec()
    );

    // Shape-mismatched checkpoints are rejected instead of silently loading.
    assert!(Linear::<2, 2>::load(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn trainer_checkpoint_writes_best_model() {
    let dir = std::env::temp_dir().join("tpt-learn-tests/ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    let (x, y) = line_dataset(12);
    let model = Trainer::new(Linear::<1, 1>::new())
        .optimizer(Adam::new(0.1))
        .epochs(50)
        .batch_size(4)
        .checkpoint(&dir)
        .fit(&x, &y, &x, &y)
        .unwrap();

    let saved = Linear::<1, 1>::load(dir.join("best_model.bin")).expect("checkpoint exists");
    assert_eq!(saved.params(), model.params());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dataloader_batch_counts() {
    let (x, y) = line_dataset(10);
    for (bs, expected) in [(1usize, 10usize), (2, 5), (3, 4), (4, 3), (10, 1), (16, 1)] {
        let loader = DataLoader::new(&x, Some(&y), bs).unwrap();
        assert_eq!(loader.num_batches(), expected, "batch_size {bs}");
        let batches: Vec<_> = DataLoader::new(&x, Some(&y), bs).unwrap().collect();
        assert_eq!(batches.len(), expected);
        assert_eq!(batches.iter().map(|b| b.len).sum::<usize>(), 10);
        for b in &batches {
            assert_eq!(b.x.shape()[0], b.len);
            assert_eq!(b.y.as_ref().unwrap().shape()[0], b.len);
            assert!(b.len <= bs);
        }
        // Last batch of an uneven split is short, never padded.
        assert_eq!(batches.last().unwrap().len, 10 - bs * (expected - 1));
    }

    assert!(DataLoader::new(&x, Some(&y), 0).is_err());
    let short = tensor(3, 1, vec![0.0, 1.0, 2.0]);
    assert!(DataLoader::new(&x, Some(&short), 2).is_err());
}

#[test]
fn dataloader_slices_content_in_order() {
    let x = tensor(4, 2, vec![0., 1., 2., 3., 4., 5., 6., 7.]);
    let batches: Vec<_> = DataLoader::new(&x, None, 3).unwrap().collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].x.to_vec(), vec![0., 1., 2., 3., 4., 5.]);
    assert_eq!(batches[1].x.to_vec(), vec![6., 7.]);
    assert_eq!(batches[1].start, 3);
}

#[test]
fn lr_schedules_behave() {
    assert_eq!(LrSchedule::Constant.lr(0.1, 7), 0.1);
    let step = LrSchedule::Step {
        step_size: 10,
        gamma: 0.5,
    };
    assert!((step.lr(1.0, 0) - 1.0).abs() < 1e-12);
    assert!((step.lr(1.0, 10) - 0.5).abs() < 1e-12);
    assert!((step.lr(1.0, 25) - 0.25).abs() < 1e-12);

    let cos = LrSchedule::Cosine {
        t_max: 10,
        min_lr: 0.0,
    };
    assert!((cos.lr(1.0, 0) - 1.0).abs() < 1e-12);
    assert!((cos.lr(1.0, 5) - 0.5).abs() < 1e-12);
    assert!(cos.lr(1.0, 10).abs() < 1e-12);
}

#[test]
fn shape_errors_are_reported_not_panics() {
    let x = tensor(4, 2, vec![0.; 8]);
    let y = tensor(4, 1, vec![0.; 4]);
    // Linear<1, 1> cannot consume 2 features.
    let err = Trainer::new(Linear::<1, 1>::new())
        .epochs(1)
        .fit(&x, &y, &x, &y)
        .unwrap_err();
    assert!(err.contains("expected 1 features"), "{err}");

    let model = Linear::<2, 1>::new();
    assert!(model.try_forward(&y).is_err());
    // `Linear` gains real exporters behind the `wasm-export`/`onnx` features.
    #[cfg(feature = "wasm-export")]
    assert!(model.export_wasm("model.wasm").is_ok());
    #[cfg(not(feature = "wasm-export"))]
    assert!(model.export_wasm("model.wasm").is_err());
    #[cfg(feature = "onnx")]
    assert!(model.to_onnx("model.onnx").is_ok());
    #[cfg(not(feature = "onnx"))]
    assert!(model.to_onnx("model.onnx").is_err());
}

#[test]
fn model_params_roundtrip_and_shapes() {
    let mut model = Linear::<3, 2>::new();
    assert_eq!(Linear::<3, 2>::IN_DIM, 3);
    assert_eq!(Linear::<3, 2>::OUT_DIM, 2);
    assert_eq!(model.num_params(), 3 * 2 + 2);

    let target: Vec<f64> = (0..8).map(|i| i as f64).collect();
    model.set_params(&target);
    assert_eq!(model.params(), target);

    let x = tensor(2, 3, vec![1., 0., 0., 0., 1., 0.]);
    let out = model.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    // Row 0 selects weight row 0 plus bias.
    assert_eq!(out.to_vec()[0], 0.0 + 6.0);
    assert_eq!(out.to_vec()[1], 1.0 + 7.0);
}
