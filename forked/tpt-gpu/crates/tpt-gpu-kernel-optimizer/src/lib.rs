//! TPT GPU kernel parameter optimizer.
//!
//! Three-phase search strategy:
//!   1. Grid search — exhaustive sweep of the full parameter space
//!   2. Hill-climbing — greedy local improvement from the best grid result
//!   3. AI-guided search — LLM suggests next candidates based on history

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Kernel evaluator modules
// ---------------------------------------------------------------------------

pub mod real_evaluator;
pub use real_evaluator::{
    generate_milestone_json, generate_milestone_report, optimize_all_gemm_problems,
    optimize_gemm_problem, standard_gemm_problems, GemmOptResult, GemmProblemConfig,
    RealGemmEvaluator,
};

pub mod attention_eval;
pub use attention_eval::{
    generate_attention_milestone_json, generate_attention_milestone_report,
    optimize_all_attention_problems, optimize_attention_problem, standard_attention_problems,
    AttentionOptResult, AttentionProblemConfig, RealAttentionEvaluator,
};

pub mod fused_eval;
pub use fused_eval::{
    fused_gemm_problems, generate_beat_cublas_json, generate_beat_cublas_report,
    optimize_fused_problem, run_beat_cublas_campaign, FusedGemmEvaluator, FusedGemmOptResult,
    FusedGemmProblem,
};

pub mod conv2d_eval;
pub use conv2d_eval::{
    generate_conv2d_milestone_json, generate_conv2d_milestone_report, optimize_all_conv2d_problems,
    optimize_conv2d_problem, standard_conv2d_problems, Conv2dOptResult, Conv2dProblemConfig,
    RealConv2dEvaluator,
};

pub mod normalization_eval;
pub use normalization_eval::{
    generate_normalization_milestone_json, generate_normalization_milestone_report,
    optimize_all_normalization_problems, optimize_normalization_problem,
    standard_normalization_problems, NormalizationOptResult, NormalizationProblemConfig,
    RealNormalizationEvaluator,
};

pub mod vector_add_eval;
pub use vector_add_eval::{
    generate_vector_add_milestone_json, generate_vector_add_milestone_report,
    optimize_all_vector_add_problems, optimize_vector_add_problem, standard_vector_add_problems,
    RealVectorAddEvaluator, VectorAddOptResult, VectorAddProblemConfig,
};

pub mod kernel_category;
pub use kernel_category::KernelCategory;

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TuningParams convenience constructors
// ---------------------------------------------------------------------------

impl<const N: usize> From<[(&'static str, u32); N]> for TuningParams {
    fn from(arr: [(&'static str, u32); N]) -> Self {
        let mut map = HashMap::with_capacity(N);
        for (k, v) in arr {
            map.insert(k.to_string(), v);
        }
        TuningParams(map)
    }
}
// Parameter types

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

pub type ParamVal = u32;

/// A concrete point in the tuning parameter space.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TuningParams(pub HashMap<String, ParamVal>);

impl TuningParams {
    pub fn get(&self, key: &str) -> Option<ParamVal> {
        self.0.get(key).copied()
    }

    pub fn set(&mut self, key: impl Into<String>, val: ParamVal) {
        self.0.insert(key.into(), val);
    }

    pub fn display(&self) -> String {
        let mut pairs: Vec<_> = self.0.iter().collect();
        pairs.sort_by_key(|(k, _)| k.as_str());
        pairs
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// A single tuning dimension: name + allowed discrete values.
#[derive(Debug, Clone)]
pub struct ParamDim {
    pub name: String,
    pub values: Vec<ParamVal>,
}

impl ParamDim {
    pub fn new(name: impl Into<String>, values: Vec<ParamVal>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    /// Powers-of-two from `min` to `max` inclusive.
    pub fn powers_of_two(name: impl Into<String>, min: u32, max: u32) -> Self {
        let mut vals = Vec::new();
        let mut v = min;
        while v <= max {
            vals.push(v);
            v *= 2;
        }
        Self {
            name: name.into(),
            values: vals,
        }
    }
}

/// The full discrete parameter space.
#[derive(Debug, Clone)]
pub struct ParamSpace {
    pub dims: Vec<ParamDim>,
}

impl ParamSpace {
    pub fn new(dims: Vec<ParamDim>) -> Self {
        Self { dims }
    }

    /// Default GEMM tuning space (~256 configurations).
    pub fn gemm_default() -> Self {
        Self::new(vec![
            ParamDim::powers_of_two("tile_m", 16, 128),
            ParamDim::powers_of_two("tile_n", 16, 128),
            ParamDim::powers_of_two("tile_k", 8, 64),
            ParamDim::powers_of_two("vec_width", 1, 8),
            ParamDim::new("unroll", vec![1, 2, 4, 8]),
        ])
    }

    /// Default Attention tuning space (~512 configurations).
    pub fn attention_default() -> Self {
        Self::new(vec![
            ParamDim::powers_of_two("tile_seq", 16, 128),
            ParamDim::powers_of_two("tile_head", 4, 64),
            ParamDim::powers_of_two("tile_k", 16, 64),
            ParamDim::powers_of_two("vec_width", 1, 8),
            ParamDim::new("unroll", vec![1, 2, 4, 8]),
        ])
    }

    /// Number of configurations (Cartesian product size).
    pub fn total_configs(&self) -> usize {
        self.dims.iter().map(|d| d.values.len()).product()
    }

    /// All configurations via Cartesian product.
    pub fn all_params(&self) -> Vec<TuningParams> {
        let mut result: Vec<HashMap<String, ParamVal>> = vec![HashMap::new()];
        for dim in &self.dims {
            let mut next = Vec::with_capacity(result.len() * dim.values.len());
            for existing in &result {
                for &val in &dim.values {
                    let mut p = existing.clone();
                    p.insert(dim.name.clone(), val);
                    next.push(p);
                }
            }
            result = next;
        }
        result.into_iter().map(TuningParams).collect()
    }

    /// All single-step neighbors of a point (vary one dim at a time).
    pub fn neighbors(&self, params: &TuningParams) -> Vec<TuningParams> {
        let mut neighbors = Vec::new();
        for dim in &self.dims {
            let current = match params.0.get(&dim.name) {
                Some(&v) => v,
                None => continue,
            };
            let idx = match dim.values.iter().position(|&v| v == current) {
                Some(i) => i,
                None => continue,
            };
            if idx > 0 {
                let mut n = params.clone();
                n.0.insert(dim.name.clone(), dim.values[idx - 1]);
                neighbors.push(n);
            }
            if idx + 1 < dim.values.len() {
                let mut n = params.clone();
                n.0.insert(dim.name.clone(), dim.values[idx + 1]);
                neighbors.push(n);
            }
        }
        neighbors
    }

    /// Nearest valid point: clamp each dim to its closest allowed value.
    pub fn clamp(&self, raw: &HashMap<String, u64>) -> TuningParams {
        let mut params = HashMap::new();
        for dim in &self.dims {
            let target = raw.get(&dim.name).copied().unwrap_or(dim.values[0] as u64) as i64;
            let closest = dim
                .values
                .iter()
                .min_by_key(|&&v| (v as i64 - target).abs())
                .copied()
                .unwrap_or(dim.values[0]);
            params.insert(dim.name.clone(), closest);
        }
        TuningParams(params)
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Optimization result for a single parameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptResult {
    pub params: TuningParams,
    /// Simulated GFLOPS (higher is better).
    pub score: f64,
    /// Cumulative evaluations used to reach this result.
    pub eval_count: usize,
}

pub trait KernelEvaluator: Send {
    fn evaluate(&self, params: &TuningParams) -> f64;
}

/// Synthetic evaluator — no hardware required.
///
/// Peak is near tile_m=64, tile_n=64, tile_k=32, vec_width=4, unroll=4.
pub struct SimulatedEvaluator {
    pub kernel_name: String,
}

impl SimulatedEvaluator {
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
        }
    }
}

impl KernelEvaluator for SimulatedEvaluator {
    fn evaluate(&self, params: &TuningParams) -> f64 {
        let category = crate::kernel_category::KernelCategory::from_name(&self.kernel_name);
        let base = match category {
            Some(crate::kernel_category::KernelCategory::Gemm) => Self::score_gemm(params),
            Some(crate::kernel_category::KernelCategory::Attention) => {
                Self::score_attention(params)
            }
            Some(crate::kernel_category::KernelCategory::Conv2d) => Self::score_conv2d(params),
            Some(crate::kernel_category::KernelCategory::VectorAdd) => {
                Self::score_vector_add(params)
            }
            Some(crate::kernel_category::KernelCategory::Normalization) => {
                Self::score_normalization(params)
            }
            None => Self::score_gemm(params),
        };
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        params.display().hash(&mut h);
        let noise = ((h.finish() % 200) as f64 / 200.0 - 0.5) * 0.04 * base;
        (base + noise).max(1.0)
    }
}

impl SimulatedEvaluator {
    pub fn score_gemm(params: &TuningParams) -> f64 {
        let tile_m = params.get("tile_m").unwrap_or(32) as f64;
        let tile_n = params.get("tile_n").unwrap_or(32) as f64;
        let tile_k = params.get("tile_k").unwrap_or(16) as f64;
        let vec_width = params.get("vec_width").unwrap_or(4) as f64;
        let unroll = params.get("unroll").unwrap_or(4) as f64;
        let tile_score = (1.0 - ((tile_m - 64.0).abs() + (tile_n - 64.0).abs()) / 256.0).max(0.2);
        let k_score = (1.0 - (tile_k - 32.0).abs() / 64.0).max(0.2);
        let vec_score = (vec_width / 4.0).min(1.0);
        let unroll_score = (unroll / 4.0).min(1.0);
        120.0 * tile_score * k_score * vec_score * unroll_score
    }

    pub fn score_attention(params: &TuningParams) -> f64 {
        let block_q = params.get("block_q").unwrap_or(32) as f64;
        let block_kv = params.get("block_kv").unwrap_or(32) as f64;
        let num_heads = params.get("num_heads").unwrap_or(4) as f64;
        let head_dim = params.get("head_dim").unwrap_or(64) as f64;
        let unroll = params.get("unroll").unwrap_or(2) as f64;
        let bq_score = (1.0 - (block_q - 64.0).abs() / 128.0).max(0.2);
        let bkv_score = (1.0 - (block_kv - 64.0).abs() / 128.0).max(0.2);
        let heads_score = (num_heads / 8.0).clamp(0.25, 1.0);
        let dim_score = (1.0 - (head_dim - 64.0).abs() / 128.0).max(0.3);
        let unroll_score = (unroll / 4.0).clamp(0.5, 1.0);
        200.0 * bq_score * bkv_score * heads_score * dim_score * unroll_score
    }

    pub fn score_conv2d(params: &TuningParams) -> f64 {
        let tile_oc = params.get("tile_oc").unwrap_or(32) as f64;
        let tile_ic = params.get("tile_ic").unwrap_or(16) as f64;
        let tile_oh = params.get("tile_oh").unwrap_or(8) as f64;
        let tile_ow = params.get("tile_ow").unwrap_or(8) as f64;
        let kernel_w = params.get("kernel_w").unwrap_or(3) as f64;
        let kernel_h = params.get("kernel_h").unwrap_or(3) as f64;
        let stride = params.get("stride").unwrap_or(1) as f64;
        let unroll = params.get("unroll").unwrap_or(2) as f64;
        let oc_score = (1.0 - (tile_oc - 64.0).abs() / 128.0).max(0.2);
        let ic_score = (1.0 - (tile_ic - 32.0).abs() / 64.0).max(0.2);
        let oh_score = (1.0 - (tile_oh - 16.0).abs() / 32.0).max(0.2);
        let ow_score = (1.0 - (tile_ow - 16.0).abs() / 32.0).max(0.2);
        let kw_score = (1.0 - (kernel_w - 3.0).abs() / 8.0).max(0.3);
        let kh_score = (1.0 - (kernel_h - 3.0).abs() / 8.0).max(0.3);
        let stride_score = if stride <= 1.0 { 1.0 } else { 0.7 };
        let unroll_score = (unroll / 4.0).clamp(0.5, 1.0);
        150.0
            * oc_score
            * ic_score
            * oh_score
            * ow_score
            * kw_score
            * kh_score
            * stride_score
            * unroll_score
    }

    pub fn score_vector_add(params: &TuningParams) -> f64 {
        let block_size = params.get("block_size").unwrap_or(128) as f64;
        let vec_width = params.get("vec_width").unwrap_or(2) as f64;
        let grid_size = params.get("grid_size").unwrap_or(16) as f64;
        let bs_score = (1.0 - (block_size - 256.0).abs() / 512.0).max(0.2);
        let vw_score = (vec_width / 4.0).clamp(0.25, 1.0);
        let gs_score = (grid_size / 32.0).clamp(0.25, 1.0);
        16.0 * bs_score * vw_score * gs_score
    }

    pub fn score_normalization(params: &TuningParams) -> f64 {
        let block_size = params.get("block_size").unwrap_or(128) as f64;
        let vec_width = params.get("vec_width").unwrap_or(2) as f64;
        let unroll = params.get("unroll").unwrap_or(2) as f64;
        let warp_reduce = params.get("warp_reduce").unwrap_or(1) as f64;
        let bs_score = (1.0 - (block_size - 256.0).abs() / 512.0).max(0.2);
        let vw_score = (vec_width / 4.0).clamp(0.25, 1.0);
        let ur_score = (unroll / 4.0).clamp(0.5, 1.0);
        let wr_score = if warp_reduce >= 1.0 { 1.0 } else { 0.6 };
        32.0 * bs_score * vw_score * ur_score * wr_score
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Grid search
// ---------------------------------------------------------------------------

/// Evaluate every point in the space; return sorted results (best first).
pub fn grid_search(space: &ParamSpace, eval: &dyn KernelEvaluator) -> Vec<OptResult> {
    let all = space.all_params();
    let total = all.len();
    let mut results: Vec<OptResult> = all
        .iter()
        .enumerate()
        .map(|(i, params)| {
            let score = eval.evaluate(params);
            if i == 0 || (i + 1) % 50 == 0 || i + 1 == total {
                eprintln!("  grid [{}/{}] best so far: {:.2}", i + 1, total, score);
            }
            OptResult {
                params: params.clone(),
                score,
                eval_count: i + 1,
            }
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ---------------------------------------------------------------------------
// Phase 2: Hill-climbing
// ---------------------------------------------------------------------------

/// Greedy hill-climb from `start`; stops when no neighbor improves the score.
pub fn hill_climb(
    space: &ParamSpace,
    start: &TuningParams,
    eval: &dyn KernelEvaluator,
    max_iters: usize,
) -> OptResult {
    let mut current = start.clone();
    let mut current_score = eval.evaluate(&current);
    let mut eval_count = 1;

    for iter in 0..max_iters {
        let neighbors = space.neighbors(&current);
        let mut improved = false;

        for neighbor in &neighbors {
            let score = eval.evaluate(neighbor);
            eval_count += 1;
            if score > current_score {
                eprintln!(
                    "  hill-climb [iter {}] {} → {:.2}",
                    iter + 1,
                    neighbor.display(),
                    score
                );
                current = neighbor.clone();
                current_score = score;
                improved = true;
                break;
            }
        }

        if !improved {
            eprintln!(
                "  hill-climb converged after {} iters ({} evals)",
                iter + 1,
                eval_count
            );
            break;
        }
    }

    OptResult {
        params: current,
        score: current_score,
        eval_count,
    }
}

// ---------------------------------------------------------------------------
// Phase 3: AI-guided search
// ---------------------------------------------------------------------------

/// LLM-guided search: ask the AI for the next candidate, evaluate, repeat.
///
/// Falls back silently if the AI response cannot be parsed — the best
/// result seen so far is always returned.
pub fn ai_guided_search(
    space: &ParamSpace,
    initial: &TuningParams,
    eval: &dyn KernelEvaluator,
    provider: &dyn tpt_gpu_shared::AiProvider,
    kernel_name: &str,
    iterations: usize,
) -> OptResult {
    let mut best = initial.clone();
    let mut best_score = eval.evaluate(initial);
    let mut history: Vec<(TuningParams, f64)> = vec![(best.clone(), best_score)];
    let mut eval_count = 1;

    for iter in 0..iterations {
        let space_desc = space
            .dims
            .iter()
            .map(|d| format!("  {}: {:?}", d.name, d.values))
            .collect::<Vec<_>>()
            .join("\n");

        // Show the 5 best trials seen so far
        let mut sorted_history = history.clone();
        sorted_history.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let history_str = sorted_history
            .iter()
            .take(5)
            .map(|(p, s)| format!("  {} → {:.2} GFLOPS", p.display(), s))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are optimizing GPU kernel tuning parameters. \
             Suggest the single best next parameter set to try.\n\n\
             Kernel: {}\n\
             Parameter space (name: allowed values):\n{}\n\n\
             Current best: {} → {:.2} GFLOPS\n\
             Top trials so far:\n{}\n\n\
             Rules:\n\
             - Each value MUST be from the allowed list above.\n\
             - Respond with ONLY a JSON object, nothing else.\n\
             - Example: {{\"tile_m\": 64, \"tile_n\": 64, \"tile_k\": 32, \"vec_width\": 4, \"unroll\": 4}}",
            kernel_name, space_desc, best.display(), best_score, history_str
        );

        eprintln!(
            "  ai-search [iter {}] querying {}...",
            iter + 1,
            provider.name()
        );

        match provider.generate(&prompt) {
            Ok(response) => {
                if let Some(candidate) = parse_json_params(&response, space) {
                    let score = eval.evaluate(&candidate);
                    eval_count += 1;
                    eprintln!(
                        "  ai-search [iter {}] {} → {:.2}",
                        iter + 1,
                        candidate.display(),
                        score
                    );
                    if score > best_score {
                        best = candidate.clone();
                        best_score = score;
                    }
                    history.push((candidate, score));
                } else {
                    eprintln!(
                        "  ai-search [iter {}] could not parse response, skipping",
                        iter + 1
                    );
                }
            }
            Err(e) => {
                eprintln!("  ai-search [iter {}] provider error: {}", iter + 1, e);
            }
        }
    }

    OptResult {
        params: best,
        score: best_score,
        eval_count,
    }
}

fn parse_json_params(response: &str, space: &ParamSpace) -> Option<TuningParams> {
    let start = response.find('{')?;
    let end = response.rfind('}')?;
    let json_str = &response[start..=end];
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = val.as_object()?;

    let mut raw: HashMap<String, u64> = HashMap::new();
    for (k, v) in obj {
        raw.insert(k.clone(), v.as_u64()?);
    }

    // Must cover all dims
    if space.dims.iter().all(|d| raw.contains_key(&d.name)) {
        Some(space.clamp(&raw))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(tm: u32, tn: u32, tk: u32, vw: u32, ur: u32) -> TuningParams {
        TuningParams(HashMap::from([
            ("tile_m".to_string(), tm),
            ("tile_n".to_string(), tn),
            ("tile_k".to_string(), tk),
            ("vec_width".to_string(), vw),
            ("unroll".to_string(), ur),
        ]))
    }

    #[test]
    fn test_param_space_total() {
        let space = ParamSpace::gemm_default();
        // tile_m: 4 vals, tile_n: 4, tile_k: 4, vec_width: 4, unroll: 4 → 4^5 = 1024
        // (Actually: 16,32,64,128 = 4; 8,16,32,64 = 4; 1,2,4,8 = 4; 1,2,4,8 = 4)
        assert_eq!(space.total_configs(), 4 * 4 * 4 * 4 * 4);
    }

    #[test]
    fn test_neighbors() {
        let space = ParamSpace::gemm_default();
        let p = make_params(64, 64, 32, 4, 4);
        let neighbors = space.neighbors(&p);
        // Each dim has 2 neighbors (prev/next), except edge dims have 1.
        // tile_m=64 has 32 and 128 (2), tile_n=64 (2), tile_k=32 (2), vec_width=4 (2), unroll=4 (2)
        assert_eq!(neighbors.len(), 10);
    }

    #[test]
    fn test_simulated_evaluator_peak() {
        let eval = SimulatedEvaluator::new("matmul");
        let best = make_params(64, 64, 32, 4, 4);
        let score = eval.evaluate(&best);
        assert!(score > 100.0, "Peak should score above 100 GFLOPS");
    }

    #[test]
    fn test_grid_search_returns_sorted() {
        let space = ParamSpace::gemm_default();
        let eval = SimulatedEvaluator::new("matmul");
        let results = grid_search(&space, &eval);
        assert_eq!(results.len(), space.total_configs());
        // Sorted descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_hill_climb_improves() {
        let space = ParamSpace::gemm_default();
        let start = make_params(16, 16, 8, 1, 1); // poor starting point
        let eval = SimulatedEvaluator::new("matmul");
        let initial_score = eval.evaluate(&start);
        let result = hill_climb(&space, &start, &eval, 50);
        assert!(result.score >= initial_score);
    }

    // ---------------------------------------------------------------------------
    // Attention evaluator tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_attention_param_space_total() {
        let space = ParamSpace::attention_default();
        let expected: usize = space.dims.iter().map(|d| d.values.len()).product();
        assert_eq!(space.total_configs(), expected);
        assert!(expected > 0);
    }

    #[test]
    fn test_attention_evaluator_reasonable_efficiency() {
        let problem = AttentionProblemConfig::new(1024, 8, 64, 0.8);
        let eval = RealAttentionEvaluator::new(problem);
        let params = TuningParams(HashMap::from([
            ("tile_seq".to_string(), 64),
            ("tile_head".to_string(), 16),
            ("tile_k".to_string(), 32),
            ("vec_width".to_string(), 4),
            ("unroll".to_string(), 4),
        ]));
        let eff = eval.evaluate(&params);
        assert!(eff > 0.0, "Efficiency should be positive, got {}", eff);
    }

    #[test]
    fn test_attention_problems_standard_count() {
        let problems = standard_attention_problems();
        assert_eq!(
            problems.len(),
            7,
            "Should have 7 standard attention problems"
        );
    }

    #[test]
    fn test_attention_problem_gflops() {
        let p = AttentionProblemConfig::new(1024, 8, 64, 0.8);
        let expected = 4.0 * 1024.0 * 1024.0 * 64.0 * 8.0;
        assert_eq!(p.flops(), expected);
    }
}
