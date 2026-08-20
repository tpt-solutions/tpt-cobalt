//! Dimensional tolerance stack-up analysis (worst-case / RSS / Monte-Carlo).

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tpt_eng_tolerance::{self, DimTol, StackupResult};

#[derive(Args)]
pub struct ToleranceArgs {
    #[command(subcommand)]
    pub cmd: ToleranceCmd,
}

#[derive(Subcommand)]
pub enum ToleranceCmd {
    /// Worst-case / RSS / Monte-Carlo stack-up of the given dimensions.
    Stackup {
        /// Dimensions as `name=nominal±tol` (symmetric) or
        /// `name=nominal;plus;minus` (asymmetric). Repeatable.
        #[arg(short = 'D', long = "dim", required = true)]
        dims: Vec<String>,
        /// Methods to report: any of `worst`,`rss`,`monte` (default: all).
        #[arg(long, default_value = "worst,rss,monte")]
        methods: String,
        /// Monte-Carlo sample count.
        #[arg(long, default_value_t = 100_000)]
        samples: usize,
        /// Spec lower bound (enables yield estimate for `monte`).
        #[arg(long)]
        low: Option<f64>,
        /// Spec upper bound.
        #[arg(long)]
        high: Option<f64>,
        /// Fixed RNG seed (otherwise time-seeded).
        #[arg(long)]
        seed: Option<u64>,
    },
}

fn parse_dim(spec: &str) -> Result<DimTol> {
    let (name, rest) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("dim must be name=nominal±tol: {spec}"))?;
    let name = name.trim().to_string();
    let rest = rest.trim();

    if let Some((n, t)) = rest.split_once('±') {
        let nominal = n.trim().parse::<f64>()?;
        let tol = t.trim().parse::<f64>()?;
        return Ok(DimTol::new(name, nominal, tol));
    }

    // Semicolon form: "nominal" / "nominal;tol" / "nominal;plus;minus".
    let parts: Vec<&str> = rest.split(';').map(str::trim).collect();
    let nominal = parts[0].parse::<f64>()?;
    match parts.len() {
        1 => Ok(DimTol::new(name, nominal, 0.0)),
        2 => {
            let tol = parts[1].parse::<f64>()?;
            Ok(DimTol::new(name, nominal, tol))
        }
        3 => {
            let plus = parts[1].parse::<f64>()?;
            let minus = parts[2].parse::<f64>()?;
            Ok(DimTol::asymmetric(name, nominal, plus, minus))
        }
        _ => bail!("invalid dim spec: {spec}"),
    }
}

pub fn run(args: ToleranceArgs) -> Result<()> {
    match args.cmd {
        ToleranceCmd::Stackup {
            dims,
            methods,
            samples,
            low,
            high,
            seed,
        } => run_stackup(&dims, &methods, samples, low, high, seed),
    }
}

fn run_stackup(
    dims: &[String],
    methods: &str,
    samples: usize,
    low: Option<f64>,
    high: Option<f64>,
    seed: Option<u64>,
) -> Result<()> {
    let parsed: Vec<DimTol> = dims.iter().map(|d| parse_dim(d)).collect::<Result<_>>()?;
    let nominal_sum: f64 = parsed.iter().map(|d| d.nominal).sum();
    let spec = match (low, high) {
        (Some(l), Some(h)) => Some((l, h)),
        _ => None,
    };

    let want = |m: &str| methods.split(',').any(|x| x.trim() == m);

    println!("Tolerance stack-up ({} dimensions)", parsed.len());
    for d in &parsed {
        println!(
            "  {}: nominal={:.4} range=[{:.4}, {:.4}]",
            d.name,
            d.nominal,
            d.min(),
            d.max()
        );
    }
    println!("  nominal sum      = {:.4}", nominal_sum);

    if want("worst") {
        let (lo, hi) = tpt_eng_tolerance::worst_case(&parsed);
        println!(
            "  worst-case range = [{:.4}, {:.4}]  (width {:.4})",
            lo,
            hi,
            hi - lo
        );
    }
    if want("rss") {
        let (lo, hi) = tpt_eng_tolerance::rss(&parsed);
        println!(
            "  RSS (3σ) range   = [{:.4}, {:.4}]  (width {:.4})",
            lo,
            hi,
            hi - lo
        );
    }
    if want("monte") {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(0xC0FFEE),
        };
        let mc: StackupResult = tpt_eng_tolerance::monte_carlo(&parsed, samples, spec, &mut rng);
        println!(
            "  Monte-Carlo (n={}): mean={:.4} std={:.4} min={:.4} max={:.4}",
            mc.n, mc.mean, mc.std, mc.min, mc.max
        );
        if let Some(y) = mc.yield_fraction {
            println!("    yield within spec = {:.4} %", y * 100.0);
        }
    }
    Ok(())
}
