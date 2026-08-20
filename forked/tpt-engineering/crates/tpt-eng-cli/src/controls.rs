//! PID controller step-response simulation over a first-order plant.

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;
use tpt_eng_controls::{Pid, PidGains};
use tpt_eng_plot::{Drawing, XyPlot, XySeries};

#[derive(Args)]
pub struct PidArgs {
    /// Proportional gain Kp.
    kp: f64,
    /// Integral gain Ki.
    ki: f64,
    /// Derivative gain Kd.
    kd: f64,
    /// Controller setpoint.
    #[arg(long, default_value_t = 1.0)]
    setpoint: f64,
    /// Plant first-order time constant (s).
    #[arg(long, default_value_t = 1.0)]
    tau: f64,
    /// Controller sample period (s).
    #[arg(long, default_value_t = 0.01)]
    dt: f64,
    /// Number of simulation steps.
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    /// Symmetric output saturation (omit for none).
    #[arg(long)]
    limit: Option<f64>,
    /// Optional PNG chart of the step response (time series of y and u).
    #[arg(long)]
    plot: Option<PathBuf>,
    /// Optional CSV time-series output (columns: t,y,u).
    #[arg(long)]
    csv: Option<PathBuf>,
}

pub fn run(args: PidArgs) -> Result<()> {
    anyhow::ensure!(args.dt > 0.0, "dt must be > 0");
    anyhow::ensure!(args.steps > 0, "steps must be > 0");

    let mut pid = Pid::new(PidGains::new(args.kp, args.ki, args.kd));
    if let Some(lim) = args.limit {
        pid = pid.with_output_limit(lim);
    }
    pid.set_setpoint(args.setpoint);

    // First-order plant: dy/dt = (u - y) / tau, forward-Euler integration.
    let mut y = 0.0_f64;
    let mut ts: Vec<f64> = Vec::with_capacity(args.steps);
    let mut ys: Vec<f64> = Vec::with_capacity(args.steps);
    let mut us: Vec<f64> = Vec::with_capacity(args.steps);
    let mut peak: f64 = 0.0;
    let mut saturated_steps = 0usize;

    for k in 0..args.steps {
        let u = pid.update(y, args.dt);
        y += args.dt * (u - y) / args.tau;
        if u.abs() >= args.limit.unwrap_or(f64::INFINITY) {
            saturated_steps += 1;
        }
        let t = (k as f64) * args.dt;
        ts.push(t);
        ys.push(y);
        us.push(u);
        if y > peak {
            peak = y;
        }
    }

    let final_y = *ys.last().unwrap();
    let overshoot = if args.setpoint > 0.0 {
        ((peak - args.setpoint) / args.setpoint).clamp(0.0, f64::INFINITY)
    } else {
        0.0
    };
    let steady_error = args.setpoint - final_y;

    println!("PID step response (first-order plant)");
    println!(
        "  gains            = Kp={} Ki={} Kd={}",
        args.kp, args.ki, args.kd
    );
    println!("  setpoint         = {}", args.setpoint);
    println!("  plant tau        = {} s", args.tau);
    println!("  sample dt        = {} s ({} steps)", args.dt, args.steps);
    if let Some(lim) = args.limit {
        println!(
            "  output limit     = ±{} (saturated {}/{} steps)",
            lim, saturated_steps, args.steps
        );
    }
    println!("  final y          = {:.6}", final_y);
    println!("  peak y           = {:.6}", peak);
    println!("  overshoot        = {:.2} %", overshoot * 100.0);
    println!("  steady-state err = {:.6e}", steady_error);

    if let Some(path) = args.csv {
        let mut s = String::from("t,y,u\n");
        for k in 0..args.steps {
            s.push_str(&format!("{},{},{}\n", ts[k], ys[k], us[k]));
        }
        std::fs::write(&path, s)?;
        println!("wrote time series {}", path.display());
    }

    if let Some(path) = args.plot {
        let y_series: Vec<(f64, f64)> = ts.iter().copied().zip(ys.iter().copied()).collect();
        let u_series: Vec<(f64, f64)> = ts.iter().copied().zip(us.iter().copied()).collect();
        let plot = XyPlot::new("PID step response")
            .with_x_label("t (s)")
            .with_y_label("signal")
            .with_series(XySeries::new("y (output)", y_series))
            .with_series(XySeries::new("u (control)", u_series));
        plot.save_png(&path)?;
        println!("wrote step-response chart {}", path.display());
    }

    Ok(())
}
