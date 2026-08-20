//! Command-line interface for the TPT engineering ecosystem.

use anyhow::{Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use tpt_eng_plot::Drawing;

mod controls;
mod materials;
mod props;
mod sections;
mod tolerance;
mod units;

use materials::{describe as describe_material, find as find_material};
use sections::{Section, describe as describe_section};
use units::convert as convert_units;

#[derive(Parser)]
#[command(name = "tpt-eng", version, about = "TPT engineering CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate an engineering input file (json / csv / stl / obj).
    Validate { path: PathBuf },
    /// Generate a calculation report (demonstrates tpt-eng-report).
    Report(ReportArgs),
    /// Unit conversions.
    Units(UnitsArgs),
    /// Inspect built-in material properties.
    Materials(MaterialsArgs),
    /// Inspect built-in cross-section properties.
    Sections(SectionsArgs),
    /// Run a simple calculation.
    Calc(CalcArgs),
    /// Fluid/fuel property lookups (water/steam, moist air, fuels).
    Props(props::PropsArgs),
    /// PID controller step-response simulation.
    Pid(controls::PidArgs),
    /// Dimensional tolerance stack-up analysis.
    Tolerance(tolerance::ToleranceArgs),
}

#[derive(Args)]
struct ReportArgs {
    /// Report title.
    #[arg(long, default_value = "TPT Calculation Report")]
    title: String,
    /// Output path; extension (.md / .html / .json) selects the format.
    #[arg(long, default_value = "report.md")]
    out: PathBuf,
    /// Optional PNG chart of the demo results.
    #[arg(long)]
    chart: Option<PathBuf>,
}

#[derive(Args)]
struct UnitsArgs {
    #[command(subcommand)]
    cmd: UnitsCmd,
}

#[derive(Subcommand)]
enum UnitsCmd {
    /// Convert a value between two units.
    Convert {
        /// Value to convert.
        value: f64,
        /// Source unit (e.g. m, mm, in, N, kN, psi, C, F, K).
        from: String,
        /// Target unit.
        to: String,
    },
    /// List supported units.
    List,
}

#[derive(Args)]
struct MaterialsArgs {
    /// Material name (steel, aluminium, concrete, timber, titanium, copper). Lists all if omitted.
    name: Option<String>,
}

#[derive(Args)]
struct SectionsArgs {
    #[command(subcommand)]
    cmd: SectionsCmd,
}

#[derive(Subcommand)]
enum SectionsCmd {
    /// Inspect a section's geometric properties.
    Inspect {
        #[command(subcommand)]
        shape: ShapeCmd,
    },
}

#[derive(Subcommand)]
enum ShapeCmd {
    /// Rectangle section.
    Rectangle {
        /// Breadth.
        b: f64,
        /// Height.
        h: f64,
    },
    /// Solid circular section.
    Circle {
        /// Diameter.
        d: f64,
    },
    /// Symmetric I-beam.
    IBeam {
        /// Total depth.
        d: f64,
        /// Flange width.
        bf: f64,
        /// Flange thickness.
        tf: f64,
        /// Web thickness.
        tw: f64,
    },
}

#[derive(Args)]
struct CalcArgs {
    #[command(subcommand)]
    cmd: CalcCmd,
}

#[derive(Subcommand)]
enum CalcCmd {
    /// Simply supported beam under a uniformly distributed load.
    Beam {
        /// Span length (m).
        span: f64,
        /// Uniformly distributed load (N/m).
        load: f64,
        /// Material name.
        #[arg(long, default_value = "steel")]
        material: String,
        /// Section geometry.
        #[command(subcommand)]
        section: ShapeCmd,
        /// Optional report output path (.md/.html/.json).
        #[arg(long)]
        report: Option<PathBuf>,
        /// Optional deflection diagram PNG path.
        #[arg(long)]
        plot: Option<PathBuf>,
    },
}

fn shape_from_cmd(cmd: &ShapeCmd) -> Section {
    match cmd {
        ShapeCmd::Rectangle { b, h } => Section::Rectangle { b: *b, h: *h },
        ShapeCmd::Circle { d } => Section::Circle { d: *d },
        ShapeCmd::IBeam { d, bf, tf, tw } => Section::IBeam {
            d: *d,
            bf: *bf,
            tf: *tf,
            tw: *tw,
        },
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate { path } => cmd_validate(&path),
        Command::Report(args) => cmd_report(args),
        Command::Units(args) => cmd_units(args),
        Command::Materials(args) => cmd_materials(args),
        Command::Sections(args) => cmd_sections(args),
        Command::Calc(args) => cmd_calc(args),
        Command::Props(args) => props::run(args),
        Command::Pid(args) => controls::run(args),
        Command::Tolerance(args) => tolerance::run(args),
    }
}

fn cmd_validate(path: &PathBuf) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| anyhow!("cannot determine file type for {}", path.display()))?;

    match ext.as_str() {
        "json" => {
            let _: serde_json::Value = tpt_eng_io::read_json(path)?;
            println!("OK: {} is valid JSON", path.display());
        }
        "csv" => {
            let records = tpt_eng_io::read_csv(path)?;
            println!(
                "OK: {} is valid CSV ({} records)",
                path.display(),
                records.len()
            );
        }
        "stl" => {
            let mesh = tpt_eng_io::read_stl(path)?;
            println!(
                "OK: {} is valid STL ({} triangles)",
                path.display(),
                mesh.triangle_count()
            );
        }
        "obj" => {
            let mesh = tpt_eng_io::read_obj(path)?;
            println!(
                "OK: {} is valid OBJ ({} vertices, {} faces)",
                path.display(),
                mesh.vertex_count(),
                mesh.face_count()
            );
        }
        other => bail!("unsupported file type: .{other} (expected json/csv/stl/obj)"),
    }
    Ok(())
}

fn cmd_report(args: ReportArgs) -> Result<()> {
    use tpt_eng_report::{NamedValue, Report, ResultEntry};

    let report = Report::new(args.title)
        .with_author("tpt-eng-cli")
        .with_summary("Example report generated by the tpt-eng CLI.")
        .assumptions(vec![
            NamedValue::new("Span", 5.0).with_unit("m"),
            NamedValue::new("Load", 10.0).with_unit("kN"),
        ])
        .results(vec![
            ResultEntry::with_limits(
                "Max moment",
                12.5,
                Some("kNm".into()),
                Some(0.0),
                Some(15.0),
            ),
            ResultEntry::with_limits(
                "Max deflection",
                18.0,
                Some("mm".into()),
                Some(0.0),
                Some(20.0),
            ),
        ])
        .heading("Conclusion")
        .paragraph("All checked quantities are within permissible limits.");

    write_report(&report, &args.out)?;

    if let Some(chart) = args.chart {
        use tpt_eng_plot::ResultChart;
        let c = ResultChart::new(
            "Results",
            vec![
                ("Max moment".to_string(), 12.5),
                ("Max deflection".to_string(), 18.0),
            ],
        )
        .with_unit("norm");
        c.save_png(&chart)?;
        println!("wrote chart {}", chart.display());
    }

    Ok(())
}

fn write_report(report: &tpt_eng_report::Report, path: &PathBuf) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "md".to_string());
    match ext.as_str() {
        "md" => tpt_eng_report::write_markdown(report, path)?,
        "html" => tpt_eng_report::write_html(report, path)?,
        "json" => tpt_eng_report::write_json(report, path)?,
        other => bail!("unsupported report format: .{other}"),
    }
    println!("wrote report {}", path.display());
    Ok(())
}

fn cmd_units(args: UnitsArgs) -> Result<()> {
    match args.cmd {
        UnitsCmd::Convert { value, from, to } => {
            let result = convert_units(value, &from, &to)?;
            println!("{value} {from} = {result} {to}");
        }
        UnitsCmd::List => {
            print!("{}", units::list_units());
        }
    }
    Ok(())
}

fn cmd_materials(args: MaterialsArgs) -> Result<()> {
    match args.name {
        Some(name) => {
            let m = find_material(&name).ok_or_else(|| anyhow!("unknown material: {name}"))?;
            println!("{}", describe_material(m));
        }
        None => {
            for m in materials::list() {
                println!("- {}", m.name);
            }
        }
    }
    Ok(())
}

fn cmd_sections(args: SectionsArgs) -> Result<()> {
    match args.cmd {
        SectionsCmd::Inspect { shape } => {
            let section = shape_from_cmd(&shape);
            println!("{}", describe_section(&section));
        }
    }
    Ok(())
}

fn cmd_calc(args: CalcArgs) -> Result<()> {
    match args.cmd {
        CalcCmd::Beam {
            span,
            load,
            material,
            section,
            report,
            plot,
        } => cmd_calc_beam(span, load, &material, section, report, plot),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_calc_beam(
    span: f64,
    load: f64,
    material_name: &str,
    section_cmd: ShapeCmd,
    report_out: Option<PathBuf>,
    plot_out: Option<PathBuf>,
) -> Result<()> {
    let m =
        find_material(material_name).ok_or_else(|| anyhow!("unknown material: {material_name}"))?;
    let section = shape_from_cmd(&section_cmd);

    let e = m.value("youngs-modulus", 0.0).unwrap_or(0.0);
    let i = section.second_moment_x();

    // Reactions and max moment via tpt-eng-structural (wrapped at the uom boundary);
    // the closed-form UDL deflection below stays in this crate.
    use tpt_eng_structural::{Beam, Load};
    use tpt_math_units::uom::si::f64::{Force, Length};
    use tpt_math_units::uom::si::{force::newton, length::meter, torque::newton_meter};
    let mut beam = Beam::new(Length::new::<meter>(span));
    beam.add(Load::uniform(
        Length::new::<meter>(0.0),
        Length::new::<meter>(span),
        Force::new::<newton>(load * span),
    ));
    let reaction = beam.reaction_a().get::<newton>();
    let max_moment = beam.max_bending_moment().get::<newton_meter>();

    // Simply supported beam under UDL: delta_max = 5 w L^4 / (384 E I)  [metres]
    let deflection_m = 5.0 * load * span.powi(4) / (384.0 * e * i);
    let deflection_mm = units::convert(deflection_m, "m", "mm")?;

    println!("Simply supported beam (UDL)");
    println!("  span L      = {span} m");
    println!("  load w      = {load} N/m");
    println!("  material     = {}", m.name);
    println!("  E           = {e:.3e} Pa");
    println!("  I_x         = {i:.3e} m^4");
    println!("  reaction RA = {reaction:.3e} N");
    println!("  reaction RB = {reaction:.3e} N");
    println!("  M_max       = {max_moment:.3e} N·m");
    println!("  delta_max   = {deflection_mm:.3e} mm");

    if let Some(path) = report_out {
        use tpt_eng_report::{NamedValue, Report, ResultEntry};
        let r = Report::new("Simply Supported Beam (UDL)")
            .with_author("tpt-eng-cli")
            .with_summary("Deflection and moment check for a simply supported beam under UDL.")
            .assumptions(vec![
                NamedValue::new("Span", span).with_unit("m"),
                NamedValue::new("Load (UDL)", load).with_unit("N/m"),
                NamedValue::new("Material", 0.0).with_description(m.name.to_string()),
                NamedValue::new("Young's modulus", e).with_unit("Pa"),
                NamedValue::new("I_x", i).with_unit("m^4"),
            ])
            .results(vec![
                ResultEntry::with_limits("Reaction A", reaction, Some("N".into()), None, None),
                ResultEntry::with_limits("Reaction B", reaction, Some("N".into()), None, None),
                ResultEntry::with_limits("Max moment", max_moment, Some("N·m".into()), None, None),
                ResultEntry::with_limits(
                    "Max deflection",
                    deflection_mm,
                    Some("mm".into()),
                    Some(0.0),
                    Some(25.0),
                ),
            ]);
        write_report(&r, &path)?;
    }

    if let Some(path) = plot_out {
        use tpt_eng_plot::{XyPlot, XySeries};
        let n = 100;
        let points: Vec<(f64, f64)> = (0..=n)
            .map(|k| {
                let x = span * (k as f64) / (n as f64);
                // deflection shape v(x) = w x (L^3 - 2 L x^2 + x^3) / (24 E I)
                let v = load * x * (span.powi(3) - 2.0 * span * x * x + x.powi(3)) / (24.0 * e * i);
                let v_mm = units::convert(v, "m", "mm").unwrap_or(v);
                (x, v_mm)
            })
            .collect();
        let plot = XyPlot::new("Beam deflection")
            .with_x_label("x (m)")
            .with_y_label("deflection (mm)")
            .with_series(XySeries::new("deflection", points));
        plot.save_png(&path)?;
        println!("wrote deflection diagram {}", path.display());
    }

    Ok(())
}
