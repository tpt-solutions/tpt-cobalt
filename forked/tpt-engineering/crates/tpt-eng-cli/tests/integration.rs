//! Integration tests for the `tpt-eng-cli` binary.

use std::io::Write;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tpt-eng-cli"))
}

#[test]
fn test_units_convert() {
    let out = bin()
        .args(["units", "convert", "1", "m", "mm"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("1000 mm"), "got: {stdout}");
}

#[test]
fn test_units_convert_temperature() {
    let out = bin()
        .args(["units", "convert", "100", "C", "F"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8(out.stdout).unwrap().contains("212"));
}

#[test]
fn test_units_incompatible() {
    let out = bin()
        .args(["units", "convert", "1", "m", "N"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn test_materials_inspect() {
    let out = bin().args(["materials", "steel"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("Young's modulus"));
}

#[test]
fn test_sections_inspect() {
    let out = bin()
        .args(["sections", "inspect", "rectangle", "0.1", "0.2"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("area:"));
}

#[test]
fn test_sections_ibeam() {
    let out = bin()
        .args([
            "sections", "inspect", "i-beam", "0.3", "0.15", "0.01", "0.006",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
}

#[test]
fn test_calc_beam() {
    let out = bin()
        .args([
            "calc",
            "beam",
            "5",
            "10000",
            "--material",
            "steel",
            "rectangle",
            "0.1",
            "0.2",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("M_max"));
}

#[test]
fn test_report_generates_all_formats() {
    let dir = std::env::temp_dir();
    for ext in ["md", "html", "json"] {
        let out_path = dir.join(format!("tpt_cli_report_test.{ext}"));
        let out = bin()
            .args(["report", "--out", &out_path.to_string_lossy()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{ext}: {:?}",
            String::from_utf8(out.stderr)
        );
        assert!(out_path.exists());
        let _ = std::fs::remove_file(&out_path);
    }
}

#[test]
fn test_validate_json() {
    let dir = std::env::temp_dir();
    let path = dir.join("tpt_cli_validate_test.json");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"{\"name\":\"test\",\"value\":1.0}").unwrap();
    }
    let out = bin()
        .args(["validate", &path.to_string_lossy()])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_validate_unknown_type_fails() {
    let dir = std::env::temp_dir();
    let path = dir.join("tpt_cli_validate_test.xyz");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"nope").unwrap();
    }
    let out = bin()
        .args(["validate", &path.to_string_lossy()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_props_water() {
    let out = bin()
        .args([
            "props",
            "water",
            "300",
            "3",
            "--temp-unit",
            "k",
            "--pressure-unit",
            "mpa",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("region One"));
    assert!(stdout.contains("enthalpy"));
}

#[test]
fn test_props_air() {
    let out = bin().args(["props", "air", "25", "50"]).output().unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("humidity ratio"));
    assert!(stdout.contains("dew point"));
}

#[test]
fn test_props_fuel() {
    let out = bin().args(["props", "fuel", "methane"]).output().unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("LHV"));
    assert!(stdout.contains("MJ/kg"));
}

#[test]
fn test_props_fuel_blend() {
    let out = bin()
        .args(["props", "fuel", "blend", "--h2", "0.3"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    assert!(
        String::from_utf8(out.stdout)
            .unwrap()
            .contains("h2 = 0.300")
    );
}

#[test]
fn test_props_unknown_fuel_fails() {
    let out = bin()
        .args(["props", "fuel", "unobtanium"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn test_pid_step_response() {
    let out = bin()
        .args([
            "pid",
            "2",
            "1",
            "0",
            "--setpoint",
            "10",
            "--tau",
            "1",
            "--dt",
            "0.01",
            "--steps",
            "1000",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("final y"));
    assert!(stdout.contains("overshoot"));
}

#[test]
fn test_pid_csv_output() {
    let dir = std::env::temp_dir();
    let path = dir.join("tpt_cli_pid_test.csv");
    let out = bin()
        .args([
            "pid",
            "2",
            "1",
            "0",
            "--setpoint",
            "1",
            "--steps",
            "50",
            "--csv",
            &path.to_string_lossy(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    assert!(path.exists());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("t,y,u"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_tolerance_stackup() {
    let out = bin()
        .args([
            "tolerance",
            "stackup",
            "-D",
            "a=10±0.1",
            "-D",
            "b=20±0.2",
            "--low",
            "29.5",
            "--high",
            "30.5",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("worst-case range"));
    assert!(stdout.contains("RSS (3σ) range"));
    assert!(stdout.contains("Monte-Carlo"));
    assert!(stdout.contains("yield within spec"));
}

#[test]
fn test_tolerance_asymmetric() {
    let out = bin()
        .args(["tolerance", "stackup", "-D", "a=10;0.2;0.1"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", String::from_utf8(out.stderr));
}

#[test]
fn test_tolerance_bad_dim_fails() {
    let out = bin()
        .args(["tolerance", "stackup", "-D", "no-equals-sign"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
