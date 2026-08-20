//! Thermal strain as a structural load — the core of the thermal–structural
//! coupling.
//!
//! A uniform temperature rise is *stress-free* (free expansion), so it produces
//! no mechanical load. Only a *gradient* in temperature produces a
//! self-equilibrated thermal-strain load `f_th = ∫ Bᵀ D ε_th dV`. This example
//! checks those invariants on a single tetrahedral element and on a small mesh,
//! and confirms the degenerate-element guard returns `None` instead of `NaN`.
//!
//! Run with: `cargo run --release --example uniform_expansion -p tpt-phys-thermal-struct`

use tpt_fem_mesh::{CellType, MeshBuilder};
use tpt_phys_core::Material;
use tpt_phys_thermal_struct::{tet4_thermal_load, thermal_load_vector};

fn main() {
    // Unit reference tetrahedron.
    let r = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mat = Material::new("Steel", 200e9, 0.3, 7850.0, 12e-6);
    let (la, mu) = (mat.lame_lambda(), mat.shear_modulus());

    // Uniform +100 K rise → free expansion → zero mechanical load.
    let f_uniform = tet4_thermal_load(&r, mat.thermal_expansion, 100.0, la, mu).unwrap();
    let sum_uniform: f64 = f_uniform.iter().sum();
    println!("Uniform +100 K rise (single tet)");
    println!("  load vector Σ      : {sum_uniform:.3e}  (≈0 ⇒ stress-free)");
    println!(
        "  |load| per node    : {:.3e}",
        f_uniform.iter().map(|v| v.abs()).sum::<f64>() / 4.0
    );
    assert!(
        sum_uniform.abs() < 1e-6,
        "uniform heating must be load-free"
    );

    // Opposite rise flips the sign of the (still self-equilibrated) load.
    let f_hot = tet4_thermal_load(&r, mat.thermal_expansion, 100.0, la, mu).unwrap();
    let f_cold = tet4_thermal_load(&r, mat.thermal_expansion, -100.0, la, mu).unwrap();
    println!(
        "  |load| at +100 K   : {:.3e}",
        f_hot.iter().map(|v| v.abs()).sum::<f64>()
    );
    println!(
        "  |load| at -100 K   : {:.3e}",
        f_cold.iter().map(|v| v.abs()).sum::<f64>()
    );

    // Degenerate (inverted / duplicate-node) element → None, never NaN.
    let bad = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], // duplicate ⇒ flat tetrahedron
    ];
    let opt = tet4_thermal_load(&bad, mat.thermal_expansion, 100.0, la, mu);
    println!("  inverted tet → None: {}", opt.is_none());
    assert!(opt.is_none());

    // Full mesh assembly.
    let mut b = MeshBuilder::new();
    let n0 = b.add_node(vec![0.0, 0.0, 0.0]);
    let n1 = b.add_node(vec![1.0, 0.0, 0.0]);
    let n2 = b.add_node(vec![0.0, 1.0, 0.0]);
    let n3 = b.add_node(vec![0.0, 0.0, 1.0]);
    b.add_element(CellType::Tet, vec![n0, n1, n2, n3]);
    let mesh = b.build();
    let temps = vec![150.0, 50.0, 50.0, 50.0]; // one hot node, T_ref = 20
    let load = thermal_load_vector(&mesh, 3, &mat, &temps, 20.0);
    let net: f64 = load.iter().sum();
    let norm = load.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!();
    println!("Mesh with one hot node (T = [150,50,50,50] K, T_ref = 20 K):");
    println!("  global load norm  : {norm:.3e}");
    println!("  net force Σ       : {net:.3e}  (≈0 ⇒ self-equilibrated)");
    assert!(net.abs() < 1e-6);

    println!();
    println!("OK: uniform heating is load-free; gradients yield self-equilibrated loads.");
}
