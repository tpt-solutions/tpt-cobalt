//! `SubModel` adapters wiring the `tpt-phys` domain coupling crates into a
//! [`tpt_sci_sim_core::Simulation`].
//!
//! Each adapter implements [`tpt_sci_sim_core::SubModel`] over a concrete
//! solver so the generic co-simulation engine can drive it:
//!
//! * [`FsiSubModel`] — partitioned fluid–structure interaction
//!   (`tpt-phys-fsi` + `tpt-phys-cfd`),
//! * [`ElectroThermalSubModel`] — Joule heating (`tpt-phys-electro-thermal`),
//! * [`ThermalStructSubModel`] — thermal-to-structural coupling
//!   (`tpt-phys-thermal-struct` + `tpt-phys-core`).
//!
//! [`build_demo_simulation`] registers one of each and couples the
//! electro-thermal temperature field into the thermal-structural model,
//! demonstrating the multi-crate orchestration end to end.

use std::fmt::Debug;

use tpt_fem_mesh::{CellType, Mesh, MeshBuilder};
use tpt_phys_cfd::Lbm2D;
use tpt_phys_core::Material;
use tpt_phys_electro_thermal::ElectroThermalRod;
use tpt_phys_fsi::{FsiDriver, LumpedStructure, StructuralModel};
use tpt_phys_thermal_struct::thermal_load_vector;
use tpt_sci_sim_core::{Coupling, SimError, Simulation, SubModel};

/// Partitioned FSI sub-model for the co-simulation engine.
pub struct FsiSubModel {
    name: String,
    fluid: Lbm2D,
    structure: LumpedStructure,
    driver: FsiDriver,
    time: f64,
    state_buf: Vec<f64>,
}

impl std::fmt::Debug for FsiSubModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsiSubModel")
            .field("name", &self.name)
            .field("structure", &self.structure)
            .field("driver", &self.driver)
            .finish_non_exhaustive()
    }
}

impl FsiSubModel {
    /// Build from a fluid lattice, a structural model, and its interface mesh.
    pub fn new(fluid: Lbm2D, structure: LumpedStructure, structural_mesh: &Mesh) -> Self {
        let driver = FsiDriver::new(&fluid, &structure, structural_mesh);
        FsiSubModel {
            name: "fsi".to_string(),
            time: 0.0,
            state_buf: vec![0.0; structure.n_interface_nodes() * 3],
            fluid,
            structure,
            driver,
        }
    }

    /// Immutable access to the fluid lattice (for inspection).
    pub fn fluid(&self) -> &Lbm2D {
        &self.fluid
    }
}

impl SubModel for FsiSubModel {
    fn id(&self) -> &str {
        &self.name
    }
    fn time(&self) -> f64 {
        self.time
    }
    fn max_step(&self) -> f64 {
        1e-4
    }
    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        self.driver.step(&mut self.fluid, &mut self.structure, dt);
        for k in 0..self.structure.n_interface_nodes() {
            let d = self.structure.displacement(k);
            self.state_buf[3 * k] = d[0];
            self.state_buf[3 * k + 1] = d[1];
            self.state_buf[3 * k + 2] = d[2];
        }
        self.time += dt;
        Ok(())
    }
    fn state(&self) -> &[f64] {
        &self.state_buf
    }
    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        self.time = time;
        if state.len() == self.state_buf.len() {
            self.state_buf.copy_from_slice(state);
        }
        Ok(())
    }
}

/// Electro-thermal sub-model (Joule heating) for the co-simulation engine.
#[derive(Debug)]
pub struct ElectroThermalSubModel {
    name: String,
    rod: ElectroThermalRod,
    time: f64,
    state_buf: Vec<f64>,
}

impl ElectroThermalSubModel {
    /// Wrap an [`ElectroThermalRod`].
    pub fn new(rod: ElectroThermalRod) -> Self {
        let n = rod.len();
        ElectroThermalSubModel {
            name: "electro-thermal".to_string(),
            time: 0.0,
            state_buf: vec![0.0; n],
            rod,
        }
    }

    /// Immutable access to the rod (for inspection).
    pub fn rod(&self) -> &ElectroThermalRod {
        &self.rod
    }
}

impl SubModel for ElectroThermalSubModel {
    fn id(&self) -> &str {
        &self.name
    }
    fn time(&self) -> f64 {
        self.time
    }
    fn max_step(&self) -> f64 {
        1e-4
    }
    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        self.rod.step(dt);
        self.state_buf.copy_from_slice(self.rod.temperatures());
        self.time += dt;
        Ok(())
    }
    fn state(&self) -> &[f64] {
        &self.state_buf
    }
    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        self.time = time;
        if state.len() == self.state_buf.len() {
            self.state_buf.copy_from_slice(state);
        }
        Ok(())
    }
}

/// Thermal-to-structural sub-model: a lumped structural response driven by the
/// thermal-strain load [`thermal_load_vector`] computes for a tetrahedral mesh.
#[derive(Debug)]
pub struct ThermalStructSubModel {
    name: String,
    mesh: Mesh,
    material: Material,
    t_ref: f64,
    mass: f64,
    stiffness: f64,
    damping: f64,
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
    force: Vec<[f64; 3]>,
    /// Per-node temperature (K), updated by the electro-thermal coupling.
    temp: Vec<f64>,
    time: f64,
    state_buf: Vec<f64>,
}

impl ThermalStructSubModel {
    /// Build from a structural tet mesh, material, and lumped oscillator params.
    pub fn new(
        mesh: Mesh,
        material: Material,
        t_ref: f64,
        mass: f64,
        stiffness: f64,
        damping: f64,
    ) -> Self {
        let n = mesh.node_count();
        ThermalStructSubModel {
            name: "thermal-struct".to_string(),
            time: 0.0,
            pos: vec![[0.0; 3]; n],
            vel: vec![[0.0; 3]; n],
            force: vec![[0.0; 3]; n],
            temp: vec![t_ref; n],
            state_buf: vec![0.0; n * 3],
            mesh,
            material,
            t_ref,
            mass,
            stiffness,
            damping,
        }
    }

    /// Immutable temperatures (K).
    pub fn temperatures(&self) -> &[f64] {
        &self.temp
    }
}

impl SubModel for ThermalStructSubModel {
    fn id(&self) -> &str {
        &self.name
    }
    fn time(&self) -> f64 {
        self.time
    }
    fn max_step(&self) -> f64 {
        1e-4
    }
    fn advance(&mut self, dt: f64) -> Result<(), SimError> {
        // Thermal strain from the current temperature field becomes a nodal
        // load through the ported coupling primitive.
        let load = thermal_load_vector(&self.mesh, 3, &self.material, &self.temp, self.t_ref);
        for (k, f) in self.force.iter_mut().enumerate() {
            f[0] += load[k * 3];
            f[1] += load[k * 3 + 1];
            f[2] += load[k * 3 + 2];
        }
        for k in 0..self.pos.len() {
            for i in 0..3 {
                let a = (self.force[k][i]
                    - self.stiffness * self.pos[k][i]
                    - self.damping * self.vel[k][i])
                    / self.mass;
                self.vel[k][i] += a * dt;
                self.pos[k][i] += self.vel[k][i] * dt;
                self.state_buf[3 * k + i] = self.pos[k][i];
            }
            self.force[k] = [0.0; 3];
        }
        self.time += dt;
        Ok(())
    }
    fn state(&self) -> &[f64] {
        &self.state_buf
    }
    fn input_mut(&mut self) -> Option<&mut [f64]> {
        // The electro-thermal coupling delivers a temperature field of equal length.
        Some(&mut self.temp)
    }
    fn restore_state(&mut self, state: &[f64], time: f64) -> Result<(), SimError> {
        self.time = time;
        if state.len() == self.state_buf.len() {
            self.state_buf.copy_from_slice(state);
        }
        Ok(())
    }
}

/// A single tetrahedral element spanning the unit cube, used as the structural
/// domain for the demo [`ThermalStructSubModel`].
fn unit_tet_mesh() -> Mesh {
    let mut b = MeshBuilder::new();
    let n0 = b.add_node(vec![0.0, 0.0, 0.0]);
    let n1 = b.add_node(vec![1.0, 0.0, 0.0]);
    let n2 = b.add_node(vec![0.0, 1.0, 0.0]);
    let n3 = b.add_node(vec![0.0, 0.0, 1.0]);
    b.add_element(CellType::Tet, vec![n0, n1, n2, n3]);
    b.build()
}

/// Build a coupled demo simulation wiring electro-thermal heating into the
/// thermal-structural model (temperature → thermal-strain load).
///
/// `material` is the structural material used by the thermal-structural model and
/// `voltage` is the applied electro-thermal rod voltage — both are parameters so
/// callers (e.g. a Monte-Carlo UQ sweep) can vary them.
pub fn build_demo_simulation_for(material: &Material, voltage: f64) -> Simulation {
    // FSI: a driven channel whose right wall is the fluid–structure interface.
    let mut fluid = Lbm2D::new(32, 16, 0.6);
    fluid.set_horizontal_walls();
    fluid.add_rect(30, 1, 30, 14);
    fluid.initialise(1.0, [0.1, 0.0]);
    let mut fsi_b = MeshBuilder::new();
    fsi_b.add_node(vec![0.0, 0.0, 0.0]);
    let fsi_mesh = fsi_b.build();
    let fsi = FsiSubModel::new(fluid, LumpedStructure::new(1, 1.0, 10.0, 2.0), &fsi_mesh);

    // Electro-thermal: a rod under voltage (steady Joule heating).
    let mut rod = ElectroThermalRod::new(11, 300.0);
    rod.dx = 0.01;
    rod.set_voltage(voltage);
    rod.convection = 50.0;
    let et = ElectroThermalSubModel::new(rod);

    // Thermal-structural: the unit tet driven by thermal strain.
    let ts = ThermalStructSubModel::new(unit_tet_mesh(), material.clone(), 300.0, 1.0, 10.0, 2.0);

    let mut sim = Simulation::new();
    sim.add_model(et).unwrap();
    sim.add_model(ts).unwrap();
    sim.add_model(fsi).unwrap();
    // Electro-thermal temperature field → thermal-structural temperature.
    sim.add_coupling(Coupling::new(
        "electro-thermal",
        "thermal-struct",
        |src: &[f64], input: &mut [f64]| {
            let n = src.len().min(input.len());
            input[..n].copy_from_slice(&src[..n]);
        },
    ));
    sim
}

/// Build the default coupled demo simulation (nominal steel, 10 V rod).
pub fn build_demo_simulation() -> Simulation {
    let mat = Material::new("Demo", 200e9, 0.3, 7850.0, 12e-6);
    build_demo_simulation_for(&mat, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_simulation_steps_and_stays_finite() {
        let mut sim = build_demo_simulation();
        sim.step_until(200.0 * 1e-4).unwrap();
        let et = sim.model("electro-thermal").unwrap().state().to_vec();
        assert!(et.iter().all(|v| v.is_finite()), "electro-thermal NaN");
        let ts = sim.model("thermal-struct").unwrap().state().to_vec();
        assert!(ts.iter().all(|v| v.is_finite()), "thermal-struct NaN");
    }
}
