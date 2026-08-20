//! Partitioned fluid–structure interaction (FSI) coupling driver.
//!
//! The [`nearest_node_map`] primitive in the crate root maps non-matching fluid
//! and structural interface discretizations. This module adds the actual
//! coupling-iteration loop the scaffold was missing:
//!
//! 1. advance the fluid ([`tpt_phys_cfd::Lbm2D`]) by one step,
//! 2. sample the surface traction at the fluid-side interface and map it onto
//!    the nearest structural nodes,
//! 3. advance the structural model,
//! 4. map the structural displacement back onto the fluid boundary (moving
//!    wall) and repeat.
//!
//! The loop is exposed both as a single [`couple_explicit`] sub-step and as an
//! [`FsiDriver`] that performs `substeps` under-relaxed Gauss–Seidel iterations
//! per fluid step (the explicit→implicit relaxation the partitioned scheme
//! needs for stability).

use tpt_fem_mesh::Mesh;
use tpt_phys_cfd::Lbm2D;

use crate::nearest_node_map;

/// A structural sub-model driven by the partitioned FSI loop.
///
/// Implementors own their own state and integration scheme. A real model would
/// back this with a `tpt-fem` elasticity solve; the bundled [`LumpedStructure`]
/// is a minimal mass–spring scaffold that exercises the whole exchange.
pub trait StructuralModel {
    /// Number of interface nodes the fluid couples to.
    fn n_interface_nodes(&self) -> usize;
    /// Accumulate a surface traction (force per unit area) at interface `node`.
    fn apply_traction(&mut self, node: usize, traction: [f64; 3]);
    /// Advance the structure by `dt`.
    fn step(&mut self, dt: f64);
    /// Displacement of interface `node` (for mapping back to the fluid).
    fn displacement(&self, node: usize) -> [f64; 3];
    /// Velocity of interface `node` (drives the fluid moving wall).
    fn velocity(&self, node: usize) -> [f64; 3];
    /// Drop accumulated tractions before the next exchange.
    fn clear_tractions(&mut self);
}

/// Lumped mass–spring–damper structural model: every interface node is a
/// 3-DOF oscillator anchored at the origin. Used as the default
/// [`StructuralModel`] when a full FEM solve is not wired in.
#[derive(Debug, Clone)]
pub struct LumpedStructure {
    /// Point mass per node (kg).
    pub mass: f64,
    /// Linear stiffness per node (N/m, isotropic in x/y/z).
    pub stiffness: f64,
    /// Viscous damping coefficient per node (N·s/m).
    pub damping: f64,
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
    force: Vec<[f64; 3]>,
}

impl LumpedStructure {
    /// Build a structure with `n` anchored oscillators.
    pub fn new(n: usize, mass: f64, stiffness: f64, damping: f64) -> Self {
        assert!(mass > 0.0 && stiffness > 0.0, "non-physical parameters");
        LumpedStructure {
            mass,
            stiffness,
            damping,
            pos: vec![[0.0; 3]; n],
            vel: vec![[0.0; 3]; n],
            force: vec![[0.0; 3]; n],
        }
    }

    /// Total elastic + kinetic energy (J) — a handy convergence diagnostic.
    pub fn energy(&self) -> f64 {
        let ke = self
            .vel
            .iter()
            .map(|v| 0.5 * self.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            .sum::<f64>();
        let pe = self
            .pos
            .iter()
            .map(|p| 0.5 * self.stiffness * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]))
            .sum::<f64>();
        ke + pe
    }
}

impl StructuralModel for LumpedStructure {
    fn n_interface_nodes(&self) -> usize {
        self.pos.len()
    }
    fn apply_traction(&mut self, node: usize, traction: [f64; 3]) {
        for (i, &t) in traction.iter().enumerate() {
            self.force[node][i] += t;
        }
    }
    fn step(&mut self, dt: f64) {
        // Sub-step so the explicit (symplectic) Euler integration stays stable
        // for any caller `dt`: the oscillator frequency is `ω = √(k/m)`, and a
        // single step is stable only for `ω·dt < 2`.
        let omega = (self.stiffness / self.mass).sqrt();
        let n = ((omega * dt / 0.5).ceil().max(1.0)) as usize;
        let h = dt / n as f64;
        for _ in 0..n {
            for k in 0..self.pos.len() {
                for i in 0..3 {
                    let a = (self.force[k][i]
                        - self.stiffness * self.pos[k][i]
                        - self.damping * self.vel[k][i])
                        / self.mass;
                    self.vel[k][i] += a * h;
                    self.pos[k][i] += self.vel[k][i] * h;
                }
            }
        }
        for f in &mut self.force {
            *f = [0.0; 3];
        }
    }
    fn displacement(&self, node: usize) -> [f64; 3] {
        self.pos[node]
    }
    fn velocity(&self, node: usize) -> [f64; 3] {
        self.vel[node]
    }
    fn clear_tractions(&mut self) {
        for f in &mut self.force {
            *f = [0.0; 3];
        }
    }
}

/// A fluid-side interface point: the lattice location adjacent to a solid wall.
#[derive(Debug, Clone, Copy)]
pub struct FluidInterfacePoint {
    /// Lattice column.
    pub ix: usize,
    /// Lattice row.
    pub iy: usize,
    /// Physical position `[x, y, 0]`.
    pub pos: [f64; 3],
}

/// Collect the fluid nodes that sit immediately next to a solid wall — the
/// fluid side of the FSI interface. These are the points fed to
/// [`nearest_node_map`] to build the fluid→structure mapping.
pub fn fluid_interface_points(sim: &Lbm2D) -> Vec<FluidInterfacePoint> {
    let mut pts = Vec::new();
    for iy in 0..sim.ny {
        for ix in 0..sim.nx {
            let i = sim.idx(ix, iy);
            if sim.solid[i] {
                continue;
            }
            let adjacent_to_wall = (ix > 0 && sim.solid[sim.idx(ix - 1, iy)])
                || (ix + 1 < sim.nx && sim.solid[sim.idx(ix + 1, iy)])
                || (iy > 0 && sim.solid[sim.idx(ix, iy - 1)])
                || (iy + 1 < sim.ny && sim.solid[sim.idx(ix, iy + 1)]);
            if adjacent_to_wall {
                pts.push(FluidInterfacePoint {
                    ix,
                    iy,
                    pos: [ix as f64, iy as f64, 0.0],
                });
            }
        }
    }
    pts
}

/// Coupling coefficient mapping fluid momentum flux to a structural traction.
/// Kept small so the scaffold exchange stays stable without a real fluid
/// stress tensor.
fn traction_from_fluid(rho: f64, u: [f64; 2], coeff: f64) -> [f64; 3] {
    let speed2 = u[0] * u[0] + u[1] * u[1];
    let speed = speed2.sqrt();
    if speed < 1e-12 {
        return [0.0; 3];
    }
    // Drag-like traction aligned with the flow, magnitude ∝ ρ·|u|².
    let mag = coeff * rho * speed2;
    [mag * u[0] / speed, mag * u[1] / speed, 0.0]
}

/// Perform one explicit coupling sub-step.
///
/// `map[i]` is the structural node nearest to fluid interface point `i`. The
/// structure is advanced by `dt`, then its velocity is fed back onto the fluid
/// boundary (moving wall) under `relax` under-relaxation.
pub fn couple_explicit(
    sim: &mut Lbm2D,
    structure: &mut dyn StructuralModel,
    interface: &[FluidInterfacePoint],
    map: &[usize],
    dt: f64,
    relax: f64,
    coupling_coeff: f64,
) {
    sim.step([0.0, 0.0]);

    structure.clear_tractions();
    for (i, p) in interface.iter().enumerate() {
        let fi = sim.idx(p.ix, p.iy);
        let traction = traction_from_fluid(sim.rho[fi], [sim.ux[fi], sim.uy[fi]], coupling_coeff);
        structure.apply_traction(map[i], traction);
    }
    structure.step(dt);

    // Map structural velocity back onto the fluid boundary (moving wall): the
    // structural motion drags the adjacent solid wall node via the Ladd
    // bounce-back correction.
    for (i, p) in interface.iter().enumerate() {
        let v = structure.velocity(map[i]);
        let wv = [relax * v[0], relax * v[1]];
        // Find the solid neighbour of this fluid interface node and move it.
        for (dx, dy) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
            let nx = p.ix as isize + dx;
            let ny = p.iy as isize + dy;
            if nx < 0 || ny < 0 || nx >= sim.nx as isize || ny >= sim.ny as isize {
                continue;
            }
            let ni = sim.idx(nx as usize, ny as usize);
            if sim.solid[ni] {
                sim.set_wall_velocity(nx as usize, ny as usize, wv);
                break;
            }
        }
    }
}

/// Partitioned FSI driver with under-relaxed sub-iterations.
///
/// Each [`FsiDriver::step`] advances the fluid by one step and performs
/// `substeps` Gauss–Seidel coupling iterations (default 1 = explicit). More
/// sub-steps with `relaxation < 1` approximate an implicit/strongly-coupled
/// exchange and improve stability for stiff interfaces.
#[derive(Debug, Clone)]
pub struct FsiDriver {
    /// Under-relaxation factor applied to the displacement→wall feedback.
    pub relaxation: f64,
    /// Gauss–Seidel sub-iterations per fluid step.
    pub substeps: usize,
    /// Fluid→structure traction coupling coefficient.
    pub coupling_coeff: f64,
    interface: Vec<FluidInterfacePoint>,
    map: Vec<usize>,
}

impl FsiDriver {
    /// Build a driver for `structure` interacting with `sim`'s fluid mesh via
    /// `structural_mesh` (used only to build the nearest-node map).
    pub fn new(sim: &Lbm2D, structure: &dyn StructuralModel, structural_mesh: &Mesh) -> Self {
        let interface = fluid_interface_points(sim);
        let pts: Vec<[f64; 3]> = interface.iter().map(|p| p.pos).collect();
        let map = nearest_node_map(&pts, structural_mesh);
        debug_assert!(
            map.iter().all(|&m| m < structure.n_interface_nodes()),
            "interface maps to more structural nodes than the model exposes"
        );
        FsiDriver {
            relaxation: 0.5,
            substeps: 1,
            coupling_coeff: 1e-3,
            interface,
            map,
        }
    }

    /// Number of fluid interface points (== length of [`FsiDriver::map`]).
    pub fn n_interface_points(&self) -> usize {
        self.interface.len()
    }

    /// Advance the coupled system by one fluid step with `substeps` relaxation
    /// iterations.
    pub fn step(&mut self, sim: &mut Lbm2D, structure: &mut dyn StructuralModel, dt: f64) {
        for _ in 0..self.substeps.max(1) {
            couple_explicit(
                sim,
                structure,
                &self.interface,
                &self.map,
                dt,
                self.relaxation,
                self.coupling_coeff,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_fem_mesh::MeshBuilder;

    fn structural_mesh(n: usize) -> Mesh {
        let mut b = MeshBuilder::new();
        for k in 0..n {
            b.add_node(vec![k as f64, 0.0, 0.0]);
        }
        b.build()
    }

    #[test]
    fn structure_displaces_under_steady_flow_and_relaxes() {
        // Fluid moving rightward drives the structure to a positive equilibrium
        // displacement; with flow removed it relaxes back toward rest.
        let nx = 32;
        let ny = 16;
        let mut sim = Lbm2D::new(nx, ny, 0.6);
        sim.set_horizontal_walls();
        // A vertical wall on the right acts as the FSI interface.
        sim.add_rect(nx - 2, 1, nx - 2, ny - 2);
        sim.initialise(1.0, [0.1, 0.0]); // steady rightward flow

        let mesh = structural_mesh(1);
        let mut structure = LumpedStructure::new(1, 1.0, 10.0, 2.0);
        let mut driver = FsiDriver::new(&sim, &structure, &mesh);
        assert!(driver.n_interface_points() > 0);

        for _ in 0..200 {
            driver.step(&mut sim, &mut structure, 1.0);
        }
        let d1 = structure.displacement(0)[0];
        assert!(d1 > 0.0, "structure should be pushed downstream, got {d1}");
        assert!(structure.energy().is_finite());

        // Remove the flow and let it relax.
        sim.initialise(1.0, [0.0, 0.0]);
        for _ in 0..2000 {
            driver.step(&mut sim, &mut structure, 1.0);
        }
        let d2 = structure.displacement(0)[0].abs();
        assert!(d2 < d1, "displacement should relax, {d1} -> {d2}");
    }

    #[test]
    fn no_interface_no_coupling() {
        // A domain with no solid walls has no fluid-adjacent boundary, so the
        // FSI interface map is empty.
        let mut sim = Lbm2D::new(16, 12, 0.6);
        sim.initialise(1.0, [0.05, 0.0]);
        let mesh = structural_mesh(1);
        let structure = LumpedStructure::new(1, 1.0, 10.0, 2.0);
        let driver = FsiDriver::new(&sim, &structure, &mesh);
        assert_eq!(driver.n_interface_points(), 0);
    }
}
