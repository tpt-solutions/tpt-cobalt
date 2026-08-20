//! Electro-thermal coupling for `tpt-physics`.
//!
//! No prior art for this exists in `tpt-physics`, `tpt-fem`, or `tpt-science`,
//! so the whole crate is net-new. It models the canonical multiphysics loop:
//!
//! * an **electric** field drives a current through a conductor,
//! * **Joule heating** (`q = σ |E|²`) deposits power,
//! * **temperature-dependent conductivity** (`σ(T)`, metallic negative-TCR) closes
//!   the loop by self-limiting the heating, and
//! * **heat conduction** + **surface convection** remove it.
//!
//! The reference solver is a 1-D resistive rod (explicit finite differences),
//! the textbook geometry for Joule heating; the same operators generalise to
//! an arbitrary mesh via the same `step` contract.
//!
//! # Roadmap: 3-D mesh-based path
//!
//! The 1-D rod is an intentionally minimal scaffold. The production target is a
//! general 3-D electro-thermal solve that reuses the sibling `tpt-fem` stack
//! instead of re-implementing FEM:
//!
//! 1. **Geometry / mesh** — build the conductor from `tpt-fem-mesh`
//!    (`MeshBuilder`, `tpt-fem-mesh-gen` for tet/hex generation).
//! 2. **Electric potential** — solve `∇·(σ(T) ∇φ) = 0` (steady current) with
//!    `tpt-fem-sparse` + `tpt-fem-solve`; `σ(T)` is the same temperature-
//!    dependent conductivity already in [`ElectroThermalRod::conductivity`].
//! 3. **Joule source** — `q = σ(T) |∇φ|²` projected onto the element
//!    quadrature (`tpt-fem-quadrature`) as a volumetric heat load.
//! 4. **Heat equation** — `ρ c_p ∂T/∂t = ∇·(k ∇T) + q − h(T − T_∞)`, solved
//!    with `tpt-fem-thermal` (steady/transient conduction) and the same
//!    `tpt-fem-sparse`/`tpt-fem-solve` back-end.
//! 5. **Coupling** — a few fixed-point iterations between steps 2-4 close the
//!    `σ(T)` loop (exactly the positive-feedback self-limiting behaviour the
//!    1-D rod already demonstrates), orchestrated via `tpt-phys-orchestrator`.
//!
//! The 1-D [`ElectroThermalRod`] remains the unit-testable smoke test for the
//! coupled physics; the 3-D path is the validated production implementation to
//! land in a later phase.

use std::fmt::Debug;

/// Boundary condition for the rod ends.
#[derive(Debug, Clone, Copy)]
pub enum EndCondition {
    /// Perfectly insulated end (zero heat flux) — adiabatic.
    Insulated,
    /// Held at a fixed temperature `T` (Dirichlet).
    Fixed(f64),
}

/// A 1-D electro-thermal rod.
///
/// Nodes are spaced `dx` apart along the axis. Electrical conductivity is
/// temperature-dependent (linearised): `σ(T) = σ₀ / (1 + α (T − T_ref))`, so a
/// hotter conductor carries more current and heats faster — the positive
/// feedback that makes electro-thermal coupling genuinely coupled.
#[derive(Debug, Clone)]
pub struct ElectroThermalRod {
    /// Node spacing (m).
    pub dx: f64,
    /// Mass density (kg/m³).
    pub density: f64,
    /// Specific heat capacity (J/kg·K).
    pub heat_capacity: f64,
    /// Thermal conductivity (W/m·K).
    pub thermal_cond: f64,
    /// Volumetric surface convection coefficient (W/m³·K).
    pub convection: f64,
    /// Ambient temperature for convection (K).
    pub ambient: f64,
    /// Reference conductivity `σ₀` at `t_ref` (S/m).
    pub sigma0: f64,
    /// Conductivity temperature coefficient `α` (1/K).
    pub alpha: f64,
    /// Reference temperature for `σ(T)` (K).
    pub t_ref: f64,
    /// Applied voltage across the rod (V).
    pub voltage: f64,
    /// Per-node temperature (K).
    temp: Vec<f64>,
    left: EndCondition,
    right: EndCondition,
}

impl ElectroThermalRod {
    /// Build a rod of `n` nodes, initialised to `t_init`.
    pub fn new(n: usize, t_init: f64) -> Self {
        assert!(n >= 2, "rod needs at least 2 nodes");
        ElectroThermalRod {
            dx: 1.0,
            density: 2700.0,
            heat_capacity: 900.0,
            thermal_cond: 200.0,
            convection: 0.0,
            ambient: t_init,
            sigma0: 3.5e7,
            alpha: 1.0 / 300.0,
            t_ref: 300.0,
            voltage: 0.0,
            temp: vec![t_init; n],
            left: EndCondition::Insulated,
            right: EndCondition::Insulated,
        }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.temp.len()
    }

    /// Whether the rod has no nodes.
    pub fn is_empty(&self) -> bool {
        self.temp.is_empty()
    }

    /// Per-node temperatures (K).
    pub fn temperatures(&self) -> &[f64] {
        &self.temp
    }

    /// Set the end conditions.
    pub fn set_ends(&mut self, left: EndCondition, right: EndCondition) {
        self.left = left;
        self.right = right;
    }

    /// Set the applied voltage (V). The uniform field is `E = V / L`.
    pub fn set_voltage(&mut self, v: f64) {
        self.voltage = v;
    }

    /// Electrical conductivity at temperature `t` (S/m).
    ///
    /// Models a metallic (negative temperature-coefficient) conductor:
    /// `σ(T) = σ₀ / (1 + α (T − T_ref))`, so a hotter conductor carries *less*
    /// current and self-limits its Joule heating.
    pub fn conductivity(&self, t: f64) -> f64 {
        self.sigma0 / (1.0 + self.alpha * (t - self.t_ref))
    }

    /// Volumetric Joule heating at node `i` (W/m³): `σ(T) |E|²`.
    pub fn joule_heating(&self, i: usize) -> f64 {
        let e = self.voltage / ((self.temp.len() as f64 - 1.0) * self.dx);
        self.conductivity(self.temp[i]) * e * e
    }

    /// Advance the temperature field by `dt` (explicit Euler). `dt` must satisfy
    /// the conduction stability limit (`dt < ρ c_p dx² / (2 k)`); callers pushing
    /// past it get clamped with a warning rather than a blow-up.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, dt: f64) {
        let n = self.temp.len();
        let rho_cp = self.density * self.heat_capacity;
        let stability = if self.thermal_cond > 0.0 {
            0.5 * rho_cp * self.dx * self.dx / self.thermal_cond
        } else {
            f64::INFINITY
        };
        let dt = if dt > stability {
            tracing::warn!(
                dt,
                stability,
                "electro-thermal dt exceeds conduction stability limit; clamping"
            );
            stability * 0.9
        } else {
            dt
        };

        let mut next = self.temp.clone();
        for i in 0..n {
            let t = self.temp[i];
            // Conduction: central difference (insulated at free ends).
            let tl = if i == 0 {
                match self.left {
                    EndCondition::Fixed(t0) => t0,
                    EndCondition::Insulated => self.temp[i + 1],
                }
            } else {
                self.temp[i - 1]
            };
            let tr = if i == n - 1 {
                match self.right {
                    EndCondition::Fixed(t0) => t0,
                    EndCondition::Insulated => self.temp[i - 1],
                }
            } else {
                self.temp[i + 1]
            };
            let laplacian = (tl - 2.0 * t + tr) / (self.dx * self.dx);
            let conduction = self.thermal_cond * laplacian;
            let joule = self.joule_heating(i);
            let conv = self.convection * (self.ambient - t);
            let dq = (conduction + joule + conv) / rho_cp;
            next[i] = t + dt * dq;
        }
        self.temp = next;
    }

    /// Total Joule power deposited in the rod (W, per-unit cross-section).
    pub fn total_joule_power(&self) -> f64 {
        self.temp
            .iter()
            .enumerate()
            .map(|(i, _)| self.joule_heating(i))
            .sum::<f64>()
            * self.dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heats_up_under_voltage_and_stays_finite() {
        let mut rod = ElectroThermalRod::new(21, 300.0);
        rod.dx = 0.01;
        rod.set_voltage(10.0);
        rod.convection = 50.0; // radiative/convective sink so a steady state exists
        let t0 = rod.temperatures().to_vec();
        for _ in 0..2000 {
            rod.step(1e-4);
        }
        assert!(rod.temperatures().iter().all(|&t| t.is_finite()));
        let t1 = rod.temperatures().to_vec();
        let rise: f64 = t1.iter().zip(&t0).map(|(a, b)| a - b).sum();
        assert!(
            rise > 0.0,
            "rod should heat under Joule load, rise = {rise}"
        );
    }

    #[test]
    fn no_voltage_stays_at_initial() {
        let mut rod = ElectroThermalRod::new(11, 300.0);
        rod.dx = 0.01;
        rod.set_voltage(0.0);
        for _ in 0..500 {
            rod.step(1e-4);
        }
        assert!(
            rod.temperatures().iter().all(|&t| (t - 300.0).abs() < 1e-9),
            "no field => no heating"
        );
    }

    #[test]
    fn temperature_dependent_conductivity_is_monotonic() {
        let rod = ElectroThermalRod::new(2, 300.0);
        let cold = rod.conductivity(250.0);
        let hot = rod.conductivity(400.0);
        // Metallic (negative TCR): conductivity falls as temperature rises.
        assert!(hot < cold, "σ must decrease as temperature rises");
    }
}
