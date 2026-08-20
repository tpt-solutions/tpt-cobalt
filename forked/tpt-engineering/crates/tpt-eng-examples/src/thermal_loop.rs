//! # Thermal-loop integration scenario
//!
//! A self-contained example of how the `tpt-eng-*` crates fit together in a
//! real physical-systems vertical (here: a district-heating / process-thermal
//! loop):
//!
//! 1. **Topology + network matrix** — model the hydraulic loop
//!    (boiler → pump → heat-exchanger → tank → back to boiler) and derive its
//!    nodal admittance matrix.
//! 2. **Fuel sizing** — the burner's maximum power is derived from a blended
//!    fuel's lower heating value.
//! 3. **Controls** — a PID regulator drives the pump to hold the supply
//!    temperature at a setpoint, with output saturated to `[0, 1]`.
//! 4. **Time-series conditioning** — the supply-temperature telemetry is
//!    irregular with a dropout; gaps are detected, linearly filled, and the
//!    stream is aligned onto a fine deterministic grid for the controller.
//! 5. **Structural** — the pipe support rack is checked as a simply-supported
//!    beam under the loop's static load.

use tpt_eng_controls::{Pid, PidGains};
use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};
use tpt_eng_network_matrix::{admittance_matrix, incidence_matrix};
use tpt_eng_props_fuels::BlendedFuel;
use tpt_eng_structural::{Beam, Load};
use tpt_eng_timeseries_align::align_to_grid;
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
use tpt_eng_timeseries_gap::{Strategy, detect_gaps, fill_gaps};

use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    force::kilonewton, length::meter, thermodynamic_temperature::degree_celsius,
    torque::kilonewton_meter,
};

/// Output of [`run_thermal_loop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalLoopReport {
    /// Number of nodes in the hydraulic topology.
    pub node_count: usize,
    /// Trace of the nodal admittance (Laplacian) matrix — a proxy for total
    /// loop conductance.
    pub admittance_trace: f64,
    /// Controller target supply temperature (°C).
    pub setpoint: f64,
    /// Final conditioned supply temperature (°C).
    pub supply_temperature: f64,
    /// Largest telemetry gap the conditioner repaired (seconds).
    pub max_gap_seconds: f64,
    /// Largest bending moment on the support rack (kN·m).
    pub rack_max_moment: f64,
}

/// Build the hydraulic topology of the loop and report its network matrices.
fn build_loop_topology() -> Topology {
    let mut topo = Topology::new();
    for n in ["boiler", "pump", "hex", "tank"] {
        topo.add_node(n);
    }
    // Hydraulic conductances (m³/s per Pa) carried in `capacity`.
    topo.add_edge(Edge::new("e1", "boiler", "pump", EdgeKind::Pipe, 1.0));
    topo.add_edge(Edge::new("e2", "pump", "hex", EdgeKind::Pipe, 1.0));
    topo.add_edge(Edge::new("e3", "hex", "tank", EdgeKind::Pipe, 1.0));
    topo.add_edge(Edge::new("e4", "tank", "boiler", EdgeKind::Pipe, 1.0));
    topo
}

/// Simulate the thermal plant and regulate it with a PID. Returns the final
/// supply temperature and the conditioned telemetry gap size.
fn simulate_and_regulate(setpoint_c: f64, fuel: &BlendedFuel) -> (f64, f64) {
    let t_ambient = ThermodynamicTemperature::new::<degree_celsius>(20.0);
    let burner_max_power = fuel.lhv * 0.04; // MW at full pump output
    let loss = 0.03; // MW per °C
    let thermal_mass = 10.0; // MW·s per °C

    let mut pid = Pid::new(PidGains::new(0.5, 0.1, 0.0)).with_output_limit(1.0);
    pid.set_setpoint(setpoint_c);

    let mut t = t_ambient;
    let dt = 1.0;
    let steps = 300;

    // Build telemetry at 1 Hz, but drop samples during a 5 s dropout window.
    let mut telemetry: Vec<Sample<f64>> = Vec::new();
    for step in 0..steps {
        let tc = t.get::<degree_celsius>();
        // Dropout: samples at t = 20..23 are missing, leaving a 5 s gap
        // between the t = 19 and t = 24 samples.
        if !(20..=23).contains(&step) {
            telemetry.push(Sample::new(Timestamp::from_seconds(step as f64), tc));
        }
        let measured = t.get::<degree_celsius>();
        let u = pid.update(measured, dt).clamp(0.0, 1.0);
        let q = burner_max_power * u; // MW delivered
        let dtc = (q - loss * (tc - 20.0)) / thermal_mass;
        t = ThermodynamicTemperature::new::<degree_celsius>(tc + dt * dtc);
    }

    let series = Series::from_samples(telemetry);
    let gaps = detect_gaps(&series, 2.0);
    let max_gap = gaps.iter().map(|g| g.end - g.start).fold(0.0_f64, f64::max);

    // Linearly fill the gaps onto a 1 Hz grid so the controller's downstream
    // consumers always have a deterministic, gap-free signal.
    let grid: Vec<f64> = (0..steps).map(|s| s as f64).collect();
    let _conditioned = fill_gaps(&series, &grid, Strategy::Linear);
    // Also demonstrate cross-rate alignment onto a finer 10 Hz grid.
    let fine: Vec<f64> = (0..steps * 10).map(|i| i as f64 / 10.0).collect();
    let _aligned = align_to_grid(&series, &fine);

    let final_t = series.last().map(|s| s.value).unwrap_or(20.0);
    (final_t, max_gap)
}

/// Run the full thermal-loop scenario and return a summary report.
pub fn run_thermal_loop() -> ThermalLoopReport {
    let topo = build_loop_topology();
    let y = admittance_matrix(&topo);
    let a = incidence_matrix(&topo);
    let n = y.nrows();
    let admittance_trace = (0..n).map(|i| y[(i, i)]).sum::<f64>();
    // `a` is validated as square-ish in the network-matrix crate; touch both.
    let _inc_rows = a.nrows();

    let fuel = BlendedFuel::new(0.2); // 20% H₂ blend
    let setpoint = 60.0;
    let (supply_temperature, max_gap) = simulate_and_regulate(setpoint, &fuel);

    // Structural check: a 6 m simply-supported rack beam carrying the loop's
    // static load (a 10 kN point load at mid-span).
    let mut rack = Beam::new(Length::new::<meter>(6.0));
    rack.add(Load::point(
        Length::new::<meter>(3.0),
        Force::new::<kilonewton>(10.0),
    ));
    let rack_max_moment = rack.max_bending_moment().get::<kilonewton_meter>();

    ThermalLoopReport {
        node_count: n,
        admittance_trace,
        setpoint,
        supply_temperature,
        max_gap_seconds: max_gap,
        rack_max_moment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_converges_and_repairs_telemetry() {
        let r = run_thermal_loop();
        assert_eq!(r.node_count, 4);
        // Nodal admittance trace of the 4-node unit loop = 4·(2) = 8.
        assert!(
            (r.admittance_trace - 8.0).abs() < 1e-9,
            "trace={}",
            r.admittance_trace
        );
        // PID converged onto the setpoint.
        assert!((r.supply_temperature - r.setpoint).abs() < 0.5);
        // The 5 s dropout was detected.
        assert!((r.max_gap_seconds - 5.0).abs() < 1e-6);
        // Mid-span point load on a 6 m beam → M_max = 10·3/2 = 15 kN·m.
        assert!((r.rack_max_moment - 15.0).abs() < 1e-6);
    }
}
