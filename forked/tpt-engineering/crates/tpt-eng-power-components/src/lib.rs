//! # tpt-eng-power-components
//!
//! Power-system component models built on the complex-arithmetic and per-unit
//! primitives of [`tpt_eng_electrical`]:
//!
//! * [`Transformer`] — two-winding equivalent circuit (series leakage
//!   impedance plus shunt magnetizing admittance) with secondary voltage,
//!   input impedance, no-load loss and voltage regulation.
//! * Generator models — shaft-to-terminal conversion
//!   ([`generator_electrical_power`]) and the induction-machine slip / rotor
//!   power split ([`induction_slip`], [`induction_mechanical_power`],
//!   [`induction_rotor_copper_loss`]).
//! * [`TransmissionLine`] — distributed-parameter line with surge impedance
//!   and the exact `ABCD` (two-port) parameters of the long-line model.
//!
//! ## Units
//!
//! All quantities are plain `f64` in SI units, or in per-unit (pu) when the
//! caller works on a [`PerUnitSystem`] base:
//!
//! * impedance in ohms (Ω), admittance in siemens (S),
//! * voltage in volts (V), current in amperes (A),
//! * real power in watts (W), reactive power in var,
//! * frequency in hertz (Hz), length in metres (m),
//! * rotational speed in revolutions per minute (rpm) — [`induction_slip`] is
//!   a ratio, so any consistent speed unit works,
//! * distributed line parameters are *per unit length*: Ω/m and S/m.
//!
//! Turns ratios, efficiencies and slips are dimensionless.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_electrical::Complex;
//! use tpt_eng_power_components::{Transformer, TransmissionLine, induction_slip};
//!
//! // Ideal 10:1 step-down transformer feeding a 10 Ω resistive load.
//! let xfmr = Transformer::ideal(10.0);
//! let v_sec = xfmr.secondary_voltage(230.0, Complex::new(10.0, 0.0));
//! assert!((v_sec.re - 23.0).abs() < 1e-12 && v_sec.im.abs() < 1e-12);
//!
//! // A zero-length line is a pass-through two-port: A = D = 1, B = C = 0.
//! let line = TransmissionLine::new(Complex::new(1.0e-4, 4.0e-4), Complex::new(0.0, 3.0e-9));
//! let (a, b, c, d) = line.abc_parameters(0.0);
//! assert!((a.re - 1.0).abs() < 1e-12 && (d.re - 1.0).abs() < 1e-12);
//! assert!(b.magnitude() < 1e-12 && c.magnitude() < 1e-12);
//!
//! // A 4-pole 50 Hz induction machine turning at its synchronous speed.
//! assert!(induction_slip(1500.0, 1500.0).abs() < 1e-15);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use tpt_eng_electrical::{
    Complex, PerUnitSystem, admittance, impedance_capacitor, impedance_inductor,
    impedance_parallel, impedance_resistor, impedance_series, three_phase_power,
};

// ---------------------------------------------------------------------------
// Complex helpers
// ---------------------------------------------------------------------------

/// Scale a complex number by a real factor: `k·z`.
pub fn complex_scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

/// Principal square root of a complex number.
///
/// Uses the branch with non-negative real part,
/// `√z = √((|z| + re)/2) + j·sgn(im)·√((|z| − re)/2)`, which is the branch
/// required for the propagation constant `γ = √(z'·y')` and the
/// characteristic impedance `Z_c = √(z'/y')` of a passive line.
pub fn complex_sqrt(z: Complex) -> Complex {
    let r = z.magnitude();
    if r == 0.0 {
        return Complex::new(0.0, 0.0);
    }
    let re = (0.5 * (r + z.re)).max(0.0).sqrt();
    let im = (0.5 * (r - z.re)).max(0.0).sqrt();
    if z.im < 0.0 {
        Complex::new(re, -im)
    } else {
        Complex::new(re, im)
    }
}

/// Hyperbolic cosine of a complex number:
/// `cosh(a + jb) = cosh a·cos b + j·sinh a·sin b`.
pub fn complex_cosh(z: Complex) -> Complex {
    Complex::new(z.re.cosh() * z.im.cos(), z.re.sinh() * z.im.sin())
}

/// Hyperbolic sine of a complex number:
/// `sinh(a + jb) = sinh a·cos b + j·cosh a·sin b`.
pub fn complex_sinh(z: Complex) -> Complex {
    Complex::new(z.re.sinh() * z.im.cos(), z.re.cosh() * z.im.sin())
}

/// Series `R + jωL` impedance (Ω) from resistance (Ω), inductance (H) and
/// frequency (Hz), i.e. the usual winding / line series branch.
pub fn series_rl_impedance(resistance: f64, inductance: f64, frequency_hz: f64) -> Complex {
    impedance_series(&[
        impedance_resistor(resistance),
        impedance_inductor(inductance, frequency_hz),
    ])
}

/// Shunt capacitive admittance `Y = jωC` (S) from capacitance (F) and
/// frequency (Hz).
///
/// # Panics
///
/// Never panics, but returns a non-finite value when `capacitance_f` or
/// `frequency_hz` is zero (an open shunt branch has infinite impedance).
pub fn shunt_capacitive_admittance(capacitance_f: f64, frequency_hz: f64) -> Complex {
    admittance(impedance_capacitor(capacitance_f, frequency_hz))
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

/// Two-winding transformer equivalent circuit, referred to the **primary**.
///
/// The model is the classical "T" circuit collapsed to an L: the total
/// leakage impedance `Z_leak = R_1 + n²·R_2 + j(X_1 + n²·X_2)` (Ω, primary
/// side) in series with the ideal `n : 1` transformation, and a shunt
/// magnetizing admittance `Y_m = G_c + jB_m` (S, primary side) representing
/// core loss and magnetization.
///
/// All quantities may equally be supplied in per-unit, in which case the
/// results are per-unit as well (a per-unit transformer typically has
/// `turns_ratio = 1`, `Z_leak ≈ 0.005 + j0.05` and `Y_m ≈ 0.002 − j0.02`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transformer {
    /// Turns ratio `n = N_1 / N_2` (primary turns per secondary turn);
    /// `n > 1` is a step-down transformer.
    pub turns_ratio: f64,
    /// Total leakage impedance referred to the primary (Ω or pu).
    pub leakage_impedance: Complex,
    /// Shunt magnetizing admittance referred to the primary (S or pu).
    pub magnetizing_admittance: Complex,
}

impl Transformer {
    /// Construct a transformer from its turns ratio, primary-referred leakage
    /// impedance and primary-referred magnetizing admittance.
    pub fn new(
        turns_ratio: f64,
        leakage_impedance: Complex,
        magnetizing_admittance: Complex,
    ) -> Self {
        Self {
            turns_ratio,
            leakage_impedance,
            magnetizing_admittance,
        }
    }

    /// Construct an **ideal** transformer: the given turns ratio, zero leakage
    /// impedance and zero magnetizing admittance (infinite core permeability,
    /// no core loss).
    pub fn ideal(turns_ratio: f64) -> Self {
        Self::new(turns_ratio, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0))
    }

    /// Construct a transformer whose shunt branch is given as a magnetizing
    /// *impedance* `Z_m` (Ω or pu) rather than an admittance;
    /// `Y_m = 1/Z_m` is formed internally.
    ///
    /// # Panics
    ///
    /// Never panics, but a zero `magnetizing_impedance` yields a non-finite
    /// admittance (a short-circuited core branch is not a physical model).
    pub fn with_magnetizing_impedance(
        turns_ratio: f64,
        leakage_impedance: Complex,
        magnetizing_impedance: Complex,
    ) -> Self {
        Self::new(
            turns_ratio,
            leakage_impedance,
            admittance(magnetizing_impedance),
        )
    }

    /// Leakage impedance referred to the **secondary**: `Z_leak / n²`
    /// (Ω or pu).
    pub fn secondary_referred_leakage(&self) -> Complex {
        complex_scale(
            self.leakage_impedance,
            1.0 / (self.turns_ratio * self.turns_ratio),
        )
    }

    /// Load impedance referred to the **primary**: `n²·Z_load` (Ω or pu).
    pub fn primary_referred_load(&self, load_impedance: Complex) -> Complex {
        complex_scale(load_impedance, self.turns_ratio * self.turns_ratio)
    }

    /// Secondary (load) terminal voltage phasor for a real primary voltage
    /// taken as the reference phasor (`V_prim ∠0°`, V or pu).
    ///
    /// The series leakage model divides the ideal open-circuit secondary
    /// voltage `V_prim / n` between the secondary-referred leakage impedance
    /// and the load:
    ///
    /// ```text
    ///           V_prim         Z_load
    /// V_sec = ────────── · ────────────────────
    ///             n         Z_load + Z_leak/n²
    /// ```
    ///
    /// The shunt magnetizing branch does not appear here: it sits across the
    /// supply, so (to the accuracy of this L-model) it draws exciting current
    /// but does not change the load voltage. Use [`Transformer::input_impedance`]
    /// when the exciting current matters.
    ///
    /// # Panics
    ///
    /// Never panics, but returns a non-finite value when
    /// `Z_load + Z_leak/n² = 0` (a resonant short circuit) or when
    /// `turns_ratio = 0`.
    pub fn secondary_voltage(&self, primary_voltage: f64, load_impedance: Complex) -> Complex {
        let open_circuit = Complex::new(primary_voltage / self.turns_ratio, 0.0);
        let divider = load_impedance / (load_impedance + self.secondary_referred_leakage());
        open_circuit * divider
    }

    /// Impedance seen from the primary terminals with `load_impedance`
    /// connected to the secondary (Ω or pu):
    /// `Z_in = Z_leak + (Z_m ∥ n²·Z_load)`.
    ///
    /// For a transformer with no magnetizing branch (`Y_m = 0`) the shunt path
    /// is an open circuit and `Z_in = Z_leak + n²·Z_load`.
    pub fn input_impedance(&self, load_impedance: Complex) -> Complex {
        let referred_load = self.primary_referred_load(load_impedance);
        let shunt = if self.magnetizing_admittance.magnitude() > 0.0 {
            let z_m = admittance(self.magnetizing_admittance);
            impedance_parallel(&[z_m, referred_load]).unwrap_or(referred_load)
        } else {
            referred_load
        };
        impedance_series(&[self.leakage_impedance, shunt])
    }

    /// Exciting (magnetizing + core-loss) current phasor drawn by the shunt
    /// branch at a real primary voltage `V_prim ∠0°`: `I_0 = V_prim·Y_m`
    /// (A or pu).
    pub fn exciting_current(&self, primary_voltage: f64) -> Complex {
        complex_scale(self.magnetizing_admittance, primary_voltage)
    }

    /// No-load (core) loss `P_0 = V_prim²·G_c` (W or pu), where
    /// `G_c = Re{Y_m}` is the core-loss conductance.
    pub fn no_load_loss(&self, primary_voltage: f64) -> f64 {
        primary_voltage * primary_voltage * self.magnetizing_admittance.re
    }

    /// Voltage regulation of the secondary terminal,
    /// `(|V_no-load| − |V_load|) / |V_load|` (dimensionless fraction).
    ///
    /// The no-load secondary voltage is the ideal `V_prim / n`; the loaded
    /// value comes from [`Transformer::secondary_voltage`]. An ideal
    /// transformer has zero regulation for any load.
    ///
    /// # Panics
    ///
    /// Never panics, but returns a non-finite value for a short-circuited
    /// secondary (`|V_load| = 0`).
    pub fn voltage_regulation(&self, primary_voltage: f64, load_impedance: Complex) -> f64 {
        let no_load = (primary_voltage / self.turns_ratio).abs();
        let loaded = self
            .secondary_voltage(primary_voltage, load_impedance)
            .magnitude();
        (no_load - loaded) / loaded
    }
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Electrical power delivered at a generator's terminals (W) from the
/// mechanical shaft power input (W) and the overall efficiency (0…1):
/// `P_elec = η·P_mech`.
///
/// Assumptions: steady state, and `η` lumps *all* losses (friction and
/// windage, stator and rotor copper loss, core loss, stray load loss). The
/// value is not clamped, so an `efficiency` outside `[0, 1]` is reported
/// as given — the caller is responsible for physically meaningful inputs.
pub fn generator_electrical_power(mechanical_power: f64, efficiency: f64) -> f64 {
    mechanical_power * efficiency
}

/// Mechanical shaft power (W) required to deliver `electrical_power` (W) at an
/// overall efficiency of `efficiency` (0…1): `P_mech = P_elec / η`.
///
/// This is the exact inverse of [`generator_electrical_power`].
///
/// # Panics
///
/// Never panics, but returns a non-finite value for `efficiency = 0`.
pub fn generator_mechanical_power(electrical_power: f64, efficiency: f64) -> f64 {
    electrical_power / efficiency
}

/// Mechanical shaft power (W) needed to sustain a balanced three-phase
/// terminal output, from the line-to-line voltage `v_ll` (V), line current
/// `i_line` (A), power factor `pf` (0…1) and overall efficiency (0…1).
///
/// The real terminal power is taken from [`three_phase_power`]
/// (`P = √3·V_ll·I_line·pf`) and divided by the efficiency.
///
/// # Panics
///
/// Never panics, but returns a non-finite value for `efficiency = 0`.
pub fn generator_shaft_power_for_output(v_ll: f64, i_line: f64, pf: f64, efficiency: f64) -> f64 {
    let (p, _q) = three_phase_power(v_ll, i_line, pf);
    generator_mechanical_power(p, efficiency)
}

/// Synchronous speed (rpm) of an AC machine: `n_s = 120·f / p`, where `f` is
/// the stator frequency (Hz) and `p` the number of poles (not pole pairs).
///
/// # Panics
///
/// Never panics, but returns a non-finite value when `poles = 0`.
pub fn synchronous_speed_rpm(frequency_hz: f64, poles: u32) -> f64 {
    120.0 * frequency_hz / f64::from(poles)
}

/// Induction-machine slip `s = (n_s − n_r) / n_s` (dimensionless).
///
/// `rotor_speed` and `synchronous_speed` must share the same unit (rpm or
/// rad/s). Sign convention:
///
/// * `s > 0` — sub-synchronous, the machine motors,
/// * `s = 0` — exactly synchronous: no rotor emf, no torque,
/// * `s < 0` — super-synchronous, the machine generates (an induction
///   generator is driven above synchronous speed).
///
/// Returns exactly `0.0` at synchronous speed and — as a defined degenerate
/// case — also when `synchronous_speed` is zero (no rotating field, so slip
/// is meaningless).
pub fn induction_slip(rotor_speed: f64, synchronous_speed: f64) -> f64 {
    if synchronous_speed == 0.0 || rotor_speed == synchronous_speed {
        return 0.0;
    }
    (synchronous_speed - rotor_speed) / synchronous_speed
}

/// Rotor (slip) frequency `f_r = s·f_s` (Hz) from the stator frequency (Hz)
/// and the slip.
pub fn induction_rotor_frequency(stator_frequency_hz: f64, slip: f64) -> f64 {
    slip * stator_frequency_hz
}

/// Rotor copper loss of an induction machine, `P_cu2 = s·P_ag` (W), from the
/// air-gap power `P_ag` (W) transferred across the gap and the slip.
pub fn induction_rotor_copper_loss(air_gap_power: f64, slip: f64) -> f64 {
    slip * air_gap_power
}

/// Gross mechanical (shaft) power developed by an induction machine,
/// `P_mech = (1 − s)·P_ag` (W).
///
/// Together with [`induction_rotor_copper_loss`] this is the classical rotor
/// power split `P_ag = P_cu2 + P_mech`; friction and windage are not
/// included.
pub fn induction_mechanical_power(air_gap_power: f64, slip: f64) -> f64 {
    (1.0 - slip) * air_gap_power
}

// ---------------------------------------------------------------------------
// Transmission line
// ---------------------------------------------------------------------------

/// Surge (characteristic) impedance magnitude `|Z_c| = √(|z'| / |y'|)` (Ω) of
/// a line with series impedance `z` (Ω/m) and shunt admittance `y` (S/m).
///
/// This is the magnitude of `√(z/y)`; for a lossless line it is the familiar
/// real surge impedance `√(L/C)`.
///
/// # Panics
///
/// Never panics, but returns a non-finite value when `|y| = 0` (no shunt
/// path).
pub fn surge_impedance(z: Complex, y: Complex) -> f64 {
    (z.magnitude() / y.magnitude()).sqrt()
}

/// Uniform transmission line described by its distributed (per-unit-length)
/// parameters.
///
/// `z_prime = R' + jωL'` (Ω/m) is the series impedance per metre and
/// `y_prime = G' + jωC'` (S/m) the shunt admittance per metre. From these the
/// exact distributed-parameter solution follows with
/// `γ = √(z'·y')` (1/m) and `Z_c = √(z'/y')` (Ω).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionLine {
    /// Series impedance per unit length (Ω/m).
    pub z_prime: Complex,
    /// Shunt admittance per unit length (S/m).
    pub y_prime: Complex,
}

impl TransmissionLine {
    /// Construct a line from its per-unit-length series impedance (Ω/m) and
    /// shunt admittance (S/m).
    pub fn new(z_prime: Complex, y_prime: Complex) -> Self {
        Self { z_prime, y_prime }
    }

    /// Construct a line from the usual overhead-line data: series resistance
    /// `R'` (Ω/m), series inductance `L'` (H/m) and shunt capacitance `C'`
    /// (F/m) at the operating frequency (Hz). Shunt conductance (leakage) is
    /// neglected, so `y' = jωC'`.
    ///
    /// # Panics
    ///
    /// Never panics, but a zero `c_prime` or `frequency_hz` gives a
    /// non-finite shunt admittance.
    pub fn from_line_parameters(
        r_prime: f64,
        l_prime: f64,
        c_prime: f64,
        frequency_hz: f64,
    ) -> Self {
        Self::new(
            series_rl_impedance(r_prime, l_prime, frequency_hz),
            shunt_capacitive_admittance(c_prime, frequency_hz),
        )
    }

    /// Complex characteristic impedance `Z_c = √(z'/y')` (Ω).
    ///
    /// # Panics
    ///
    /// Never panics, but returns a non-finite value when `y' = 0`.
    pub fn characteristic_impedance(&self) -> Complex {
        complex_sqrt(self.z_prime / self.y_prime)
    }

    /// Surge-impedance magnitude `|Z_c| = √(|z'|/|y'|)` (Ω); see
    /// [`surge_impedance`].
    pub fn surge_impedance(&self) -> f64 {
        surge_impedance(self.z_prime, self.y_prime)
    }

    /// Propagation constant `γ = √(z'·y') = α + jβ` (1/m), with attenuation
    /// `α` (Np/m) and phase constant `β` (rad/m).
    pub fn propagation_constant(&self) -> Complex {
        complex_sqrt(self.z_prime * self.y_prime)
    }

    /// Total series impedance of a `length`-metre section, `Z = z'·l` (Ω).
    pub fn total_series_impedance(&self, length: f64) -> Complex {
        complex_scale(self.z_prime, length)
    }

    /// Total shunt admittance of a `length`-metre section, `Y = y'·l` (S).
    pub fn total_shunt_admittance(&self, length: f64) -> Complex {
        complex_scale(self.y_prime, length)
    }

    /// Total series impedance of a `length`-metre section in per-unit on the
    /// given base.
    pub fn series_impedance_pu(&self, length: f64, base: PerUnitSystem) -> Complex {
        let z = self.total_series_impedance(length);
        Complex::new(base.impedance_to_pu(z.re), base.impedance_to_pu(z.im))
    }

    /// Total shunt admittance of a `length`-metre section in per-unit on the
    /// given base (`y_pu = Y·Z_base`).
    pub fn shunt_admittance_pu(&self, length: f64, base: PerUnitSystem) -> Complex {
        complex_scale(self.total_shunt_admittance(length), base.base_impedance())
    }

    /// Exact `ABCD` (two-port) parameters of a `length`-metre section,
    /// returned as `(A, B, C, D)` for
    ///
    /// ```text
    /// ⎡V_send⎤   ⎡A  B⎤ ⎡V_recv⎤          ⎡A  B⎤   ⎡  cosh(γl)     Z_c·sinh(γl)⎤
    /// ⎢      ⎥ = ⎢    ⎥ ⎢      ⎥    with  ⎢    ⎥ = ⎢                           ⎥
    /// ⎣I_send⎦   ⎣C  D⎦ ⎣I_recv⎦          ⎣C  D⎦   ⎣sinh(γl)/Z_c    cosh(γl)   ⎦
    /// ```
    ///
    /// where `γ = √(z'·y')` and `Z_c = √(z'/y')`. `A` and `D` are
    /// dimensionless, `B` is an impedance (Ω) and `C` an admittance (S). The
    /// network is symmetric and reciprocal, so `A = D` and `A·D − B·C = 1`.
    ///
    /// A `length` of exactly `0.0` returns the identity two-port
    /// (`A = D = 1`, `B = C = 0`) without evaluating `γ` or `Z_c`, so a
    /// degenerate line (e.g. `y' = 0`) still yields a clean pass-through.
    ///
    /// # Panics
    ///
    /// Never panics, but returns non-finite entries for a non-zero length
    /// when `y' = 0`, because `Z_c` is then unbounded.
    pub fn abc_parameters(&self, length: f64) -> (Complex, Complex, Complex, Complex) {
        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        if length == 0.0 {
            return (one, zero, zero, one);
        }
        let gamma_l = complex_scale(self.propagation_constant(), length);
        let z_c = self.characteristic_impedance();
        let cosh_gl = complex_cosh(gamma_l);
        let sinh_gl = complex_sinh(gamma_l);
        (cosh_gl, z_c * sinh_gl, sinh_gl / z_c, cosh_gl)
    }

    /// Sending-end voltage and current phasors `(V_send, I_send)` for given
    /// receiving-end phasors, using the exact `ABCD` parameters of a
    /// `length`-metre section:
    /// `V_s = A·V_r + B·I_r`, `I_s = C·V_r + D·I_r`.
    ///
    /// Voltages are per phase (line-to-neutral, V) and currents per phase
    /// (A), or both in per-unit.
    pub fn sending_end(
        &self,
        receiving_voltage: Complex,
        receiving_current: Complex,
        length: f64,
    ) -> (Complex, Complex) {
        let (a, b, c, d) = self.abc_parameters(length);
        (
            a * receiving_voltage + b * receiving_current,
            c * receiving_voltage + d * receiving_current,
        )
    }

    /// Surge-impedance loading (SIL) of the line, `P_SIL = V_ll² / |Z_c|` (W),
    /// for a line-to-line voltage `v_ll` (V).
    ///
    /// This is the three-phase real power delivered when the line is
    /// terminated in its own surge impedance, at which point it neither
    /// absorbs nor generates net reactive power.
    ///
    /// # Panics
    ///
    /// Never panics, but returns a non-finite value when `|Z_c| = 0`.
    pub fn surge_impedance_loading(&self, v_ll: f64) -> f64 {
        v_ll * v_ll / self.surge_impedance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn complex_sqrt_squares_back() {
        for z in [
            Complex::new(3.0, 4.0),
            Complex::new(-3.0, 4.0),
            Complex::new(-3.0, -4.0),
            Complex::new(0.0, -2.0),
            Complex::new(5.0, 0.0),
        ] {
            let s = complex_sqrt(z);
            let back = s * s;
            assert!((back.re - z.re).abs() < 1e-9, "re mismatch for {z:?}");
            assert!((back.im - z.im).abs() < 1e-9, "im mismatch for {z:?}");
            // Principal branch: non-negative real part.
            assert!(s.re >= 0.0);
        }
        assert_eq!(complex_sqrt(Complex::new(0.0, 0.0)), Complex::new(0.0, 0.0));
    }

    #[test]
    fn hyperbolic_identity_holds() {
        // cosh² − sinh² = 1 for complex arguments.
        let z = Complex::new(0.3, 1.1);
        let c = complex_cosh(z);
        let s = complex_sinh(z);
        let id = c * c - s * s;
        assert!((id.re - 1.0).abs() < 1e-12);
        assert!(id.im.abs() < 1e-12);
        // Purely imaginary argument reduces to cos / j·sin.
        let jb = complex_cosh(Complex::new(0.0, 0.7));
        assert!((jb.re - 0.7_f64.cos()).abs() < TOL);
        assert!(jb.im.abs() < TOL);
    }

    #[test]
    fn rl_and_c_branch_helpers() {
        let f = 50.0;
        let z = series_rl_impedance(2.0, 0.1, f);
        assert!((z.re - 2.0).abs() < TOL);
        assert!((z.im - 2.0 * std::f64::consts::PI * f * 0.1).abs() < 1e-9);
        // 1/(−j/(ωC)) = jωC.
        let y = shunt_capacitive_admittance(1.0e-6, f);
        assert!(y.re.abs() < 1e-15);
        assert!((y.im - 2.0 * std::f64::consts::PI * f * 1.0e-6).abs() < 1e-15);
    }

    #[test]
    fn transformer_referrals_are_inverse() {
        let xfmr = Transformer::new(10.0, Complex::new(1.0, 5.0), Complex::new(1.0e-4, -1.0e-3));
        let secondary = xfmr.secondary_referred_leakage();
        assert!((secondary.re - 0.01).abs() < TOL);
        assert!((secondary.im - 0.05).abs() < TOL);
        let load = Complex::new(4.0, 3.0);
        let referred = xfmr.primary_referred_load(load);
        assert!((referred.re - 400.0).abs() < 1e-9);
        assert!((referred.im - 300.0).abs() < 1e-9);
    }

    #[test]
    fn input_impedance_without_shunt_is_series_sum() {
        let xfmr = Transformer::new(2.0, Complex::new(0.5, 2.0), Complex::new(0.0, 0.0));
        let z_in = xfmr.input_impedance(Complex::new(10.0, 0.0));
        // Z_in = Z_leak + n²·Z_load = 0.5 + j2 + 40.
        assert!((z_in.re - 40.5).abs() < 1e-9);
        assert!((z_in.im - 2.0).abs() < 1e-9);
    }

    #[test]
    fn input_impedance_includes_parallel_magnetizing_branch() {
        let leakage = Complex::new(0.005, 0.05);
        let load = Complex::new(1.0, 0.0);
        let z_m = Complex::new(0.0, 50.0); // highly inductive core branch
        let xfmr = Transformer::with_magnetizing_impedance(1.0, leakage, z_m);

        // Z_in = Z_leak + Z_m·Z_load / (Z_m + Z_load) for a unity ratio.
        let expected = leakage + (z_m * load) / (z_m + load);
        assert!((xfmr.input_impedance(load) - expected).magnitude() < 1e-12);

        // The shunt path diverts current, so the parallel branch has a smaller
        // magnitude than the referred load on its own.
        let no_shunt = Transformer::new(1.0, leakage, Complex::new(0.0, 0.0));
        let with_branch = (xfmr.input_impedance(load) - leakage).magnitude();
        let without_branch = (no_shunt.input_impedance(load) - leakage).magnitude();
        assert!(with_branch < without_branch);
    }

    #[test]
    fn exciting_current_and_core_loss() {
        // Y_m = 0.002 − j0.02 pu at 1.0 pu volts.
        let xfmr = Transformer::new(1.0, Complex::new(0.0, 0.05), Complex::new(0.002, -0.02));
        let i0 = xfmr.exciting_current(1.0);
        assert!((i0.re - 0.002).abs() < TOL);
        assert!((i0.im + 0.02).abs() < TOL);
        assert!((xfmr.no_load_loss(1.0) - 0.002).abs() < TOL);
        // Loss scales with the square of the applied voltage.
        assert!((xfmr.no_load_loss(2.0) - 0.008).abs() < TOL);
        // An ideal unit has no core loss and no exciting current.
        let ideal = Transformer::ideal(1.0);
        assert_eq!(ideal.no_load_loss(1.0), 0.0);
        assert_eq!(ideal.exciting_current(1.0).magnitude(), 0.0);
    }

    #[test]
    fn slip_sign_convention() {
        let n_s = synchronous_speed_rpm(50.0, 4);
        assert!((n_s - 1500.0).abs() < TOL);
        // Motoring below synchronous speed.
        assert!(induction_slip(1455.0, n_s) > 0.0);
        // Generating above synchronous speed.
        assert!(induction_slip(1545.0, n_s) < 0.0);
        // Degenerate: no rotating field.
        assert_eq!(induction_slip(900.0, 0.0), 0.0);
    }

    #[test]
    fn rotor_power_split_sums_to_air_gap_power() {
        let p_ag = 20.0e3;
        let s = 0.03;
        let cu = induction_rotor_copper_loss(p_ag, s);
        let mech = induction_mechanical_power(p_ag, s);
        assert!((cu - 600.0).abs() < 1e-9);
        assert!((cu + mech - p_ag).abs() < 1e-9);
        assert!((induction_rotor_frequency(50.0, s) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn shaft_power_round_trip() {
        let p_mech = 1.0e6;
        let eta = 0.96;
        let p_elec = generator_electrical_power(p_mech, eta);
        assert!((generator_mechanical_power(p_elec, eta) - p_mech).abs() < 1e-6);
    }

    #[test]
    fn line_pu_conversion_matches_base() {
        // 100 km of 0.05 Ω/km + j0.4 Ω/km on a 100 MVA / 230 kV base (529 Ω).
        let line = TransmissionLine::new(Complex::new(5.0e-5, 4.0e-4), Complex::new(0.0, 3.0e-9));
        let base = PerUnitSystem::new(100.0e6, 230.0e3);
        let z_pu = line.series_impedance_pu(100.0e3, base);
        assert!((z_pu.re - 5.0 / 529.0).abs() < 1e-12);
        assert!((z_pu.im - 40.0 / 529.0).abs() < 1e-12);
        let y_pu = line.shunt_admittance_pu(100.0e3, base);
        assert!((y_pu.im - 3.0e-4 * 529.0).abs() < 1e-12);
    }

    #[test]
    fn surge_impedance_loading_is_positive() {
        let line = TransmissionLine::from_line_parameters(5.0e-5, 1.0e-6, 1.1e-11, 50.0);
        let sil = line.surge_impedance_loading(230.0e3);
        assert!(sil > 0.0 && sil.is_finite());
        // SIL = V² / |Zc|.
        assert!((sil - 230.0e3 * 230.0e3 / line.surge_impedance()).abs() < 1e-6);
    }
}
