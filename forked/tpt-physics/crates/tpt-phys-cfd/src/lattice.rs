//! The D2Q9 lattice definition: velocity set, weights, and equilibrium.
//!
//! All constants use the standard lattice-unit convention `cs² = 1/3`, `Δx =
//! Δt = 1`.

/// The 9-velocity D2Q9 lattice.
pub struct D2Q9;

impl D2Q9 {
    /// Squared lattice sound speed `cs² = 1/3`.
    pub const CS2: f64 = 1.0 / 3.0;

    /// The nine discrete velocities `(ex, ey)`, ordered
    /// `0:rest, 1:E, 2:N, 3:W, 4:S, 5:NE, 6:NW, 7:SW, 8:SE`.
    pub const E: [(i32, i32); 9] = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ];

    /// The nine lattice weights.
    pub const W: [f64; 9] = [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];

    /// Index of the opposite velocity for each direction (used for bounce-back).
    pub const OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

    /// Second-order equilibrium distribution `f_i^eq(ρ, u)`.
    ///
    /// `f_i^eq = w_i ρ (1 + 3 e·u + 4.5 (e·u)² − 1.5 |u|²)`.
    #[inline]
    pub fn equilibrium(rho: f64, u: [f64; 2]) -> [f64; 9] {
        let u2 = u[0] * u[0] + u[1] * u[1];
        let mut feq = [0.0_f64; 9];
        for (i, slot) in feq.iter_mut().enumerate() {
            let (ex, ey) = D2Q9::E[i];
            let eu = ex as f64 * u[0] + ey as f64 * u[1];
            *slot = D2Q9::W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
        }
        feq
    }
}
