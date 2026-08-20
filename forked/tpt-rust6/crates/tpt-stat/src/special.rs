//! Special functions used by distribution CDFs (compact Numerical-Recipes style
//! implementations): error function, log-gamma, regularized incomplete gamma and
//! beta functions.

pub fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

pub fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Regularized lower incomplete gamma `P(a, x)` via series (x < a+1) or
/// continued fraction (otherwise). Standard NR routine.
pub fn gammap(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut del = sum;
        let gln = lgamma(a);
        for n in 1.. {
            let an = n as f64;
            del *= x / (a + an);
            sum += del;
            if del.abs() < sum.abs() * 1e-12 {
                break;
            }
            if n > 1000 {
                break;
            }
        }
        sum * (-x + a * x.ln() - gln).exp()
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        let gln = lgamma(a);
        for i in 1.. {
            let i = i as f64;
            let an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-12 {
                break;
            }
            if i > 1000.0 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - gln).exp() * h
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let fpmin = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    let mut d = 1.0 / d;
    let mut h = d;
    for m in 1.. {
        let m = m as f64;
        let m2 = m * 2.0;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
        if m > 500.0 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_sanity() {
        assert!((erf(0.0)).abs() < 1e-9);
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
    }

    #[test]
    fn gamma_p_basic() {
        // P(1, x) = 1 - e^{-x}
        assert!((gammap(1.0, 2.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn beta_i_symmetric() {
        let v = betai(2.0, 2.0, 0.5);
        assert!((v - 0.5).abs() < 1e-9);
    }
}
