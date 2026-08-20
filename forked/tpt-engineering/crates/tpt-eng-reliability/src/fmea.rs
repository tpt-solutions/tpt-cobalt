//! Failure Mode and Effects Analysis (FMEA) data structures.

/// A single FMEA line item.
#[derive(Debug, Clone, PartialEq)]
pub struct FmeaItem {
    /// Unique identifier.
    pub id: String,
    /// Function or item being analyzed.
    pub function: String,
    /// Potential failure mode.
    pub failure_mode: String,
    /// Potential cause of the failure.
    pub cause: String,
    /// Effect of the failure.
    pub effect: String,
    /// Severity ranking (typically 1–10).
    pub severity: u8,
    /// Occurrence ranking (typically 1–10).
    pub occurrence: u8,
    /// Detection ranking (typically 1–10, higher = harder to detect).
    pub detection: u8,
}

impl FmeaItem {
    /// Build an FMEA item. Rankings are clamped to the 1–10 scale.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        function: impl Into<String>,
        failure_mode: impl Into<String>,
        cause: impl Into<String>,
        effect: impl Into<String>,
        severity: u8,
        occurrence: u8,
        detection: u8,
    ) -> Self {
        let clamp = |v: u8| v.clamp(1, 10);
        Self {
            id: id.into(),
            function: function.into(),
            failure_mode: failure_mode.into(),
            cause: cause.into(),
            effect: effect.into(),
            severity: clamp(severity),
            occurrence: clamp(occurrence),
            detection: clamp(detection),
        }
    }

    /// Risk Priority Number `RPN = severity * occurrence * detection`.
    pub fn rpn(&self) -> u32 {
        self.severity as u32 * self.occurrence as u32 * self.detection as u32
    }

    /// Criticality using a severity-weighted RPN (`severity^2 * occurrence * detection`).
    pub fn criticality(&self) -> u32 {
        (self.severity as u32).pow(2) * self.occurrence as u32 * self.detection as u32
    }
}

/// Rank FMEA items by RPN in descending order (highest risk first).
pub fn rank_by_rpn(items: &[FmeaItem]) -> Vec<&FmeaItem> {
    let mut v: Vec<&FmeaItem> = items.iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.rpn()));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<FmeaItem> {
        vec![
            FmeaItem::new("A", "pump", "leak", "seal wear", "fluid loss", 7, 3, 4),
            FmeaItem::new("B", "pump", "stall", "bearing seize", "no flow", 9, 2, 5),
            FmeaItem::new("C", "valve", "stick", "corrosion", "reduced flow", 4, 2, 2),
        ]
    }

    #[test]
    fn rpn_and_ranking() {
        let items = sample();
        assert_eq!(items[0].rpn(), 7 * 3 * 4);
        let ranked = rank_by_rpn(&items);
        // B (9*2*5=90) > A (84) > C (16)
        assert_eq!(ranked[0].id, "B");
        assert_eq!(ranked[1].id, "A");
        assert_eq!(ranked[2].id, "C");
    }
}
