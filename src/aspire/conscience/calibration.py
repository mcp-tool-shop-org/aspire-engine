"""Calibration framework for ASPIRE thresholds.

This module implements distribution-based calibration, replacing fixed
magic numbers with empirically-derived thresholds. The key insight:

    Any threshold that encodes "good judgment" should be relative, not absolute.

Absolute thresholds are acceptable only for:
- Sanity checks
- Null-condition rejection
- Invariants (e.g., "this should never happen")

Everything else should be:
- Distribution-aware
- Regime-aware
- Preferably self-normalizing

Literature Foundation:
- Percentile-based thresholds: Standard in anomaly detection and control charts
- Z-score normalization: Foundational in statistical process control
- Null distribution comparison: Basis of hypothesis testing (Fisher, Neyman-Pearson)
- Calibration in ML: Platt scaling, isotonic regression for probability calibration

See docs/LITERATURE_REVIEW.md for complete citations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy import stats


class ConfidenceLevel(Enum):
    """Standard confidence levels for threshold computation."""
    LOW = 0.90      # P90 - permissive
    MEDIUM = 0.95   # P95 - standard
    HIGH = 0.99     # P99 - strict
    VERY_HIGH = 0.999  # P99.9 - very strict


class ConscienceLevel(Enum):
    """Tiered conscience classification based on null distribution."""
    NONE = "none"           # ≤ P90 of null
    WEAK = "weak"           # P90 - P95
    MODERATE = "moderate"   # P95 - P99
    STRONG = "strong"       # ≥ P99


@dataclass
class NullDistribution:
    """Represents a null distribution from ablation runs.

    Used to compute percentile-based thresholds that answer:
    "Is this metric better than what we'd expect by chance?"
    """
    values: List[float]
    source: str  # e.g., "RANDOM_PROFESSORS", "NO_CRITIC"
    n_runs: int = 0

    # Cached percentiles
    _percentiles: Dict[float, float] = field(default_factory=dict)

    def __post_init__(self):
        self.n_runs = len(self.values)
        if self.n_runs > 0:
            self._compute_percentiles()

    def _compute_percentiles(self):
        """Pre-compute common percentiles."""
        for p in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
            self._percentiles[p] = float(np.percentile(self.values, p * 100))

    def percentile(self, p: float) -> float:
        """Get value at percentile p (0-1)."""
        if p in self._percentiles:
            return self._percentiles[p]
        return float(np.percentile(self.values, p * 100))

    def z_score(self, value: float) -> float:
        """Compute z-score relative to this distribution."""
        if self.n_runs < 2:
            return 0.0
        mean = np.mean(self.values)
        std = np.std(self.values, ddof=1)
        if std < 1e-10:
            return 0.0
        return (value - mean) / std

    def percentile_rank(self, value: float) -> float:
        """What percentile does this value fall at? (0-1)"""
        return stats.percentileofscore(self.values, value) / 100.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    @property
    def p90(self) -> float:
        return self.percentile(0.90)

    @property
    def p95(self) -> float:
        return self.percentile(0.95)

    @property
    def p99(self) -> float:
        return self.percentile(0.99)


@dataclass
class CalibrationProfile:
    """Calibrated thresholds derived from baseline runs.

    This replaces magic numbers with empirically-derived thresholds.
    All thresholds become artifacts of the calibration process, not
    arbitrary constants.

    Usage:
        # Collect baseline runs
        full_scores = [run.conscience_score for run in full_runs]
        null_scores = [run.conscience_score for run in random_prof_runs]

        # Build calibration profile
        profile = CalibrationProfile.from_runs(
            full_runs=full_runs,
            null_runs=random_prof_runs,
            confidence_level=ConfidenceLevel.MEDIUM
        )

        # Use calibrated thresholds
        has_conscience = score > profile.conscience_threshold
        level = profile.classify_conscience(score)
    """

    # Null distributions for different metrics
    null_conscience: NullDistribution
    null_surprise_stability: NullDistribution
    null_generalization: NullDistribution

    # Full (non-ablated) distributions for comparison
    full_conscience: NullDistribution
    full_surprise_stability: NullDistribution
    full_curvature: NullDistribution
    full_persistence: NullDistribution

    # Confidence level used for threshold computation
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM

    # Computed thresholds (populated by compute_thresholds)
    conscience_threshold: float = 0.0
    surprise_stability_z_threshold: float = 1.5
    collapse_rate_ratio_threshold: float = 2.0
    curvature_band: Tuple[float, float] = (0.0, 1.0)
    persistence_threshold: float = 0.0
    leakage_correlation_threshold: float = 0.0

    def __post_init__(self):
        self.compute_thresholds()

    def compute_thresholds(self):
        """Derive all thresholds from distributions."""
        conf = self.confidence_level.value

        # Conscience threshold: Must exceed P_conf of null distribution
        # This ensures we're "better than random professors"
        if self.null_conscience.n_runs > 0:
            self.conscience_threshold = self.null_conscience.percentile(conf)

        # Surprise stability: Use z-score threshold
        # z > 1.5 means "stabilizing more than expected by chance"
        # This is kept as a constant since z-scores have standard meaning
        self.surprise_stability_z_threshold = 1.5

        # Collapse rate ratio: Early vs mid acceleration
        # Flag if early rate > 2x mid rate
        # This is a relative measure, so constant ratio is acceptable
        self.collapse_rate_ratio_threshold = 2.0

        # Curvature band: Based on full runs distribution
        # Acceptable = [P25, P90] of healthy runs
        if self.full_curvature.n_runs > 0:
            self.curvature_band = (
                self.full_curvature.percentile(0.25),
                self.full_curvature.percentile(0.90),
            )

        # Persistence: Must exceed lower quartile of full runs
        if self.full_persistence.n_runs > 0:
            self.persistence_threshold = self.full_persistence.percentile(0.25)

        # Leakage correlation: Flag if |ρ| > P_conf of null
        # Use random professors to establish expected correlation noise
        if self.null_generalization.n_runs > 0:
            # For leakage, we care about absolute correlation
            abs_values = [abs(v) for v in self.null_generalization.values]
            null_abs = NullDistribution(abs_values, "null_abs_correlation")
            self.leakage_correlation_threshold = null_abs.percentile(conf)

    def classify_conscience(self, score: float) -> ConscienceLevel:
        """Classify conscience level using tiered thresholds.

        Returns one of: NONE, WEAK, MODERATE, STRONG
        Based on where score falls relative to null distribution.
        """
        if self.null_conscience.n_runs == 0:
            # No null distribution - fall back to absolute thresholds
            if score < 0.5:
                return ConscienceLevel.NONE
            elif score < 0.6:
                return ConscienceLevel.WEAK
            elif score < 0.7:
                return ConscienceLevel.MODERATE
            else:
                return ConscienceLevel.STRONG

        percentile = self.null_conscience.percentile_rank(score)

        if percentile < 0.90:
            return ConscienceLevel.NONE
        elif percentile < 0.95:
            return ConscienceLevel.WEAK
        elif percentile < 0.99:
            return ConscienceLevel.MODERATE
        else:
            return ConscienceLevel.STRONG

    def evaluate_surprise_stability(
        self,
        stability: float,
        trend: float,
    ) -> Tuple[bool, str]:
        """Evaluate surprise stability using z-score normalization.

        Args:
            stability: Raw stability value
            trend: Surprise trend (should be negative for healthy learning)

        Returns:
            (passes, reason): Whether stability is acceptable and why
        """
        if self.null_surprise_stability.n_runs == 0:
            # Fallback to absolute check
            if stability < 0.5:
                return False, "stability below 0.5 (no baseline)"
            if trend > 0.001:
                return False, f"surprise trend positive: {trend:.4f}"
            return True, "passes absolute checks"

        z = self.null_surprise_stability.z_score(stability)

        if z < self.surprise_stability_z_threshold:
            return False, f"z-score {z:.2f} < threshold {self.surprise_stability_z_threshold}"

        if trend > 0:
            return False, f"surprise trend positive: {trend:.4f} (should decrease)"

        return True, f"z-score {z:.2f} with negative trend {trend:.4f}"

    def evaluate_collapse_rate(
        self,
        early_collapse_rate: float,
        mid_collapse_rate: float,
    ) -> Tuple[bool, str]:
        """Evaluate dimensional collapse using relative acceleration.

        Instead of absolute thresholds, we compare early vs mid rates.
        Flag if early rate is disproportionately faster.

        Args:
            early_collapse_rate: Collapse rate in first 20% of training
            mid_collapse_rate: Collapse rate in 20-60% of training

        Returns:
            (passes, reason): Whether collapse rate is acceptable
        """
        if mid_collapse_rate < 1e-10:
            # No mid collapse - check if early collapse is minimal too
            if early_collapse_rate > 0.1:
                return False, "early collapse with no mid-training collapse"
            return True, "minimal collapse overall"

        ratio = early_collapse_rate / mid_collapse_rate

        if ratio > self.collapse_rate_ratio_threshold:
            return False, (
                f"early/mid collapse ratio {ratio:.2f} > "
                f"threshold {self.collapse_rate_ratio_threshold}"
            )

        return True, f"collapse ratio {ratio:.2f} within acceptable range"

    def evaluate_curvature(
        self,
        total_curvature: float,
        trajectory_length: int,
    ) -> Tuple[bool, str]:
        """Evaluate trajectory curvature.

        Curvature indicates phase transitions during learning.
        Too little = no real learning transitions
        Too much = unstable/thrashing

        Args:
            total_curvature: Sum of curvature over trajectory
            trajectory_length: Number of steps in trajectory

        Returns:
            (passes, reason): Whether curvature is in acceptable band
        """
        if trajectory_length == 0:
            return False, "empty trajectory"

        # Normalize by trajectory length
        normalized = total_curvature / trajectory_length

        low, high = self.curvature_band

        if normalized < low:
            return False, f"curvature {normalized:.4f} < lower bound {low:.4f} (no transitions)"

        if normalized > high:
            return False, f"curvature {normalized:.4f} > upper bound {high:.4f} (unstable)"

        return True, f"curvature {normalized:.4f} in band [{low:.4f}, {high:.4f}]"

    def evaluate_persistence(self, persistence: float) -> Tuple[bool, str]:
        """Evaluate geometric persistence.

        Persistence measures how well structure survives perturbation.
        Must exceed lower quartile of healthy runs.
        """
        if persistence < self.persistence_threshold:
            return False, (
                f"persistence {persistence:.3f} < "
                f"threshold {self.persistence_threshold:.3f}"
            )

        return True, f"persistence {persistence:.3f} acceptable"

    def evaluate_leakage(
        self,
        correlation: float,
        feature_type: str = "style",
    ) -> Tuple[bool, str]:
        """Evaluate feature leakage correlation.

        Args:
            correlation: Correlation between text features and token scores
            feature_type: "style" (hedge, markers) or "semantic"

        Returns:
            (passes, reason): Whether correlation indicates leakage
        """
        abs_corr = abs(correlation)

        # Style features (hedge_count, markers) should have LOW correlation
        # Semantic features can have higher expected correlation
        if feature_type == "style":
            threshold = self.leakage_correlation_threshold
        else:
            # Allow higher correlation for semantic features
            threshold = min(0.8, self.leakage_correlation_threshold * 1.5)

        if abs_corr > threshold:
            return False, (
                f"{feature_type} correlation |{correlation:.3f}| > "
                f"threshold {threshold:.3f}"
            )

        return True, f"{feature_type} correlation {correlation:.3f} acceptable"

    @classmethod
    def from_runs(
        cls,
        full_runs: List[Dict[str, Any]],
        null_runs: List[Dict[str, Any]],
        confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> "CalibrationProfile":
        """Build calibration profile from baseline runs.

        Args:
            full_runs: Results from FULL (non-ablated) training runs
                Each dict should have: conscience_score, surprise_stability,
                total_curvature, persistence, trajectory_length
            null_runs: Results from RANDOM_PROFESSORS ablation runs
                Each dict should have: conscience_score, surprise_stability,
                generalization_correlations
            confidence_level: How strict to make thresholds

        Returns:
            Calibrated profile with empirically-derived thresholds
        """
        # Extract null distributions
        null_conscience = NullDistribution(
            values=[r.get("conscience_score", 0.0) for r in null_runs],
            source="RANDOM_PROFESSORS",
        )

        null_surprise = NullDistribution(
            values=[r.get("surprise_stability", 0.0) for r in null_runs],
            source="RANDOM_PROFESSORS",
        )

        # For generalization, collect all correlation values
        null_gen_values = []
        for r in null_runs:
            corrs = r.get("generalization_correlations", {})
            null_gen_values.extend(corrs.values())
        null_gen = NullDistribution(
            values=null_gen_values if null_gen_values else [0.0],
            source="RANDOM_PROFESSORS",
        )

        # Extract full distributions
        full_conscience = NullDistribution(
            values=[r.get("conscience_score", 0.0) for r in full_runs],
            source="FULL",
        )

        full_surprise = NullDistribution(
            values=[r.get("surprise_stability", 0.0) for r in full_runs],
            source="FULL",
        )

        # Normalize curvature by trajectory length
        full_curvature_values = []
        for r in full_runs:
            curv = r.get("total_curvature", 0.0)
            length = r.get("trajectory_length", 1)
            full_curvature_values.append(curv / max(1, length))
        full_curvature = NullDistribution(
            values=full_curvature_values if full_curvature_values else [0.0],
            source="FULL",
        )

        full_persistence = NullDistribution(
            values=[r.get("persistence", 0.0) for r in full_runs],
            source="FULL",
        )

        return cls(
            null_conscience=null_conscience,
            null_surprise_stability=null_surprise,
            null_generalization=null_gen,
            full_conscience=full_conscience,
            full_surprise_stability=full_surprise,
            full_curvature=full_curvature,
            full_persistence=full_persistence,
            confidence_level=confidence_level,
        )

    @classmethod
    def default(cls) -> "CalibrationProfile":
        """Create a default profile with reasonable fallback values.

        Use this when no baseline runs are available yet.
        The thresholds are conservative but not calibrated.
        """
        # Create empty distributions
        empty_null = NullDistribution([], "empty")
        empty_full = NullDistribution([], "empty")

        profile = cls(
            null_conscience=empty_null,
            null_surprise_stability=empty_null,
            null_generalization=empty_null,
            full_conscience=empty_full,
            full_surprise_stability=empty_full,
            full_curvature=empty_full,
            full_persistence=empty_full,
            confidence_level=ConfidenceLevel.MEDIUM,
        )

        # Set conservative defaults
        profile.conscience_threshold = 0.6  # Historical default
        profile.surprise_stability_z_threshold = 1.5
        profile.collapse_rate_ratio_threshold = 2.0
        profile.curvature_band = (0.001, 0.1)  # Conservative
        profile.persistence_threshold = 0.3
        profile.leakage_correlation_threshold = 0.5

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Export profile as dictionary for serialization."""
        return {
            "confidence_level": self.confidence_level.name,
            "thresholds": {
                "conscience": self.conscience_threshold,
                "surprise_stability_z": self.surprise_stability_z_threshold,
                "collapse_rate_ratio": self.collapse_rate_ratio_threshold,
                "curvature_band": self.curvature_band,
                "persistence": self.persistence_threshold,
                "leakage_correlation": self.leakage_correlation_threshold,
            },
            "null_stats": {
                "conscience": {
                    "mean": self.null_conscience.mean,
                    "std": self.null_conscience.std,
                    "p90": self.null_conscience.p90 if self.null_conscience.n_runs > 0 else None,
                    "p95": self.null_conscience.p95 if self.null_conscience.n_runs > 0 else None,
                    "n_runs": self.null_conscience.n_runs,
                },
                "surprise_stability": {
                    "mean": self.null_surprise_stability.mean,
                    "std": self.null_surprise_stability.std,
                    "n_runs": self.null_surprise_stability.n_runs,
                },
            },
            "full_stats": {
                "curvature": {
                    "p25": self.full_curvature.percentile(0.25) if self.full_curvature.n_runs > 0 else None,
                    "p90": self.full_curvature.p90 if self.full_curvature.n_runs > 0 else None,
                    "n_runs": self.full_curvature.n_runs,
                },
                "persistence": {
                    "p25": self.full_persistence.percentile(0.25) if self.full_persistence.n_runs > 0 else None,
                    "n_runs": self.full_persistence.n_runs,
                },
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary of calibration profile."""
        lines = [
            "=" * 60,
            "CALIBRATION PROFILE",
            "=" * 60,
            f"Confidence Level: {self.confidence_level.name} ({self.confidence_level.value})",
            "",
            "THRESHOLDS:",
            f"  Conscience:           {self.conscience_threshold:.4f}",
            f"  Surprise Stability z: {self.surprise_stability_z_threshold:.2f}",
            f"  Collapse Rate Ratio:  {self.collapse_rate_ratio_threshold:.1f}x",
            f"  Curvature Band:       [{self.curvature_band[0]:.4f}, {self.curvature_band[1]:.4f}]",
            f"  Persistence:          {self.persistence_threshold:.4f}",
            f"  Leakage Correlation:  {self.leakage_correlation_threshold:.4f}",
            "",
            "NULL DISTRIBUTION (RANDOM_PROFESSORS):",
            f"  Conscience:  mean={self.null_conscience.mean:.4f}, "
            f"std={self.null_conscience.std:.4f}, n={self.null_conscience.n_runs}",
            f"  Stability:   mean={self.null_surprise_stability.mean:.4f}, "
            f"std={self.null_surprise_stability.std:.4f}, n={self.null_surprise_stability.n_runs}",
            "",
            "FULL DISTRIBUTION:",
            f"  Curvature:   n={self.full_curvature.n_runs}",
            f"  Persistence: n={self.full_persistence.n_runs}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class CalibratedResult:
    """Result of evaluating metrics against calibrated thresholds."""

    # Overall verdict
    has_conscience: bool
    conscience_level: ConscienceLevel
    conscience_score: float

    # Component evaluations
    surprise_passes: bool
    surprise_reason: str

    collapse_passes: bool
    collapse_reason: str

    curvature_passes: bool
    curvature_reason: str

    persistence_passes: bool
    persistence_reason: str

    leakage_passes: bool
    leakage_reason: str

    # Warnings (non-fatal issues)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_checks_pass(self) -> bool:
        """Do all component checks pass?"""
        return all([
            self.surprise_passes,
            self.collapse_passes,
            self.curvature_passes,
            self.persistence_passes,
            self.leakage_passes,
        ])

    @property
    def failure_reasons(self) -> List[str]:
        """List all failure reasons."""
        reasons = []
        if not self.surprise_passes:
            reasons.append(f"Surprise: {self.surprise_reason}")
        if not self.collapse_passes:
            reasons.append(f"Collapse: {self.collapse_reason}")
        if not self.curvature_passes:
            reasons.append(f"Curvature: {self.curvature_reason}")
        if not self.persistence_passes:
            reasons.append(f"Persistence: {self.persistence_reason}")
        if not self.leakage_passes:
            reasons.append(f"Leakage: {self.leakage_reason}")
        return reasons

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ CONSCIENCE PRESENT" if self.has_conscience else "✗ NO CONSCIENCE"

        lines = [
            f"{status} (level: {self.conscience_level.value})",
            f"Score: {self.conscience_score:.4f}",
            "",
            "Component Checks:",
            f"  {'✓' if self.surprise_passes else '✗'} Surprise: {self.surprise_reason}",
            f"  {'✓' if self.collapse_passes else '✗'} Collapse: {self.collapse_reason}",
            f"  {'✓' if self.curvature_passes else '✗'} Curvature: {self.curvature_reason}",
            f"  {'✓' if self.persistence_passes else '✗'} Persistence: {self.persistence_reason}",
            f"  {'✓' if self.leakage_passes else '✗'} Leakage: {self.leakage_reason}",
        ]

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


def calibrated_evaluation(
    profile: CalibrationProfile,
    conscience_score: float,
    surprise_stability: float,
    surprise_trend: float,
    early_collapse_rate: float,
    mid_collapse_rate: float,
    total_curvature: float,
    trajectory_length: int,
    persistence: float,
    style_correlations: Dict[str, float],
) -> CalibratedResult:
    """Evaluate all metrics using calibrated thresholds.

    This is the main entry point for calibrated conscience evaluation.

    Args:
        profile: Calibration profile with thresholds
        conscience_score: Overall conscience score (0-1)
        surprise_stability: Surprise stability metric
        surprise_trend: Trend of surprise over training
        early_collapse_rate: Collapse rate in first 20%
        mid_collapse_rate: Collapse rate in 20-60%
        total_curvature: Total trajectory curvature
        trajectory_length: Number of trajectory steps
        persistence: Geometric persistence score
        style_correlations: Text feature correlations (hedge_count, etc.)

    Returns:
        CalibratedResult with all evaluations
    """
    # Classify conscience level
    conscience_level = profile.classify_conscience(conscience_score)
    has_conscience = conscience_level != ConscienceLevel.NONE

    # Evaluate components
    surprise_passes, surprise_reason = profile.evaluate_surprise_stability(
        surprise_stability, surprise_trend
    )

    collapse_passes, collapse_reason = profile.evaluate_collapse_rate(
        early_collapse_rate, mid_collapse_rate
    )

    curvature_passes, curvature_reason = profile.evaluate_curvature(
        total_curvature, trajectory_length
    )

    persistence_passes, persistence_reason = profile.evaluate_persistence(persistence)

    # Evaluate all style correlations
    leakage_passes = True
    leakage_reasons = []
    for feat_name, corr in style_correlations.items():
        passes, reason = profile.evaluate_leakage(corr, feature_type="style")
        if not passes:
            leakage_passes = False
            leakage_reasons.append(f"{feat_name}: {reason}")

    leakage_reason = "; ".join(leakage_reasons) if leakage_reasons else "no leakage detected"

    # Collect warnings
    warnings = []

    # Warn if conscience is weak
    if conscience_level == ConscienceLevel.WEAK:
        warnings.append("Conscience is weak - consider longer training or different professors")

    # Warn if any component is borderline
    if surprise_passes and surprise_stability < 0.6:
        warnings.append(f"Surprise stability is borderline: {surprise_stability:.3f}")

    if persistence_passes and persistence < 0.5:
        warnings.append(f"Persistence is borderline: {persistence:.3f}")

    # If score is high but components fail, warn about potential gaming
    if conscience_score > 0.7 and not collapse_passes:
        warnings.append("High score but collapse detected - possible gaming")

    return CalibratedResult(
        has_conscience=has_conscience,
        conscience_level=conscience_level,
        conscience_score=conscience_score,
        surprise_passes=surprise_passes,
        surprise_reason=surprise_reason,
        collapse_passes=collapse_passes,
        collapse_reason=collapse_reason,
        curvature_passes=curvature_passes,
        curvature_reason=curvature_reason,
        persistence_passes=persistence_passes,
        persistence_reason=persistence_reason,
        leakage_passes=leakage_passes,
        leakage_reason=leakage_reason,
        warnings=warnings,
    )
