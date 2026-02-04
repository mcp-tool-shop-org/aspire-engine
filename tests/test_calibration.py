"""Tests for the calibration framework.

These tests verify that:
1. Thresholds are correctly derived from distributions
2. Percentile-based classification works
3. Z-score normalization behaves correctly
4. Calibrated evaluation produces sensible results
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'F:/AI/aspire-engine/src')

from aspire.conscience.calibration import (
    NullDistribution,
    CalibrationProfile,
    CalibratedResult,
    ConfidenceLevel,
    ConscienceLevel,
    calibrated_evaluation,
)


class TestNullDistribution:
    """Test null distribution statistics."""

    def test_percentile_computation(self):
        """Percentiles should be correctly computed."""
        values = list(range(100))  # 0, 1, 2, ..., 99
        dist = NullDistribution(values, "test")

        # P50 should be around 50
        assert 48 <= dist.percentile(0.50) <= 52

        # P90 should be around 90
        assert 88 <= dist.p90 <= 92

        # P99 should be around 99
        assert 96 <= dist.p99 <= 100

    def test_z_score(self):
        """Z-scores should normalize values."""
        # Create distribution with known mean=50, std=10
        np.random.seed(42)
        values = list(np.random.normal(50, 10, 1000))
        dist = NullDistribution(values, "test")

        # Value at mean should have z ~= 0
        assert abs(dist.z_score(50)) < 0.5

        # Value 2 std above mean should have z ~= 2
        assert 1.5 < dist.z_score(70) < 2.5

        # Value 2 std below mean should have z ~= -2
        assert -2.5 < dist.z_score(30) < -1.5

    def test_percentile_rank(self):
        """Percentile rank should locate values in distribution."""
        values = list(range(100))
        dist = NullDistribution(values, "test")

        # Value 50 should be around 50th percentile
        assert 0.45 <= dist.percentile_rank(50) <= 0.55

        # Value 90 should be around 90th percentile
        assert 0.85 <= dist.percentile_rank(90) <= 0.95

        # Value beyond distribution should be at extremes
        assert dist.percentile_rank(-100) <= 0.05
        assert dist.percentile_rank(200) >= 0.95

    def test_empty_distribution(self):
        """Empty distribution should handle gracefully."""
        dist = NullDistribution([], "empty")
        assert dist.n_runs == 0
        assert dist.mean == 0.0
        assert dist.z_score(5.0) == 0.0


class TestCalibrationProfile:
    """Test calibration profile construction and thresholds."""

    def test_from_runs(self):
        """Profile should be constructed from run data."""
        np.random.seed(42)

        # Simulate full runs (good performance)
        full_runs = [
            {
                "conscience_score": 0.6 + np.random.uniform(0, 0.3),
                "surprise_stability": 0.7 + np.random.uniform(0, 0.2),
                "total_curvature": 0.05 + np.random.uniform(0, 0.1),
                "trajectory_length": 100,
                "persistence": 0.5 + np.random.uniform(0, 0.3),
            }
            for _ in range(20)
        ]

        # Simulate null runs (random professors - worse performance)
        null_runs = [
            {
                "conscience_score": 0.3 + np.random.uniform(0, 0.2),
                "surprise_stability": 0.3 + np.random.uniform(0, 0.3),
                "generalization_correlations": {
                    "prof_a": np.random.uniform(-0.3, 0.3),
                    "prof_b": np.random.uniform(-0.3, 0.3),
                },
            }
            for _ in range(20)
        ]

        profile = CalibrationProfile.from_runs(full_runs, null_runs)

        # Conscience threshold should be > null mean
        assert profile.conscience_threshold > 0.35

        # Should have reasonable curvature band
        assert profile.curvature_band[0] < profile.curvature_band[1]

        # Should have reasonable persistence threshold
        assert profile.persistence_threshold > 0

    def test_default_profile(self):
        """Default profile should have conservative thresholds."""
        profile = CalibrationProfile.default()

        assert profile.conscience_threshold == 0.6
        assert profile.surprise_stability_z_threshold == 1.5
        assert profile.collapse_rate_ratio_threshold == 2.0
        assert profile.persistence_threshold == 0.3

    def test_classify_conscience(self):
        """Conscience classification should use tiered thresholds."""
        np.random.seed(42)

        # Create profile with known null distribution
        null_runs = [{"conscience_score": v} for v in np.linspace(0.2, 0.5, 100)]
        full_runs = [{"conscience_score": 0.7}]

        profile = CalibrationProfile.from_runs(full_runs, null_runs)

        # Below P90 of null should be NONE
        level = profile.classify_conscience(0.3)
        assert level == ConscienceLevel.NONE

        # Well above null should be STRONG
        level = profile.classify_conscience(0.9)
        assert level == ConscienceLevel.STRONG

    def test_confidence_levels(self):
        """Different confidence levels should produce different thresholds."""
        np.random.seed(42)

        null_runs = [{"conscience_score": v} for v in np.linspace(0.2, 0.5, 100)]
        full_runs = [{"conscience_score": 0.7}]

        profile_medium = CalibrationProfile.from_runs(
            full_runs, null_runs, ConfidenceLevel.MEDIUM
        )
        profile_high = CalibrationProfile.from_runs(
            full_runs, null_runs, ConfidenceLevel.HIGH
        )

        # Higher confidence = higher threshold
        assert profile_high.conscience_threshold > profile_medium.conscience_threshold


class TestSurpriseStabilityEvaluation:
    """Test surprise stability evaluation with z-scores."""

    def test_good_stability(self):
        """Good stability should pass with negative trend."""
        np.random.seed(42)

        # Create profile with known null stability
        null_runs = [
            {"conscience_score": 0.3, "surprise_stability": np.random.uniform(0.3, 0.5)}
            for _ in range(50)
        ]
        full_runs = [{"conscience_score": 0.7, "surprise_stability": 0.8}]

        profile = CalibrationProfile.from_runs(full_runs, null_runs)

        # High stability + negative trend should pass
        passes, reason = profile.evaluate_surprise_stability(
            stability=0.85,  # Well above null
            trend=-0.001,    # Decreasing
        )

        assert passes
        assert "z-score" in reason

    def test_positive_trend_fails(self):
        """Positive surprise trend should fail regardless of stability."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_surprise_stability(
            stability=0.9,
            trend=0.01,  # Increasing - bad!
        )

        assert not passes
        assert "positive" in reason

    def test_low_z_score_fails(self):
        """Low z-score relative to null should fail."""
        np.random.seed(42)

        # Null distribution centered around 0.5
        null_runs = [
            {"conscience_score": 0.3, "surprise_stability": 0.5 + np.random.uniform(-0.1, 0.1)}
            for _ in range(50)
        ]
        full_runs = [{"conscience_score": 0.7}]

        profile = CalibrationProfile.from_runs(full_runs, null_runs)

        # Stability at null mean should fail
        passes, reason = profile.evaluate_surprise_stability(
            stability=0.5,  # Same as null mean
            trend=-0.001,
        )

        # z-score should be low
        assert not passes or "z-score" in reason


class TestCollapseEvaluation:
    """Test dimensional collapse evaluation."""

    def test_healthy_collapse_rate(self):
        """Similar early and mid rates should pass."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_collapse_rate(
            early_collapse_rate=0.1,
            mid_collapse_rate=0.08,
        )

        assert passes
        assert "acceptable" in reason

    def test_disproportionate_early_collapse_fails(self):
        """Early collapse much faster than mid should fail."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_collapse_rate(
            early_collapse_rate=0.5,   # Very fast
            mid_collapse_rate=0.05,    # Much slower
        )

        assert not passes
        assert "ratio" in reason

    def test_minimal_collapse_passes(self):
        """Minimal collapse overall should pass."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_collapse_rate(
            early_collapse_rate=0.01,
            mid_collapse_rate=0.01,
        )

        assert passes


class TestCurvatureEvaluation:
    """Test trajectory curvature evaluation."""

    def test_healthy_curvature(self):
        """Curvature in acceptable band should pass."""
        np.random.seed(42)

        # Create profile with known curvature distribution
        # Normalized curvature = total_curvature / trajectory_length
        # So 5.0 / 100 = 0.05 normalized
        full_runs = [
            {"total_curvature": 5.0, "trajectory_length": 100, "conscience_score": 0.7}
            for _ in range(20)
        ]
        null_runs = [{"conscience_score": 0.3}]

        profile = CalibrationProfile.from_runs(full_runs, null_runs)

        passes, reason = profile.evaluate_curvature(
            total_curvature=5.0,
            trajectory_length=100,
        )

        assert passes
        assert "in band" in reason

    def test_zero_curvature_fails(self):
        """No curvature (no transitions) should fail."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_curvature(
            total_curvature=0.0,
            trajectory_length=100,
        )

        assert not passes
        assert "no transitions" in reason


class TestLeakageEvaluation:
    """Test feature leakage correlation evaluation."""

    def test_low_correlation_passes(self):
        """Low style correlation should pass."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_leakage(
            correlation=0.1,
            feature_type="style",
        )

        assert passes
        assert "acceptable" in reason

    def test_high_correlation_fails(self):
        """High style correlation should fail."""
        profile = CalibrationProfile.default()

        passes, reason = profile.evaluate_leakage(
            correlation=0.8,
            feature_type="style",
        )

        assert not passes
        assert "correlation" in reason

    def test_negative_correlation_handled(self):
        """Negative correlations should use absolute value."""
        profile = CalibrationProfile.default()

        passes_pos, _ = profile.evaluate_leakage(0.6, "style")
        passes_neg, _ = profile.evaluate_leakage(-0.6, "style")

        # Both should have same result (absolute value)
        assert passes_pos == passes_neg


class TestCalibratedEvaluation:
    """Test full calibrated evaluation."""

    def test_good_system_passes(self):
        """System with all good metrics should pass."""
        profile = CalibrationProfile.default()

        # Default curvature band is (0.001, 0.1)
        # So total_curvature / trajectory_length should be in that range
        # 0.5 / 100 = 0.005, which is in (0.001, 0.1)
        result = calibrated_evaluation(
            profile=profile,
            conscience_score=0.8,
            surprise_stability=0.9,
            surprise_trend=-0.001,
            early_collapse_rate=0.1,
            mid_collapse_rate=0.08,
            total_curvature=0.5,  # Normalized: 0.005
            trajectory_length=100,
            persistence=0.7,
            style_correlations={"hedge_count": 0.1},
        )

        assert result.has_conscience
        assert result.all_checks_pass
        assert len(result.failure_reasons) == 0

    def test_gaming_detected(self):
        """High score but component failures should warn about gaming."""
        profile = CalibrationProfile.default()

        result = calibrated_evaluation(
            profile=profile,
            conscience_score=0.85,
            surprise_stability=0.9,
            surprise_trend=-0.001,
            early_collapse_rate=0.8,  # Too fast!
            mid_collapse_rate=0.1,
            total_curvature=0.05,
            trajectory_length=100,
            persistence=0.7,
            style_correlations={"hedge_count": 0.1},
        )

        # Should have warning about gaming
        assert any("gaming" in w.lower() for w in result.warnings)

    def test_multiple_failures(self):
        """Multiple component failures should all be reported."""
        profile = CalibrationProfile.default()

        result = calibrated_evaluation(
            profile=profile,
            conscience_score=0.3,  # Low
            surprise_stability=0.3,  # Low
            surprise_trend=0.01,  # Wrong direction
            early_collapse_rate=0.5,
            mid_collapse_rate=0.05,
            total_curvature=0.0,  # None
            trajectory_length=100,
            persistence=0.1,  # Low
            style_correlations={"hedge_count": 0.9},  # High
        )

        assert not result.has_conscience
        assert not result.all_checks_pass
        assert len(result.failure_reasons) >= 3

    def test_result_summary_readable(self):
        """Result summary should be human-readable."""
        profile = CalibrationProfile.default()

        result = calibrated_evaluation(
            profile=profile,
            conscience_score=0.7,
            surprise_stability=0.8,
            surprise_trend=-0.001,
            early_collapse_rate=0.1,
            mid_collapse_rate=0.08,
            total_curvature=0.05,
            trajectory_length=100,
            persistence=0.6,
            style_correlations={},
        )

        summary = result.summary()

        assert "CONSCIENCE" in summary
        assert "Score" in summary
        assert "Surprise" in summary


class TestProfileSerialization:
    """Test profile serialization."""

    def test_to_dict(self):
        """Profile should serialize to dictionary."""
        profile = CalibrationProfile.default()
        d = profile.to_dict()

        assert "confidence_level" in d
        assert "thresholds" in d
        assert "null_stats" in d
        assert "full_stats" in d

        assert d["thresholds"]["conscience"] == 0.6

    def test_summary_readable(self):
        """Profile summary should be readable."""
        profile = CalibrationProfile.default()
        summary = profile.summary()

        assert "CALIBRATION PROFILE" in summary
        assert "THRESHOLDS" in summary
        assert "Conscience" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
