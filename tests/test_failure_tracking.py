"""Tests for failure tracking infrastructure."""

import pytest
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, 'F:/AI/aspire-engine/src')

from aspire.experiments.failure_tracking import (
    FailureTracker,
    FailureReport,
    FailureCase,
    FailureCategory,
    generate_failure_atlas_data,
)


@dataclass
class MockResult:
    """Mock result for testing."""
    final_conscience_score: float = 0.5
    final_surprise_stability: float = 0.7
    student_type: str = "honest"
    gaming_detected: bool = False
    behavior_changed: bool = False
    holdout_professor_correlation: float = 0.5
    transfer_ratio: float = 0.8
    max_leakage_correlation: float = 0.2


@dataclass
class MockConfig:
    """Mock config for testing."""
    n_training_cycles: int = 50
    n_training_items: int = 100
    n_runs_per_condition: int = 5


class TestFailureCase:
    """Test FailureCase dataclass."""

    def test_create_failure_case(self):
        """Can create a failure case."""
        case = FailureCase(
            category=FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            experiment="test_exp",
            condition="scalar_reward",
            run_idx=0,
            seed=42,
            description="Test failure",
            expected=True,
            observed_value=0.4,
            expected_range=(0.0, 0.3),
            training_cycles=50,
            sample_size=100,
        )

        assert case.category == FailureCategory.SCALAR_PARTIAL_CONSCIENCE
        assert case.expected is True
        assert case.observed_value == 0.4

    def test_failure_case_to_dict(self):
        """FailureCase serializes to dict."""
        case = FailureCase(
            category=FailureCategory.RANDOM_MATCHES_FULL,
            experiment="test",
            condition="random",
            run_idx=1,
            seed=123,
            description="Random matched full",
            expected=False,
            observed_value=0.7,
            expected_range=(0.0, 0.5),
            training_cycles=50,
            sample_size=100,
            falsifies_theory=True,
        )

        d = case.to_dict()

        assert d["category"] == "random_matches_full"
        assert d["falsifies_theory"] is True
        assert d["observed_value"] == 0.7
        assert d["expected_range"] == [0.0, 0.5]


class TestFailureReport:
    """Test FailureReport aggregation."""

    def test_add_failure(self):
        """Can add failures to report."""
        report = FailureReport(experiment_name="test", total_runs=10)

        case1 = FailureCase(
            category=FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            experiment="test",
            condition="scalar",
            run_idx=0,
            seed=42,
            description="Expected failure",
            expected=True,
            observed_value=0.4,
            expected_range=(0.0, 0.3),
            training_cycles=50,
            sample_size=100,
        )

        case2 = FailureCase(
            category=FailureCategory.RANDOM_MATCHES_FULL,
            experiment="test",
            condition="random",
            run_idx=1,
            seed=43,
            description="Unexpected failure",
            expected=False,
            observed_value=0.7,
            expected_range=(0.0, 0.5),
            training_cycles=50,
            sample_size=100,
            falsifies_theory=True,
        )

        report.add_failure(case1)
        report.add_failure(case2)

        assert len(report.failures) == 2
        assert report.expected_failure_count == 1
        assert report.unexpected_failure_count == 1
        assert report.falsification_count == 1

    def test_compute_statistics(self):
        """Statistics are computed correctly."""
        report = FailureReport(experiment_name="test", total_runs=20)

        # Add multiple failures of same category
        for i in range(5):
            report.add_failure(FailureCase(
                category=FailureCategory.SEED_VARIANCE,
                experiment="test",
                condition="full",
                run_idx=i,
                seed=i,
                description=f"Variance failure {i}",
                expected=True,
                observed_value=0.4 + i * 0.05,
                expected_range=(0.0, 0.3),
                training_cycles=50,
                sample_size=100,
            ))

        stats = report.compute_statistics()

        assert stats["total_failures"] == 5
        assert stats["expected_failures"] == 5
        assert stats["failure_rate"] == 0.25  # 5/20
        assert "seed_variance" in stats["by_category"]
        assert stats["by_category"]["seed_variance"]["count"] == 5


class TestFailureTracker:
    """Test FailureTracker tracking logic."""

    def test_tracker_initialization(self):
        """Tracker initializes correctly."""
        tracker = FailureTracker("test_experiment")

        assert tracker.experiment_name == "test_experiment"
        assert tracker.report.total_runs == 0

    def test_track_scalar_partial_conscience(self):
        """Detects SCALAR_REWARD with partial conscience."""
        tracker = FailureTracker("exp1")
        config = MockConfig()

        result = MockResult(final_conscience_score=0.4)

        tracker.track_run("scalar_reward", 0, 42, result, config)

        report = tracker.get_report()
        assert report.total_runs == 1
        assert len(report.failures) == 1
        assert report.failures[0].category == FailureCategory.SCALAR_PARTIAL_CONSCIENCE

    def test_no_failure_for_low_scalar_conscience(self):
        """No failure when SCALAR has appropriately low conscience."""
        tracker = FailureTracker("exp1")
        config = MockConfig()

        result = MockResult(final_conscience_score=0.2)

        tracker.track_run("scalar_reward", 0, 42, result, config)

        report = tracker.get_report()
        assert len(report.failures) == 0

    def test_track_early_holdout_failure(self):
        """Detects holdout failure in early training."""
        tracker = FailureTracker("exp2")
        config = MockConfig(n_training_cycles=20)  # Short training

        result = MockResult(holdout_professor_correlation=0.2)

        tracker.track_run("holdout_one", 0, 42, result, config)

        report = tracker.get_report()
        assert any(
            f.category == FailureCategory.EARLY_HOLDOUT_FAILURE
            for f in report.failures
        )

    def test_track_holdout_success_with_long_training(self):
        """No holdout failure with sufficient training and good transfer."""
        tracker = FailureTracker("exp2")
        config = MockConfig(n_training_cycles=50)

        result = MockResult(holdout_professor_correlation=0.6)

        tracker.track_run("holdout_one", 0, 42, result, config)

        report = tracker.get_report()
        # Should not have early holdout failure
        assert not any(
            f.category == FailureCategory.EARLY_HOLDOUT_FAILURE
            for f in report.failures
        )

    def test_track_no_transfer_falsification(self):
        """Detects no transfer even with long training (falsification)."""
        tracker = FailureTracker("exp2")
        config = MockConfig(n_training_cycles=60)

        result = MockResult(holdout_professor_correlation=0.1)

        tracker.track_run("holdout_one", 0, 42, result, config)

        report = tracker.get_report()
        falsification_failures = [
            f for f in report.failures if f.falsifies_theory
        ]
        assert len(falsification_failures) == 1
        assert falsification_failures[0].category == FailureCategory.NO_TRANSFER_TO_HOLDOUT

    def test_analyze_variance(self):
        """Variance analysis detects high run-to-run variance."""
        tracker = FailureTracker("exp1")

        # Create results with high variance
        results = [
            MockResult(final_conscience_score=0.3),
            MockResult(final_conscience_score=0.7),
            MockResult(final_conscience_score=0.2),
            MockResult(final_conscience_score=0.8),
            MockResult(final_conscience_score=0.4),
        ]

        tracker.analyze_variance({"full_aspire": results})

        report = tracker.get_report()
        variance_failures = [
            f for f in report.failures
            if f.category == FailureCategory.SEED_VARIANCE
        ]
        assert len(variance_failures) >= 1

    def test_analyze_comparison_random_vs_full(self):
        """Comparison analysis detects RANDOM matching FULL."""
        tracker = FailureTracker("exp1")

        # Full results with tight variance (std ≈ 0.05)
        full_results = [
            MockResult(final_conscience_score=0.65),
            MockResult(final_conscience_score=0.70),
            MockResult(final_conscience_score=0.68),
        ]

        # Random results that actually match (mean=0.67 is within std of full mean=0.677)
        random_results = [
            MockResult(final_conscience_score=0.66),  # Very close to full!
            MockResult(final_conscience_score=0.68),
            MockResult(final_conscience_score=0.67),
        ]

        tracker.analyze_comparison(full_results, random_results, "random_vs_full")

        report = tracker.get_report()
        falsification_failures = [
            f for f in report.failures
            if f.category == FailureCategory.RANDOM_MATCHES_FULL
        ]
        assert len(falsification_failures) == 1
        assert falsification_failures[0].falsifies_theory is True

    def test_identify_boundary_conditions(self):
        """Boundary conditions are identified from config."""
        tracker = FailureTracker("exp1")
        config = MockConfig(
            n_training_cycles=20,  # Too short
            n_training_items=30,   # Too small
            n_runs_per_condition=3,  # Too few
        )

        tracker.identify_boundary_conditions(config)

        report = tracker.get_report()
        assert len(report.boundary_conditions) >= 2

    def test_generate_negative_results_section(self):
        """Generates negative results markdown text."""
        tracker = FailureTracker("exp1")
        tracker.report.total_runs = 15

        # Add some failures
        tracker.report.add_failure(FailureCase(
            category=FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            experiment="exp1",
            condition="scalar",
            run_idx=0,
            seed=42,
            description="Scalar had partial conscience",
            expected=True,
            observed_value=0.35,
            expected_range=(0.0, 0.3),
            training_cycles=50,
            sample_size=100,
            interpretation="SCALAR averaging doesn't fully destroy signal",
        ))

        text = tracker.generate_negative_results_section()

        assert "When Conscience Does Not Form" in text
        assert "15 runs" in text
        assert "Expected Failure Patterns" in text

    def test_save_report(self):
        """Report saves to JSON correctly."""
        tracker = FailureTracker("test")
        tracker.report.total_runs = 5

        tracker.report.add_failure(FailureCase(
            category=FailureCategory.SEED_VARIANCE,
            experiment="test",
            condition="full",
            run_idx=0,
            seed=42,
            description="High variance",
            expected=True,
            observed_value=0.4,
            expected_range=(0.0, 0.3),
            training_cycles=50,
            sample_size=100,
        ))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        tracker.save_report(path)

        import json
        with open(path) as f:
            data = json.load(f)

        assert data["experiment"] == "test"
        assert data["total_runs"] == 5
        assert len(data["failures"]) == 1

        path.unlink()


class TestFailureAtlasData:
    """Test failure atlas data generation."""

    def test_generate_atlas_data(self):
        """Generates atlas visualization data."""
        tracker = FailureTracker("exp1")
        tracker.report.total_runs = 20

        # Add failures in different quadrants
        tracker.report.add_failure(FailureCase(
            category=FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            experiment="exp1",
            condition="scalar",
            run_idx=0,
            seed=42,
            description="Expected recoverable",
            expected=True,
            observed_value=0.35,
            expected_range=(0.0, 0.3),
            training_cycles=50,
            sample_size=100,
        ))

        tracker.report.add_failure(FailureCase(
            category=FailureCategory.RANDOM_MATCHES_FULL,
            experiment="exp1",
            condition="random",
            run_idx=1,
            seed=43,
            description="Unexpected critical",
            expected=False,
            observed_value=0.7,
            expected_range=(0.0, 0.5),
            training_cycles=50,
            sample_size=100,
            falsifies_theory=True,
        ))

        atlas_data = generate_failure_atlas_data(tracker)

        assert atlas_data["total_runs"] == 20
        assert "quadrants" in atlas_data
        assert len(atlas_data["quadrants"]["expected_recoverable"]) >= 1
        assert len(atlas_data["quadrants"]["unexpected_critical"]) >= 1

    def test_atlas_data_has_timeline(self):
        """Atlas data includes failure timeline."""
        tracker = FailureTracker("exp1")
        tracker.report.total_runs = 10

        for i in range(3):
            tracker.report.add_failure(FailureCase(
                category=FailureCategory.SEED_VARIANCE,
                experiment="exp1",
                condition="full",
                run_idx=i,
                seed=i,
                description=f"Variance {i}",
                expected=True,
                observed_value=0.4,
                expected_range=(0.0, 0.3),
                training_cycles=50,
                sample_size=100,
            ))

        atlas_data = generate_failure_atlas_data(tracker)

        assert len(atlas_data["failure_timeline"]) == 3
        assert atlas_data["failure_timeline"][0]["run_idx"] == 0


class TestFailureCategories:
    """Test failure category classifications."""

    def test_expected_categories(self):
        """Verify expected failure categories exist."""
        expected = [
            FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            FailureCategory.EARLY_HOLDOUT_FAILURE,
            FailureCategory.SLOWROLL_EVASION,
            FailureCategory.SEED_VARIANCE,
        ]

        for cat in expected:
            assert cat.value is not None

    def test_unexpected_categories(self):
        """Verify unexpected (falsification) categories exist."""
        unexpected = [
            FailureCategory.RANDOM_MATCHES_FULL,
            FailureCategory.HONEST_LOSES_TO_ADVERSARIAL,
            FailureCategory.NO_TRANSFER_TO_HOLDOUT,
        ]

        for cat in unexpected:
            assert cat.value is not None

    def test_boundary_categories(self):
        """Verify boundary condition categories exist."""
        boundary = [
            FailureCategory.INSUFFICIENT_TRAINING,
            FailureCategory.SAMPLE_SIZE_TOO_SMALL,
            FailureCategory.HIGH_VARIANCE_INCONCLUSIVE,
        ]

        for cat in boundary:
            assert cat.value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
