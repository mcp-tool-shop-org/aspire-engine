"""Failure case tracking and negative results reporting.

Per scientific rigor, failures are first-class results. This module provides
infrastructure to:
1. Track and categorize failure cases
2. Report negative results systematically
3. Document boundary conditions of conscience formation
4. Generate "failure atlas" visualizations

Expected failure cases (to be tracked, not hidden):
    A. SCALAR_REWARD achieving partial ConscienceScore (if real judgment leaks through)
    B. Holdout professor failing in early training runs
    C. SlowRollDeceiver evading detection in short windows
    D. Adversarial student "winning" if defense is misconfigured
    E. High run-to-run variance under small seeds
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import numpy as np
from datetime import datetime


class FailureCategory(Enum):
    """Categories of expected and unexpected failures."""

    # Expected failures (theory predicts these)
    SCALAR_PARTIAL_CONSCIENCE = "scalar_partial_conscience"
    EARLY_HOLDOUT_FAILURE = "early_holdout_failure"
    SLOWROLL_EVASION = "slowroll_evasion"
    DEFENSE_MISCONFIGURATION = "defense_misconfiguration"
    SEED_VARIANCE = "seed_variance"

    # Unexpected failures (may falsify theory)
    RANDOM_MATCHES_FULL = "random_matches_full"
    HONEST_LOSES_TO_ADVERSARIAL = "honest_loses_to_adversarial"
    NO_TRANSFER_TO_HOLDOUT = "no_transfer_to_holdout"

    # Boundary conditions (neither support nor falsify)
    INSUFFICIENT_TRAINING = "insufficient_training"
    SAMPLE_SIZE_TOO_SMALL = "sample_size_too_small"
    HIGH_VARIANCE_INCONCLUSIVE = "high_variance_inconclusive"


@dataclass
class FailureCase:
    """A single tracked failure case."""

    category: FailureCategory
    experiment: str
    condition: str
    run_idx: int
    seed: int

    # What happened
    description: str
    expected: bool  # Was this failure expected by theory?

    # Quantitative details
    observed_value: float
    expected_range: Tuple[float, float]

    # Context
    training_cycles: int
    sample_size: int

    # Interpretation
    interpretation: str = ""
    falsifies_theory: bool = False

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category": self.category.value,
            "experiment": self.experiment,
            "condition": self.condition,
            "run_idx": self.run_idx,
            "seed": self.seed,
            "description": self.description,
            "expected": self.expected,
            "observed_value": self.observed_value,
            "expected_range": list(self.expected_range),
            "training_cycles": self.training_cycles,
            "sample_size": self.sample_size,
            "interpretation": self.interpretation,
            "falsifies_theory": self.falsifies_theory,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FailureReport:
    """Aggregated failure report across experiments."""

    experiment_name: str
    total_runs: int

    # Tracked failures by category
    failures: List[FailureCase] = field(default_factory=list)
    failures_by_category: Dict[str, List[FailureCase]] = field(default_factory=dict)

    # Summary statistics
    expected_failure_count: int = 0
    unexpected_failure_count: int = 0
    falsification_count: int = 0

    # Boundary conditions identified
    boundary_conditions: List[str] = field(default_factory=list)

    def add_failure(self, failure: FailureCase):
        """Add a failure case to the report."""
        self.failures.append(failure)

        category_key = failure.category.value
        if category_key not in self.failures_by_category:
            self.failures_by_category[category_key] = []
        self.failures_by_category[category_key].append(failure)

        if failure.expected:
            self.expected_failure_count += 1
        else:
            self.unexpected_failure_count += 1

        if failure.falsifies_theory:
            self.falsification_count += 1

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute failure statistics."""
        stats = {
            "total_failures": len(self.failures),
            "expected_failures": self.expected_failure_count,
            "unexpected_failures": self.unexpected_failure_count,
            "falsifications": self.falsification_count,
            "failure_rate": len(self.failures) / max(1, self.total_runs),
            "by_category": {},
        }

        for category, cases in self.failures_by_category.items():
            stats["by_category"][category] = {
                "count": len(cases),
                "rate": len(cases) / max(1, self.total_runs),
                "expected_count": sum(1 for c in cases if c.expected),
            }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "experiment": self.experiment_name,
            "total_runs": self.total_runs,
            "statistics": self.compute_statistics(),
            "failures": [f.to_dict() for f in self.failures],
            "boundary_conditions": self.boundary_conditions,
        }


class FailureTracker:
    """Tracks failure cases across experiment runs."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.report = FailureReport(experiment_name=experiment_name, total_runs=0)

    def track_run(
        self,
        condition: str,
        run_idx: int,
        seed: int,
        result: Any,
        config: Any,
    ):
        """Track failures from a single run."""
        self.report.total_runs += 1

        # Check for expected failure patterns
        self._check_scalar_partial_conscience(condition, run_idx, seed, result, config)
        self._check_early_holdout_failure(condition, run_idx, seed, result, config)
        self._check_slowroll_evasion(condition, run_idx, seed, result, config)
        self._check_defense_issues(condition, run_idx, seed, result, config)
        self._check_seed_variance(condition, run_idx, seed, result, config)

        # Check for unexpected failures (potential falsifications)
        self._check_random_matches_full(condition, run_idx, seed, result, config)
        self._check_honest_vs_adversarial(condition, run_idx, seed, result, config)
        self._check_holdout_transfer(condition, run_idx, seed, result, config)

    def _check_scalar_partial_conscience(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check for SCALAR_REWARD achieving partial conscience (expected)."""
        if condition != "scalar_reward":
            return

        conscience = getattr(result, "final_conscience_score", 0)

        # Expected: SCALAR should have LOW conscience (< 0.4)
        # But if real judgment leaks through professor aggregation, may be higher
        if conscience > 0.3:
            self.report.add_failure(FailureCase(
                category=FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
                experiment=self.experiment_name,
                condition=condition,
                run_idx=run_idx,
                seed=seed,
                description=(
                    f"SCALAR_REWARD achieved conscience={conscience:.3f}, "
                    "higher than expected. Real judgment may leak through scalar collapse."
                ),
                expected=True,  # This is a known edge case
                observed_value=conscience,
                expected_range=(0.0, 0.3),
                training_cycles=getattr(config, "n_training_cycles", 0),
                sample_size=getattr(config, "n_training_items", 0),
                interpretation=(
                    "SCALAR averaging doesn't fully destroy multi-dimensional signal. "
                    "Document as boundary condition, not falsification."
                ),
            ))

    def _check_early_holdout_failure(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check for holdout professor failing in early training."""
        if condition != "holdout_one":
            return

        training_cycles = getattr(config, "n_training_cycles", 50)
        holdout_corr = getattr(result, "holdout_professor_correlation", None)
        transfer_ratio = getattr(result, "transfer_ratio", None)

        # Early runs (< 30 cycles) may fail to transfer
        if training_cycles < 30 and holdout_corr is not None:
            if holdout_corr < 0.3:
                self.report.add_failure(FailureCase(
                    category=FailureCategory.EARLY_HOLDOUT_FAILURE,
                    experiment=self.experiment_name,
                    condition=condition,
                    run_idx=run_idx,
                    seed=seed,
                    description=(
                        f"Holdout transfer failed (corr={holdout_corr:.3f}) with only "
                        f"{training_cycles} training cycles. Expected with short training."
                    ),
                    expected=True,
                    observed_value=holdout_corr,
                    expected_range=(0.3, 1.0),
                    training_cycles=training_cycles,
                    sample_size=getattr(config, "n_training_items", 0),
                    interpretation=(
                        "Transfer learning requires sufficient training. "
                        "Report minimum training threshold for reliable transfer."
                    ),
                ))

    def _check_slowroll_evasion(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check for SlowRollDeceiver evading detection."""
        student_type = getattr(result, "student_type", "")
        gaming_detected = getattr(result, "gaming_detected", False)
        training_cycles = getattr(config, "n_training_cycles", 50)

        if "slow_roll" in student_type.lower() and not gaming_detected:
            if training_cycles < 40:
                self.report.add_failure(FailureCase(
                    category=FailureCategory.SLOWROLL_EVASION,
                    experiment=self.experiment_name,
                    condition=condition,
                    run_idx=run_idx,
                    seed=seed,
                    description=(
                        f"SlowRollDeceiver evaded detection with {training_cycles} cycles. "
                        "This adversarial strategy is designed to evade short-window detection."
                    ),
                    expected=True,
                    observed_value=0.0,  # No detection
                    expected_range=(0.6, 1.0),  # Should detect
                    training_cycles=training_cycles,
                    sample_size=getattr(config, "n_training_items", 0),
                    interpretation=(
                        "SlowRoll exploits finite observation windows. "
                        "Document detection window requirements."
                    ),
                ))

    def _check_defense_issues(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check for defense misconfiguration."""
        if condition != "adversarial_with_defense":
            return

        gaming_detected = getattr(result, "gaming_detected", False)
        behavior_changed = getattr(result, "behavior_changed", False)
        conscience = getattr(result, "final_conscience_score", 0)

        # Defense should detect gaming and cause behavior change
        if gaming_detected and not behavior_changed and conscience > 0.5:
            self.report.add_failure(FailureCase(
                category=FailureCategory.DEFENSE_MISCONFIGURATION,
                experiment=self.experiment_name,
                condition=condition,
                run_idx=run_idx,
                seed=seed,
                description=(
                    "Gaming detected but no behavior change, yet high conscience score. "
                    "Defense may be too lenient or scoring is miscalibrated."
                ),
                expected=False,
                observed_value=conscience,
                expected_range=(0.0, 0.3),
                training_cycles=getattr(config, "n_training_cycles", 0),
                sample_size=getattr(config, "n_training_items", 0),
                interpretation=(
                    "Defense penalties may not propagate to conscience scoring. "
                    "Review penalty application logic."
                ),
            ))

    def _check_seed_variance(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Track high variance for seed sensitivity analysis."""
        # This is called per-run; aggregate analysis happens in finalize()
        pass  # Variance computed across runs, not per-run

    def _check_random_matches_full(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check if RANDOM_PROFESSORS matches FULL_ASPIRE (falsification!)."""
        # This is an aggregate check done in analyze_comparison()
        pass

    def _check_honest_vs_adversarial(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check if adversarial beats honest (potential falsification)."""
        # This is an aggregate check done in analyze_comparison()
        pass

    def _check_holdout_transfer(
        self, condition: str, run_idx: int, seed: int, result: Any, config: Any
    ):
        """Check for complete holdout transfer failure."""
        if condition != "holdout_one":
            return

        training_cycles = getattr(config, "n_training_cycles", 50)
        holdout_corr = getattr(result, "holdout_professor_correlation", None)

        # With sufficient training, holdout should transfer
        if training_cycles >= 50 and holdout_corr is not None and holdout_corr < 0.2:
            self.report.add_failure(FailureCase(
                category=FailureCategory.NO_TRANSFER_TO_HOLDOUT,
                experiment=self.experiment_name,
                condition=condition,
                run_idx=run_idx,
                seed=seed,
                description=(
                    f"No transfer to holdout professor (corr={holdout_corr:.3f}) "
                    f"despite {training_cycles} training cycles. "
                    "Theory predicts transfer should occur."
                ),
                expected=False,
                observed_value=holdout_corr,
                expected_range=(0.4, 1.0),
                training_cycles=training_cycles,
                sample_size=getattr(config, "n_training_items", 0),
                interpretation=(
                    "Conscience may not generalize to unseen professors. "
                    "This is a potential falsification if consistent."
                ),
                falsifies_theory=True,
            ))

    def analyze_comparison(
        self,
        full_results: List[Any],
        comparison_results: List[Any],
        comparison_type: str,
    ):
        """Analyze comparisons between conditions for falsification."""
        if not full_results or not comparison_results:
            return

        full_scores = [r.final_conscience_score for r in full_results]
        comp_scores = [r.final_conscience_score for r in comparison_results]

        full_mean = np.mean(full_scores)
        full_std = np.std(full_scores) if len(full_scores) > 1 else 0.1
        comp_mean = np.mean(comp_scores)

        if comparison_type == "random_vs_full":
            # RANDOM should NOT match FULL
            if comp_mean >= full_mean - full_std:
                self.report.add_failure(FailureCase(
                    category=FailureCategory.RANDOM_MATCHES_FULL,
                    experiment=self.experiment_name,
                    condition="random_professors",
                    run_idx=-1,  # Aggregate
                    seed=-1,
                    description=(
                        f"RANDOM_PROFESSORS (mean={comp_mean:.3f}) matches "
                        f"FULL_ASPIRE (mean={full_mean:.3f}±{full_std:.3f}). "
                        "Random evaluation should not achieve comparable conscience."
                    ),
                    expected=False,
                    observed_value=comp_mean,
                    expected_range=(0.0, full_mean - full_std),
                    training_cycles=0,
                    sample_size=len(comp_scores),
                    interpretation=(
                        "If random evaluation matches structured evaluation, "
                        "conscience formation may be an artifact of training dynamics, "
                        "not meaningful judgment learning."
                    ),
                    falsifies_theory=True,
                ))

        elif comparison_type == "adversarial_vs_honest":
            # Adversarial should NOT beat honest
            honest_scores = full_scores
            adversarial_scores = comp_scores

            honest_mean = np.mean(honest_scores)
            adversarial_mean = np.mean(adversarial_scores)

            if adversarial_mean > honest_mean:
                self.report.add_failure(FailureCase(
                    category=FailureCategory.HONEST_LOSES_TO_ADVERSARIAL,
                    experiment=self.experiment_name,
                    condition="adversarial_no_defense",
                    run_idx=-1,
                    seed=-1,
                    description=(
                        f"Adversarial student (mean={adversarial_mean:.3f}) "
                        f"outperforms honest student (mean={honest_mean:.3f}). "
                        "Honest behavior should be rewarded over gaming."
                    ),
                    expected=False,
                    observed_value=adversarial_mean,
                    expected_range=(0.0, honest_mean),
                    training_cycles=0,
                    sample_size=len(comp_scores),
                    interpretation=(
                        "If gaming strategies outperform honest behavior, "
                        "the conscience metric is gameable and may not reflect "
                        "genuine judgment development."
                    ),
                    falsifies_theory=True,
                ))

    def analyze_variance(self, results_by_condition: Dict[str, List[Any]]):
        """Analyze run-to-run variance for stability assessment."""
        for condition, results in results_by_condition.items():
            if len(results) < 3:
                continue

            scores = [r.final_conscience_score for r in results]
            variance = np.var(scores)
            std = np.std(scores)
            mean = np.mean(scores)
            cv = std / mean if mean > 0 else float('inf')  # Coefficient of variation

            # High variance (CV > 0.3) indicates instability
            if cv > 0.3:
                self.report.add_failure(FailureCase(
                    category=FailureCategory.SEED_VARIANCE,
                    experiment=self.experiment_name,
                    condition=condition,
                    run_idx=-1,
                    seed=-1,
                    description=(
                        f"High variance in {condition}: CV={cv:.2f}, std={std:.3f}, "
                        f"mean={mean:.3f}. Results may not be reliable."
                    ),
                    expected=True,  # Some variance is expected
                    observed_value=cv,
                    expected_range=(0.0, 0.3),
                    training_cycles=0,
                    sample_size=len(results),
                    interpretation=(
                        "High run-to-run variance suggests sensitivity to initialization. "
                        "Report confidence intervals and recommend larger n_runs."
                    ),
                ))

            # Also check if variance is too high to draw conclusions
            if cv > 0.5:
                if "High variance inconclusive" not in self.report.boundary_conditions:
                    self.report.boundary_conditions.append(
                        f"High variance (CV={cv:.2f}) in {condition} makes conclusions uncertain"
                    )

    def identify_boundary_conditions(self, config: Any):
        """Identify boundary conditions from configuration."""
        n_cycles = getattr(config, "n_training_cycles", 50)
        n_items = getattr(config, "n_training_items", 100)
        n_runs = getattr(config, "n_runs_per_condition", 5)

        if n_cycles < 30:
            self.report.boundary_conditions.append(
                f"Training duration ({n_cycles} cycles) may be insufficient for stable conscience formation"
            )

        if n_items < 50:
            self.report.boundary_conditions.append(
                f"Sample size ({n_items} items) may be too small for reliable evaluation"
            )

        if n_runs < 5:
            self.report.boundary_conditions.append(
                f"Number of runs ({n_runs}) may be too small for statistical significance"
            )

    def get_report(self) -> FailureReport:
        """Get the complete failure report."""
        return self.report

    def save_report(self, path: Path):
        """Save failure report to JSON."""
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)

    def generate_negative_results_section(self) -> str:
        """Generate text for the Negative Results section of a paper."""
        stats = self.report.compute_statistics()

        lines = [
            "## When Conscience Does Not Form",
            "",
            "We tracked failure cases systematically to document the boundary conditions",
            "of conscience formation. Not all failures indicate theoretical problems;",
            "expected failures validate our understanding of the mechanism's limits.",
            "",
        ]

        # Summary statistics
        lines.append(f"Across {self.report.total_runs} runs:")
        lines.append(f"- {stats['expected_failures']} expected failures (theory predicts these)")
        lines.append(f"- {stats['unexpected_failures']} unexpected failures")
        lines.append(f"- {stats['falsifications']} potential falsifications")
        lines.append("")

        # Expected failures (these are OK)
        expected_categories = [
            FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
            FailureCategory.EARLY_HOLDOUT_FAILURE,
            FailureCategory.SLOWROLL_EVASION,
            FailureCategory.SEED_VARIANCE,
        ]

        lines.append("### Expected Failure Patterns")
        lines.append("")
        for cat in expected_categories:
            cases = self.report.failures_by_category.get(cat.value, [])
            if cases:
                lines.append(f"**{cat.value}** ({len(cases)} cases)")
                lines.append(f"> {cases[0].interpretation}")
                lines.append("")

        # Unexpected failures (potential problems)
        unexpected_categories = [
            FailureCategory.RANDOM_MATCHES_FULL,
            FailureCategory.HONEST_LOSES_TO_ADVERSARIAL,
            FailureCategory.NO_TRANSFER_TO_HOLDOUT,
        ]

        unexpected_found = False
        for cat in unexpected_categories:
            cases = self.report.failures_by_category.get(cat.value, [])
            if cases:
                if not unexpected_found:
                    lines.append("### Unexpected Failures (Potential Falsifications)")
                    lines.append("")
                    unexpected_found = True

                lines.append(f"**{cat.value}** ({len(cases)} cases)")
                lines.append(f"> {cases[0].description}")
                lines.append(f"> Interpretation: {cases[0].interpretation}")
                lines.append("")

        if not unexpected_found:
            lines.append("### Unexpected Failures")
            lines.append("No unexpected failures observed in this experimental run.")
            lines.append("")

        # Boundary conditions
        if self.report.boundary_conditions:
            lines.append("### Boundary Conditions")
            lines.append("")
            for bc in self.report.boundary_conditions:
                lines.append(f"- {bc}")
            lines.append("")

        return "\n".join(lines)


def generate_failure_atlas_data(tracker: FailureTracker) -> Dict[str, Any]:
    """Generate data for a failure atlas visualization.

    Returns structured data for creating a visual summary of all failure modes.
    """
    report = tracker.get_report()
    stats = report.compute_statistics()

    # Organize failures by severity and type
    atlas_data = {
        "title": f"Failure Atlas: {report.experiment_name}",
        "total_runs": report.total_runs,
        "overall_failure_rate": stats["failure_rate"],

        # Quadrant data (for 2x2 visualization)
        "quadrants": {
            "expected_recoverable": [],  # Expected and doesn't hurt conclusions
            "expected_limiting": [],     # Expected but limits what we can claim
            "unexpected_minor": [],      # Unexpected but doesn't falsify
            "unexpected_critical": [],   # Unexpected and potentially falsifying
        },

        # Timeline data (for temporal visualization)
        "failure_timeline": [],

        # Category breakdown
        "by_category": {},
    }

    for failure in report.failures:
        # Categorize into quadrants
        if failure.expected:
            if failure.category in [
                FailureCategory.SCALAR_PARTIAL_CONSCIENCE,
                FailureCategory.EARLY_HOLDOUT_FAILURE,
            ]:
                quadrant = "expected_recoverable"
            else:
                quadrant = "expected_limiting"
        else:
            if failure.falsifies_theory:
                quadrant = "unexpected_critical"
            else:
                quadrant = "unexpected_minor"

        atlas_data["quadrants"][quadrant].append({
            "category": failure.category.value,
            "condition": failure.condition,
            "value": failure.observed_value,
            "description": failure.description[:100],
        })

        # Timeline entry
        atlas_data["failure_timeline"].append({
            "run_idx": failure.run_idx,
            "category": failure.category.value,
            "severity": "critical" if failure.falsifies_theory else (
                "expected" if failure.expected else "unexpected"
            ),
        })

    # Category summary
    for cat, cases in report.failures_by_category.items():
        atlas_data["by_category"][cat] = {
            "count": len(cases),
            "expected_rate": sum(1 for c in cases if c.expected) / len(cases) if cases else 0,
            "mean_value": np.mean([c.observed_value for c in cases]) if cases else 0,
        }

    return atlas_data
