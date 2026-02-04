"""ASPIRE Falsification Experiments.

This module implements the three core experiments that validate or falsify
ASPIRE's conscience formation claims:

Experiment 1: Null vs Structured Judgment
    - Does conscience emerge only under meaningful evaluation?
    - Compares FULL, SCALAR_REWARD, RANDOM_PROFESSORS conditions
    - Falsifies if: RANDOM achieves comparable ConscienceScore to FULL

Experiment 2: Holdout Judgment Transfer
    - Is the student learning how to judge, or just who trained it?
    - Trains without one professor, tests generalization
    - Falsifies if: Holdout correlation ≈ 0

Experiment 3: Adversarial Pressure
    - Does ASPIRE resist actively deceptive students?
    - Compares honest vs adversarial students
    - Falsifies if: Adversarial student achieves high scores without behavior change

Together, these form a publication-grade falsification suite.
"""

from .experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    Condition,
)
from .experiment1_null_vs_structured import (
    NullVsStructuredExperiment,
    NullVsStructuredResult,
)
from .experiment2_holdout_transfer import (
    HoldoutTransferExperiment,
    HoldoutTransferResult,
)
from .experiment3_adversarial import (
    AdversarialPressureExperiment,
    AdversarialPressureResult,
)
from .figures import (
    FigureGenerator,
    generate_all_figures,
    generate_failure_atlas,
    generate_boundary_conditions_figure,
)
from .failure_tracking import (
    FailureTracker,
    FailureReport,
    FailureCase,
    FailureCategory,
    generate_failure_atlas_data,
)

__all__ = [
    # Core runner
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "Condition",
    # Experiment 1
    "NullVsStructuredExperiment",
    "NullVsStructuredResult",
    # Experiment 2
    "HoldoutTransferExperiment",
    "HoldoutTransferResult",
    # Experiment 3
    "AdversarialPressureExperiment",
    "AdversarialPressureResult",
    # Figures
    "FigureGenerator",
    "generate_all_figures",
    "generate_failure_atlas",
    "generate_boundary_conditions_figure",
    # Failure tracking
    "FailureTracker",
    "FailureReport",
    "FailureCase",
    "FailureCategory",
    "generate_failure_atlas_data",
]
