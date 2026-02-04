"""Conscience metrics and validation for ASPIRE.

This module operationalizes the "conscience" concept with falsifiable metrics.
A conscience is not a metaphor - it's a measurable property of the trained system.

OPERATIONAL DEFINITION:
A system has internalized conscience when it exhibits:
1. Low surprise variance (stable predictive alignment)
2. Stable anisotropy across tasks (consistent evaluation structure)
3. Cross-professor generalization (not just pleasing specific evaluators)
4. Geometric stability (dimensional structure persists under perturbation)

FAILURE MODES (equally important):
1. Heuristic collapse: dimensional collapse too fast = surface learning
2. Professor pleasing: high accuracy but no generalization
3. Feature gaming: text hedging without logit uncertainty
4. Geometric instability: structure doesn't persist across domains

This module provides tools to measure and validate conscience formation.

Literature Foundation:
- Surprise/Prediction Error: Based on actor-critic RL and RPE research
- Dimensional Collapse: Neural collapse and SSL dimensional collapse prevention
- Phase Transitions: Grokking phenomenon and complexity phase transitions
- Ensemble Uncertainty: Deep ensembles for calibrated uncertainty
- Self-Correction: SCoRe and CRITIC frameworks for LLM revision
- Intrinsic Motivation: Motif and moral alignment research

See docs/LITERATURE_REVIEW.md for complete citations and theoretical grounding.
"""

from .metrics import (
    ConscienceMetrics,
    ConscienceScore,
    compute_conscience_score,
    SurpriseStability,
    AnisotropyStability,
    GeneralizationScore,
)
from .validation import (
    ConscienceValidator,
    ValidationResult,
    FailureMode,
    detect_failure_modes,
)
from .ablation import (
    AblationConfig,
    AblationRunner,
    AblationResult,
    compare_ablations,
)
from .leakage import (
    FeatureLeakageDetector,
    LeakageReport,
    LeakageIndicator,
    detect_early_collapse,
    detect_velocity_anomaly,
    detect_curvature_anomaly,
    detect_text_token_shortcut,
)
from .calibration import (
    CalibrationProfile,
    CalibratedResult,
    NullDistribution,
    ConfidenceLevel,
    ConscienceLevel,
    calibrated_evaluation,
)

__all__ = [
    # Metrics
    "ConscienceMetrics",
    "ConscienceScore",
    "compute_conscience_score",
    "SurpriseStability",
    "AnisotropyStability",
    "GeneralizationScore",
    # Validation
    "ConscienceValidator",
    "ValidationResult",
    "FailureMode",
    "detect_failure_modes",
    # Ablation
    "AblationConfig",
    "AblationRunner",
    "AblationResult",
    "compare_ablations",
    # Leakage detection
    "FeatureLeakageDetector",
    "LeakageReport",
    "LeakageIndicator",
    "detect_early_collapse",
    "detect_velocity_anomaly",
    "detect_curvature_anomaly",
    "detect_text_token_shortcut",
    # Calibration
    "CalibrationProfile",
    "CalibratedResult",
    "NullDistribution",
    "ConfidenceLevel",
    "ConscienceLevel",
    "calibrated_evaluation",
]
