"""ASPIRE training engine."""

from .loop import AspireEngine, CycleResult, TrainingMetrics
from .revision import (
    RevisionEngine,
    RevisionConfig,
    RevisionDecision,
    RevisionResult,
    RevisionTrigger,
)
from .revision_engine import RevisionAspireEngine, RevisionCycleResult, RevisionMetrics

__all__ = [
    "AspireEngine",
    "CycleResult",
    "TrainingMetrics",
    "RevisionEngine",
    "RevisionConfig",
    "RevisionDecision",
    "RevisionResult",
    "RevisionTrigger",
    "RevisionAspireEngine",
    "RevisionCycleResult",
    "RevisionMetrics",
]

# Re-export governor for convenience
from ..governor import TokenPool, GovernorConfig
