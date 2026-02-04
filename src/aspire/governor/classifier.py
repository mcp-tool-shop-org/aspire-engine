"""Failure classification for ASPIRE Governor."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from .config import GovernorConfig


class FailureClassification(Enum):
    """Classification of why an inference operation failed."""
    SUCCESS = "success"
    OOM = "oom"                        # Out of memory
    LIKELY_OOM = "likely_oom"          # Probably OOM based on evidence
    PAGING_DEATH = "paging_death"      # Severe memory pressure
    INFERENCE_ERROR = "inference_error"  # Model/code error, not resource
    TIMEOUT = "timeout"                # Operation took too long
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of failure classification."""
    classification: FailureClassification
    confidence: float                  # 0.0 - 1.0
    message: Optional[str] = None
    should_retry: bool = False
    retry_with_tokens: Optional[int] = None
    reasons: List[str] = field(default_factory=list)


class FailureClassifier:
    """Classify inference failures to enable intelligent retries.

    Uses evidence-based scoring to determine if a failure was:
    - OOM (out of memory) → retry with fewer tokens
    - Paging death (memory thrashing) → reduce parallelism
    - Actual inference error → don't retry
    """

    def __init__(self, config: GovernorConfig):
        self.config = config

    def classify(
        self,
        exit_code: int,
        memory_ratio_at_acquire: float,
        memory_ratio_at_exit: float,
        peak_memory_mb: int = 0,
        duration_ms: float = 0,
        stderr_text: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify a failure based on evidence."""
        if exit_code == 0:
            return ClassificationResult(
                classification=FailureClassification.SUCCESS,
                confidence=1.0,
            )

        evidence_score = 0.0
        reasons = []

        # Evidence: High memory at exit
        if memory_ratio_at_exit >= self.config.hard_stop_ratio:
            evidence_score += 0.4
            reasons.append(f"Memory at exit: {memory_ratio_at_exit:.0%} (critical)")
        elif memory_ratio_at_exit >= self.config.soft_stop_ratio:
            evidence_score += 0.25
            reasons.append(f"Memory at exit: {memory_ratio_at_exit:.0%} (high)")

        # Evidence: Memory increased significantly during operation
        memory_delta = memory_ratio_at_exit - memory_ratio_at_acquire
        if memory_delta > 0.15:
            evidence_score += 0.2
            reasons.append(f"Memory increased by {memory_delta:.0%} during operation")

        # Evidence: Peak memory exceeded threshold
        peak_gb = peak_memory_mb / 1024
        if peak_gb >= self.config.oom_peak_gb_threshold:
            evidence_score += 0.3
            reasons.append(f"Peak memory: {peak_gb:.1f}GB (exceeded threshold)")
        elif peak_gb >= self.config.oom_peak_gb_threshold * 0.8:
            evidence_score += 0.15
            reasons.append(f"Peak memory: {peak_gb:.1f}GB (near threshold)")

        # Evidence: Very short duration (crashed early)
        if duration_ms > 0 and duration_ms < 1000 and memory_ratio_at_exit > 0.7:
            evidence_score += 0.15
            reasons.append("Very short duration with high memory")

        # Evidence: Stderr contains OOM indicators
        if stderr_text:
            oom_keywords = [
                "out of memory",
                "oom",
                "cuda error",
                "memory allocation",
                "alloc failed",
                "cudnn",
                "cublas",
            ]
            stderr_lower = stderr_text.lower()
            for kw in oom_keywords:
                if kw in stderr_lower:
                    evidence_score += 0.3
                    reasons.append(f"Stderr contains '{kw}'")
                    break

            # Counter-evidence: Looks like actual code error
            code_error_keywords = [
                "traceback",
                "assertion",
                "valueerror",
                "typeerror",
                "keyerror",
                "indexerror",
            ]
            for kw in code_error_keywords:
                if kw in stderr_lower:
                    evidence_score -= 0.2
                    reasons.append(f"Stderr suggests code error ('{kw}')")

        # Classify based on evidence
        evidence_score = max(0.0, min(1.0, evidence_score))

        if evidence_score >= 0.6:
            classification = FailureClassification.OOM
            should_retry = True
            retry_tokens = 1  # Minimum
            message = self._format_oom_message(
                memory_ratio_at_exit, peak_gb, reasons
            )
        elif evidence_score >= 0.4:
            classification = FailureClassification.LIKELY_OOM
            should_retry = True
            retry_tokens = 1
            message = self._format_likely_oom_message(
                memory_ratio_at_exit, evidence_score, reasons
            )
        elif evidence_score >= 0.25:
            classification = FailureClassification.PAGING_DEATH
            should_retry = True
            retry_tokens = 1
            message = "Memory pressure detected. Reducing parallelism recommended."
        else:
            classification = FailureClassification.INFERENCE_ERROR
            should_retry = False
            retry_tokens = None
            message = "Failure appears to be an inference/code error, not resource exhaustion."

        return ClassificationResult(
            classification=classification,
            confidence=evidence_score,
            message=message,
            should_retry=should_retry,
            retry_with_tokens=retry_tokens,
            reasons=reasons,
        )

    def _format_oom_message(
        self,
        memory_ratio: float,
        peak_gb: float,
        reasons: List[str],
    ) -> str:
        """Format a detailed OOM failure message."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║  INFERENCE FAILED: Out of Memory                             ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  GPU memory: {memory_ratio:.0%}                                              ║",
            f"║  Peak usage: {peak_gb:.1f} GB                                            ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Recommendation: Retry with reduced batch size               ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def _format_likely_oom_message(
        self,
        memory_ratio: float,
        confidence: float,
        reasons: List[str],
    ) -> str:
        """Format a likely-OOM failure message."""
        reason_text = "; ".join(reasons[:3])
        return (
            f"Likely OOM (confidence: {confidence:.0%}). "
            f"Memory at {memory_ratio:.0%}. "
            f"Evidence: {reason_text}"
        )
