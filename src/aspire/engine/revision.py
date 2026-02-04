"""Revision behavior for ASPIRE - the self-correction loop.

This is where internalization becomes behavior. Instead of just
predicting and grading, the student now:

1. Produces draft response
2. Receives critique + tokens
3. Critic computes misalignment
4. If misalignment high → student revises
5. Revision re-evaluated

The revision loop teaches the student to:
- Notice when reasoning is weak
- Repair under pressure
- Calibrate hedging when disagreement is high
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from ..core import (
    TrainingItem,
    StudentResponse,
    TokenVector,
    TokenDimension,
    EnsembleEvaluation,
)
from ..critic import CriticPrediction, MisalignmentSignal


class RevisionTrigger(Enum):
    """Why a revision was triggered."""
    NONE = "none"                          # No revision needed
    HIGH_NEGATIVE_SURPRISE = "negative_surprise"  # Predicted high, got low
    HIGH_DISAGREEMENT = "disagreement"     # Professors disagreed significantly
    OVERCONFIDENCE = "overconfidence"      # Student confident but wrong/uncertain
    LOW_TOKENS = "low_tokens"              # Total tokens below threshold
    EXPLICIT_REQUEST = "explicit"          # Forced revision (for testing)


@dataclass
class RevisionDecision:
    """Decision about whether to revise."""
    should_revise: bool
    triggers: List[RevisionTrigger]
    confidence: float                      # How confident are we revision will help
    reason: str                            # Human-readable explanation


@dataclass
class RevisionConfig:
    """Configuration for revision behavior."""
    enabled: bool = True

    # Trigger thresholds
    negative_surprise_threshold: float = 0.4    # Aggregate negative surprise
    disagreement_threshold: float = 0.35        # Professor disagreement score
    overconfidence_gap_threshold: float = 0.3   # |confidence - actual_correctness|
    low_token_threshold: float = 0.4            # Total tokens below this

    # Revision behavior
    max_revisions_per_item: int = 1             # Limit revision attempts
    include_gold_in_revision: bool = False      # Teaching mode vs self-correction
    include_critique_in_revision: bool = True   # Show professor feedback
    include_prediction_flags: bool = True       # Show critic warnings

    # Prompt constraints
    require_change_summary: bool = True         # Require "Changes:" section
    max_revision_tokens: int = 300              # Allow longer for revision


@dataclass
class RevisionResult:
    """Result of a revision attempt."""
    original_response: StudentResponse
    revised_response: StudentResponse
    triggers: List[RevisionTrigger]

    # Token changes
    original_tokens: TokenVector
    revised_tokens: TokenVector

    # Metrics
    token_uplift: float                    # Total improvement
    uplift_by_dim: dict                    # Per-dimension improvement
    revision_time_ms: float


class RevisionEngine:
    """Manages the revision decision and prompt construction.

    The revision engine encapsulates the "internal voice" logic:
    - When to revise (based on critic signals)
    - How to prompt for revision (what to show the student)
    - How to measure improvement
    """

    def __init__(self, config: Optional[RevisionConfig] = None):
        self.config = config or RevisionConfig()

    def should_revise(
        self,
        prediction: CriticPrediction,
        misalignment: MisalignmentSignal,
        evaluation: EnsembleEvaluation,
        response: StudentResponse,
    ) -> RevisionDecision:
        """Decide whether the student should revise.

        Uses multiple signals to determine if revision would help:
        - Critic's negative surprise (predicted better than actual)
        - Professor disagreement (ambiguous case)
        - Student overconfidence (high confidence but wrong)
        - Low absolute tokens (just did poorly)
        """
        if not self.config.enabled:
            return RevisionDecision(
                should_revise=False,
                triggers=[RevisionTrigger.NONE],
                confidence=0.0,
                reason="Revision disabled",
            )

        triggers = []
        confidence_sum = 0.0

        # Check negative surprise
        # (predicted tokens high but got low = student didn't meet expectations)
        negative_surprise = 0.0
        for dim in TokenDimension:
            pred = prediction.expected_tokens.values[dim]
            actual = evaluation.aggregated_tokens.values[dim]
            if pred > actual:
                negative_surprise += (pred - actual)

        if negative_surprise > self.config.negative_surprise_threshold:
            triggers.append(RevisionTrigger.HIGH_NEGATIVE_SURPRISE)
            confidence_sum += 0.4

        # Check disagreement
        if evaluation.disagreement_score > self.config.disagreement_threshold:
            triggers.append(RevisionTrigger.HIGH_DISAGREEMENT)
            confidence_sum += 0.3

        # Check overconfidence
        # Student was confident but professors didn't agree it was correct
        if response.confidence > 0.7 and not evaluation.consensus_correct:
            confidence_gap = response.confidence - (1.0 if evaluation.consensus_correct else 0.0)
            if abs(confidence_gap) > self.config.overconfidence_gap_threshold:
                triggers.append(RevisionTrigger.OVERCONFIDENCE)
                confidence_sum += 0.3

        # Check low tokens overall
        total_tokens = evaluation.aggregated_tokens.total / len(TokenDimension)
        if total_tokens < self.config.low_token_threshold:
            triggers.append(RevisionTrigger.LOW_TOKENS)
            confidence_sum += 0.2

        should = len(triggers) > 0
        confidence = min(1.0, confidence_sum)

        # Build reason string
        if not triggers:
            reason = "Response acceptable"
        else:
            reasons = []
            if RevisionTrigger.HIGH_NEGATIVE_SURPRISE in triggers:
                reasons.append(f"negative surprise ({negative_surprise:.2f})")
            if RevisionTrigger.HIGH_DISAGREEMENT in triggers:
                reasons.append(f"high disagreement ({evaluation.disagreement_score:.2f})")
            if RevisionTrigger.OVERCONFIDENCE in triggers:
                reasons.append("overconfident but wrong")
            if RevisionTrigger.LOW_TOKENS in triggers:
                reasons.append(f"low tokens ({total_tokens:.2f})")
            reason = "Revise due to: " + ", ".join(reasons)

        return RevisionDecision(
            should_revise=should,
            triggers=triggers if triggers else [RevisionTrigger.NONE],
            confidence=confidence,
            reason=reason,
        )

    def build_revision_prompt(
        self,
        item: TrainingItem,
        draft_response: StudentResponse,
        evaluation: EnsembleEvaluation,
        prediction: CriticPrediction,
        decision: RevisionDecision,
    ) -> str:
        """Build the prompt for revision.

        The revision prompt includes:
        - Original question
        - Student's draft response
        - Critique feedback (if enabled)
        - Prediction flags / warnings (if enabled)
        - Gold answer (if teaching mode)
        - Clear instructions for revision format
        """
        parts = []

        # Context
        parts.append("You are revising your previous response based on feedback.")
        parts.append("")

        # Original question
        parts.append("ORIGINAL QUESTION:")
        parts.append(item.prompt)
        parts.append("")

        # Student's draft
        parts.append("YOUR DRAFT RESPONSE:")
        parts.append(f"REASONING: {draft_response.reasoning_trace}")
        parts.append(f"CONFIDENCE: {self._confidence_to_word(draft_response.confidence)}")
        parts.append(f"ANSWER: {draft_response.answer}")
        parts.append("")

        # Critique feedback
        if self.config.include_critique_in_revision and evaluation.critiques:
            parts.append("FEEDBACK FROM REVIEWERS:")
            for critique in evaluation.critiques:
                verdict = "✓" if critique.is_correct else "✗"
                parts.append(f"- [{critique.professor_id}] {verdict}: {critique.critique_text}")
                if critique.specific_weaknesses:
                    for weakness in critique.specific_weaknesses[:2]:
                        parts.append(f"  • {weakness}")
            parts.append("")

        # Prediction flags (critic's warnings)
        if self.config.include_prediction_flags:
            warnings = []
            if RevisionTrigger.HIGH_DISAGREEMENT in decision.triggers:
                warnings.append("⚠ Reviewers disagreed significantly - consider acknowledging multiple perspectives")
            if RevisionTrigger.OVERCONFIDENCE in decision.triggers:
                warnings.append("⚠ Your confidence was high but reviewers weren't convinced - consider hedging")
            if RevisionTrigger.HIGH_NEGATIVE_SURPRISE in decision.triggers:
                warnings.append("⚠ Response didn't meet expected quality - strengthen reasoning")

            if warnings:
                parts.append("WARNINGS:")
                for w in warnings:
                    parts.append(w)
                parts.append("")

        # Gold answer (teaching mode)
        if self.config.include_gold_in_revision:
            parts.append("CORRECT APPROACH:")
            parts.append(f"Answer: {item.gold_answer}")
            parts.append(f"Rationale: {item.gold_rationale}")
            parts.append("")

        # Revision instructions
        parts.append("REVISION INSTRUCTIONS:")
        parts.append("1. Consider the feedback carefully")
        parts.append("2. Fix identified weaknesses")
        parts.append("3. Adjust confidence if appropriate")
        parts.append("4. Keep the same format")
        parts.append("")

        if self.config.require_change_summary:
            parts.append("YOUR REVISED RESPONSE (start with CHANGES: listing what you fixed):")
        else:
            parts.append("YOUR REVISED RESPONSE:")

        parts.append("CHANGES: <list 1-2 key changes>")
        parts.append("REASONING: <your revised step-by-step thinking>")
        parts.append("CONFIDENCE: <low/medium/high>")
        parts.append("ANSWER: <your revised answer - one line only>")

        return "\n".join(parts)

    def _confidence_to_word(self, confidence: float) -> str:
        """Convert numeric confidence to word."""
        if confidence < 0.4:
            return "low"
        elif confidence < 0.7:
            return "medium"
        else:
            return "high"

    def compute_uplift(
        self,
        original_tokens: TokenVector,
        revised_tokens: TokenVector,
    ) -> Tuple[float, dict]:
        """Compute improvement from revision.

        Returns:
            total_uplift: Sum of token improvements
            uplift_by_dim: Per-dimension improvements
        """
        uplift_by_dim = {}
        total = 0.0

        for dim in TokenDimension:
            orig = original_tokens.values[dim]
            rev = revised_tokens.values[dim]
            diff = rev - orig
            uplift_by_dim[dim.value] = diff
            total += diff

        return total, uplift_by_dim

    def parse_revision_changes(self, text: str) -> List[str]:
        """Extract the CHANGES: section from revision."""
        import re

        match = re.search(
            r"CHANGES:\s*(.+?)(?:REASONING:|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            return []

        changes_text = match.group(1).strip()

        # Parse bullet points or numbered list
        lines = changes_text.split("\n")
        changes = []
        for line in lines:
            line = line.strip()
            if line.startswith(("-", "•", "*", "1", "2", "3")):
                # Remove bullet/number
                line = re.sub(r"^[-•*\d.)\s]+", "", line).strip()
                if line:
                    changes.append(line)
            elif line and not changes:
                # First line without bullet
                changes.append(line)

        return changes[:3]  # Limit to 3 changes
