"""ASPIRE Engine with revision support - the self-correction loop."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Iterator
import time

from ..core import (
    TrainingItem,
    StudentResponse,
    EnsembleEvaluation,
    TeachingMoment,
    TokenVector,
    TokenLedger,
    TokenDimension,
)
from ..student import StudentModel, TrainingSignal
from ..professors import ProfessorEnsemble
from ..critic import Critic, CriticPrediction, MisalignmentSignal
from ..governor import TokenPool, GovernorConfig, ThrottleLevel

from .revision import (
    RevisionEngine,
    RevisionConfig,
    RevisionDecision,
    RevisionResult,
    RevisionTrigger,
)


@dataclass
class RevisionCycleResult:
    """Result of a training cycle with possible revision."""
    item: TrainingItem

    # Draft pass
    draft_response: StudentResponse
    draft_evaluation: EnsembleEvaluation
    draft_tokens: TokenVector
    draft_prediction: CriticPrediction
    draft_misalignment: MisalignmentSignal

    # Revision decision
    revision_decision: RevisionDecision
    did_revise: bool

    # Revision pass (if any)
    revised_response: Optional[StudentResponse] = None
    revised_evaluation: Optional[EnsembleEvaluation] = None
    revised_tokens: Optional[TokenVector] = None

    # Final tokens (draft or revised)
    final_tokens: TokenVector = None
    final_response: StudentResponse = None

    # Metrics
    token_uplift: float = 0.0
    uplift_by_dim: dict = field(default_factory=dict)
    cycle_time_ms: float = 0.0
    revision_time_ms: float = 0.0
    changes_made: List[str] = field(default_factory=list)

    # Teaching moment
    teaching_moment: TeachingMoment = None

    # Governor
    governor_wait_ms: float = 0.0
    governor_throttled: bool = False


@dataclass
class RevisionMetrics:
    """Metrics tracking revision behavior."""
    total_cycles: int = 0
    correct_count: int = 0
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    critic_surprise_history: List[float] = field(default_factory=list)
    avg_cycle_time_ms: float = 0.0

    # Governor metrics
    throttled_cycles: int = 0
    total_governor_wait_ms: float = 0.0
    oom_retries: int = 0

    # Revision-specific metrics
    revision_count: int = 0
    revision_triggers: dict = field(default_factory=dict)
    total_uplift: float = 0.0
    uplift_history: List[float] = field(default_factory=list)
    avg_revision_time_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.correct_count / self.total_cycles

    @property
    def revision_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.revision_count / self.total_cycles

    @property
    def avg_uplift(self) -> float:
        if not self.uplift_history:
            return 0.0
        return sum(self.uplift_history) / len(self.uplift_history)

    @property
    def avg_surprise(self) -> float:
        if not self.critic_surprise_history:
            return 0.0
        return sum(self.critic_surprise_history) / len(self.critic_surprise_history)

    @property
    def throttle_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.throttled_cycles / self.total_cycles


class RevisionAspireEngine:
    """ASPIRE Engine with two-pass revision support.

    Extends the base training loop with self-correction:

    1. Draft: Student generates initial response
    2. Critique: Professors evaluate, critic predicts
    3. Decision: Should student revise? (based on surprise/disagreement)
    4. Revise: If yes, student rewrites with critique feedback
    5. Re-evaluate: Professors score revision
    6. Learn: Update student and critic on both passes

    This teaches the student to:
    - Notice when reasoning is weak (via critic signals)
    - Repair under pressure (revision pass)
    - Calibrate hedging when disagreement is high
    """

    def __init__(
        self,
        student: StudentModel,
        professors: Optional[ProfessorEnsemble] = None,
        critic: Optional[Critic] = None,
        revision_config: Optional[RevisionConfig] = None,
        governor: Optional[TokenPool] = None,
        governor_config: Optional[GovernorConfig] = None,
        on_cycle_complete: Optional[Callable[[RevisionCycleResult], None]] = None,
    ):
        from ..critic import HeuristicCritic

        self.student = student
        self.professors = professors or ProfessorEnsemble()
        self.critic = critic or HeuristicCritic()
        self.revision_engine = RevisionEngine(revision_config or RevisionConfig())
        self.on_cycle_complete = on_cycle_complete

        # Governor
        if governor is not None:
            self.governor = governor
        elif governor_config is not None:
            self.governor = TokenPool(governor_config)
        else:
            self.governor = None

        self.metrics = RevisionMetrics()
        self._running = False

    def run_cycle(self, item: TrainingItem) -> RevisionCycleResult:
        """Run a single training cycle with possible revision."""
        cycle_start = time.perf_counter()
        governor_wait_ms = 0.0
        governor_throttled = False
        lease_id = None

        # 0. Acquire governor token
        if self.governor is not None:
            acquire_result = self.governor.try_acquire(
                operation=f"inference:{item.id}",
                requested_tokens=2,  # Request 2 for potential revision
            )
            governor_wait_ms = acquire_result.wait_time_ms
            governor_throttled = acquire_result.throttle_level != ThrottleLevel.NORMAL

            if not acquire_result.success:
                raise RuntimeError(f"Governor refused: {acquire_result.reason}")

            lease_id = acquire_result.lease_id

        inference_start = time.perf_counter()

        # === DRAFT PASS ===

        # 1. Student generates draft
        draft_response = self.student.generate(item)

        # 2. Critic predicts
        draft_prediction = self.critic.predict(item, draft_response)

        # 3. Professors evaluate draft
        draft_evaluation = self.professors.evaluate(item, draft_response)

        # 4. Compute misalignment
        draft_misalignment = self.critic.compute_misalignment(
            draft_prediction,
            draft_evaluation.aggregated_tokens,
            draft_evaluation.disagreement_score,
            draft_response.confidence,
        )

        # 5. Compute draft tokens
        draft_tokens = self._compute_tokens(draft_evaluation, draft_misalignment)

        # === REVISION DECISION ===

        revision_decision = self.revision_engine.should_revise(
            draft_prediction,
            draft_misalignment,
            draft_evaluation,
            draft_response,
        )

        # Initialize revision fields
        revised_response = None
        revised_evaluation = None
        revised_tokens = None
        token_uplift = 0.0
        uplift_by_dim = {}
        revision_time_ms = 0.0
        changes_made = []

        # === REVISION PASS (if triggered) ===

        if revision_decision.should_revise:
            revision_start = time.perf_counter()

            # Build revision prompt
            revision_prompt = self.revision_engine.build_revision_prompt(
                item,
                draft_response,
                draft_evaluation,
                draft_prediction,
                revision_decision,
            )

            # Create revision item (reuse item with new prompt)
            revision_item = TrainingItem(
                id=f"{item.id}_revision",
                prompt=revision_prompt,
                gold_answer=item.gold_answer,
                gold_rationale=item.gold_rationale,
                difficulty=item.difficulty,
                domain=item.domain,
            )

            # Generate revision
            revised_response = self.student.generate(
                revision_item,
                max_tokens=self.revision_engine.config.max_revision_tokens,
            )

            # Extract changes
            changes_made = self.revision_engine.parse_revision_changes(
                revised_response.reasoning_trace
            )

            # Evaluate revision
            revised_evaluation = self.professors.evaluate(item, revised_response)

            # Compute revised tokens
            revised_misalignment = self.critic.compute_misalignment(
                draft_prediction,  # Use same prediction for fair comparison
                revised_evaluation.aggregated_tokens,
                revised_evaluation.disagreement_score,
                revised_response.confidence,
            )
            revised_tokens = self._compute_tokens(revised_evaluation, revised_misalignment)

            # Compute uplift
            token_uplift, uplift_by_dim = self.revision_engine.compute_uplift(
                draft_tokens, revised_tokens
            )

            revision_time_ms = (time.perf_counter() - revision_start) * 1000

            # Track revision triggers
            for trigger in revision_decision.triggers:
                key = trigger.value
                self.metrics.revision_triggers[key] = \
                    self.metrics.revision_triggers.get(key, 0) + 1

        # === DETERMINE FINAL RESULT ===

        if revised_response is not None and token_uplift > 0:
            # Revision improved - use revised
            final_tokens = revised_tokens
            final_response = revised_response
            final_evaluation = revised_evaluation
        else:
            # No revision or no improvement - use draft
            final_tokens = draft_tokens
            final_response = draft_response
            final_evaluation = draft_evaluation

        # === UPDATE STUDENT AND CRITIC ===

        # Training signal for draft
        draft_signal = TrainingSignal(
            item=item,
            response=draft_response,
            token_reward=draft_tokens.total / len(TokenDimension),
            gold_answer=item.gold_answer,
            gold_rationale=item.gold_rationale,
            critiques=[c.critique_text for c in draft_evaluation.critiques],
        )
        self.student.update(draft_signal)

        # Training signal for revision (if any)
        if revised_response is not None:
            revision_signal = TrainingSignal(
                item=item,
                response=revised_response,
                token_reward=revised_tokens.total / len(TokenDimension),
                gold_answer=item.gold_answer,
                gold_rationale=item.gold_rationale,
                critiques=[c.critique_text for c in revised_evaluation.critiques],
            )
            self.student.update(revision_signal)

        # Update critic
        self.critic.update(
            draft_prediction,
            draft_evaluation.aggregated_tokens,
            draft_evaluation.disagreement_score,
        )

        # === RELEASE GOVERNOR ===

        if self.governor is not None and lease_id is not None:
            inference_time_ms = (time.perf_counter() - inference_start) * 1000
            release_result = self.governor.release(
                lease_id=lease_id,
                peak_memory_mb=0,
                exit_code=0,
                duration_ms=inference_time_ms,
            )
            if release_result.should_retry:
                self.metrics.oom_retries += 1

        # === CREATE TEACHING MOMENT ===

        teaching_moment = TeachingMoment(
            item=item,
            student_response=final_response,
            evaluation=final_evaluation,
            tokens_earned=final_tokens,
            should_revise=False,  # Already handled
        )

        cycle_time = (time.perf_counter() - cycle_start) * 1000

        # === UPDATE METRICS ===

        self.metrics.total_cycles += 1
        if final_evaluation.consensus_correct:
            self.metrics.correct_count += 1
        self.metrics.token_ledger.record(final_tokens)
        self.metrics.critic_surprise_history.append(draft_misalignment.total_surprise)
        self.metrics.avg_cycle_time_ms = (
            self.metrics.avg_cycle_time_ms * (self.metrics.total_cycles - 1) +
            cycle_time
        ) / self.metrics.total_cycles

        if governor_throttled:
            self.metrics.throttled_cycles += 1
        self.metrics.total_governor_wait_ms += governor_wait_ms

        # Revision metrics
        if revision_decision.should_revise:
            self.metrics.revision_count += 1
            self.metrics.total_uplift += token_uplift
            self.metrics.uplift_history.append(token_uplift)
            if self.metrics.revision_count > 0:
                self.metrics.avg_revision_time_ms = (
                    self.metrics.avg_revision_time_ms * (self.metrics.revision_count - 1) +
                    revision_time_ms
                ) / self.metrics.revision_count

        # === BUILD RESULT ===

        result = RevisionCycleResult(
            item=item,
            draft_response=draft_response,
            draft_evaluation=draft_evaluation,
            draft_tokens=draft_tokens,
            draft_prediction=draft_prediction,
            draft_misalignment=draft_misalignment,
            revision_decision=revision_decision,
            did_revise=revision_decision.should_revise,
            revised_response=revised_response,
            revised_evaluation=revised_evaluation,
            revised_tokens=revised_tokens,
            final_tokens=final_tokens,
            final_response=final_response,
            token_uplift=token_uplift,
            uplift_by_dim=uplift_by_dim,
            cycle_time_ms=cycle_time,
            revision_time_ms=revision_time_ms,
            changes_made=changes_made,
            teaching_moment=teaching_moment,
            governor_wait_ms=governor_wait_ms,
            governor_throttled=governor_throttled,
        )

        if self.on_cycle_complete:
            self.on_cycle_complete(result)

        return result

    def _compute_tokens(
        self,
        evaluation: EnsembleEvaluation,
        misalignment: MisalignmentSignal,
    ) -> TokenVector:
        """Compute final tokens with bonuses/penalties."""
        tokens = evaluation.aggregated_tokens

        # Bonus for surviving disagreement
        if evaluation.disagreement_score > 0.3 and evaluation.consensus_correct:
            bonus = TokenVector({d: 0.1 for d in tokens.values.keys()})
            tokens = tokens + bonus

        # Penalty for overconfidence
        if misalignment.overconfidence_penalty > 0:
            penalty_factor = 1.0 - (misalignment.overconfidence_penalty * 0.2)
            tokens = TokenVector({
                d: v * penalty_factor
                for d, v in tokens.values.items()
            })

        return tokens

    def train(
        self,
        items: Iterator[TrainingItem],
        max_cycles: Optional[int] = None,
        target_accuracy: Optional[float] = None,
        target_uplift: Optional[float] = None,
    ) -> RevisionMetrics:
        """Run training loop with revision support."""
        self._running = True
        recent_correct = []

        for item in items:
            if not self._running:
                break

            if max_cycles and self.metrics.total_cycles >= max_cycles:
                break

            result = self.run_cycle(item)

            # Track recent accuracy
            recent_correct.append(1 if result.final_response and
                                 result.draft_evaluation.consensus_correct else 0)
            if len(recent_correct) > 100:
                recent_correct.pop(0)

            # Check stopping conditions
            if target_accuracy and len(recent_correct) >= 100:
                if sum(recent_correct) / len(recent_correct) >= target_accuracy:
                    break

            if target_uplift and len(self.metrics.uplift_history) >= 20:
                recent_uplift = sum(self.metrics.uplift_history[-20:]) / 20
                if recent_uplift >= target_uplift:
                    break

        self._running = False
        return self.metrics

    def stop(self):
        """Stop training loop."""
        self._running = False

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = RevisionMetrics()
