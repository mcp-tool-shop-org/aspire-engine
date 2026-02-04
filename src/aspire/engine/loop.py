"""ASPIRE training loop - the core engine."""

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
)
from ..student import StudentModel, TrainingSignal
from ..professors import ProfessorEnsemble
from ..critic import Critic, CriticPrediction, MisalignmentSignal
from ..governor import TokenPool, GovernorConfig, ThrottleLevel


@dataclass
class CycleResult:
    """Result of a single training cycle."""
    item: TrainingItem
    response: StudentResponse
    evaluation: EnsembleEvaluation
    critic_prediction: CriticPrediction
    misalignment: MisalignmentSignal
    tokens_earned: TokenVector
    cycle_time_ms: float
    teaching_moment: TeachingMoment
    governor_wait_ms: float = 0.0      # Time spent waiting for governor
    governor_throttled: bool = False   # Was this cycle throttled?


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    total_cycles: int = 0
    correct_count: int = 0
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    critic_surprise_history: List[float] = field(default_factory=list)
    avg_cycle_time_ms: float = 0.0

    # Governor metrics
    throttled_cycles: int = 0
    total_governor_wait_ms: float = 0.0
    oom_retries: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.correct_count / self.total_cycles

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


class AspireEngine:
    """The ASPIRE training engine.

    Orchestrates the test→critique→reveal→update loop.

    Each cycle:
    0. Acquire governor token (wait if throttled)
    1. Present test item to student
    2. Student generates reasoning + answer
    3. Critic predicts token outcomes (gut feeling)
    4. Professors evaluate and award tokens
    5. Compute misalignment (surprise)
    6. Reveal gold answer + rationale (teaching moment)
    7. Update student and critic
    8. Release governor token (with OOM classification)
    9. Move to next test

    Target: 5-20 seconds per cycle
    """

    def __init__(
        self,
        student: StudentModel,
        professors: Optional[ProfessorEnsemble] = None,
        critic: Optional[Critic] = None,
        governor: Optional[TokenPool] = None,
        governor_config: Optional[GovernorConfig] = None,
        on_cycle_complete: Optional[Callable[[CycleResult], None]] = None,
    ):
        from ..critic import HeuristicCritic

        self.student = student
        self.professors = professors or ProfessorEnsemble()
        self.critic = critic or HeuristicCritic()
        self.on_cycle_complete = on_cycle_complete

        # Governor for resource protection
        if governor is not None:
            self.governor = governor
        elif governor_config is not None:
            self.governor = TokenPool(governor_config)
        else:
            # No governor - run unprotected (for testing/mock)
            self.governor = None

        self.metrics = TrainingMetrics()
        self._running = False

    def run_cycle(self, item: TrainingItem) -> CycleResult:
        """Run a single training cycle."""
        cycle_start = time.perf_counter()
        governor_wait_ms = 0.0
        governor_throttled = False
        lease_id = None

        # 0. Acquire governor token (if enabled)
        if self.governor is not None:
            acquire_result = self.governor.try_acquire(
                operation=f"inference:{item.id}",
                requested_tokens=1,
            )
            governor_wait_ms = acquire_result.wait_time_ms
            governor_throttled = acquire_result.throttle_level != ThrottleLevel.NORMAL

            if not acquire_result.success:
                # Governor refused - create a failed result
                # This shouldn't happen often if timeouts are reasonable
                raise RuntimeError(
                    f"Governor refused inference: {acquire_result.reason}"
                )

            lease_id = acquire_result.lease_id

        inference_start = time.perf_counter()

        # 1. Student generates response
        response = self.student.generate(item)

        # 2. Critic predicts (before seeing professors)
        critic_pred = self.critic.predict(item, response)

        # 3. Professors evaluate
        evaluation = self.professors.evaluate(item, response)

        # 4. Compute misalignment
        misalignment = self.critic.compute_misalignment(
            critic_pred,
            evaluation.aggregated_tokens,
            evaluation.disagreement_score,
            response.confidence,
        )

        # 5. Determine final token payout
        # Bonus for surviving disagreement
        tokens_earned = evaluation.aggregated_tokens
        if evaluation.disagreement_score > 0.3 and evaluation.consensus_correct:
            # Defended position under scrutiny - bonus!
            bonus = TokenVector({d: 0.1 for d in tokens_earned.values.keys()})
            tokens_earned = tokens_earned + bonus

        # Penalty for overconfidence
        if misalignment.overconfidence_penalty > 0:
            penalty_factor = 1.0 - (misalignment.overconfidence_penalty * 0.2)
            tokens_earned = TokenVector({
                d: v * penalty_factor
                for d, v in tokens_earned.values.items()
            })

        # 6. Create teaching moment
        teaching_moment = TeachingMoment(
            item=item,
            student_response=response,
            evaluation=evaluation,
            tokens_earned=tokens_earned,
            should_revise=misalignment.total_surprise > 0.5,
        )

        # 7. Update student
        signal = TrainingSignal(
            item=item,
            response=response,
            token_reward=tokens_earned.total / len(tokens_earned.values),  # Normalized
            gold_answer=item.gold_answer,
            gold_rationale=item.gold_rationale,
            critiques=[c.critique_text for c in evaluation.critiques],
        )
        self.student.update(signal)

        # 8. Update critic
        self.critic.update(
            critic_pred,
            evaluation.aggregated_tokens,
            evaluation.disagreement_score,
        )

        # 9. Release governor token
        if self.governor is not None and lease_id is not None:
            inference_time_ms = (time.perf_counter() - inference_start) * 1000
            release_result = self.governor.release(
                lease_id=lease_id,
                peak_memory_mb=0,  # TODO: track actual peak
                exit_code=0,  # Success
                duration_ms=inference_time_ms,
            )

            # Track OOM retries if classified
            if release_result.should_retry:
                self.metrics.oom_retries += 1

        cycle_time = (time.perf_counter() - cycle_start) * 1000

        # Update metrics
        self.metrics.total_cycles += 1
        if evaluation.consensus_correct:
            self.metrics.correct_count += 1
        self.metrics.token_ledger.record(tokens_earned)
        self.metrics.critic_surprise_history.append(misalignment.total_surprise)
        self.metrics.avg_cycle_time_ms = (
            self.metrics.avg_cycle_time_ms * (self.metrics.total_cycles - 1) +
            cycle_time
        ) / self.metrics.total_cycles

        # Governor metrics
        if governor_throttled:
            self.metrics.throttled_cycles += 1
        self.metrics.total_governor_wait_ms += governor_wait_ms

        result = CycleResult(
            item=item,
            response=response,
            evaluation=evaluation,
            critic_prediction=critic_pred,
            misalignment=misalignment,
            tokens_earned=tokens_earned,
            cycle_time_ms=cycle_time,
            teaching_moment=teaching_moment,
            governor_wait_ms=governor_wait_ms,
            governor_throttled=governor_throttled,
        )

        if self.on_cycle_complete:
            self.on_cycle_complete(result)

        return result

    def train(
        self,
        items: Iterator[TrainingItem],
        max_cycles: Optional[int] = None,
        target_accuracy: Optional[float] = None,
        target_surprise: Optional[float] = None,
    ) -> TrainingMetrics:
        """Run training loop over items.

        Stops when:
        - max_cycles reached
        - target_accuracy achieved (over last 100 cycles)
        - target_surprise achieved (critic predictions converged)
        - items exhausted
        """
        self._running = True
        recent_correct = []

        for item in items:
            if not self._running:
                break

            if max_cycles and self.metrics.total_cycles >= max_cycles:
                break

            result = self.run_cycle(item)

            # Track recent accuracy
            recent_correct.append(1 if result.evaluation.consensus_correct else 0)
            if len(recent_correct) > 100:
                recent_correct.pop(0)

            # Check stopping conditions
            if target_accuracy and len(recent_correct) >= 100:
                recent_acc = sum(recent_correct) / len(recent_correct)
                if recent_acc >= target_accuracy:
                    break

            if target_surprise and len(self.metrics.critic_surprise_history) >= 50:
                recent_surprise = sum(self.metrics.critic_surprise_history[-50:]) / 50
                if recent_surprise <= target_surprise:
                    break

        self._running = False
        return self.metrics

    def stop(self):
        """Stop training loop."""
        self._running = False

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = TrainingMetrics()
