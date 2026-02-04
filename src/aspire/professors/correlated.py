"""Correlated Professor Implementations.

These professors share latent evaluative structure by design.
Unlike the orthogonal professors (accuracy, clarity, calibration),
these professors all measure the SAME underlying quality from
different perspectives or with different noise.

This enables meaningful holdout transfer tests:
- If professors share structure, transfer SHOULD succeed
- If transfer fails with correlated professors, ASPIRE has a genuine limitation

Key design principles:
1. All professors derive scores from a common latent "quality" function
2. Each professor adds perspective-specific noise/weighting
3. Inter-professor correlation should be 0.5-0.8 (moderate to strong)
4. Factor analysis should reveal 1 dominant shared factor (>50% variance)
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random
import math

from ..core import (
    TrainingItem,
    StudentResponse,
    ProfessorCritique,
    TokenVector,
    TokenDimension,
    EnsembleEvaluation,
)
from .base import Professor


@dataclass
class LatentQuality:
    """The shared latent quality that all correlated professors measure.

    This represents the "true" quality of a response that different
    professors observe through different lenses.
    """

    # Core quality dimensions (shared)
    correctness: float  # 0-1: factual accuracy
    reasoning: float    # 0-1: quality of reasoning process
    calibration: float  # 0-1: confidence-accuracy alignment

    # Derived aggregate
    @property
    def overall(self) -> float:
        """Weighted aggregate of core qualities."""
        return 0.5 * self.correctness + 0.3 * self.reasoning + 0.2 * self.calibration


def compute_latent_quality(
    item: TrainingItem,
    response: StudentResponse,
    seed: Optional[int] = None,
) -> LatentQuality:
    """Compute the latent quality of a response.

    This is the ground truth that all correlated professors will
    observe (with their own noise/perspective).
    """
    if seed is not None:
        random.seed(seed)

    # Correctness: does the answer match?
    answer_match = (
        item.gold_answer.lower() in response.answer.lower() or
        response.answer.lower() in item.gold_answer.lower()
    )
    correctness = 0.9 if answer_match else 0.2

    # Add small noise to avoid perfect correlation
    correctness = max(0, min(1, correctness + random.gauss(0, 0.05)))

    # Reasoning: quality markers in reasoning trace
    trace = response.reasoning_trace.lower()
    reasoning_markers = ["because", "therefore", "since", "given", "implies"]
    structure_markers = ["first", "second", "finally", "step"]

    reasoning = 0.4
    reasoning += 0.1 * sum(1 for m in reasoning_markers if m in trace)
    reasoning += 0.1 * sum(1 for m in structure_markers if m in trace)
    reasoning = max(0, min(1, reasoning + random.gauss(0, 0.05)))

    # Calibration: confidence-correctness alignment
    if answer_match and response.confidence > 0.7:
        calibration = 0.9
    elif not answer_match and response.confidence < 0.4:
        calibration = 0.8
    elif answer_match and response.confidence < 0.4:
        calibration = 0.5  # Underconfident
    elif not answer_match and response.confidence > 0.7:
        calibration = 0.2  # Overconfident
    else:
        calibration = 0.6

    calibration = max(0, min(1, calibration + random.gauss(0, 0.05)))

    return LatentQuality(
        correctness=correctness,
        reasoning=reasoning,
        calibration=calibration,
    )


class CorrelatedProfessor(Professor):
    """Base class for professors that share latent structure.

    All correlated professors observe the same LatentQuality
    but with different weights and noise levels.
    """

    def __init__(
        self,
        professor_id: str,
        name: str,
        description: str,
        correctness_weight: float = 0.33,
        reasoning_weight: float = 0.33,
        calibration_weight: float = 0.34,
        noise_std: float = 0.1,
    ):
        super().__init__(professor_id, name, description)
        self.correctness_weight = correctness_weight
        self.reasoning_weight = reasoning_weight
        self.calibration_weight = calibration_weight
        self.noise_std = noise_std

    def _observe_quality(
        self,
        latent: LatentQuality,
        seed: Optional[int] = None,
    ) -> float:
        """Observe the latent quality with this professor's weights and noise."""
        if seed is not None:
            random.seed(seed)

        # Weighted observation
        observed = (
            self.correctness_weight * latent.correctness +
            self.reasoning_weight * latent.reasoning +
            self.calibration_weight * latent.calibration
        )

        # Add perspective-specific noise
        observed += random.gauss(0, self.noise_std)

        return max(0, min(1, observed))

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        """Evaluate based on observed latent quality."""
        # Compute shared latent quality
        latent = compute_latent_quality(item, response)

        # Observe through this professor's lens
        observed_quality = self._observe_quality(latent)

        # Generate tokens based on observed quality
        tokens = self._generate_tokens(observed_quality, latent, response)

        # Generate critique
        is_correct = latent.correctness > 0.5
        critique_text = self._generate_critique(observed_quality, latent)
        weaknesses = self._identify_weaknesses(latent)

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text=critique_text,
            specific_weaknesses=weaknesses,
        )

    @abstractmethod
    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        """Generate token vector from observed quality."""
        pass

    def _generate_critique(
        self,
        observed_quality: float,
        latent: LatentQuality,
    ) -> str:
        """Generate critique text."""
        if observed_quality > 0.8:
            return "Excellent response with clear reasoning."
        elif observed_quality > 0.6:
            return "Good response with minor issues."
        elif observed_quality > 0.4:
            return "Adequate response but room for improvement."
        else:
            return "Response needs significant improvement."

    def _identify_weaknesses(self, latent: LatentQuality) -> List[str]:
        """Identify specific weaknesses."""
        weaknesses = []
        if latent.correctness < 0.5:
            weaknesses.append("Incorrect or unclear answer")
        if latent.reasoning < 0.5:
            weaknesses.append("Weak reasoning structure")
        if latent.calibration < 0.5:
            weaknesses.append("Poor confidence calibration")
        return weaknesses


class RigorProfessor(CorrelatedProfessor):
    """Emphasizes correctness and reasoning rigor.

    Weighs correctness and reasoning more heavily.
    Lower noise - more consistent evaluations.
    """

    def __init__(self):
        super().__init__(
            professor_id="rigor",
            name="Professor Rigor",
            description="Emphasizes correctness and sound reasoning",
            correctness_weight=0.50,
            reasoning_weight=0.35,
            calibration_weight=0.15,
            noise_std=0.08,
        )

    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        return {
            TokenDimension.CORRECTNESS: latent.correctness,
            TokenDimension.COHERENCE: 0.7 * latent.reasoning + 0.3 * observed_quality,
            TokenDimension.CALIBRATION: latent.calibration,
            TokenDimension.TRADEOFFS: observed_quality * 0.8,
            TokenDimension.CLARITY: observed_quality * 0.9,
        }


class NuanceProfessor(CorrelatedProfessor):
    """Emphasizes calibration and reasoning nuance.

    Weighs calibration and reasoning more heavily.
    Moderate noise - balanced perspective.
    """

    def __init__(self):
        super().__init__(
            professor_id="nuance",
            name="Professor Nuance",
            description="Values epistemic calibration and nuanced reasoning",
            correctness_weight=0.25,
            reasoning_weight=0.35,
            calibration_weight=0.40,
            noise_std=0.10,
        )

    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        return {
            TokenDimension.CORRECTNESS: latent.correctness * 0.9,
            TokenDimension.COHERENCE: latent.reasoning,
            TokenDimension.CALIBRATION: 0.6 * latent.calibration + 0.4 * observed_quality,
            TokenDimension.TRADEOFFS: observed_quality,
            TokenDimension.CLARITY: 0.5 * latent.reasoning + 0.5 * observed_quality,
        }


class HolisticProfessor(CorrelatedProfessor):
    """Takes a balanced view of all quality dimensions.

    Equal weights on all dimensions.
    Higher noise - more variable but still correlated.
    """

    def __init__(self):
        super().__init__(
            professor_id="holistic",
            name="Professor Holistic",
            description="Balanced evaluation of all response qualities",
            correctness_weight=0.33,
            reasoning_weight=0.34,
            calibration_weight=0.33,
            noise_std=0.12,
        )

    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        return {
            TokenDimension.CORRECTNESS: latent.correctness,
            TokenDimension.COHERENCE: latent.reasoning,
            TokenDimension.CALIBRATION: latent.calibration,
            TokenDimension.TRADEOFFS: observed_quality,
            TokenDimension.CLARITY: observed_quality,
        }


class PragmatistProfessor(CorrelatedProfessor):
    """Emphasizes correctness with practical lens.

    Strong emphasis on getting the answer right.
    Moderate noise.
    """

    def __init__(self):
        super().__init__(
            professor_id="pragmatist",
            name="Professor Pragmatist",
            description="Focuses on practical correctness and clear communication",
            correctness_weight=0.55,
            reasoning_weight=0.25,
            calibration_weight=0.20,
            noise_std=0.10,
        )

    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        return {
            TokenDimension.CORRECTNESS: 0.8 * latent.correctness + 0.2 * observed_quality,
            TokenDimension.COHERENCE: 0.6 * latent.reasoning + 0.4 * latent.correctness,
            TokenDimension.CALIBRATION: latent.calibration * 0.8,
            TokenDimension.TRADEOFFS: observed_quality * 0.7,
            TokenDimension.CLARITY: 0.4 * latent.reasoning + 0.6 * observed_quality,
        }


class TheoristProfessor(CorrelatedProfessor):
    """Emphasizes reasoning and calibration.

    Strong emphasis on reasoning quality.
    Lower noise - theoreticians are consistent.
    """

    def __init__(self):
        super().__init__(
            professor_id="theorist",
            name="Professor Theorist",
            description="Values rigorous reasoning and proper uncertainty",
            correctness_weight=0.20,
            reasoning_weight=0.45,
            calibration_weight=0.35,
            noise_std=0.08,
        )

    def _generate_tokens(
        self,
        observed_quality: float,
        latent: LatentQuality,
        response: StudentResponse,
    ) -> Dict[TokenDimension, float]:
        return {
            TokenDimension.CORRECTNESS: latent.correctness * 0.9,
            TokenDimension.COHERENCE: 0.7 * latent.reasoning + 0.3 * observed_quality,
            TokenDimension.CALIBRATION: 0.5 * latent.calibration + 0.5 * latent.reasoning,
            TokenDimension.TRADEOFFS: latent.reasoning,
            TokenDimension.CLARITY: 0.6 * latent.reasoning + 0.4 * observed_quality,
        }


class CorrelatedProfessorEnsemble:
    """Ensemble of correlated professors for testing holdout transfer.

    Unlike ProfessorEnsemble, these professors share latent structure
    by design, enabling meaningful transfer tests.

    Expected properties:
    - Inter-professor correlation: 0.5-0.8
    - First factor variance: >50%
    - Effective dimensionality: <2 for 5 professors
    """

    def __init__(
        self,
        professors: Optional[List[CorrelatedProfessor]] = None,
        holdout_idx: Optional[int] = None,
    ):
        self.all_professors = professors or [
            RigorProfessor(),
            NuanceProfessor(),
            HolisticProfessor(),
            PragmatistProfessor(),
            TheoristProfessor(),
        ]
        self.holdout_idx = holdout_idx

        # Active professors (excluding holdout if set)
        if holdout_idx is not None:
            self.active_professors = [
                p for i, p in enumerate(self.all_professors)
                if i != holdout_idx
            ]
            self.holdout_professor = self.all_professors[holdout_idx]
        else:
            self.active_professors = self.all_professors
            self.holdout_professor = None

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
        strict_dims: Optional[List[str]] = None,
    ) -> EnsembleEvaluation:
        """Get critiques from active professors and aggregate."""
        critiques = [
            prof.evaluate(item, response)
            for prof in self.active_professors
        ]
        return EnsembleEvaluation.from_critiques(critiques, strict_dims)

    def evaluate_holdout(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> Optional[ProfessorCritique]:
        """Get critique from holdout professor only."""
        if self.holdout_professor is None:
            return None
        return self.holdout_professor.evaluate(item, response)

    @property
    def professor_names(self) -> List[str]:
        """Names of all professors (for analysis)."""
        return [p.name for p in self.all_professors]

    @property
    def active_professor_names(self) -> List[str]:
        """Names of active (non-holdout) professors."""
        return [p.name for p in self.active_professors]


def verify_correlation_structure(
    n_samples: int = 200,
    seed: int = 42,
) -> Dict:
    """Verify that correlated professors actually share structure.

    Returns analysis showing:
    - Pairwise correlations (expected: 0.5-0.8)
    - Factor loadings (expected: one dominant factor)
    - Effective dimensionality (expected: <2 for 5 professors)
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Create ensemble
    ensemble = CorrelatedProfessorEnsemble()

    # Generate test items
    items = []
    for i in range(n_samples):
        items.append(TrainingItem(
            id=f"verify_{i}",
            prompt=f"Test question {i}",
            gold_answer="Correct answer",
            gold_rationale="",
            domain="test",
        ))

    # Collect scores
    scores = {p.professor_id: [] for p in ensemble.all_professors}

    for item in items:
        response = StudentResponse(
            item_id=item.id,
            answer=random.choice([item.gold_answer, "Wrong", "Partial"]),
            reasoning_trace=random.choice([
                "Because of X, therefore Y.",
                "First, analyze. Second, conclude. Finally, verify.",
                "I think maybe the answer is this.",
                "The answer follows from the given premises since they imply the conclusion.",
            ]),
            confidence=random.uniform(0.2, 0.95),
        )

        for prof in ensemble.all_professors:
            critique = prof.evaluate(item, response)
            # Use overall token average as score
            tokens = critique.tokens.values
            score = sum(tokens.values()) / len(tokens)
            scores[prof.professor_id].append(score)

    # Compute correlations
    prof_ids = list(scores.keys())
    score_matrix = np.array([scores[pid] for pid in prof_ids])

    correlations = {}
    for i in range(len(prof_ids)):
        for j in range(i + 1, len(prof_ids)):
            corr = np.corrcoef(score_matrix[i], score_matrix[j])[0, 1]
            correlations[f"{prof_ids[i]}_vs_{prof_ids[j]}"] = float(corr)

    mean_corr = np.mean(list(correlations.values()))

    # Factor analysis
    centered = score_matrix - score_matrix.mean(axis=1, keepdims=True)
    cov = np.cov(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]

    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues

    # Effective dimensionality
    normalized = eigenvalues / total_var if total_var > 0 else eigenvalues
    sum_sq = np.sum(normalized ** 2)
    eff_dim = 1.0 / sum_sq if sum_sq > 0 else len(eigenvalues)

    return {
        "pairwise_correlations": correlations,
        "mean_correlation": float(mean_corr),
        "first_factor_variance": float(var_explained[0]),
        "second_factor_variance": float(var_explained[1]) if len(var_explained) > 1 else 0,
        "effective_dimensionality": float(eff_dim),
        "n_professors": len(prof_ids),
        "transfer_viable": bool(mean_corr > 0.4 and var_explained[0] > 0.4),
    }
