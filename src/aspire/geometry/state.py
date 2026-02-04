"""State vectors for ASPIRE training geometry.

A StateVector captures all relevant scalar dimensions of an ASPIRE system
at a single point in training. This enables tracking dimensional evolution
and visualizing the transition from untrained to trained states.

Scalar dimensions captured:
1. Token dimension statistics (per-dimension mean, std, min, max)
2. Critic prediction error (per-dimension MAE)
3. Professor disagreement distribution
4. Student confidence calibration
5. Revision behavior statistics

The key insight from ML geometry research: training doesn't increase
dimensionality - it reshapes and compresses it. ASPIRE's "conscience"
is literally a dimensional collapse from isotropic to task-aligned.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core import TokenDimension, TokenVector


@dataclass
class TokenDimensionStats:
    """Statistics for a single token dimension over recent cycles."""
    dimension: TokenDimension
    mean: float = 0.5
    std: float = 0.25
    min_val: float = 0.0
    max_val: float = 1.0

    # Prediction error (critic)
    prediction_mae: float = 0.5

    # Sensitivity (how much this dimension varies with input)
    sensitivity: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert to 6-element vector."""
        return np.array([
            self.mean,
            self.std,
            self.min_val,
            self.max_val,
            self.prediction_mae,
            self.sensitivity,
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, dim: TokenDimension, vec: np.ndarray) -> "TokenDimensionStats":
        """Create from 6-element vector."""
        return cls(
            dimension=dim,
            mean=float(vec[0]),
            std=float(vec[1]),
            min_val=float(vec[2]),
            max_val=float(vec[3]),
            prediction_mae=float(vec[4]),
            sensitivity=float(vec[5]),
        )


@dataclass
class CriticState:
    """State of the critic's prediction capability."""
    total_predictions: int = 0

    # Per-dimension MAE
    token_mae: Dict[str, float] = field(default_factory=dict)
    disagreement_mae: float = 0.5

    # Confidence calibration
    confidence_mean: float = 0.3
    confidence_std: float = 0.2

    # Surprise statistics
    surprise_mean: float = 0.5
    surprise_std: float = 0.3
    negative_surprise_rate: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert to vector representation."""
        # 5 token MAEs + disagreement + 3 confidence + 3 surprise = 12
        token_maes = [
            self.token_mae.get(dim.value, 0.5)
            for dim in TokenDimension
        ]
        return np.array([
            *token_maes,
            self.disagreement_mae,
            self.confidence_mean,
            self.confidence_std,
            self.surprise_mean,
            self.surprise_std,
            self.negative_surprise_rate,
        ], dtype=np.float32)


@dataclass
class RevisionState:
    """State of the revision behavior."""
    revision_rate: float = 0.0

    # Uplift statistics
    avg_uplift: float = 0.0
    uplift_std: float = 0.0
    positive_uplift_rate: float = 0.0

    # Trigger distribution
    trigger_rates: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert to vector representation."""
        # Standard triggers
        trigger_names = [
            "negative_surprise",
            "disagreement",
            "overconfidence",
            "low_tokens",
        ]
        trigger_vals = [
            self.trigger_rates.get(name, 0.0)
            for name in trigger_names
        ]
        return np.array([
            self.revision_rate,
            self.avg_uplift,
            self.uplift_std,
            self.positive_uplift_rate,
            *trigger_vals,
        ], dtype=np.float32)


@dataclass
class StateVector:
    """Complete state vector capturing all ASPIRE scalars.

    This is the fundamental object for geometry analysis.
    It represents a single point in ASPIRE's training manifold.

    Dimensions (total ~55):
    - Token dimensions: 5 dims × 6 stats = 30
    - Critic state: 11 scalars
    - Revision state: 8 scalars
    - Global: 6 scalars
    """
    # Training progress
    cycle: int = 0
    accuracy: float = 0.0

    # Component states
    token_stats: Dict[TokenDimension, TokenDimensionStats] = field(default_factory=dict)
    critic_state: CriticState = field(default_factory=CriticState)
    revision_state: RevisionState = field(default_factory=RevisionState)

    # Global statistics
    avg_tokens_total: float = 2.5
    disagreement_mean: float = 0.3
    confidence_calibration: float = 0.0  # correlation(confidence, correct)

    # Logit features (V1 critic)
    has_logit_features: bool = False
    entropy_mean: float = 0.0
    margin_mean: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Convert to flat numpy vector for analysis."""
        parts = []

        # Token dimension stats (5 × 6 = 30)
        for dim in TokenDimension:
            if dim in self.token_stats:
                parts.append(self.token_stats[dim].to_vector())
            else:
                # Default untrained stats
                parts.append(np.array([0.5, 0.25, 0.0, 1.0, 0.5, 0.5], dtype=np.float32))

        # Critic state (11)
        parts.append(self.critic_state.to_vector())

        # Revision state (8)
        parts.append(self.revision_state.to_vector())

        # Global (6)
        parts.append(np.array([
            self.accuracy,
            self.avg_tokens_total / 5.0,  # Normalize to 0-1
            self.disagreement_mean,
            self.confidence_calibration,
            self.entropy_mean,
            self.margin_mean,
        ], dtype=np.float32))

        return np.concatenate(parts)

    @property
    def dimensionality(self) -> int:
        """Total number of scalar dimensions."""
        return len(self.to_numpy())

    def distance_to(self, other: "StateVector") -> float:
        """Euclidean distance to another state."""
        return float(np.linalg.norm(self.to_numpy() - other.to_numpy()))

    def cosine_similarity(self, other: "StateVector") -> float:
        """Cosine similarity to another state."""
        v1 = self.to_numpy()
        v2 = other.to_numpy()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


@dataclass
class StateSnapshot:
    """A timestamped state vector with metadata."""
    state: StateVector
    timestamp_ms: float = 0.0

    # What changed since last snapshot
    delta_accuracy: float = 0.0
    delta_tokens: float = 0.0

    # Events this cycle
    had_revision: bool = False
    revision_helped: bool = False

    @classmethod
    def create_untrained(cls) -> "StateSnapshot":
        """Create the canonical 'before training' state.

        Before training, the system has:
        - Uniform token predictions (~0.5)
        - High variance / low structure
        - No task alignment
        - Isotropic scalar distribution
        """
        # Create uniform token stats
        token_stats = {}
        for dim in TokenDimension:
            token_stats[dim] = TokenDimensionStats(
                dimension=dim,
                mean=0.5,
                std=0.25,  # High variance
                min_val=0.0,
                max_val=1.0,
                prediction_mae=0.5,  # Random predictions
                sensitivity=0.5,  # Uniform sensitivity
            )

        # Untrained critic
        critic = CriticState(
            total_predictions=0,
            token_mae={dim.value: 0.5 for dim in TokenDimension},
            disagreement_mae=0.5,
            confidence_mean=0.3,
            confidence_std=0.2,
            surprise_mean=0.5,
            surprise_std=0.3,
            negative_surprise_rate=0.5,
        )

        # No revision history
        revision = RevisionState(
            revision_rate=0.0,
            avg_uplift=0.0,
            uplift_std=0.0,
            positive_uplift_rate=0.0,
            trigger_rates={},
        )

        state = StateVector(
            cycle=0,
            accuracy=0.5,  # Random chance
            token_stats=token_stats,
            critic_state=critic,
            revision_state=revision,
            avg_tokens_total=2.5,
            disagreement_mean=0.3,
            confidence_calibration=0.0,  # No correlation
        )

        return cls(state=state, timestamp_ms=0.0)


def compute_effective_dimensionality(states: List[StateVector]) -> float:
    """Compute effective dimensionality using participation ratio.

    This measures how many scalar dimensions are "active" (not collapsed).
    Before training: ~full dimensionality (isotropic)
    After training: reduced (task-aligned, anisotropic)

    PR = (sum(λ))² / sum(λ²) where λ are eigenvalues of covariance
    """
    if len(states) < 2:
        return float(states[0].dimensionality) if states else 0.0

    # Stack state vectors
    X = np.array([s.to_numpy() for s in states])

    # Center
    X_centered = X - X.mean(axis=0)

    # Covariance eigenvalues
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability

    # Participation ratio
    total = eigenvalues.sum()
    if total < 1e-10:
        return 0.0

    pr = (total ** 2) / (eigenvalues ** 2).sum()
    return float(pr)


def compute_anisotropy(states: List[StateVector]) -> float:
    """Compute anisotropy ratio (largest / smallest non-zero eigenvalue).

    Before training: ~1 (isotropic)
    After training: >> 1 (anisotropic, task-aligned)
    """
    if len(states) < 2:
        return 1.0

    X = np.array([s.to_numpy() for s in states])
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1]

    # Find smallest non-zero eigenvalue
    nonzero = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero) < 2:
        return 1.0

    return float(nonzero[0] / nonzero[-1])
