"""Training trajectory through ASPIRE's state space.

A trajectory is a sequence of StateVectors over training, representing
the path the system takes through its scalar manifold.

Key metrics:
- Path length: Total distance traveled in state space
- Curvature: How sharply the trajectory bends (phase transitions)
- Dimensional collapse: Reduction in effective dimensionality
- Alignment: How much the trajectory aligns with task structure
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np

from .state import (
    StateVector,
    StateSnapshot,
    compute_effective_dimensionality,
    compute_anisotropy,
)


@dataclass
class TrajectoryMetrics:
    """Metrics computed over a training trajectory."""
    # Path geometry
    total_path_length: float = 0.0
    avg_step_size: float = 0.0
    max_step_size: float = 0.0

    # Curvature (phase transitions)
    avg_curvature: float = 0.0
    max_curvature: float = 0.0
    curvature_peaks: List[int] = field(default_factory=list)  # Cycle indices

    # Dimensional evolution
    initial_effective_dim: float = 0.0
    final_effective_dim: float = 0.0
    dimensional_collapse_ratio: float = 1.0

    # Anisotropy evolution
    initial_anisotropy: float = 1.0
    final_anisotropy: float = 1.0

    # Task alignment (accuracy correlation with state)
    task_alignment: float = 0.0


class TrainingTrajectory:
    """A trajectory through ASPIRE's training manifold.

    Collects StateSnapshots and computes geometric properties
    that reveal how training reshapes the scalar structure.

    Usage:
        trajectory = TrainingTrajectory()

        for cycle_result in training:
            snapshot = trajectory.record(engine, cycle_result)

        metrics = trajectory.compute_metrics()
        trajectory.save("training_geometry.npz")
    """

    def __init__(self, window_size: int = 50):
        """Initialize trajectory collector.

        Args:
            window_size: Window for computing rolling statistics
        """
        self.window_size = window_size
        self.snapshots: List[StateSnapshot] = []
        self._vectors_cache: Optional[np.ndarray] = None

    def add_snapshot(self, snapshot: StateSnapshot):
        """Add a state snapshot to the trajectory."""
        self.snapshots.append(snapshot)
        self._vectors_cache = None  # Invalidate cache

    def record_from_metrics(
        self,
        cycle: int,
        accuracy: float,
        token_ledger,
        critic_metrics: dict,
        revision_metrics: Optional[dict] = None,
        had_revision: bool = False,
        revision_helped: bool = False,
        timestamp_ms: float = 0.0,
    ) -> StateSnapshot:
        """Record a snapshot from training metrics.

        Args:
            cycle: Current training cycle
            accuracy: Current accuracy
            token_ledger: TokenLedger from engine
            critic_metrics: Metrics from critic.get_metrics_summary()
            revision_metrics: Optional revision metrics
            had_revision: Whether revision occurred this cycle
            revision_helped: Whether revision improved tokens

        Returns:
            The created snapshot
        """
        from .state import TokenDimensionStats, CriticState, RevisionState
        from ..core import TokenDimension

        # Build token stats from ledger
        token_stats = {}
        for dim in TokenDimension:
            history = token_ledger.get_dimension_history(dim)
            if history:
                recent = history[-self.window_size:] if len(history) > self.window_size else history
                token_stats[dim] = TokenDimensionStats(
                    dimension=dim,
                    mean=float(np.mean(recent)),
                    std=float(np.std(recent)),
                    min_val=float(np.min(recent)),
                    max_val=float(np.max(recent)),
                    prediction_mae=critic_metrics.get("mae_per_dim", {}).get(dim.value, 0.5),
                    sensitivity=float(np.std(recent)),  # Use std as proxy for sensitivity
                )
            else:
                token_stats[dim] = TokenDimensionStats(dimension=dim)

        # Build critic state
        critic_state = CriticState(
            total_predictions=critic_metrics.get("total_predictions", 0),
            token_mae=critic_metrics.get("mae_per_dim", {}),
            disagreement_mae=critic_metrics.get("disagreement_mae", 0.5),
            surprise_mean=critic_metrics.get("avg_surprise", 0.5) if "avg_surprise" in critic_metrics else 0.5,
            negative_surprise_rate=critic_metrics.get("negative_surprise_rate", 0.5),
        )

        # Build revision state
        if revision_metrics:
            revision_state = RevisionState(
                revision_rate=revision_metrics.get("revision_rate", 0.0),
                avg_uplift=revision_metrics.get("avg_uplift", 0.0),
                positive_uplift_rate=revision_metrics.get("positive_uplift_rate", 0.0),
                trigger_rates=revision_metrics.get("trigger_rates", {}),
            )
        else:
            revision_state = RevisionState()

        # Compute aggregates
        avg_tokens_total = sum(
            token_stats[dim].mean for dim in TokenDimension
        )

        state = StateVector(
            cycle=cycle,
            accuracy=accuracy,
            token_stats=token_stats,
            critic_state=critic_state,
            revision_state=revision_state,
            avg_tokens_total=avg_tokens_total,
        )

        # Compute deltas from previous
        delta_accuracy = 0.0
        delta_tokens = 0.0
        if self.snapshots:
            prev = self.snapshots[-1].state
            delta_accuracy = accuracy - prev.accuracy
            delta_tokens = avg_tokens_total - prev.avg_tokens_total

        snapshot = StateSnapshot(
            state=state,
            timestamp_ms=timestamp_ms,
            delta_accuracy=delta_accuracy,
            delta_tokens=delta_tokens,
            had_revision=had_revision,
            revision_helped=revision_helped,
        )

        self.add_snapshot(snapshot)
        return snapshot

    @property
    def vectors(self) -> np.ndarray:
        """Get stacked state vectors as numpy array."""
        if self._vectors_cache is None:
            self._vectors_cache = np.array([
                s.state.to_numpy() for s in self.snapshots
            ])
        return self._vectors_cache

    def compute_path_length(self) -> float:
        """Compute total path length through state space."""
        if len(self.snapshots) < 2:
            return 0.0

        vectors = self.vectors
        deltas = np.diff(vectors, axis=0)
        step_sizes = np.linalg.norm(deltas, axis=1)
        return float(step_sizes.sum())

    def compute_step_sizes(self) -> np.ndarray:
        """Compute step sizes between consecutive states."""
        if len(self.snapshots) < 2:
            return np.array([])

        vectors = self.vectors
        deltas = np.diff(vectors, axis=0)
        return np.linalg.norm(deltas, axis=1)

    def compute_curvature(self) -> np.ndarray:
        """Compute local curvature at each point.

        Curvature is computed as the angle between consecutive
        velocity vectors. High curvature indicates phase transitions.
        """
        if len(self.snapshots) < 3:
            return np.array([])

        vectors = self.vectors
        velocities = np.diff(vectors, axis=0)

        # Normalize velocities
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        velocities_normalized = velocities / norms

        # Compute angles between consecutive velocities
        curvatures = []
        for i in range(len(velocities_normalized) - 1):
            v1 = velocities_normalized[i]
            v2 = velocities_normalized[i + 1]
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot)
            curvatures.append(angle)

        return np.array(curvatures)

    def find_phase_transitions(self, threshold: float = 0.5) -> List[int]:
        """Find cycles where curvature exceeds threshold.

        These indicate potential phase transitions in training.
        """
        curvatures = self.compute_curvature()
        if len(curvatures) == 0:
            return []

        peaks = []
        for i, c in enumerate(curvatures):
            if c > threshold:
                # Return the cycle index (offset by 1 due to diff)
                peaks.append(i + 1)

        return peaks

    def compute_dimensional_evolution(
        self,
        window: int = 20,
    ) -> Tuple[List[float], List[float]]:
        """Compute effective dimensionality and anisotropy over training.

        Args:
            window: Window size for computing local statistics

        Returns:
            (effective_dims, anisotropies) lists over time
        """
        if len(self.snapshots) < window:
            return [], []

        states = [s.state for s in self.snapshots]
        effective_dims = []
        anisotropies = []

        for i in range(window, len(states) + 1):
            window_states = states[i - window:i]
            ed = compute_effective_dimensionality(window_states)
            an = compute_anisotropy(window_states)
            effective_dims.append(ed)
            anisotropies.append(an)

        return effective_dims, anisotropies

    def compute_metrics(self) -> TrajectoryMetrics:
        """Compute all trajectory metrics."""
        metrics = TrajectoryMetrics()

        if len(self.snapshots) < 2:
            return metrics

        # Path geometry
        step_sizes = self.compute_step_sizes()
        metrics.total_path_length = float(step_sizes.sum())
        metrics.avg_step_size = float(step_sizes.mean())
        metrics.max_step_size = float(step_sizes.max())

        # Curvature
        curvatures = self.compute_curvature()
        if len(curvatures) > 0:
            metrics.avg_curvature = float(curvatures.mean())
            metrics.max_curvature = float(curvatures.max())
            metrics.curvature_peaks = self.find_phase_transitions(threshold=np.pi / 4)

        # Dimensional evolution
        effective_dims, anisotropies = self.compute_dimensional_evolution()
        if effective_dims:
            metrics.initial_effective_dim = effective_dims[0]
            metrics.final_effective_dim = effective_dims[-1]
            if metrics.initial_effective_dim > 0:
                metrics.dimensional_collapse_ratio = (
                    metrics.final_effective_dim / metrics.initial_effective_dim
                )
            metrics.initial_anisotropy = anisotropies[0]
            metrics.final_anisotropy = anisotropies[-1]

        # Task alignment (correlation of state norm with accuracy)
        accuracies = np.array([s.state.accuracy for s in self.snapshots])
        state_norms = np.linalg.norm(self.vectors, axis=1)
        if len(accuracies) > 1 and np.std(accuracies) > 0 and np.std(state_norms) > 0:
            metrics.task_alignment = float(np.corrcoef(accuracies, state_norms)[0, 1])

        return metrics

    def save(self, path: str):
        """Save trajectory to numpy archive."""
        np.savez(
            path,
            vectors=self.vectors,
            cycles=np.array([s.state.cycle for s in self.snapshots]),
            accuracies=np.array([s.state.accuracy for s in self.snapshots]),
            timestamps=np.array([s.timestamp_ms for s in self.snapshots]),
            had_revisions=np.array([s.had_revision for s in self.snapshots]),
            revision_helped=np.array([s.revision_helped for s in self.snapshots]),
        )

    @classmethod
    def load(cls, path: str) -> "TrainingTrajectory":
        """Load trajectory from numpy archive."""
        data = np.load(path)
        trajectory = cls()

        # Reconstruct snapshots (simplified - just vectors and metadata)
        for i in range(len(data["cycles"])):
            state = StateVector(
                cycle=int(data["cycles"][i]),
                accuracy=float(data["accuracies"][i]),
            )
            snapshot = StateSnapshot(
                state=state,
                timestamp_ms=float(data["timestamps"][i]),
                had_revision=bool(data["had_revisions"][i]),
                revision_helped=bool(data["revision_helped"][i]),
            )
            trajectory.snapshots.append(snapshot)

        # Cache the original vectors
        trajectory._vectors_cache = data["vectors"]

        return trajectory

    def get_before_after_comparison(
        self,
        early_cycles: int = 20,
        late_cycles: int = 20,
    ) -> dict:
        """Compare early vs late training statistics.

        Returns dictionary with before/after comparisons for key metrics.
        """
        if len(self.snapshots) < early_cycles + late_cycles:
            return {}

        early_states = [s.state for s in self.snapshots[:early_cycles]]
        late_states = [s.state for s in self.snapshots[-late_cycles:]]

        early_vecs = np.array([s.to_numpy() for s in early_states])
        late_vecs = np.array([s.to_numpy() for s in late_states])

        return {
            "before": {
                "mean_accuracy": float(np.mean([s.accuracy for s in early_states])),
                "effective_dim": compute_effective_dimensionality(early_states),
                "anisotropy": compute_anisotropy(early_states),
                "state_norm_mean": float(np.linalg.norm(early_vecs, axis=1).mean()),
                "state_norm_std": float(np.linalg.norm(early_vecs, axis=1).std()),
            },
            "after": {
                "mean_accuracy": float(np.mean([s.accuracy for s in late_states])),
                "effective_dim": compute_effective_dimensionality(late_states),
                "anisotropy": compute_anisotropy(late_states),
                "state_norm_mean": float(np.linalg.norm(late_vecs, axis=1).mean()),
                "state_norm_std": float(np.linalg.norm(late_vecs, axis=1).std()),
            },
            "delta": {
                "accuracy_change": float(
                    np.mean([s.accuracy for s in late_states]) -
                    np.mean([s.accuracy for s in early_states])
                ),
                "dim_collapse": (
                    compute_effective_dimensionality(late_states) /
                    max(compute_effective_dimensionality(early_states), 1e-6)
                ),
                "anisotropy_ratio": (
                    compute_anisotropy(late_states) /
                    max(compute_anisotropy(early_states), 1e-6)
                ),
            },
        }
