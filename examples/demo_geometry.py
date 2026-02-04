"""Demo: Visualizing ASPIRE training geometry.

Shows how to capture and visualize the dimensional evolution
of ASPIRE training as vector images.

Key visualizations:
1. Trajectory plot: Path through 2D state space (PCA)
2. Dimensional collapse: Effective dimensionality over time
3. Before/after comparison: How scalars change

The core insight: training induces dimensional collapse from
isotropic (uniform) to anisotropic (task-aligned) structure.
This is the "conscience formation" made visible.

Usage:
    python examples/demo_geometry.py --cycles 100
"""

import argparse
import random
from typing import List

from aspire.core import TrainingItem, StudentResponse, TokenVector, TokenDimension
from aspire.student import MockStudent
from aspire.professors import ProfessorEnsemble
from aspire.critic import LearnedCriticV0
from aspire.geometry import (
    StateVector,
    StateSnapshot,
    TrainingTrajectory,
    TrajectoryMetrics,
    GeometryRenderer,
)


def create_test_items(count: int = 100) -> List[TrainingItem]:
    """Create diverse test items."""
    templates = [
        {
            "prompt": "Should we use microservices or monolith?",
            "gold_answer": "Depends on team size and complexity",
            "difficulty": 0.7,
            "domain": "architecture",
        },
        {
            "prompt": "Is it ethical to use AI for hiring?",
            "gold_answer": "Requires transparency and bias auditing",
            "difficulty": 0.8,
            "domain": "ethics",
        },
        {
            "prompt": "Should passwords be hashed?",
            "gold_answer": "Yes, always hash with salt",
            "difficulty": 0.2,
            "domain": "security",
        },
        {
            "prompt": "SQL or NoSQL for e-commerce?",
            "gold_answer": "SQL for transactions, consider hybrid",
            "difficulty": 0.5,
            "domain": "technical",
        },
    ]

    items = []
    for i in range(count):
        template = random.choice(templates)
        items.append(TrainingItem(
            id=f"{template['domain']}_{i:03d}",
            prompt=template["prompt"],
            gold_answer=template["gold_answer"],
            gold_rationale="See gold answer.",
            difficulty=template["difficulty"],
            domain=template["domain"],
        ))
    return items


def run_training_with_geometry(
    num_cycles: int,
    verbose: bool = False,
) -> TrainingTrajectory:
    """Run training while collecting geometry data."""
    print("=" * 80)
    print("ASPIRE Training Geometry Demo")
    print("=" * 80)
    print("""
This demo captures the dimensional evolution of ASPIRE training.

Key insight from ML geometry research:
- Before training: isotropic, high-dimensional, random
- After training: anisotropic, low-dimensional, task-aligned

Watch for:
- Dimensional collapse (effective dim decreases)
- Anisotropy increase (eigenvalue ratio increases)
- Phase transitions (curvature peaks)

This is "conscience formation" made visible.
""")

    # Create components
    student = MockStudent(correct_rate=0.3)  # Start weak
    professors = ProfessorEnsemble()
    critic = LearnedCriticV0(learning_rate=0.02)

    # Create trajectory collector
    trajectory = TrainingTrajectory(window_size=20)

    # Add initial "untrained" state
    untrained = StateSnapshot.create_untrained()
    trajectory.add_snapshot(untrained)

    print(f"\nRunning {num_cycles} training cycles...")
    print("-" * 80)

    items = create_test_items(num_cycles)
    correct_count = 0

    for i, item in enumerate(items):
        # Generate response
        response = student.generate(item)

        # Critic predicts
        prediction = critic.predict(item, response)

        # Professors evaluate
        evaluation = professors.evaluate(item, response)

        # Update critic
        critic.update(
            prediction,
            evaluation.aggregated_tokens,
            evaluation.disagreement_score,
        )

        # Update student
        from aspire.student import TrainingSignal
        signal = TrainingSignal(
            item=item,
            response=response,
            token_reward=evaluation.aggregated_tokens.total / len(TokenDimension),
            gold_answer=item.gold_answer,
            gold_rationale=item.gold_rationale,
            critiques=[c.critique_text for c in evaluation.critiques],
        )
        student.update(signal)

        # Track accuracy
        if evaluation.consensus_correct:
            correct_count += 1
        accuracy = correct_count / (i + 1)

        # Record state
        # Build simple metrics for demonstration
        snapshot = trajectory.record_from_metrics(
            cycle=i + 1,
            accuracy=accuracy,
            token_ledger=SimpleTokenLedger(evaluation.aggregated_tokens),
            critic_metrics=critic.get_metrics_summary(),
            had_revision=False,
            revision_helped=False,
            timestamp_ms=i * 100,  # Simulated
        )

        if verbose or i % 20 == 0:
            state = snapshot.state
            print(f"[Cycle {i+1:3d}] Accuracy: {accuracy:.1%} | "
                  f"Critic MAE: {state.critic_state.disagreement_mae:.3f} | "
                  f"Tokens: {state.avg_tokens_total:.2f}")

    print("-" * 80)
    print(f"Training complete: {num_cycles} cycles, {accuracy:.1%} final accuracy")

    return trajectory


class SimpleTokenLedger:
    """Simple token ledger for demo purposes."""

    def __init__(self, tokens: TokenVector):
        self._history = {dim: [tokens.values[dim]] for dim in TokenDimension}

    def get_dimension_history(self, dim: TokenDimension) -> List[float]:
        return self._history.get(dim, [0.5])


def analyze_and_visualize(trajectory: TrainingTrajectory):
    """Analyze trajectory and create visualizations."""
    print("\n" + "=" * 80)
    print("GEOMETRY ANALYSIS")
    print("=" * 80)

    # Compute metrics
    metrics = trajectory.compute_metrics()

    print(f"\nPath Geometry:")
    print(f"  Total path length: {metrics.total_path_length:.2f}")
    print(f"  Average step size: {metrics.avg_step_size:.4f}")
    print(f"  Max step size:     {metrics.max_step_size:.4f}")

    print(f"\nCurvature (phase transitions):")
    print(f"  Average curvature: {metrics.avg_curvature:.4f}")
    print(f"  Max curvature:     {metrics.max_curvature:.4f}")
    if metrics.curvature_peaks:
        print(f"  Phase transitions at cycles: {metrics.curvature_peaks[:5]}")

    print(f"\nDimensional Evolution:")
    print(f"  Initial effective dim: {metrics.initial_effective_dim:.1f}")
    print(f"  Final effective dim:   {metrics.final_effective_dim:.1f}")
    print(f"  Collapse ratio:        {metrics.dimensional_collapse_ratio:.2f}")
    print(f"  Initial anisotropy:    {metrics.initial_anisotropy:.1f}")
    print(f"  Final anisotropy:      {metrics.final_anisotropy:.1f}")

    print(f"\nTask Alignment:")
    print(f"  State-accuracy correlation: {metrics.task_alignment:.3f}")

    # Before/after comparison
    comparison = trajectory.get_before_after_comparison()
    if comparison:
        print(f"\n" + "=" * 80)
        print("BEFORE vs AFTER TRAINING")
        print("=" * 80)

        print(f"\nBefore (early cycles):")
        print(f"  Mean accuracy:     {comparison['before']['mean_accuracy']:.1%}")
        print(f"  Effective dim:     {comparison['before']['effective_dim']:.1f}")
        print(f"  Anisotropy:        {comparison['before']['anisotropy']:.1f}")
        print(f"  State norm (mean): {comparison['before']['state_norm_mean']:.3f}")

        print(f"\nAfter (late cycles):")
        print(f"  Mean accuracy:     {comparison['after']['mean_accuracy']:.1%}")
        print(f"  Effective dim:     {comparison['after']['effective_dim']:.1f}")
        print(f"  Anisotropy:        {comparison['after']['anisotropy']:.1f}")
        print(f"  State norm (mean): {comparison['after']['state_norm_mean']:.3f}")

        print(f"\nChange:")
        print(f"  Accuracy change:   {comparison['delta']['accuracy_change']:+.1%}")
        print(f"  Dim collapse:      {comparison['delta']['dim_collapse']:.2f}x")
        print(f"  Anisotropy ratio:  {comparison['delta']['anisotropy_ratio']:.2f}x")

    # Create visualizations
    print(f"\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)

    renderer = GeometryRenderer()

    # Trajectory image
    traj_image = renderer.render_trajectory_pca(trajectory, color_by="accuracy")
    print(f"\nTrajectory (PCA):")
    print(f"  Coordinates shape: {traj_image.coordinates.shape}")
    print(f"  Explained variance: {traj_image.explained_variance[0]:.1%}, {traj_image.explained_variance[1]:.1%}")
    print(f"  Start: ({traj_image.x[0]:.2f}, {traj_image.y[0]:.2f})")
    print(f"  End:   ({traj_image.x[-1]:.2f}, {traj_image.y[-1]:.2f})")
    print(f"  Distance traveled: {((traj_image.x[-1]-traj_image.x[0])**2 + (traj_image.y[-1]-traj_image.y[0])**2)**0.5:.2f}")

    # Vector field
    field_image = renderer.render_vector_field(trajectory, grid_size=10)
    print(f"\nVector Field:")
    print(f"  Grid size: {field_image.vectors.shape[0]}x{field_image.vectors.shape[1]}")
    mean_magnitude = field_image.scalar_field.mean() if field_image.scalar_field is not None else 0
    print(f"  Mean flow magnitude: {mean_magnitude:.4f}")

    # Heatmap
    heatmap = renderer.render_heatmap(trajectory, subsample=5)
    print(f"\nDimension Heatmap:")
    print(f"  Dimensions tracked: {len(heatmap.row_labels)}")
    print(f"  Time points: {heatmap.data.shape[1]}")

    # Print ASCII visualization of trajectory
    print(f"\n" + "=" * 80)
    print("ASCII TRAJECTORY PLOT (PCA projection)")
    print("=" * 80)
    print_ascii_trajectory(traj_image)


def print_ascii_trajectory(traj_image):
    """Print simple ASCII visualization of trajectory."""
    width = 60
    height = 20

    # Normalize coordinates to grid
    x = traj_image.x
    y = traj_image.y

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Add padding
    x_range = max(x_max - x_min, 0.001)
    y_range = max(y_max - y_min, 0.001)

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for i in range(len(x)):
        xi = int((x[i] - x_min) / x_range * (width - 1))
        yi = int((y[i] - y_min) / y_range * (height - 1))
        yi = height - 1 - yi  # Flip y axis

        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))

        # Use different chars for progression
        if i == 0:
            grid[yi][xi] = 'S'  # Start
        elif i == len(x) - 1:
            grid[yi][xi] = 'E'  # End
        elif i % 10 == 0:
            grid[yi][xi] = str((i // 10) % 10)
        else:
            grid[yi][xi] = '.'

    # Print grid
    print('+' + '-' * width + '+')
    for row in grid:
        print('|' + ''.join(row) + '|')
    print('+' + '-' * width + '+')
    print("S=Start, E=End, numbers=every 10 cycles, .=intermediate")


def main():
    parser = argparse.ArgumentParser(description="ASPIRE Training Geometry Demo")
    parser.add_argument("--cycles", type=int, default=100, help="Training cycles")
    parser.add_argument("--verbose", action="store_true", help="Show all cycles")
    args = parser.parse_args()

    trajectory = run_training_with_geometry(args.cycles, args.verbose)
    analyze_and_visualize(trajectory)

    print(f"\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
What the geometry reveals:

1. DIMENSIONAL COLLAPSE
   Effective dimensionality decreases = training is finding structure
   The "active" scalar dimensions shrink from ~full to task-relevant

2. ANISOTROPY INCREASE
   Eigenvalue ratio grows = training creates preferred directions
   Some dimensions become much more important than others

3. PHASE TRANSITIONS
   Curvature peaks = sudden changes in learning direction
   Often correspond to "aha moments" in internalization

4. TRAJECTORY SHAPE
   Start: random, spread (isotropic)
   End: focused, directional (task-aligned)

This is the "conscience" emerging as geometric structure.
The model learns not just WHAT to answer, but HOW to judge.
""")


if __name__ == "__main__":
    main()
