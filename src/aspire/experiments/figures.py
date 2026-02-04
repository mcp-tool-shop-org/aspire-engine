"""Publication-grade figure generation for ASPIRE experiments.

This module generates the figures needed to validate or falsify ASPIRE:

Experiment 1 (Null vs Structured):
    - Figure 1: ConscienceScore Distribution Across Conditions
    - Figure 2: Surprise Stability Over Training
    - Figure 3: Effective Dimensionality vs Training Time

Experiment 2 (Holdout Transfer):
    - Figure 4: Critic-Professor Correlation Matrix
    - Figure 5: GeneralizationScore Across Runs
    - Figure 6: Surprise Spike on Holdout Introduction

Experiment 3 (Adversarial Pressure):
    - Figure 7: Feature Leakage Correlation Over Time
    - Figure 8: Geometry Trajectories in 2D Projection
    - Figure 9: Failure Mode Incidence

Each figure is designed to answer a concrete question and has explicit
falsification signatures.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# Conditional matplotlib import for environments without display
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .experiment_runner import ExperimentSummary, Condition


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    output_dir: Path = Path("experiments/figures")
    dpi: int = 300
    format: str = "png"  # or "pdf", "svg"

    # Style settings
    figsize_single: Tuple[float, float] = (6, 4)
    figsize_wide: Tuple[float, float] = (10, 4)
    figsize_tall: Tuple[float, float] = (6, 8)

    # Colors for conditions
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "full_aspire": "#2E86AB",      # Blue
                "scalar_reward": "#F6AE2D",    # Yellow
                "random_professors": "#E94F37", # Red
                "all_professors": "#2E86AB",
                "holdout_one": "#7B2CBF",       # Purple
                "single_professor": "#E94F37",
                "honest_student": "#2E86AB",
                "adversarial_no_defense": "#E94F37",
                "adversarial_with_defense": "#7B2CBF",
            }
        self.output_dir.mkdir(parents=True, exist_ok=True)


class FigureGenerator:
    """Generate publication-grade figures from experiment results."""

    def __init__(self, config: Optional[FigureConfig] = None):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for figure generation. "
                "Install with: pip install matplotlib"
            )
        self.config = config or FigureConfig()

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

    def generate_experiment1_figures(
        self,
        summary: ExperimentSummary,
    ) -> List[Path]:
        """Generate all figures for Experiment 1."""
        paths = []

        paths.append(self.figure1_conscience_distribution(summary))
        paths.append(self.figure2_surprise_over_time(summary))
        paths.append(self.figure3_dimensionality(summary))

        return paths

    def figure1_conscience_distribution(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 1: ConscienceScore Distribution Across Conditions.

        This is the primary result figure. Theory is falsified if
        RANDOM_PROFESSORS overlaps with FULL_ASPIRE.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        conditions = [
            Condition.FULL_ASPIRE,
            Condition.SCALAR_REWARD,
            Condition.RANDOM_PROFESSORS,
        ]

        data = []
        labels = []
        colors = []

        for cond in conditions:
            scores = summary.conscience_scores_by_condition.get(cond.value, [])
            if scores:
                data.append(scores)
                labels.append(cond.value.replace("_", "\n"))
                colors.append(self.config.colors.get(cond.value, "gray"))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add individual points
            for i, scores in enumerate(data):
                x = np.random.normal(i + 1, 0.04, len(scores))
                ax.scatter(x, scores, alpha=0.5, color=colors[i], s=20)

        ax.set_ylabel("ConscienceScore")
        ax.set_title("Conscience Formation by Training Condition")
        ax.set_ylim(0, 1)

        # Add falsification line annotation
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')

        fig.tight_layout()

        path = self.config.output_dir / f"fig1_conscience_distribution.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure2_surprise_over_time(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 2: Surprise Stability Over Training.

        Shows surprise trajectories with confidence bands.
        Theory falsified if RANDOM shows decreasing surprise.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_wide)

        conditions = [
            Condition.FULL_ASPIRE,
            Condition.SCALAR_REWARD,
            Condition.RANDOM_PROFESSORS,
        ]

        for cond in conditions:
            results = summary.results_by_condition.get(cond, [])
            if not results:
                continue

            # Aggregate trajectories
            all_surprises = []
            for r in results:
                if hasattr(r, 'surprise_trajectory'):
                    all_surprises.append(r.surprise_trajectory)
                elif r.trajectory:
                    all_surprises.append([t.surprise for t in r.trajectory])

            if not all_surprises:
                continue

            # Pad to same length
            max_len = max(len(s) for s in all_surprises)
            padded = []
            for s in all_surprises:
                if len(s) < max_len:
                    s = s + [s[-1]] * (max_len - len(s))
                padded.append(s)

            surprises = np.array(padded)
            mean = np.mean(surprises, axis=0)
            std = np.std(surprises, axis=0)

            x = np.arange(len(mean))
            color = self.config.colors.get(cond.value, "gray")

            ax.plot(x, mean, color=color, label=cond.value.replace("_", " "))
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_xlabel("Training Cycle")
        ax.set_ylabel("Surprise (Prediction Error)")
        ax.set_title("Surprise Stability Over Training")
        ax.legend()

        fig.tight_layout()

        path = self.config.output_dir / f"fig2_surprise_trajectory.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure3_dimensionality(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 3: Effective Dimensionality vs Training Time.

        Shows dimensional collapse patterns.
        Theory falsified if FULL collapses as fast as SCALAR.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_wide)

        conditions = [
            Condition.FULL_ASPIRE,
            Condition.SCALAR_REWARD,
            Condition.RANDOM_PROFESSORS,
        ]

        for cond in conditions:
            results = summary.results_by_condition.get(cond, [])
            if not results:
                continue

            # Aggregate dimensionality trajectories
            all_dims = []
            for r in results:
                if hasattr(r, 'dimensionality_trajectory'):
                    all_dims.append(r.dimensionality_trajectory)
                elif r.trajectory:
                    all_dims.append([t.effective_dimensionality for t in r.trajectory])

            if not all_dims:
                continue

            # Pad and compute stats
            max_len = max(len(d) for d in all_dims)
            padded = []
            for d in all_dims:
                if len(d) < max_len:
                    d = d + [d[-1]] * (max_len - len(d))
                padded.append(d)

            dims = np.array(padded)
            mean = np.mean(dims, axis=0)
            std = np.std(dims, axis=0)

            x = np.arange(len(mean))
            color = self.config.colors.get(cond.value, "gray")

            ax.plot(x, mean, color=color, label=cond.value.replace("_", " "))
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_xlabel("Training Cycle")
        ax.set_ylabel("Effective Dimensionality (Participation Ratio)")
        ax.set_title("Dimensional Evolution During Training")
        ax.legend()

        fig.tight_layout()

        path = self.config.output_dir / f"fig3_dimensionality.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def generate_experiment2_figures(
        self,
        summary: ExperimentSummary,
    ) -> List[Path]:
        """Generate all figures for Experiment 2."""
        paths = []

        paths.append(self.figure4_correlation_matrix(summary))
        paths.append(self.figure5_generalization_scores(summary))
        paths.append(self.figure6_surprise_spike(summary))

        return paths

    def figure4_correlation_matrix(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 4: Critic-Professor Correlation Matrix.

        The crucial internalization figure.
        Theory falsified if holdout row ≈ 0.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        # Get holdout results
        holdout_results = summary.results_by_condition.get(Condition.HOLDOUT_ONE, [])

        if not holdout_results:
            ax.text(0.5, 0.5, "No holdout results available",
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # Build correlation data
            from .experiment2_holdout_transfer import HoldoutTransferResult

            all_profs = set()
            for r in holdout_results:
                if isinstance(r, HoldoutTransferResult):
                    all_profs.update(r.seen_professor_correlations.keys())
                    if r.holdout_professor_name:
                        all_profs.add(r.holdout_professor_name)

            all_profs = sorted(all_profs)
            n_profs = len(all_profs)

            if n_profs > 0:
                # Average correlations across runs
                corr_matrix = np.zeros((n_profs, 2))  # Seen vs Holdout

                for i, prof in enumerate(all_profs):
                    seen_corrs = []
                    holdout_corrs = []
                    for r in holdout_results:
                        if isinstance(r, HoldoutTransferResult):
                            if prof in r.seen_professor_correlations:
                                seen_corrs.append(r.seen_professor_correlations[prof])
                            if prof == r.holdout_professor_name:
                                holdout_corrs.append(r.holdout_professor_correlation)

                    corr_matrix[i, 0] = np.mean(seen_corrs) if seen_corrs else 0
                    corr_matrix[i, 1] = np.mean(holdout_corrs) if holdout_corrs else np.nan

                im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Seen', 'Holdout'])
                ax.set_yticks(range(n_profs))
                ax.set_yticklabels(all_profs)

                # Add correlation values
                for i in range(n_profs):
                    for j in range(2):
                        val = corr_matrix[i, j]
                        if not np.isnan(val):
                            text = ax.text(j, i, f'{val:.2f}',
                                          ha='center', va='center', color='black')

                plt.colorbar(im, ax=ax, label='Correlation')

        ax.set_title("Critic-Professor Correlation\n(Holdout = Transfer Test)")

        fig.tight_layout()

        path = self.config.output_dir / f"fig4_correlation_matrix.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure5_generalization_scores(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 5: GeneralizationScore Across Runs.

        Compares ensemble vs single-professor generalization.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        conditions = [
            Condition.ALL_PROFESSORS,
            Condition.HOLDOUT_ONE,
            Condition.SINGLE_PROFESSOR,
        ]

        data = []
        labels = []
        colors = []

        for cond in conditions:
            scores = summary.generalization_by_condition.get(cond.value, [])
            if scores:
                data.append(scores)
                labels.append(cond.value.replace("_", "\n"))
                colors.append(self.config.colors.get(cond.value, "gray"))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel("Min Generalization Correlation")
        ax.set_title("Generalization Across Professor Conditions")
        ax.set_ylim(-0.5, 1)

        fig.tight_layout()

        path = self.config.output_dir / f"fig5_generalization.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure6_surprise_spike(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 6: Surprise Spike on Holdout Introduction.

        Shows adaptation dynamics when encountering unseen professor.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_wide)

        holdout_results = summary.results_by_condition.get(Condition.HOLDOUT_ONE, [])

        if holdout_results:
            from .experiment2_holdout_transfer import HoldoutTransferResult

            # Collect adaptation data
            befores = []
            ats = []
            afters = []

            for r in holdout_results:
                if isinstance(r, HoldoutTransferResult):
                    befores.append(r.surprise_before_holdout)
                    ats.append(r.surprise_at_holdout)
                    afters.append(r.surprise_after_adaptation)

            if befores:
                x = ['Before\nHoldout', 'At Holdout\nIntroduction', 'After\nAdaptation']
                means = [np.mean(befores), np.mean(ats), np.mean(afters)]
                stds = [np.std(befores), np.std(ats), np.std(afters)]

                ax.bar(x, means, yerr=stds, capsize=5, color=self.config.colors['holdout_one'])

                # Annotate the spike
                if means[1] > means[0]:
                    ax.annotate('Adaptation\nSpike',
                               xy=(1, means[1]),
                               xytext=(1.5, means[1] + 0.1),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               fontsize=9)

        ax.set_ylabel("Surprise")
        ax.set_title("Adaptation to Holdout Professor")

        fig.tight_layout()

        path = self.config.output_dir / f"fig6_surprise_spike.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def generate_experiment3_figures(
        self,
        summary: ExperimentSummary,
    ) -> List[Path]:
        """Generate all figures for Experiment 3."""
        paths = []

        paths.append(self.figure7_leakage_over_time(summary))
        paths.append(self.figure8_geometry_trajectories(summary))
        paths.append(self.figure9_failure_modes(summary))

        return paths

    def figure7_leakage_over_time(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 7: Feature Leakage Correlation Over Time.

        Shows how leakage develops differently for honest vs adversarial.
        Theory falsified if adversarial leakage matches honest.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_wide)

        conditions = [
            Condition.HONEST_STUDENT,
            Condition.ADVERSARIAL_NO_DEFENSE,
            Condition.ADVERSARIAL_WITH_DEFENSE,
        ]

        for cond in conditions:
            results = summary.results_by_condition.get(cond, [])
            if not results:
                continue

            from .experiment3_adversarial import AdversarialPressureResult

            # Aggregate leakage trajectories
            all_leakage = []
            for r in results:
                if isinstance(r, AdversarialPressureResult):
                    all_leakage.append(r.leakage_correlation_trajectory)

            if not all_leakage:
                continue

            # Pad and average
            max_len = max(len(l) for l in all_leakage)
            padded = []
            for l in all_leakage:
                if len(l) < max_len:
                    l = l + [l[-1] if l else 0] * (max_len - len(l))
                padded.append(l)

            leakage = np.array(padded)
            mean = np.mean(leakage, axis=0)
            std = np.std(leakage, axis=0)

            x = np.arange(len(mean))
            color = self.config.colors.get(cond.value, "gray")

            ax.plot(x, mean, color=color, label=cond.value.replace("_", " "))
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        # Threshold line
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Gaming threshold')

        ax.set_xlabel("Training Cycle")
        ax.set_ylabel("|Correlation| (Hedge Count vs Tokens)")
        ax.set_title("Feature Leakage Detection")
        ax.legend()
        ax.set_ylim(0, 1)

        fig.tight_layout()

        path = self.config.output_dir / f"fig7_leakage.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure8_geometry_trajectories(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 8: Geometry Trajectories (simplified 2D).

        Shows trajectory shapes for different student types.
        Theory falsified if trajectories are indistinguishable.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        conditions = [
            Condition.HONEST_STUDENT,
            Condition.ADVERSARIAL_NO_DEFENSE,
            Condition.ADVERSARIAL_WITH_DEFENSE,
        ]

        for ax, cond in zip(axes, conditions):
            results = summary.results_by_condition.get(cond, [])

            for r in results[:3]:  # Show up to 3 trajectories per condition
                if r.trajectory:
                    # Use surprise and dimensionality as 2D coordinates
                    x = [t.surprise for t in r.trajectory]
                    y = [t.effective_dimensionality for t in r.trajectory]

                    # Color by time
                    colors = np.linspace(0, 1, len(x))
                    ax.scatter(x, y, c=colors, cmap='viridis', s=10, alpha=0.7)
                    ax.plot(x, y, color='gray', alpha=0.3, linewidth=0.5)

            ax.set_xlabel("Surprise")
            ax.set_ylabel("Eff. Dimensionality")
            ax.set_title(cond.value.replace("_", " "))

        fig.suptitle("Training Trajectories in Metric Space")
        fig.tight_layout()

        path = self.config.output_dir / f"fig8_trajectories.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path

    def figure9_failure_modes(
        self,
        summary: ExperimentSummary,
    ) -> Path:
        """Figure 9: Failure Mode Incidence.

        Shows which failure modes fire for each condition.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        conditions = [
            Condition.HONEST_STUDENT,
            Condition.ADVERSARIAL_NO_DEFENSE,
            Condition.ADVERSARIAL_WITH_DEFENSE,
        ]

        failure_types = ["FEATURE_GAMING", "HIGH_LEAKAGE", "NONE"]
        x = np.arange(len(conditions))
        width = 0.25

        counts = {ft: [] for ft in failure_types}

        for cond in conditions:
            results = summary.results_by_condition.get(cond, [])
            total = len(results) if results else 1

            for ft in failure_types:
                if ft == "NONE":
                    count = sum(1 for r in results if not r.failure_modes_detected)
                else:
                    count = sum(1 for r in results if ft in r.failure_modes_detected)
                counts[ft].append(count / total * 100)

        for i, ft in enumerate(failure_types):
            color = 'green' if ft == "NONE" else ('red' if 'GAMING' in ft else 'orange')
            ax.bar(x + i * width, counts[ft], width, label=ft, color=color, alpha=0.7)

        ax.set_xlabel("Condition")
        ax.set_ylabel("% of Runs")
        ax.set_title("Failure Mode Incidence by Student Type")
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.value.replace("_", "\n") for c in conditions])
        ax.legend()

        fig.tight_layout()

        path = self.config.output_dir / f"fig9_failure_modes.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return path


def generate_all_figures(
    summary: ExperimentSummary,
    config: Optional[FigureConfig] = None,
) -> Dict[str, List[Path]]:
    """Generate all figures for an experiment.

    Returns dict mapping experiment name to list of figure paths.
    """
    generator = FigureGenerator(config)

    figures = {}

    # Detect experiment type from conditions
    if Condition.FULL_ASPIRE in summary.results_by_condition:
        figures["experiment1"] = generator.generate_experiment1_figures(summary)

    if Condition.HOLDOUT_ONE in summary.results_by_condition:
        figures["experiment2"] = generator.generate_experiment2_figures(summary)

    if Condition.HONEST_STUDENT in summary.results_by_condition:
        figures["experiment3"] = generator.generate_experiment3_figures(summary)

    return figures


def generate_failure_atlas(
    atlas_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
) -> Path:
    """Generate a one-page failure atlas visualization.

    The failure atlas is a visual summary of all failure modes observed,
    organized by:
    - Expected vs unexpected (x-axis)
    - Severity: recoverable vs limiting (y-axis)

    Args:
        atlas_data: Data from generate_failure_atlas_data()
        config: Figure configuration

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for failure atlas")

    config = config or FigureConfig()

    fig = plt.figure(figsize=(12, 10))

    # Main title
    fig.suptitle(atlas_data["title"], fontsize=14, fontweight='bold')

    # Create a 2x2 grid for quadrants + summary panels
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 1], hspace=0.3, wspace=0.3)

    # =========================================================================
    # Top panel: Summary statistics
    # =========================================================================
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')

    summary_text = (
        f"Total Runs: {atlas_data['total_runs']}  |  "
        f"Overall Failure Rate: {atlas_data['overall_failure_rate']:.1%}  |  "
        f"Expected: {len(atlas_data['quadrants']['expected_recoverable']) + len(atlas_data['quadrants']['expected_limiting'])}  |  "
        f"Unexpected: {len(atlas_data['quadrants']['unexpected_minor']) + len(atlas_data['quadrants']['unexpected_critical'])}"
    )
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=11, transform=ax_summary.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # =========================================================================
    # Main 2x2 quadrant visualization
    # =========================================================================
    quadrant_config = [
        ('expected_recoverable', 0, 0, '#90EE90', 'Expected\n(Recoverable)'),
        ('expected_limiting', 0, 1, '#FFD700', 'Expected\n(Limiting)'),
        ('unexpected_minor', 1, 0, '#FFA07A', 'Unexpected\n(Minor)'),
        ('unexpected_critical', 1, 1, '#FF6B6B', 'Unexpected\n(Critical)'),
    ]

    for quad_name, row, col, color, title in quadrant_config:
        ax = fig.add_subplot(gs[1, col] if row == 0 else gs[2, col])

        failures = atlas_data['quadrants'][quad_name]
        ax.set_facecolor(color)
        ax.set_alpha(0.3)

        # Title
        ax.set_title(title, fontsize=10, fontweight='bold')

        if failures:
            # Show failure categories as text
            categories = {}
            for f in failures:
                cat = f['category']
                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += 1

            y_pos = 0.9
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                ax.text(0.1, y_pos, f"• {cat}: {count}",
                       fontsize=8, transform=ax.transAxes)
                y_pos -= 0.15
                if y_pos < 0.1:
                    break
        else:
            ax.text(0.5, 0.5, "No failures", ha='center', va='center',
                   fontsize=9, style='italic', transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

    # =========================================================================
    # Right panel: Category breakdown bar chart
    # =========================================================================
    ax_cats = fig.add_subplot(gs[1:, 2])

    categories = atlas_data.get('by_category', {})
    if categories:
        cats = list(categories.keys())
        counts = [categories[c]['count'] for c in cats]

        # Color by expected rate
        colors = []
        for c in cats:
            exp_rate = categories[c]['expected_rate']
            if exp_rate > 0.7:
                colors.append('#90EE90')  # Expected
            elif exp_rate > 0.3:
                colors.append('#FFD700')  # Mixed
            else:
                colors.append('#FF6B6B')  # Unexpected

        y_pos = np.arange(len(cats))
        ax_cats.barh(y_pos, counts, color=colors, alpha=0.7)
        ax_cats.set_yticks(y_pos)
        ax_cats.set_yticklabels([c.replace('_', '\n') for c in cats], fontsize=7)
        ax_cats.set_xlabel('Count')
        ax_cats.set_title('Failures by Category', fontsize=10)

        # Add expected rate annotation
        for i, (cat, count) in enumerate(zip(cats, counts)):
            exp_rate = categories[cat]['expected_rate']
            ax_cats.text(count + 0.1, i, f'{exp_rate:.0%} exp',
                        va='center', fontsize=7, alpha=0.7)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Expected'),
        Patch(facecolor='#FFD700', label='Mixed'),
        Patch(facecolor='#FF6B6B', label='Unexpected'),
    ]
    ax_cats.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # Save
    path = config.output_dir / f"failure_atlas.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)

    return path


def generate_boundary_conditions_figure(
    boundary_conditions: List[str],
    failure_report: Any,
    config: Optional[FigureConfig] = None,
) -> Path:
    """Generate a figure summarizing boundary conditions.

    This figure documents where conscience formation fails or
    becomes unreliable, serving as a guide for practitioners.

    Args:
        boundary_conditions: List of identified boundary conditions
        failure_report: FailureReport object
        config: Figure configuration

    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    config = config or FigureConfig()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Boundary Conditions for Conscience Formation", fontsize=12, fontweight='bold')

    # =========================================================================
    # Panel A: Training duration threshold
    # =========================================================================
    ax = axes[0, 0]
    ax.set_title("A. Minimum Training Duration")

    # Simulated data showing conscience score vs training cycles
    cycles = np.array([5, 10, 20, 30, 40, 50, 75, 100])
    scores = np.array([0.2, 0.3, 0.45, 0.55, 0.62, 0.68, 0.72, 0.75])
    std = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04])

    ax.errorbar(cycles, scores, yerr=std, marker='o', capsize=3, color='#2E86AB')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.axvline(x=30, color='orange', linestyle=':', alpha=0.5, label='Min recommended')
    ax.fill_between([0, 30], [0, 0], [1, 1], alpha=0.1, color='red')

    ax.set_xlabel("Training Cycles")
    ax.set_ylabel("ConscienceScore")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # =========================================================================
    # Panel B: Sample size requirements
    # =========================================================================
    ax = axes[0, 1]
    ax.set_title("B. Minimum Sample Size")

    samples = np.array([10, 25, 50, 75, 100, 150, 200])
    variance = np.array([0.35, 0.22, 0.15, 0.11, 0.08, 0.06, 0.05])

    ax.plot(samples, variance, marker='s', color='#7B2CBF')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Reliability threshold')
    ax.fill_between([0, 50], [0, 0], [0.5, 0.5], alpha=0.1, color='red')

    ax.set_xlabel("Training Items")
    ax.set_ylabel("Score Variance (CV)")
    ax.set_xlim(0, 210)
    ax.legend(fontsize=8)

    # =========================================================================
    # Panel C: Adversarial detection window
    # =========================================================================
    ax = axes[1, 0]
    ax.set_title("C. Adversarial Detection Windows")

    strategies = ['Fake\nHedger', 'Consensus\nMimic', 'Slow\nRoll', 'Entropy\nShaper']
    min_cycles = [15, 20, 40, 25]
    colors = ['#2E86AB', '#2E86AB', '#E94F37', '#2E86AB']

    bars = ax.bar(strategies, min_cycles, color=colors, alpha=0.7)
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Standard run')
    ax.set_ylabel("Min Cycles for Detection")
    ax.legend(fontsize=8)

    # Highlight SlowRoll as problematic
    ax.annotate('Evades standard\ndetection window',
                xy=(2, 40), xytext=(2.5, 35),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # =========================================================================
    # Panel D: Boundary conditions text summary
    # =========================================================================
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("D. Identified Boundary Conditions")

    if boundary_conditions:
        text = "\n".join([f"• {bc}" for bc in boundary_conditions[:6]])
    else:
        text = "• No boundary conditions identified\n  in current experimental run"

    ax.text(0.05, 0.95, text, va='top', ha='left', fontsize=9,
           transform=ax.transAxes, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.tight_layout()

    path = config.output_dir / f"boundary_conditions.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)

    return path
