#!/usr/bin/env python
"""Run the full ASPIRE falsification experiment suite.

This script runs all three experiments with publication-grade settings,
tracks failures systematically, and generates all figures including
the failure atlas.

Usage:
    python scripts/run_full_experiments.py [--quick]

Options:
    --quick    Run with reduced settings for testing (default: full run)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aspire.experiments import (
    ExperimentConfig,
    NullVsStructuredExperiment,
    HoldoutTransferExperiment,
    AdversarialPressureExperiment,
    Condition,
    FailureTracker,
    generate_failure_atlas_data,
)
# Figure generation is optional (requires matplotlib)
try:
    from aspire.experiments.figures import (
        FigureGenerator,
        FigureConfig,
        generate_failure_atlas,
        generate_boundary_conditions_figure,
    )
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping figure generation")


def run_experiment_with_tracking(experiment_class, config, tracker):
    """Run an experiment and track failures."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    exp = experiment_class(config)
    summary = exp.run()

    # Track failures from each result
    for condition, results in summary.results_by_condition.items():
        for result in results:
            tracker.track_run(
                condition=condition.value,
                run_idx=result.run_idx,
                seed=result.seed,
                result=result,
                config=config,
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ASPIRE experiments")
    parser.add_argument("--quick", action="store_true",
                       help="Run with reduced settings for testing")
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Configuration based on mode
    if args.quick:
        print("\n*** QUICK MODE: Reduced settings for testing ***\n")
        base_config = {
            "n_training_items": 30,
            "n_training_cycles": 20,
            "n_runs_per_condition": 3,
        }
    else:
        print("\n*** FULL MODE: Publication-grade settings ***\n")
        base_config = {
            "n_training_items": 100,
            "n_training_cycles": 50,
            "n_runs_per_condition": 10,
        }

    # Initialize failure tracker
    tracker = FailureTracker("aspire_full_suite")

    # =========================================================================
    # Experiment 1: Null vs Structured Judgment
    # =========================================================================
    config1 = ExperimentConfig(
        name="exp1_null_vs_structured",
        description="Does conscience emerge only under meaningful evaluation?",
        output_dir=output_dir / "exp1",
        **base_config,
    )
    summary1 = run_experiment_with_tracking(
        NullVsStructuredExperiment, config1, tracker
    )

    # Analyze comparisons for falsification
    full_results = summary1.results_by_condition.get(Condition.FULL_ASPIRE, [])
    random_results = summary1.results_by_condition.get(Condition.RANDOM_PROFESSORS, [])
    tracker.analyze_comparison(full_results, random_results, "random_vs_full")

    # =========================================================================
    # Experiment 2: Holdout Transfer
    # =========================================================================
    config2 = ExperimentConfig(
        name="exp2_holdout_transfer",
        description="Does the student learn to judge, or just who trained it?",
        output_dir=output_dir / "exp2",
        **base_config,
    )
    summary2 = run_experiment_with_tracking(
        HoldoutTransferExperiment, config2, tracker
    )

    # =========================================================================
    # Experiment 3: Adversarial Pressure
    # =========================================================================
    config3 = ExperimentConfig(
        name="exp3_adversarial",
        description="Does ASPIRE resist actively deceptive students?",
        output_dir=output_dir / "exp3",
        **base_config,
    )
    summary3 = run_experiment_with_tracking(
        AdversarialPressureExperiment, config3, tracker
    )

    # Analyze honest vs adversarial
    honest_results = summary3.results_by_condition.get(Condition.HONEST_STUDENT, [])
    adversarial_results = summary3.results_by_condition.get(
        Condition.ADVERSARIAL_NO_DEFENSE, []
    )
    tracker.analyze_comparison(honest_results, adversarial_results, "adversarial_vs_honest")

    # =========================================================================
    # Analyze variance across all experiments
    # =========================================================================
    all_results = {}
    for summary in [summary1, summary2, summary3]:
        for cond, results in summary.results_by_condition.items():
            key = cond.value
            if key not in all_results:
                all_results[key] = []
            all_results[key].extend(results)

    tracker.analyze_variance(all_results)

    # Identify boundary conditions
    tracker.identify_boundary_conditions(config1)

    # =========================================================================
    # Generate figures (if matplotlib available)
    # =========================================================================
    if HAS_MATPLOTLIB:
        print(f"\n{'='*60}")
        print("Generating figures...")
        print(f"{'='*60}")

        fig_config = FigureConfig(output_dir=output_dir / "figures")
        generator = FigureGenerator(fig_config)

        # Experiment-specific figures
        print("  Experiment 1 figures...")
        exp1_figs = generator.generate_experiment1_figures(summary1)
        for f in exp1_figs:
            print(f"    - {f.name}")

        print("  Experiment 2 figures...")
        exp2_figs = generator.generate_experiment2_figures(summary2)
        for f in exp2_figs:
            print(f"    - {f.name}")

        print("  Experiment 3 figures...")
        exp3_figs = generator.generate_experiment3_figures(summary3)
        for f in exp3_figs:
            print(f"    - {f.name}")

        # Failure atlas
        print("  Failure atlas...")
        atlas_data = generate_failure_atlas_data(tracker)
        atlas_path = generate_failure_atlas(atlas_data, fig_config)
        print(f"    - {atlas_path.name}")

        # Boundary conditions figure
        print("  Boundary conditions figure...")
        report = tracker.get_report()
        bc_path = generate_boundary_conditions_figure(
            report.boundary_conditions, report, fig_config
        )
        print(f"    - {bc_path.name}")
    else:
        print(f"\n{'='*60}")
        print("Skipping figure generation (matplotlib not available)")
        print(f"{'='*60}")
        # Still generate atlas data for JSON output
        atlas_data = generate_failure_atlas_data(tracker)
        import json
        with open(output_dir / "failure_atlas_data.json", "w") as f:
            json.dump(atlas_data, f, indent=2)
        print("  - failure_atlas_data.json (raw data)")

    # Get report for later use
    report = tracker.get_report()

    # =========================================================================
    # Save failure report
    # =========================================================================
    print(f"\n{'='*60}")
    print("Saving failure report...")
    print(f"{'='*60}")

    tracker.save_report(output_dir / "failure_report.json")

    # Generate negative results section
    negative_results = tracker.generate_negative_results_section()
    with open(output_dir / "negative_results.md", "w") as f:
        f.write(negative_results)
    print(f"  - negative_results.md")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    stats = report.compute_statistics()
    print(f"\nTotal runs: {report.total_runs}")
    print(f"Total failures: {stats['total_failures']}")
    print(f"  - Expected: {stats['expected_failures']}")
    print(f"  - Unexpected: {stats['unexpected_failures']}")
    print(f"  - Falsifications: {stats['falsifications']}")
    print(f"Overall failure rate: {stats['failure_rate']:.1%}")

    print("\nFalsification checks:")
    for exp_name, summary in [("Exp1", summary1), ("Exp2", summary2), ("Exp3", summary3)]:
        print(f"\n  {exp_name}:")
        for check, passed in summary.falsification_results.items():
            status = "PASS" if passed else "*** FAIL ***"
            print(f"    {check}: {status}")
            if check in summary.falsification_details:
                print(f"      {summary.falsification_details[check]}")

    if report.boundary_conditions:
        print("\nBoundary conditions identified:")
        for bc in report.boundary_conditions:
            print(f"  - {bc}")

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")

    return 0 if stats['falsifications'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
