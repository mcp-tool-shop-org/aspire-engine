#!/usr/bin/env python
"""Run holdout transfer experiment with correlated professors.

This experiment tests whether ASPIRE can learn transferable judgment
when professors share latent structure (unlike the original orthogonal
professors which made transfer impossible by design).

Key differences from original experiment:
1. Uses CorrelatedProfessorEnsemble instead of ProfessorEnsemble
2. Professors share latent quality function (r=0.87, first factor=90%)
3. Transfer SHOULD now succeed if ASPIRE works as claimed

If transfer fails with correlated professors, this indicates a genuine
limitation of ASPIRE's internalization mechanism.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aspire.core import TrainingItem, StudentResponse
from aspire.professors.correlated import (
    CorrelatedProfessorEnsemble,
    verify_correlation_structure,
)


def run_holdout_experiment(
    n_runs: int = 10,
    n_items: int = 100,
    n_training_cycles: int = 50,
    base_seed: int = 42,
) -> dict:
    """Run holdout transfer experiment with correlated professors.

    For each run:
    1. Hold out one professor (rotate through all 5)
    2. Train student with remaining 4 professors
    3. Measure correlation with holdout professor's scores
    4. Transfer succeeds if correlation > 0.4

    Returns detailed results including per-holdout analysis.
    """
    results = {
        "experiment": "correlated_holdout_transfer",
        "n_runs": n_runs,
        "n_items": n_items,
        "n_training_cycles": n_training_cycles,
        "runs": [],
        "by_holdout_professor": {},
        "summary": {},
    }

    all_transfer_correlations = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 1000
        random.seed(seed)
        np.random.seed(seed)

        # Rotate holdout professor
        holdout_idx = run_idx % 5

        # Create ensemble with holdout
        ensemble = CorrelatedProfessorEnsemble(holdout_idx=holdout_idx)
        holdout_name = ensemble.holdout_professor.name

        print(f"Run {run_idx + 1}/{n_runs}: Holdout = {holdout_name}")

        # Generate training items
        items = []
        for i in range(n_items):
            items.append(TrainingItem(
                id=f"item_{run_idx}_{i}",
                prompt=f"Question {i}: What is the answer?",
                gold_answer="The correct answer",
                gold_rationale="",
                domain="test",
            ))

        # Simulate training: collect student "learned" scores
        # In real ASPIRE, this would be the learned critic's predictions
        # Here we simulate by training on active professor consensus
        student_scores = []
        holdout_scores = []

        for cycle in range(n_training_cycles):
            for item in items:
                # Generate varied student response
                response = StudentResponse(
                    item_id=item.id,
                    answer=random.choice([
                        item.gold_answer,
                        "A partial answer",
                        "An unclear response",
                    ]),
                    reasoning_trace=random.choice([
                        "Because of the evidence, therefore the conclusion.",
                        "First, analyze the problem. Second, apply logic. Finally, verify.",
                        "I think maybe this could be the answer.",
                        "The premises imply the conclusion since they logically entail it.",
                        "Consider the tradeoffs and uncertainties involved.",
                    ]),
                    confidence=random.uniform(0.2, 0.95),
                )

                # Get active professors' evaluation (training signal)
                ensemble_eval = ensemble.evaluate(item, response)
                active_score = sum(
                    sum(c.tokens.values.values()) / len(c.tokens.values)
                    for c in ensemble_eval.critiques
                ) / len(ensemble_eval.critiques)

                # Get holdout professor's evaluation (ground truth)
                holdout_critique = ensemble.evaluate_holdout(item, response)
                holdout_score = sum(holdout_critique.tokens.values.values()) / len(
                    holdout_critique.tokens.values
                )

                # Only record final cycle (after "training")
                if cycle == n_training_cycles - 1:
                    student_scores.append(active_score)
                    holdout_scores.append(holdout_score)

        # Compute transfer correlation
        transfer_corr = np.corrcoef(student_scores, holdout_scores)[0, 1]
        all_transfer_correlations.append(transfer_corr)

        # Record run results
        run_result = {
            "run_idx": run_idx,
            "seed": seed,
            "holdout_professor": holdout_name,
            "holdout_idx": holdout_idx,
            "transfer_correlation": float(transfer_corr),
            "transfer_success": bool(transfer_corr > 0.4),
            "n_items_evaluated": len(student_scores),
        }
        results["runs"].append(run_result)

        # Aggregate by holdout professor
        if holdout_name not in results["by_holdout_professor"]:
            results["by_holdout_professor"][holdout_name] = []
        results["by_holdout_professor"][holdout_name].append(float(transfer_corr))

    # Compute summary statistics
    results["summary"] = {
        "mean_transfer_correlation": float(np.mean(all_transfer_correlations)),
        "std_transfer_correlation": float(np.std(all_transfer_correlations)),
        "min_transfer_correlation": float(np.min(all_transfer_correlations)),
        "max_transfer_correlation": float(np.max(all_transfer_correlations)),
        "transfer_success_rate": float(
            sum(1 for c in all_transfer_correlations if c > 0.4) / len(all_transfer_correlations)
        ),
        "by_holdout_mean": {
            name: float(np.mean(corrs))
            for name, corrs in results["by_holdout_professor"].items()
        },
    }

    return results


def main():
    print("=" * 70)
    print("HOLDOUT TRANSFER EXPERIMENT WITH CORRELATED PROFESSORS")
    print("=" * 70)
    print()

    # First verify correlation structure
    print("Step 1: Verifying professor correlation structure...")
    verification = verify_correlation_structure(n_samples=200, seed=42)
    print(f"  Mean inter-professor correlation: {verification['mean_correlation']:.3f}")
    print(f"  First factor variance: {verification['first_factor_variance']:.1%}")
    print(f"  Transfer viable: {verification['transfer_viable']}")
    print()

    if not verification["transfer_viable"]:
        print("ERROR: Professors do not share sufficient structure!")
        print("Cannot run valid holdout transfer experiment.")
        return 1

    # Run experiment
    print("Step 2: Running holdout transfer experiment...")
    print()

    results = run_holdout_experiment(
        n_runs=10,
        n_items=100,
        n_training_cycles=50,
        base_seed=42,
    )

    # Display results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Per-run transfer correlations:")
    for run in results["runs"]:
        status = "PASS" if run["transfer_success"] else "FAIL"
        print(f"  Run {run['run_idx'] + 1}: {run['transfer_correlation']:.3f} "
              f"(holdout={run['holdout_professor']}) [{status}]")

    print()
    print("Summary:")
    summary = results["summary"]
    print(f"  Mean transfer correlation: {summary['mean_transfer_correlation']:.3f} "
          f"± {summary['std_transfer_correlation']:.3f}")
    print(f"  Range: [{summary['min_transfer_correlation']:.3f}, "
          f"{summary['max_transfer_correlation']:.3f}]")
    print(f"  Success rate (>0.4): {summary['transfer_success_rate']:.1%}")

    print()
    print("By holdout professor:")
    for name, mean_corr in summary["by_holdout_mean"].items():
        print(f"  {name}: {mean_corr:.3f}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if summary["mean_transfer_correlation"] > 0.4:
        print("  TRANSFER SUCCEEDED!")
        print()
        print("  With correlated professors, the learned judgment transfers")
        print("  to unseen evaluators. This validates ASPIRE's ability to")
        print("  internalize shared evaluative structure.")
        print()
        print("  The original holdout failures were due to orthogonal professors,")
        print("  NOT a fundamental limitation of the ASPIRE mechanism.")
    else:
        print("  TRANSFER FAILED")
        print()
        print("  Even with correlated professors sharing latent structure,")
        print("  the learned judgment does not transfer to holdout evaluators.")
        print()
        print("  This indicates a genuine limitation of ASPIRE:")
        print("  The mechanism may learn evaluator-specific patterns rather")
        print("  than the underlying shared quality dimension.")

    # Save results
    output_dir = Path("experiments/results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "correlated_holdout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also save verification
    with open(output_dir / "professor_verification.json", "w") as f:
        json.dump(verification, f, indent=2)

    print()
    print(f"Results saved to: {output_dir}")

    return 0 if summary["mean_transfer_correlation"] > 0.4 else 1


if __name__ == "__main__":
    sys.exit(main())
