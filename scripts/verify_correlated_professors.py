#!/usr/bin/env python
"""Verify that correlated professors share latent structure.

This script confirms that the new CorrelatedProfessor implementations
actually have the properties needed for meaningful holdout transfer tests:

1. Inter-professor correlation > 0.4 (ideally 0.5-0.8)
2. First factor explains > 40% variance (ideally > 50%)
3. Effective dimensionality < 2 for 5 professors

If these conditions hold, holdout transfer SHOULD succeed.
If transfer then fails, it indicates a genuine ASPIRE limitation.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aspire.professors.correlated import verify_correlation_structure


def main():
    print("=" * 60)
    print("VERIFYING CORRELATED PROFESSOR STRUCTURE")
    print("=" * 60)
    print()

    # Run verification
    print("Generating 200 test items and collecting professor scores...")
    results = verify_correlation_structure(n_samples=200, seed=42)

    print()
    print("=" * 60)
    print("PAIRWISE CORRELATIONS")
    print("=" * 60)
    print()

    for pair, corr in sorted(results["pairwise_correlations"].items()):
        status = "OK" if corr > 0.4 else "LOW"
        print(f"  {pair}: {corr:.3f}  [{status}]")

    print()
    print(f"  Mean correlation: {results['mean_correlation']:.3f}")
    print()

    print("=" * 60)
    print("FACTOR ANALYSIS")
    print("=" * 60)
    print()
    print(f"  First factor variance:  {results['first_factor_variance']:.1%}")
    print(f"  Second factor variance: {results['second_factor_variance']:.1%}")
    print(f"  Effective dimensionality: {results['effective_dimensionality']:.2f} / {results['n_professors']}")
    print()

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print()

    if results["transfer_viable"]:
        print("  PASS: Professors share sufficient latent structure.")
        print()
        print("  Holdout transfer tests are NOW VALID.")
        print("  If transfer fails with these professors, it indicates")
        print("  a genuine limitation of ASPIRE's internalization mechanism.")
    else:
        print("  FAIL: Professors do not share sufficient structure.")
        print()
        print("  Need to increase correlation by:")
        print("    - Reducing noise_std in professor configs")
        print("    - Making weight distributions more similar")
        print("    - Checking latent quality computation")

    print()

    # Save results
    output_dir = Path("experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "correlated_professor_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_dir / 'correlated_professor_verification.json'}")

    return 0 if results["transfer_viable"] else 1


if __name__ == "__main__":
    sys.exit(main())
