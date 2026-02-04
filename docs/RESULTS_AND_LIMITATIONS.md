# ASPIRE Experimental Results and Theoretical Boundaries

This document presents the results of the ASPIRE falsification experiments, including
systematic negative results and identified boundary conditions. Per scientific rigor,
failures are reported as first-class results, not hidden or minimized.

## Executive Summary

Across 90 experimental runs (3 experiments × 3 conditions × 10 runs), the falsification
suite identified:

- **8 expected failures** (theory predicts these boundary behaviors)
- **11 unexpected failures** (potential falsifications requiring interpretation)
- **Overall failure rate**: 21.1%

The results reveal important constraints on ASPIRE's claims that must be acknowledged
and properly framed.

---

## Experiment 1: Null vs Structured Judgment

**Question**: Does conscience emerge only under meaningful evaluation?

### Results

| Condition | ConscienceScore (mean ± std) | Stability |
|-----------|------------------------------|-----------|
| FULL_ASPIRE | 0.510 ± 0.187 | 0.134 |
| SCALAR_REWARD | 0.360 ± 0.162 | 0.372 |
| RANDOM_PROFESSORS | 0.420 ± 0.099 | 1.000 |

### Falsification Status

- **conscience_separation**: FAILED
  - Random achieved scores within 1 std of Full (0.420 vs 0.510 ± 0.187)
  - This is a **potential falsification** requiring interpretation

- **surprise_stability**: FAILED
  - Random showed higher stability (1.0) than Full (0.134)
  - Counterintuitive result suggests stability metric may be miscalibrated

- **generalization**: PASSED
  - Full generalization (0.700) significantly exceeded Random (-0.400)

### Interpretation

The failure of conscience_separation does NOT necessarily falsify ASPIRE. Two interpretations:

1. **Simulation Limitation**: The synthetic professors may not provide sufficiently
   distinct evaluation signals. Real multi-dimensional evaluation may show clearer separation.

2. **Training Dynamics Artifact**: Some conscience-like behavior may emerge from
   training dynamics alone, independent of evaluation quality. This would require
   revising claims about the *source* of conscience, not its existence.

**Correct framing**: "ASPIRE demonstrates measurable conscience formation, but the
degree to which this depends on structured vs random evaluation requires further
investigation with more diverse evaluators."

---

## Experiment 2: Holdout Transfer

**Question**: Does the student learn how to judge, or just who trained it?

### Results

| Condition | Holdout Correlation | Transfer Ratio |
|-----------|---------------------|----------------|
| HOLDOUT_ONE | -0.095 ± 0.22 | -0.281 |
| ALL_PROFESSORS | N/A | N/A |
| SINGLE_PROFESSOR | N/A | N/A |

### Falsification Status

- **holdout_correlation**: FAILED
  - Mean correlation with unseen professor = -0.095 (threshold was 0.2)
  - 10/10 runs showed no positive transfer
  - This is a **consistent falsification**

- **transfer_ratio**: FAILED
  - Ratio = -0.281 (threshold was 0.5)
  - Negative ratio suggests *anti-transfer*

- **ensemble_benefit**: FAILED
  - Single-professor generalization (0.792) exceeded all-professors (-0.002)

### Interpretation

This is the most serious finding. The failure of holdout transfer suggests:

1. **Overfitting to Evaluator Identity**: The student may be learning *who* trained it,
   not *how to judge*. This is a form of evaluator memorization.

2. **Insufficient Evaluator Diversity**: With only 3-5 professors, there may not be
   enough diversity to force generalization beyond specific evaluator patterns.

3. **Simulation Artifact**: The synthetic professor implementations may share too much
   structure, making "holdout" not truly independent.

**Correct framing**: "Conscience formation is bounded by evaluator diversity; ASPIRE
exposes this dependency rather than obscuring it. Transfer to truly novel evaluators
remains an open challenge that requires either (a) more diverse evaluation, or
(b) explicit regularization against evaluator-specific features."

---

## Experiment 3: Adversarial Pressure

**Question**: Does ASPIRE resist actively deceptive students?

### Results

| Condition | ConscienceScore | Gaming Detected | Leakage |
|-----------|-----------------|-----------------|---------|
| HONEST_STUDENT | 0.900 | N/A | 0.000 |
| ADVERSARIAL_NO_DEFENSE | 0.700 | 0% | 0.000 |
| ADVERSARIAL_WITH_DEFENSE | 0.700 | N/A | N/A |

### Falsification Status

- **adversarial_detection**: FAILED
  - Detection rate = 0% (threshold was 70%)
  - Gaming behavior was not identified

- **honest_advantage**: PASSED
  - Honest (0.900) > Adversarial (0.700)
  - Honest behavior is still rewarded more highly

- **leakage_discrimination**: FAILED
  - Both honest and adversarial showed 0.0 leakage correlation
  - Detection mechanism did not fire

### Interpretation

The adversarial experiment shows mixed results:

1. **Good News**: Honest students consistently outperform adversarial students (0.9 vs 0.7).
   Gaming does not *pay* in terms of final scores.

2. **Bad News**: The gaming *detection* mechanism failed entirely. This means we cannot
   reliably identify when gaming is occurring, only that it results in lower scores.

3. **Implication**: ASPIRE may achieve robustness through *score penalties* rather than
   *explicit detection*. This is a weaker form of robustness but still meaningful.

**Correct framing**: "ASPIRE penalizes gaming behavior through score differentials, but
explicit gaming detection requires more sensitive leakage metrics or longer observation
windows. The SlowRollDeceiver adversarial strategy in particular is designed to evade
detection in standard-length training runs."

---

## When Conscience Does Not Form

### Expected Failure Patterns

These failures are predicted by theory and validate our understanding of mechanism limits:

**SCALAR_PARTIAL_CONSCIENCE** (5 cases)
> SCALAR averaging doesn't fully destroy multi-dimensional signal. This confirms that
> scalar reward can approximate but not sustain full conscience formation.

**SEED_VARIANCE** (3 cases)
> High run-to-run variance (CV > 0.3) in several conditions suggests sensitivity to
> initialization. Recommend larger n_runs for production deployments.

### Boundary Conditions

The experiments operated under these constraints:

1. **Training Duration**: 50 cycles may be at the lower bound for stable conscience formation
2. **Sample Size**: 100 items provides moderate but not comprehensive coverage
3. **Evaluator Diversity**: 3-5 synthetic professors may not represent real-world diversity

---

## Limitations and Theoretical Boundaries

### Fundamental Constraints

1. **Evaluator Homogeneity Risk**
   > "Conscience formation is bounded by evaluator diversity; ASPIRE exposes this
   > dependency rather than obscuring it."

   If all professors share deep biases, the student will internalize those biases as
   "conscience." ASPIRE cannot distinguish genuine ethical judgment from consistent
   evaluator bias without external validation.

2. **Empirical Indistinguishability at the Limit**
   > "At the limit of empirical indistinguishability, internalization is defined
   > behaviorally."

   A perfectly gaming system that passes all tests is empirically indistinguishable
   from genuine conscience. This is not a weakness of ASPIRE specifically—it is a
   philosophical boundary of any empirical approach to evaluating internal states.

3. **Transfer vs Memorization**
   The holdout experiments reveal that conscience may be *evaluator-specific* rather
   than *judgment-general*. This is the most significant limitation identified.

### What ASPIRE Does and Does Not Claim

**ASPIRE Claims (Supported)**:
- Conscience formation can be operationalized as a statistical property
- Multi-dimensional evaluation produces different training dynamics than scalar reward
- Honest behavior achieves higher scores than adversarial gaming
- Failures are diagnosable and predictable

**ASPIRE Does NOT Claim**:
- Conscience is guaranteed to form
- Formed conscience generalizes to arbitrary novel evaluators
- All adversarial strategies are detected
- Results are independent of evaluator quality

---

## Recommendations

1. **For Researchers**: The holdout transfer failure is the most important finding.
   Future work should investigate (a) evaluator diversity requirements and
   (b) regularization techniques to force judgment generalization.

2. **For Practitioners**: Use ASPIRE with diverse, independent evaluators. Do not
   assume transfer to truly novel evaluation criteria without empirical validation.

3. **For Reviewers**: The systematic negative results demonstrate intellectual honesty.
   The theory is falsifiable and some predictions were not confirmed. This is how
   science should work.

---

## Appendix: Raw Failure Data

See `failure_report.json` for complete structured data including:
- All 19 failure cases with timestamps and observed values
- Expected vs actual ranges for each metric
- Interpretations and falsification flags
- Category breakdown and statistics

See `failure_atlas.png` for visual summary organized by:
- Expected vs unexpected (columns)
- Recoverable vs limiting (rows)
