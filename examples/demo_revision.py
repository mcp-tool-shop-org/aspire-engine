"""Demo: Revision behavior - the self-correction loop.

Shows how the student revises its response when the critic detects
misalignment (negative surprise, high disagreement, overconfidence).

This demonstrates the core ASPIRE concept: internalization of judgment
through active repair under pressure.

Output shows 3-way comparison per cycle:
1. Draft tokens + critic prediction
2. Revision decision (why/why not)
3. Revised tokens + uplift (if revised)

Usage:
    python examples/demo_revision.py --cycles 50
"""

import argparse
from typing import List
import random

from aspire.core import TrainingItem, TokenDimension
from aspire.student import MockStudent
from aspire.professors import ProfessorEnsemble
from aspire.critic import LearnedCriticV0
from aspire.engine import (
    RevisionAspireEngine,
    RevisionConfig,
    RevisionTrigger,
    RevisionCycleResult,
)


def create_test_items(count: int = 50) -> List[TrainingItem]:
    """Create diverse test items that will trigger various revision scenarios."""
    templates = [
        # High disagreement expected (ethical dilemmas)
        {
            "prompt": "Should companies be required to disclose when content is AI-generated?",
            "gold_answer": "Context-dependent - consider transparency vs practicality tradeoffs",
            "gold_rationale": "Multiple valid perspectives exist.",
            "difficulty": 0.8,
            "domain": "ethics",
        },
        {
            "prompt": "Is it ethical to use predictive policing algorithms?",
            "gold_answer": "Depends on implementation, oversight, and bias mitigation",
            "gold_rationale": "Balancing public safety vs civil liberties.",
            "difficulty": 0.9,
            "domain": "ethics",
        },
        # Medium difficulty (technical decisions)
        {
            "prompt": "Should we use a NoSQL or SQL database for our e-commerce platform?",
            "gold_answer": "SQL for transactional data, consider hybrid for catalog",
            "gold_rationale": "ACID compliance matters for orders.",
            "difficulty": 0.5,
            "domain": "technical",
        },
        {
            "prompt": "Is it better to deploy on Kubernetes or serverless for a startup?",
            "gold_answer": "Serverless for early stage, K8s when scale/control needed",
            "gold_rationale": "Operational overhead vs flexibility tradeoff.",
            "difficulty": 0.6,
            "domain": "technical",
        },
        # Low disagreement (clearer answers)
        {
            "prompt": "Should we write unit tests for critical financial calculations?",
            "gold_answer": "Yes, absolutely - financial code needs high test coverage",
            "gold_rationale": "Bugs in financial code have serious consequences.",
            "difficulty": 0.2,
            "domain": "code_review",
        },
        {
            "prompt": "Should passwords be stored in plain text?",
            "gold_answer": "No, always hash passwords with salt",
            "gold_rationale": "Basic security requirement.",
            "difficulty": 0.1,
            "domain": "security",
        },
        # Ambiguous business decisions
        {
            "prompt": "Should we pursue a freemium or paid-only model?",
            "gold_answer": "Depends on market, product complexity, and customer acquisition cost",
            "gold_rationale": "No universal right answer.",
            "difficulty": 0.7,
            "domain": "business",
        },
        {
            "prompt": "Should we expand to a new market or deepen in our current one?",
            "gold_answer": "Evaluate market saturation, competitive dynamics, and resources",
            "gold_rationale": "Strategic decision with multiple valid paths.",
            "difficulty": 0.8,
            "domain": "business",
        },
    ]

    items = []
    for i in range(count):
        template = random.choice(templates)
        items.append(TrainingItem(
            id=f"{template['domain']}_{i:03d}",
            prompt=template["prompt"],
            gold_answer=template["gold_answer"],
            gold_rationale=template["gold_rationale"],
            difficulty=template["difficulty"],
            domain=template["domain"],
        ))

    return items


def format_tokens(tokens, label: str = "") -> str:
    """Format token vector for display."""
    parts = []
    for dim in TokenDimension:
        val = tokens.values[dim]
        bar = "█" * int(val * 10)
        parts.append(f"{dim.value[:4]:4}={val:.2f} {bar}")
    return f"{label}[{' | '.join(parts)}]"


def on_cycle_complete(result: RevisionCycleResult):
    """Detailed callback showing the 3-way comparison."""
    print(f"\n{'='*80}")
    print(f"Cycle: {result.item.id}")
    print(f"Question: {result.item.prompt[:70]}...")
    print(f"{'='*80}")

    # Draft pass
    print(f"\n[DRAFT]")
    print(f"  Answer: {result.draft_response.answer[:60]}...")
    print(f"  Confidence: {result.draft_response.confidence:.2f}")
    print(f"  Tokens: {result.draft_tokens.total:.2f}")
    print(f"  {format_tokens(result.draft_tokens)}")

    # Critic prediction vs actual
    pred_total = result.draft_prediction.expected_tokens.total
    actual_total = result.draft_tokens.total
    surprise = actual_total - pred_total
    print(f"\n[CRITIC]")
    print(f"  Predicted: {pred_total:.2f} | Actual: {actual_total:.2f} | Surprise: {surprise:+.2f}")
    print(f"  Expected disagreement: {result.draft_prediction.expected_disagreement:.2f}")
    print(f"  Actual disagreement:   {result.draft_evaluation.disagreement_score:.2f}")

    # Revision decision
    print(f"\n[DECISION]")
    if result.did_revise:
        triggers = [t.value for t in result.revision_decision.triggers]
        print(f"  ⚡ REVISING - Triggers: {', '.join(triggers)}")
        print(f"  Reason: {result.revision_decision.reason}")
    else:
        print(f"  ✓ No revision needed")
        print(f"  Reason: {result.revision_decision.reason}")

    # Revision result
    if result.did_revise:
        print(f"\n[REVISION]")
        print(f"  Answer: {result.revised_response.answer[:60]}...")
        print(f"  Confidence: {result.revised_response.confidence:.2f}")
        print(f"  Tokens: {result.revised_tokens.total:.2f}")
        print(f"  {format_tokens(result.revised_tokens)}")

        # Uplift
        print(f"\n[UPLIFT]")
        print(f"  Total: {result.token_uplift:+.3f}")
        for dim, uplift in result.uplift_by_dim.items():
            indicator = "↑" if uplift > 0 else "↓" if uplift < 0 else "="
            print(f"    {dim:15} {indicator} {uplift:+.3f}")

        if result.changes_made:
            print(f"\n[CHANGES]")
            for change in result.changes_made:
                print(f"    • {change}")

    # Final result
    print(f"\n[FINAL]")
    correct = "✓" if result.draft_evaluation.consensus_correct else "✗"
    print(f"  {correct} Consensus: {'Correct' if result.draft_evaluation.consensus_correct else 'Incorrect'}")
    print(f"  Final tokens: {result.final_tokens.total:.2f}")
    print(f"  Time: {result.cycle_time_ms:.0f}ms (revision: {result.revision_time_ms:.0f}ms)")


def print_summary(metrics):
    """Print training summary."""
    print("\n" + "=" * 80)
    print("REVISION TRAINING SUMMARY")
    print("=" * 80)

    print(f"\nOverall:")
    print(f"  Total cycles:    {metrics.total_cycles}")
    print(f"  Accuracy:        {metrics.accuracy:.1%}")
    print(f"  Avg cycle time:  {metrics.avg_cycle_time_ms:.0f}ms")

    print(f"\nRevision Behavior:")
    print(f"  Revisions:       {metrics.revision_count} ({metrics.revision_rate:.1%} of cycles)")
    print(f"  Total uplift:    {metrics.total_uplift:+.2f}")
    print(f"  Avg uplift:      {metrics.avg_uplift:+.3f}")
    print(f"  Avg revision ms: {metrics.avg_revision_time_ms:.0f}ms")

    print(f"\nRevision Triggers:")
    for trigger, count in sorted(metrics.revision_triggers.items(), key=lambda x: -x[1]):
        pct = count / max(1, metrics.revision_count) * 100
        print(f"    {trigger:25} {count:3} ({pct:.0f}%)")

    print(f"\nCritic Performance:")
    print(f"  Avg surprise:    {metrics.avg_surprise:.3f}")

    # Uplift trend
    if len(metrics.uplift_history) >= 10:
        early = sum(metrics.uplift_history[:10]) / 10
        late = sum(metrics.uplift_history[-10:]) / 10
        print(f"\nUplift Trend:")
        print(f"  Early avg: {early:+.3f}")
        print(f"  Late avg:  {late:+.3f}")
        if late > early:
            print(f"  → Revisions improving over time! ✓")
        else:
            print(f"  → Revisions not improving (may need tuning)")

    # Assessment
    print(f"\n{'='*80}")
    print("INTERNALIZATION ASSESSMENT")
    print("{'='*80}")

    if metrics.revision_rate > 0.1 and metrics.avg_uplift > 0:
        print("✓ Student is learning to revise productively")
    elif metrics.revision_rate > 0.3 and metrics.avg_uplift <= 0:
        print("○ High revision rate but no improvement - critic may be too sensitive")
    elif metrics.revision_rate < 0.05:
        print("○ Low revision rate - critic may be too lenient or student is strong")
    else:
        print("◐ Moderate revision behavior - system is calibrating")


def main():
    parser = argparse.ArgumentParser(description="Revision Behavior Demo")
    parser.add_argument("--cycles", type=int, default=30, help="Training cycles")
    parser.add_argument("--verbose", action="store_true", help="Show all cycles")
    parser.add_argument("--print-interval", type=int, default=5, help="Print every N cycles")
    args = parser.parse_args()

    print("=" * 80)
    print("ASPIRE Revision Behavior Demo")
    print("=" * 80)
    print("""
This demo shows the self-correction loop:
1. Student generates draft
2. Critic predicts tokens (gut feeling)
3. Professors evaluate
4. If misalignment detected → Student revises
5. Revision re-evaluated → Measure uplift

Watch for:
- Revisions triggered by negative surprise (predicted > actual)
- Revisions triggered by high disagreement
- Positive uplift = revision improved the response
""")

    # Create components
    student = MockStudent(correct_rate=0.4)  # Start weaker to trigger revisions
    professors = ProfessorEnsemble()
    critic = LearnedCriticV0(learning_rate=0.02)

    # Revision config - moderate thresholds
    revision_config = RevisionConfig(
        enabled=True,
        negative_surprise_threshold=0.35,
        disagreement_threshold=0.3,
        include_gold_in_revision=False,  # Self-correction mode
        include_critique_in_revision=True,
    )

    # Track which cycles to print
    cycles_printed = [0]

    def selective_callback(result):
        cycles_printed[0] += 1
        if args.verbose or cycles_printed[0] % args.print_interval == 0 or result.did_revise:
            on_cycle_complete(result)

    # Create engine
    engine = RevisionAspireEngine(
        student=student,
        professors=professors,
        critic=critic,
        revision_config=revision_config,
        on_cycle_complete=selective_callback,
    )

    # Run training
    print(f"\nStarting {args.cycles} cycles...")
    print(f"(Showing every {args.print_interval} cycles + all revisions)")

    items = create_test_items(args.cycles)
    metrics = engine.train(iter(items), max_cycles=args.cycles)

    # Summary
    print_summary(metrics)


if __name__ == "__main__":
    main()
