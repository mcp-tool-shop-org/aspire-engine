"""Demo: ASPIRE training loop with Governor resource protection.

This demonstrates the governor protecting against GPU memory exhaustion
during parallel/rapid inference cycles.

Usage:
    # With real GPU monitoring
    python examples/demo_governor.py --device cuda

    # With mock GPU (for testing without GPU)
    python examples/demo_governor.py --mock-gpu
"""

import argparse
import time

from aspire.core import TrainingItem
from aspire.student import MockStudent
from aspire.professors import ProfessorEnsemble
from aspire.critic import HeuristicCritic
from aspire.engine import AspireEngine
from aspire.governor import (
    TokenPool,
    GovernorConfig,
    ThrottleLevel,
)
from aspire.governor.metrics import GPUMetrics, MockGPUMetrics


def create_test_items(count: int = 20):
    """Create sample training items."""
    base_items = [
        TrainingItem(
            id="ethics_001",
            prompt="A self-driving car must choose between hitting one pedestrian or swerving into five.",
            gold_answer="This is a trolley problem with no objectively correct answer",
            gold_rationale="Competing ethical frameworks create genuine tension.",
            difficulty=0.8,
            domain="ethics",
        ),
        TrainingItem(
            id="code_review_001",
            prompt="Should we merge this PR that adds 2000 lines but has no tests?",
            gold_answer="No, require tests before merging",
            gold_rationale="Tests prevent regressions and document behavior.",
            difficulty=0.4,
            domain="code_review",
        ),
        TrainingItem(
            id="strategy_001",
            prompt="Our competitor cut prices 30%. Should we match?",
            gold_answer="Depends on cost structure and brand positioning",
            gold_rationale="Price wars can destroy margins.",
            difficulty=0.7,
            domain="business",
        ),
        TrainingItem(
            id="architecture_001",
            prompt="Should we use microservices for our MVP?",
            gold_answer="Probably not - monolith is simpler for small teams",
            gold_rationale="Microservices add operational complexity.",
            difficulty=0.6,
            domain="architecture",
        ),
    ]

    # Repeat to fill count
    items = []
    for i in range(count):
        base = base_items[i % len(base_items)]
        items.append(TrainingItem(
            id=f"{base.id}_{i}",
            prompt=base.prompt,
            gold_answer=base.gold_answer,
            gold_rationale=base.gold_rationale,
            difficulty=base.difficulty,
            domain=base.domain,
        ))
    return items


def on_cycle_complete(result):
    """Callback for each training cycle."""
    correct = "✓" if result.evaluation.consensus_correct else "✗"
    tokens = result.tokens_earned.total

    throttle_marker = ""
    if result.governor_throttled:
        throttle_marker = f" [THROTTLED +{result.governor_wait_ms:.0f}ms]"

    print(
        f"[{result.item.id:20}] {correct} "
        f"tokens={tokens:.2f} "
        f"time={result.cycle_time_ms:.0f}ms"
        f"{throttle_marker}"
    )


def simulate_memory_pressure(mock_metrics: MockGPUMetrics, cycle: int):
    """Simulate increasing memory pressure over cycles."""
    # Start at 4GB, increase by 500MB every 3 cycles
    base = 4096
    increase = (cycle // 3) * 512
    mock_metrics.set_memory_used(min(base + increase, 15000))


def main():
    parser = argparse.ArgumentParser(description="ASPIRE Governor Demo")
    parser.add_argument("--mock-gpu", action="store_true", help="Use mock GPU metrics")
    parser.add_argument("--cycles", type=int, default=20, help="Number of training cycles")
    parser.add_argument("--simulate-pressure", action="store_true",
                       help="Simulate increasing memory pressure (mock only)")
    args = parser.parse_args()

    print("=" * 70)
    print("ASPIRE Engine - Governor Demo")
    print("=" * 70)

    # Configure governor
    if args.mock_gpu:
        print("\nUsing MOCK GPU metrics (no real GPU required)")
        config = GovernorConfig.for_testing()
        metrics = MockGPUMetrics(memory_total_mb=16384, memory_used_mb=4096)
    else:
        print("\nUsing REAL GPU metrics via nvidia-smi")
        config = GovernorConfig.for_rtx_5080()
        metrics = GPUMetrics(gpu_index=0)

    # Create governor
    governor = TokenPool(config=config, metrics=metrics)

    # Show initial status
    status = governor.get_status()
    print(f"\nInitial GPU status:")
    print(f"  Memory: {status.memory_ratio:.0%} used")
    print(f"  Free: {status.memory_free_gb:.1f} GB")
    print(f"  Available tokens: {status.available_tokens}")
    print(f"  Throttle level: {status.throttle_level.value}")

    # Create engine with governor
    student = MockStudent(correct_rate=0.5)
    professors = ProfessorEnsemble()
    critic = HeuristicCritic()

    engine = AspireEngine(
        student=student,
        professors=professors,
        critic=critic,
        governor=governor,
        on_cycle_complete=on_cycle_complete,
    )

    # Run training
    print("\n" + "-" * 70)
    print("Starting training loop with governor protection")
    print("-" * 70 + "\n")

    items = create_test_items(args.cycles)

    if args.simulate_pressure and args.mock_gpu:
        # Run with simulated pressure
        for i, item in enumerate(items):
            simulate_memory_pressure(metrics, i)

            # Show pressure changes
            if i % 5 == 0:
                status = governor.get_status()
                print(f"\n  [Pressure check] Memory: {status.memory_ratio:.0%}, "
                      f"Throttle: {status.throttle_level.value}\n")

            engine.run_cycle(item)
    else:
        # Normal run
        engine.train(iter(items), max_cycles=args.cycles)

    # Report
    m = engine.metrics
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Total cycles:      {m.total_cycles}")
    print(f"Accuracy:          {m.accuracy:.1%}")
    print(f"Avg cycle time:    {m.avg_cycle_time_ms:.0f}ms")
    print()
    print("Governor Metrics:")
    print(f"  Throttled cycles: {m.throttled_cycles} ({m.throttle_rate:.0%})")
    print(f"  Total wait time:  {m.total_governor_wait_ms:.0f}ms")
    print(f"  OOM retries:      {m.oom_retries}")

    # Final status
    final_status = governor.get_status()
    print(f"\nFinal GPU status:")
    print(f"  Memory: {final_status.memory_ratio:.0%} used")
    print(f"  Active leases: {final_status.active_leases}")


if __name__ == "__main__":
    main()
