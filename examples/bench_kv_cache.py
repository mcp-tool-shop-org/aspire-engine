"""Benchmark: KV-cache vs no-cache inference performance.

Compares generation speed with and without KV-cache to quantify
the performance improvement.

Usage:
    python examples/bench_kv_cache.py --model models/phi3-mini.onnx --tokenizer microsoft/Phi-3-mini-4k-instruct

Output:
    - First token latency (TTFT)
    - Steady-state tokens/second
    - Total generation time
    - Peak VRAM usage
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import time

from aspire.core import TrainingItem
from aspire.student import ONNXStudentV2, GenerationConfig, CacheConfig
from aspire.governor import TokenPool, GovernorConfig
from aspire.governor.metrics import GPUMetrics, MockGPUMetrics


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    prompt: str
    mode: str  # "cache" or "no_cache"
    prompt_tokens: int
    generated_tokens: int
    first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    peak_memory_mb: int = 0
    throttled: bool = False


def create_test_prompts() -> List[TrainingItem]:
    """Create prompts of varying complexity."""
    return [
        TrainingItem(
            id="short_001",
            prompt="What is 2 + 2?",
            gold_answer="4",
            gold_rationale="Basic arithmetic.",
            difficulty=0.1,
            domain="math",
        ),
        TrainingItem(
            id="medium_001",
            prompt="Explain the tradeoffs between microservices and monolithic architectures for a startup.",
            gold_answer="Monoliths are simpler for small teams; microservices add operational complexity.",
            gold_rationale="Consider team size, scale expectations, and operational maturity.",
            difficulty=0.5,
            domain="architecture",
        ),
        TrainingItem(
            id="long_001",
            prompt=(
                "A tech company is deciding between three cloud providers for their new ML platform. "
                "Provider A offers the lowest cost but has limited GPU availability. "
                "Provider B has excellent GPU support but charges premium prices. "
                "Provider C is moderately priced with good support but requires significant migration effort. "
                "What factors should they consider and what would you recommend?"
            ),
            gold_answer="Consider TCO, GPU availability, migration effort, and lock-in risks.",
            gold_rationale="Multi-factor decision requiring tradeoff analysis.",
            difficulty=0.8,
            domain="business",
        ),
    ]


class KVCacheBenchmark:
    """Benchmark runner for KV-cache comparison."""

    def __init__(
        self,
        model_path: str,
        tokenizer: str,
        device: str = "cuda",
        use_governor: bool = True,
    ):
        self.model_path = Path(model_path)
        self.tokenizer = tokenizer
        self.device = device
        self.use_governor = use_governor

        # Create student
        self.config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.0,  # Deterministic for fair comparison
            do_sample=False,
        )

        self.student = ONNXStudentV2(
            model_path=str(model_path),
            tokenizer_name_or_path=tokenizer,
            device=device,
            generation_config=self.config,
        )

        # Governor for memory tracking
        if use_governor:
            if device == "cuda":
                self.governor = TokenPool(GovernorConfig.for_rtx_5080())
            else:
                self.governor = TokenPool(
                    GovernorConfig.for_testing(),
                    metrics=MockGPUMetrics(),
                )
        else:
            self.governor = None

    def run_single(
        self,
        item: TrainingItem,
        max_tokens: int,
        force_no_cache: bool = False,
    ) -> BenchmarkResult:
        """Run a single generation and collect metrics."""

        # Temporarily disable cache if requested
        original_supports = self.student._supports_kv_cache
        if force_no_cache:
            self.student._supports_kv_cache = False

        # Get memory before
        memory_before = 0
        if self.governor:
            status = self.governor.get_status()
            memory_before = int((status.memory_ratio * 16) * 1024)  # Rough estimate

        # Run generation with metrics
        response, metrics = self.student.generate_with_metrics(item, max_tokens=max_tokens)

        # Get memory after
        memory_after = 0
        if self.governor:
            status = self.governor.get_status()
            memory_after = int((status.memory_ratio * 16) * 1024)

        # Restore cache setting
        if force_no_cache:
            self.student._supports_kv_cache = original_supports

        return BenchmarkResult(
            prompt=item.prompt[:50] + "...",
            mode="no_cache" if force_no_cache else ("cache" if metrics.used_kv_cache else "no_cache"),
            prompt_tokens=metrics.prompt_tokens,
            generated_tokens=metrics.generated_tokens,
            first_token_ms=metrics.first_token_ms,
            total_time_ms=metrics.total_time_ms,
            tokens_per_second=metrics.tokens_per_second,
            peak_memory_mb=max(memory_before, memory_after),
            throttled=False,
        )

    def run_comparison(
        self,
        items: List[TrainingItem],
        max_tokens: int = 200,
        warmup_runs: int = 1,
    ) -> dict:
        """Run full comparison benchmark."""
        results = {
            "cache": [],
            "no_cache": [],
        }

        print(f"\nModel: {self.model_path.name}")
        print(f"KV-cache supported: {self.student.supports_kv_cache}")
        print(f"Max tokens: {max_tokens}")
        print()

        # Warmup
        if warmup_runs > 0:
            print(f"Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                self.run_single(items[0], max_tokens=50)
            print()

        # Run benchmarks
        for item in items:
            print(f"Benchmarking: {item.id}")

            # With cache (if supported)
            if self.student.supports_kv_cache:
                result_cache = self.run_single(item, max_tokens, force_no_cache=False)
                results["cache"].append(result_cache)
                print(f"  [cache]    TTFT: {result_cache.first_token_ms:6.1f}ms | "
                      f"Total: {result_cache.total_time_ms:7.1f}ms | "
                      f"{result_cache.tokens_per_second:5.1f} tok/s | "
                      f"{result_cache.generated_tokens} tokens")

            # Without cache
            result_no_cache = self.run_single(item, max_tokens, force_no_cache=True)
            results["no_cache"].append(result_no_cache)
            print(f"  [no_cache] TTFT: {result_no_cache.first_token_ms:6.1f}ms | "
                  f"Total: {result_no_cache.total_time_ms:7.1f}ms | "
                  f"{result_no_cache.tokens_per_second:5.1f} tok/s | "
                  f"{result_no_cache.generated_tokens} tokens")

            # Speedup
            if self.student.supports_kv_cache:
                speedup = result_no_cache.total_time_ms / result_cache.total_time_ms
                print(f"  [speedup]  {speedup:.2f}x faster with cache")

            print()

        return results


def print_summary(results: dict, supports_cache: bool):
    """Print benchmark summary."""
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    def avg(lst, key):
        vals = [getattr(r, key) for r in lst]
        return sum(vals) / len(vals) if vals else 0

    if supports_cache and results["cache"]:
        cache_results = results["cache"]
        no_cache_results = results["no_cache"]

        print(f"\n{'Metric':<25} {'With Cache':>15} {'Without Cache':>15} {'Speedup':>10}")
        print("-" * 70)

        cache_ttft = avg(cache_results, "first_token_ms")
        no_cache_ttft = avg(no_cache_results, "first_token_ms")
        print(f"{'Avg TTFT (ms)':<25} {cache_ttft:>15.1f} {no_cache_ttft:>15.1f} {no_cache_ttft/cache_ttft:>9.2f}x")

        cache_total = avg(cache_results, "total_time_ms")
        no_cache_total = avg(no_cache_results, "total_time_ms")
        print(f"{'Avg Total Time (ms)':<25} {cache_total:>15.1f} {no_cache_total:>15.1f} {no_cache_total/cache_total:>9.2f}x")

        cache_tps = avg(cache_results, "tokens_per_second")
        no_cache_tps = avg(no_cache_results, "tokens_per_second")
        print(f"{'Avg Tokens/sec':<25} {cache_tps:>15.1f} {no_cache_tps:>15.1f} {cache_tps/no_cache_tps:>9.2f}x")

        print("\nConclusion: ", end="")
        overall_speedup = no_cache_total / cache_total
        if overall_speedup > 2:
            print(f"KV-cache provides {overall_speedup:.1f}x speedup - significant improvement!")
        elif overall_speedup > 1.2:
            print(f"KV-cache provides {overall_speedup:.1f}x speedup - moderate improvement.")
        else:
            print(f"KV-cache provides minimal speedup ({overall_speedup:.1f}x) - model may not benefit much.")

    else:
        no_cache_results = results["no_cache"]
        print("\nKV-cache not supported by this model. No-cache baseline:")
        print(f"  Avg TTFT: {avg(no_cache_results, 'first_token_ms'):.1f}ms")
        print(f"  Avg Total: {avg(no_cache_results, 'total_time_ms'):.1f}ms")
        print(f"  Avg Tokens/sec: {avg(no_cache_results, 'tokens_per_second'):.1f}")


def main():
    parser = argparse.ArgumentParser(description="KV-cache Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "dml", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--no-governor", action="store_true", help="Disable governor")
    args = parser.parse_args()

    # Verify model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nExport a model first:")
        print("  python scripts/export_model.py --model microsoft/Phi-3-mini-4k-instruct")
        return

    print("=" * 70)
    print("ASPIRE KV-Cache Benchmark")
    print("=" * 70)

    benchmark = KVCacheBenchmark(
        model_path=args.model,
        tokenizer=args.tokenizer,
        device=args.device,
        use_governor=not args.no_governor,
    )

    prompts = create_test_prompts()
    results = benchmark.run_comparison(prompts, max_tokens=args.max_tokens, warmup_runs=args.warmup)

    print_summary(results, benchmark.student.supports_kv_cache)


if __name__ == "__main__":
    main()
