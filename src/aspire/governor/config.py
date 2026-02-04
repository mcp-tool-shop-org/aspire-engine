"""Configuration for ASPIRE Governor."""

from dataclasses import dataclass


@dataclass
class GovernorConfig:
    """Configuration for the token-based resource governor.

    The governor protects the system from parallelism overload by:
    1. Monitoring GPU memory and compute utilization
    2. Granting tokens (permits) for inference operations
    3. Throttling when resources are constrained
    4. Classifying failures as OOM vs actual errors

    Tokens represent "inference permits" - each inference cycle
    requires tokens proportional to expected resource usage.
    """

    # Token budget calculation
    gb_per_token: float = 1.0           # GPU GB per token (e.g., 1 token = 1GB)
    safety_reserve_gb: float = 2.0       # Always keep this much free
    min_tokens: int = 1                  # Floor (never grant less)
    max_tokens: int = 16                 # Ceiling (never grant more)

    # Throttle thresholds (based on GPU memory utilization ratio)
    caution_ratio: float = 0.70          # Start throttling at 70%
    soft_stop_ratio: float = 0.82        # Aggressive throttling at 82%
    hard_stop_ratio: float = 0.90        # Refuse new leases at 90%

    # Timing
    monitor_interval_ms: int = 500       # How often to check resources
    lease_ttl_minutes: int = 5           # Max lease duration before auto-reclaim
    acquire_timeout_ms: int = 30000      # Max wait time for token acquisition
    retry_delay_normal_ms: int = 50      # Delay between retries (normal)
    retry_delay_caution_ms: int = 200    # Delay between retries (caution)
    retry_delay_soft_stop_ms: int = 500  # Delay between retries (soft stop)

    # Inference cost estimation
    tokens_per_small_inference: int = 1   # <100 output tokens
    tokens_per_medium_inference: int = 2  # 100-500 output tokens
    tokens_per_large_inference: int = 4   # >500 output tokens

    # Failure classification thresholds
    oom_memory_ratio_threshold: float = 0.88  # Classify as OOM if memory was this high
    oom_peak_gb_threshold: float = 12.0       # Classify as OOM if peak exceeded this

    @classmethod
    def for_rtx_5080(cls) -> "GovernorConfig":
        """Preset for RTX 5080 (16GB VRAM)."""
        return cls(
            gb_per_token=1.5,
            safety_reserve_gb=2.5,
            max_tokens=10,
            oom_peak_gb_threshold=14.0,
        )

    @classmethod
    def for_rtx_4090(cls) -> "GovernorConfig":
        """Preset for RTX 4090 (24GB VRAM)."""
        return cls(
            gb_per_token=2.0,
            safety_reserve_gb=4.0,
            max_tokens=12,
            oom_peak_gb_threshold=22.0,
        )

    @classmethod
    def for_testing(cls) -> "GovernorConfig":
        """Lenient config for testing without real GPU."""
        return cls(
            gb_per_token=0.5,
            safety_reserve_gb=0.5,
            min_tokens=4,
            max_tokens=32,
            caution_ratio=0.90,
            soft_stop_ratio=0.95,
            hard_stop_ratio=0.98,
            lease_ttl_minutes=1,
        )
