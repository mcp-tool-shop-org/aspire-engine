"""Token pool for ASPIRE Governor - manages inference permits."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from threading import Lock
import uuid
import asyncio
import time

from .config import GovernorConfig
from .metrics import GPUMetrics, ResourceStatus, ThrottleLevel


@dataclass
class Lease:
    """A granted token lease for an inference operation."""
    lease_id: str
    operation: str                    # e.g., "inference", "batch_inference"
    tokens: int                       # Number of tokens granted
    acquired_at: datetime
    expires_at: datetime
    memory_ratio_at_acquire: float    # Snapshot for failure classification
    peak_memory_mb: int = 0           # Updated during operation
    warning_logged: bool = False      # For TTL warning


@dataclass
class AcquireResult:
    """Result of attempting to acquire tokens."""
    success: bool
    lease_id: Optional[str] = None
    granted_tokens: int = 0
    recommended_parallelism: int = 1
    memory_ratio: float = 0.0
    throttle_level: ThrottleLevel = ThrottleLevel.NORMAL
    reason: Optional[str] = None
    wait_time_ms: float = 0.0         # How long we waited


@dataclass
class ReleaseResult:
    """Result of releasing a lease."""
    acknowledged: bool
    classification: str = "success"   # success, oom, error, expired
    message: Optional[str] = None
    should_retry: bool = False
    retry_with_tokens: Optional[int] = None


@dataclass
class PoolStatus:
    """Current state of the token pool."""
    total_budget: int
    available_tokens: int
    active_leases: int
    active_tokens: int                # Sum of tokens in active leases
    throttle_level: ThrottleLevel
    memory_ratio: float
    memory_free_gb: float
    leases: List[Lease] = field(default_factory=list)


class TokenPool:
    """Manages token budget and active leases for inference operations.

    The pool:
    1. Monitors GPU resources (every 500ms by default)
    2. Calculates dynamic token budget based on available memory
    3. Grants leases (permits) for inference operations
    4. Reclaims expired leases automatically
    5. Classifies failures to enable intelligent retries

    Thread-safe via Lock for all state mutations.
    """

    def __init__(
        self,
        config: Optional[GovernorConfig] = None,
        metrics: Optional[GPUMetrics] = None,
    ):
        self.config = config or GovernorConfig()
        self.metrics = metrics or GPUMetrics()

        self._leases: Dict[str, Lease] = {}
        self._lock = Lock()
        self._last_status: Optional[ResourceStatus] = None
        self._last_status_time: float = 0

        # Background monitoring (optional, for async usage)
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    def _get_resource_status(self) -> ResourceStatus:
        """Get current resource status, with caching."""
        now = time.time()

        # Cache for monitor_interval_ms
        if (
            self._last_status is not None
            and (now - self._last_status_time) * 1000 < self.config.monitor_interval_ms
        ):
            return self._last_status

        self._last_status = self.metrics.get_status(self.config)
        self._last_status_time = now
        return self._last_status

    def _reclaim_expired_leases(self):
        """Reclaim any expired leases (called under lock)."""
        now = datetime.now()
        expired = [
            lid for lid, lease in self._leases.items()
            if lease.expires_at < now
        ]
        for lid in expired:
            del self._leases[lid]

    def _active_token_count(self) -> int:
        """Sum of tokens in active leases (called under lock)."""
        return sum(lease.tokens for lease in self._leases.values())

    def _calculate_budget(self, status: ResourceStatus) -> int:
        """Calculate current token budget."""
        # Start with what metrics say is available
        base = status.available_tokens

        # Subtract active leases
        with self._lock:
            active = self._active_token_count()

        return max(0, base - active)

    def _get_retry_delay(self, level: ThrottleLevel) -> int:
        """Get retry delay in ms based on throttle level."""
        if level == ThrottleLevel.NORMAL:
            return self.config.retry_delay_normal_ms
        elif level == ThrottleLevel.CAUTION:
            return self.config.retry_delay_caution_ms
        elif level == ThrottleLevel.SOFT_STOP:
            return self.config.retry_delay_soft_stop_ms
        else:
            return self.config.retry_delay_soft_stop_ms * 2

    def try_acquire(
        self,
        operation: str = "inference",
        requested_tokens: int = 1,
        timeout_ms: Optional[int] = None,
    ) -> AcquireResult:
        """Try to acquire tokens for an inference operation.

        Blocks until tokens are available or timeout is reached.
        Uses graduated backoff based on throttle level.
        """
        timeout_ms = timeout_ms or self.config.acquire_timeout_ms
        deadline = time.time() + (timeout_ms / 1000)
        wait_start = time.time()

        while True:
            status = self._get_resource_status()

            # Hard stop = refuse immediately
            if status.throttle_level == ThrottleLevel.HARD_STOP:
                return AcquireResult(
                    success=False,
                    throttle_level=status.throttle_level,
                    memory_ratio=status.memory_ratio,
                    reason="GPU memory at hard stop threshold",
                    wait_time_ms=(time.time() - wait_start) * 1000,
                )

            with self._lock:
                self._reclaim_expired_leases()

                available = self._calculate_budget(status)

                # Can we grant?
                grant = min(requested_tokens, available)

                if grant > 0:
                    # Create lease
                    lease_id = uuid.uuid4().hex[:12]
                    now = datetime.now()
                    lease = Lease(
                        lease_id=lease_id,
                        operation=operation,
                        tokens=grant,
                        acquired_at=now,
                        expires_at=now + timedelta(minutes=self.config.lease_ttl_minutes),
                        memory_ratio_at_acquire=status.memory_ratio,
                    )
                    self._leases[lease_id] = lease

                    # Calculate recommended parallelism
                    rec_parallel = max(1, int(status.memory_free_gb / 3.0))

                    return AcquireResult(
                        success=True,
                        lease_id=lease_id,
                        granted_tokens=grant,
                        recommended_parallelism=rec_parallel,
                        memory_ratio=status.memory_ratio,
                        throttle_level=status.throttle_level,
                        wait_time_ms=(time.time() - wait_start) * 1000,
                    )

            # Check timeout
            if time.time() >= deadline:
                return AcquireResult(
                    success=False,
                    throttle_level=status.throttle_level,
                    memory_ratio=status.memory_ratio,
                    reason=f"Timeout waiting for tokens (requested {requested_tokens})",
                    wait_time_ms=(time.time() - wait_start) * 1000,
                )

            # Wait and retry
            delay = self._get_retry_delay(status.throttle_level)
            time.sleep(delay / 1000)

    def release(
        self,
        lease_id: str,
        peak_memory_mb: int = 0,
        exit_code: int = 0,
        duration_ms: float = 0,
    ) -> ReleaseResult:
        """Release a lease and classify any failure."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)

        if lease is None:
            return ReleaseResult(
                acknowledged=False,
                classification="expired",
                message=f"Lease {lease_id} not found (may have expired)",
            )

        # Get current status for classification
        status = self._get_resource_status()

        # Classify the result
        if exit_code == 0:
            return ReleaseResult(
                acknowledged=True,
                classification="success",
            )

        # Check for OOM indicators
        from .classifier import FailureClassifier

        classifier = FailureClassifier(self.config)
        result = classifier.classify(
            exit_code=exit_code,
            memory_ratio_at_acquire=lease.memory_ratio_at_acquire,
            memory_ratio_at_exit=status.memory_ratio,
            peak_memory_mb=peak_memory_mb,
            duration_ms=duration_ms,
        )

        return ReleaseResult(
            acknowledged=True,
            classification=result.classification.value,
            message=result.message,
            should_retry=result.should_retry,
            retry_with_tokens=result.retry_with_tokens,
        )

    def get_status(self) -> PoolStatus:
        """Get current pool status."""
        status = self._get_resource_status()

        with self._lock:
            self._reclaim_expired_leases()
            leases = list(self._leases.values())
            active_tokens = self._active_token_count()

        return PoolStatus(
            total_budget=status.available_tokens,
            available_tokens=max(0, status.available_tokens - active_tokens),
            active_leases=len(leases),
            active_tokens=active_tokens,
            throttle_level=status.throttle_level,
            memory_ratio=status.memory_ratio,
            memory_free_gb=status.memory_free_gb,
            leases=leases,
        )

    async def start_monitor(self):
        """Start background monitoring task."""
        if self._running:
            return

        self._running = True

        async def monitor_loop():
            while self._running:
                # Refresh status
                self._get_resource_status()

                # Check for lease warnings
                with self._lock:
                    now = datetime.now()
                    warning_threshold = timedelta(minutes=self.config.lease_ttl_minutes - 1)

                    for lease in self._leases.values():
                        if not lease.warning_logged:
                            if now - lease.acquired_at > warning_threshold:
                                print(
                                    f"[Governor] Warning: Lease {lease.lease_id} "
                                    f"({lease.operation}) approaching TTL"
                                )
                                lease.warning_logged = True

                await asyncio.sleep(self.config.monitor_interval_ms / 1000)

        self._monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitor(self):
        """Stop background monitoring task."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
