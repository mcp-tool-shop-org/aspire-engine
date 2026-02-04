"""GPU and system resource metrics for ASPIRE Governor."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import subprocess
import re


class ThrottleLevel(Enum):
    """Current throttle state based on resource pressure."""
    NORMAL = "normal"          # Full speed, immediate grants
    CAUTION = "caution"        # Light throttling, small delays
    SOFT_STOP = "soft_stop"    # Heavy throttling, longer delays
    HARD_STOP = "hard_stop"    # Refuse new leases


@dataclass
class GPUStatus:
    """Status of a single GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    temperature_c: int

    @property
    def memory_ratio(self) -> float:
        """Memory utilization ratio (0.0 - 1.0)."""
        if self.memory_total_mb == 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024

    @property
    def memory_free_gb(self) -> float:
        return self.memory_free_mb / 1024


@dataclass
class ResourceStatus:
    """Aggregated resource status for governor decisions."""
    gpus: List[GPUStatus]
    primary_gpu: Optional[GPUStatus]
    memory_ratio: float              # Primary GPU memory utilization
    memory_free_gb: float            # Primary GPU free memory
    memory_total_gb: float           # Primary GPU total memory
    throttle_level: ThrottleLevel
    available_tokens: int
    reason: str                      # Human-readable status

    @classmethod
    def unavailable(cls, reason: str = "GPU metrics unavailable") -> "ResourceStatus":
        """Create a status indicating metrics couldn't be collected."""
        return cls(
            gpus=[],
            primary_gpu=None,
            memory_ratio=0.0,
            memory_free_gb=0.0,
            memory_total_gb=0.0,
            throttle_level=ThrottleLevel.NORMAL,  # Fail open
            available_tokens=4,  # Conservative default
            reason=reason,
        )


class GPUMetrics:
    """Collect GPU metrics via nvidia-smi.

    On Windows with CUDA, nvidia-smi is the standard way to query
    GPU memory and utilization. Falls back gracefully if unavailable.
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvidia_smi_path = self._find_nvidia_smi()

    def _find_nvidia_smi(self) -> Optional[str]:
        """Locate nvidia-smi executable."""
        import shutil

        # Common locations on Windows
        paths = [
            shutil.which("nvidia-smi"),
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]

        for path in paths:
            if path and self._test_nvidia_smi(path):
                return path

        return None

    def _test_nvidia_smi(self, path: str) -> bool:
        """Test if nvidia-smi works at this path."""
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def query(self) -> List[GPUStatus]:
        """Query all GPUs and return their status."""
        if not self._nvidia_smi_path:
            return []

        try:
            result = subprocess.run(
                [
                    self._nvidia_smi_path,
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append(GPUStatus(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_used_mb=int(parts[3]),
                        memory_free_mb=int(parts[4]),
                        utilization_percent=int(parts[5]),
                        temperature_c=int(parts[6]),
                    ))

            return gpus

        except Exception:
            return []

    def get_status(self, config: "GovernorConfig") -> ResourceStatus:
        """Get aggregated resource status for governor decisions."""
        from .config import GovernorConfig

        gpus = self.query()

        if not gpus:
            return ResourceStatus.unavailable("No GPUs found or nvidia-smi unavailable")

        # Use primary GPU (by index)
        primary = None
        for gpu in gpus:
            if gpu.index == self.gpu_index:
                primary = gpu
                break

        if not primary:
            primary = gpus[0]

        # Calculate throttle level
        ratio = primary.memory_ratio

        if ratio >= config.hard_stop_ratio:
            level = ThrottleLevel.HARD_STOP
            reason = f"GPU memory critical ({ratio:.0%})"
        elif ratio >= config.soft_stop_ratio:
            level = ThrottleLevel.SOFT_STOP
            reason = f"GPU memory high ({ratio:.0%})"
        elif ratio >= config.caution_ratio:
            level = ThrottleLevel.CAUTION
            reason = f"GPU memory elevated ({ratio:.0%})"
        else:
            level = ThrottleLevel.NORMAL
            reason = f"GPU memory OK ({ratio:.0%})"

        # Calculate available tokens
        usable_gb = max(0, primary.memory_free_gb - config.safety_reserve_gb)
        raw_tokens = int(usable_gb / config.gb_per_token)
        available = max(config.min_tokens, min(config.max_tokens, raw_tokens))

        # Override to 0 at hard stop
        if level == ThrottleLevel.HARD_STOP:
            available = 0

        return ResourceStatus(
            gpus=gpus,
            primary_gpu=primary,
            memory_ratio=ratio,
            memory_free_gb=primary.memory_free_gb,
            memory_total_gb=primary.memory_total_gb,
            throttle_level=level,
            available_tokens=available,
            reason=reason,
        )


class MockGPUMetrics(GPUMetrics):
    """Mock GPU metrics for testing without a real GPU."""

    def __init__(
        self,
        memory_total_mb: int = 16384,  # 16GB
        memory_used_mb: int = 4096,    # 4GB used
        gpu_index: int = 0,
    ):
        super().__init__(gpu_index)
        self._memory_total = memory_total_mb
        self._memory_used = memory_used_mb

    def set_memory_used(self, mb: int):
        """Simulate memory pressure changes."""
        self._memory_used = min(mb, self._memory_total)

    def query(self) -> List[GPUStatus]:
        """Return mock GPU status."""
        return [GPUStatus(
            index=0,
            name="Mock GPU (Testing)",
            memory_total_mb=self._memory_total,
            memory_used_mb=self._memory_used,
            memory_free_mb=self._memory_total - self._memory_used,
            utilization_percent=50,
            temperature_c=65,
        )]
