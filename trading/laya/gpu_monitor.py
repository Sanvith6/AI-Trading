"""
Laya GPU Resource Isolation & Telemetry Monitor.

Enforces Directive 21:
Monitors NVIDIA RTX 3050 Laptop GPU (4GB VRAM):
- GPU utilization (%)
- VRAM (Total, Used, Free MiB)
- Temperature (C)
- Host process memory (psutil)

If Laya consumes excessive resources or approaches OOM:
- Safe degradation to baseline without stopping trading operations.
- The trading engine remains 100% functional without Laya.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import subprocess
import logging
import os
import psutil

from .config import laya_config

logger = logging.getLogger(__name__)


@dataclass
class GpuTelemetry:
    """Snapshot of hardware utilization and temperature."""
    is_available: bool
    device_name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    gpu_util_pct: float
    temperature_c: float
    process_memory_mb: float
    is_safe: bool
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GpuMonitor:
    """Hardware watchdog protecting trading processes from GPU exhaustion or overheating."""

    def __init__(
        self,
        max_vram_mb: float = laya_config.max_vram_mb,
        max_temp_c: float = laya_config.max_temp_c,
    ):
        self.max_vram_mb = max_vram_mb
        self.max_temp_c = max_temp_c
        self._last_telemetry: Optional[GpuTelemetry] = None

    def check_telemetry(self) -> GpuTelemetry:
        """Queries nvidia-smi and system memory to determine hardware safety."""
        process_mem = 0.0
        try:
            p = psutil.Process(os.getpid())
            process_mem = p.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        try:
            # Run nvidia-smi with minimal footprint and 2s timeout
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False
            )

            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 6:
                    name = parts[0]
                    total = float(parts[1])
                    used = float(parts[2])
                    free = float(parts[3])
                    util = float(parts[4])
                    temp = float(parts[5])

                    warning = None
                    is_safe = True

                    if used >= self.max_vram_mb:
                        warning = f"High VRAM warning: {used:.0f} MiB used (limit: {self.max_vram_mb} MiB)"
                        is_safe = False
                    elif temp >= self.max_temp_c:
                        warning = f"High GPU temp warning: {temp:.0f} C (limit: {self.max_temp_c} C)"
                        is_safe = False

                    telemetry = GpuTelemetry(
                        is_available=True,
                        device_name=name,
                        vram_total_mb=total,
                        vram_used_mb=used,
                        vram_free_mb=free,
                        gpu_util_pct=util,
                        temperature_c=temp,
                        process_memory_mb=round(process_mem, 1),
                        is_safe=is_safe,
                        warning=warning
                    )
                    self._last_telemetry = telemetry
                    return telemetry

        except Exception as e:
            logger.debug(f"GPU monitor query exception: {e}")

        # If nvidia-smi is unavailable or failed, fail safely: report unavailable but do not crash
        fallback = GpuTelemetry(
            is_available=False,
            device_name="Unknown / CPU",
            vram_total_mb=0.0,
            vram_used_mb=0.0,
            vram_free_mb=0.0,
            gpu_util_pct=0.0,
            temperature_c=0.0,
            process_memory_mb=round(process_mem, 1),
            is_safe=True,
            warning="nvidia-smi telemetry unavailable"
        )
        self._last_telemetry = fallback
        return fallback

    def is_safe_for_inference(self) -> bool:
        """Returns True if GPU resources permit continued neural inference."""
        t = self.check_telemetry()
        return t.is_safe


# Global singleton monitor
gpu_monitor = GpuMonitor()
