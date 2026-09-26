"""
Laya Configuration & Operational Parameters.

Defines operational modes, task-specific probability thresholds,
latency budgets, inference limits, and GPU safety boundaries.
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path


@dataclass
class LayaConfig:
    # Operational Mode: 'off', 'shadow', 'advisory', 'gate' (Default: 'shadow')
    mode: str = os.getenv("LAYA_MODE", "shadow").lower()

    # Local SystemOne Service Endpoints
    endpoint: str = os.getenv("LAYA_ENDPOINT", "http://127.0.0.1:8000/v1/systemone")
    health_endpoint: str = os.getenv("LAYA_HEALTH_ENDPOINT", "http://127.0.0.1:8000/health")

    # Timeouts & Latency Budget (in milliseconds)
    request_timeout_seconds: float = float(os.getenv("LAYA_TIMEOUT_SEC", "3.5"))
    latency_budget_ms: float = float(os.getenv("LAYA_LATENCY_BUDGET_MS", "1800.0"))
    max_retries: int = int(os.getenv("LAYA_MAX_RETRIES", "1"))

    # Inference Volume Controls (Directive 5)
    # Only the Top N pre-screened candidates are sent to Laya
    max_candidates_per_scan: int = int(os.getenv("LAYA_MAX_CANDIDATES", "5"))
    max_concurrent_workers: int = int(os.getenv("LAYA_MAX_CONCURRENT", "2"))

    # Task-Specific Thresholds (Directive 3 - No generic threshold)
    trade_quality_threshold: float = float(os.getenv("LAYA_TQ_THRESHOLD", "0.60"))
    risk_threshold: float = float(os.getenv("LAYA_RISK_THRESHOLD", "0.65"))
    regime_threshold: float = float(os.getenv("LAYA_REGIME_THRESHOLD", "0.50"))

    # Decision Multipliers for Bounded Risk Policy (Directive 2)
    reduce_position_multiplier: float = float(os.getenv("LAYA_REDUCE_MULTIPLIER", "0.50"))

    # Reproducibility & Caching Settings (Directive 4)
    cache_enabled: bool = os.getenv("LAYA_CACHE_ENABLED", "true").lower() == "true"
    cache_dir: Path = Path(__file__).parent.parent.parent / "data_lake" / "laya_cache"

    # GPU Guardrails (RTX 3050 4GB)
    max_vram_mb: float = 3800.0  # Alert if VRAM usage exceeds 3800 MiB / 4096 MiB
    max_temp_c: float = 82.0

    # Model & Version Metadata
    model_version: str = "convaiinnovations/laya-rl-agent"
    task_version: str = "v1.0"
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        return d


# Global default configuration instance
laya_config = LayaConfig()
