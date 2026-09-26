"""
Data & Experiment Versioning — Immutable Experiment References for 100% Reproducibility.

Every backtest, ML experiment, and evolutionary cycle must record and verify:
- dataset_version
- universe_version
- feature_version
- strategy_version
- model_version
- code_version
- random_seed
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

REGISTRY_DIR = Path(os.getenv("EXPERIMENT_REGISTRY_DIR", "trading/experiments"))


@dataclass
class ExperimentMetadata:
    experiment_id: str
    dataset_version: str
    universe_version: str
    feature_version: str
    strategy_version: str
    model_version: str
    code_version: str
    random_seed: int
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_signature(self) -> str:
        """Deterministic fingerprint of experiment inputs."""
        payload = f"{self.dataset_version}|{self.universe_version}|{self.feature_version}|{self.strategy_version}|{self.model_version}|{self.code_version}|{self.random_seed}|{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ExperimentRegistry:
    """Registry to record and reload experiments for exact reproducibility."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or REGISTRY_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def record_experiment(self, metadata: ExperimentMetadata) -> Path:
        """Persist experiment metadata to immutable JSON record."""
        file_name = f"{metadata.experiment_id}.json"
        target_path = self.storage_dir / file_name
        data = metadata.to_dict()
        data["signature"] = metadata.compute_signature()
        target_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return target_path

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment metadata."""
        file_path = self.storage_dir / f"{experiment_id}.json"
        if file_path.exists():
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def list_experiments(self) -> list[Dict[str, Any]]:
        """List all recorded experiments."""
        experiments = []
        for p in self.storage_dir.glob("*.json"):
            try:
                experiments.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)


def create_experiment_context(
    strategy_version: str = "v1.0.0",
    dataset_version: str = "2026-NSE-DAILY",
    universe_version: str = "UNIV-100-2026",
    feature_version: str = "FEAT-V2-TIMESAFE",
    model_version: str = "META-ENS-V1",
    code_version: str = "GIT-CLEAN",
    random_seed: int = 42,
    parameters: Optional[Dict[str, Any]] = None,
) -> ExperimentMetadata:
    """Factory helper to generate initialized ExperimentMetadata."""
    exp_id = f"exp_{int(time.time())}_{hashlib.md5(strategy_version.encode()).hexdigest()[:6]}"
    return ExperimentMetadata(
        experiment_id=exp_id,
        dataset_version=dataset_version,
        universe_version=universe_version,
        feature_version=feature_version,
        strategy_version=strategy_version,
        model_version=model_version,
        code_version=code_version,
        random_seed=random_seed,
        parameters=parameters or {},
    )
