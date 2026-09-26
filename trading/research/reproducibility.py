"""
Quantitative Reproducibility Engine.

Captures total environment state for every experiment:
- Git commit SHA
- Dataset SHA & version
- Universe version & eligibility rules
- Feature registry version & parameter hash
- Model architecture version
- StrategyDNA & Random seed
- Python environment & package manifests

Provides `reproduce_experiment(experiment_id)` to re-verify results.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import subprocess
import hashlib
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPRO_DIR = Path(__file__).parent.parent.parent / "data_lake" / "reproducibility"


@dataclass
class EnvironmentSnapshot:
    git_commit_sha: str
    git_dirty: bool
    python_version: str
    key_packages: Dict[str, str]
    random_seed: int
    config_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityBundle:
    experiment_id: str
    strategy_id: str
    strategy_dna: Dict[str, Any]
    dataset_sha256: str
    dataset_version: str
    feature_version: str
    universe_version: str
    model_version: str
    environment: EnvironmentSnapshot
    baseline_metrics: Dict[str, float]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "strategy_id": self.strategy_id,
            "strategy_dna": self.strategy_dna,
            "dataset_sha256": self.dataset_sha256,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "universe_version": self.universe_version,
            "model_version": self.model_version,
            "environment": self.environment.to_dict(),
            "baseline_metrics": self.baseline_metrics,
            "created_at": self.created_at,
        }


class ReproducibilityEngine:
    """
    Manages cryptographic provenance and deterministic experiment replication.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DEFAULT_REPRO_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_file = self.storage_dir / "bundles.json"
        self.bundles: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.bundles_file.exists():
            try:
                return json.loads(self.bundles_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load reproducibility bundles: {e}")
        return {}

    def _save(self) -> None:
        try:
            self.bundles_file.write_text(json.dumps(self.bundles, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save reproducibility bundles: {e}")

    @staticmethod
    def capture_environment(seed: int = 42, config_dict: Optional[Dict[str, Any]] = None) -> EnvironmentSnapshot:
        """Capture live runtime state."""
        # 1. Git Commit SHA
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit = res.stdout.strip()
            dirty_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            dirty = len(dirty_res.stdout.strip()) > 0
        except Exception:
            commit = "UNKNOWN_NO_GIT"
            dirty = False

        # 2. Key quantitative packages
        packages = {}
        for p in ["duckdb", "pyarrow", "polars", "optuna", "pandas", "numpy", "scipy", "sklearn"]:
            try:
                mod = __import__(p)
                packages[p] = getattr(mod, "__version__", "installed")
            except ImportError:
                packages[p] = "not_installed"

        # 3. Config Hash
        cfg_str = json.dumps(config_dict or {}, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]

        return EnvironmentSnapshot(
            git_commit_sha=commit,
            git_dirty=dirty,
            python_version=sys.version.split()[0],
            key_packages=packages,
            random_seed=seed,
            config_hash=cfg_hash,
            timestamp=datetime.utcnow().isoformat(),
        )

    def freeze_experiment(
        self,
        experiment_id: str,
        strategy_id: str,
        strategy_dna: Dict[str, Any],
        dataset_sha256: str,
        dataset_version: str,
        feature_version: str,
        universe_version: str,
        model_version: str,
        baseline_metrics: Dict[str, float],
        seed: int = 42,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> ReproducibilityBundle:
        """
        Freezes complete reproducibility bundle for an experiment.
        """
        env = self.capture_environment(seed=seed, config_dict=config_dict)
        bundle = ReproducibilityBundle(
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            strategy_dna=strategy_dna,
            dataset_sha256=dataset_sha256,
            dataset_version=dataset_version,
            feature_version=feature_version,
            universe_version=universe_version,
            model_version=model_version,
            environment=env,
            baseline_metrics=baseline_metrics,
            created_at=datetime.utcnow().isoformat(),
        )

        self.bundles[experiment_id] = bundle.to_dict()
        self._save()
        logger.info(f"Frozen reproducibility bundle for: {experiment_id}")
        return bundle

    def get_bundle(self, experiment_id: str) -> Optional[ReproducibilityBundle]:
        """Retrieve frozen bundle for experiment ID."""
        data = self.bundles.get(experiment_id)
        if not data:
            return None
        env_data = data.pop("environment")
        env = EnvironmentSnapshot(**env_data)
        return ReproducibilityBundle(environment=env, **data)

    def reproduce_experiment(
        self,
        experiment_id: str,
        reproduced_metrics: Dict[str, float],
        tolerance_pct: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Verifies whether a reproduction attempt strictly matches recorded baseline metrics.
        """
        bundle = self.get_bundle(experiment_id)
        if not bundle:
            return {"status": "NOT_FOUND", "experiment_id": experiment_id, "reproduced": False}

        discrepancies = {}
        passed = True

        for k, exp_val in bundle.baseline_metrics.items():
            curr_val = reproduced_metrics.get(k)
            if curr_val is None:
                discrepancies[k] = {"expected": exp_val, "actual": "MISSING"}
                passed = False
                continue

            pct_diff = abs(curr_val - exp_val) / max(abs(exp_val), 1e-4)
            if pct_diff > tolerance_pct:
                discrepancies[k] = {
                    "expected": exp_val,
                    "actual": curr_val,
                    "pct_diff": round(pct_diff * 100.0, 2),
                    "tolerance": f"{tolerance_pct*100}%"
                }
                passed = False

        return {
            "status": "VERIFIED" if passed else "REPRODUCTION_FAILED",
            "experiment_id": experiment_id,
            "strategy_id": bundle.strategy_id,
            "recorded_git_commit": bundle.environment.git_commit_sha,
            "recorded_dataset_sha": bundle.dataset_sha256[:12],
            "discrepancies": discrepancies,
            "reproduced": passed,
        }
