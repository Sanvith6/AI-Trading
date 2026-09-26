"""
Experiment Tracking & Model Lineage Engine (MLflow Integration).

Maintains end-to-end auditability of every model, alpha search, and backtest:
- Experiment ID & Run ID
- Strategy ID & Strategy DNA
- Model architecture & parameters
- Dataset version, Feature version, Universe version
- Code commit SHA & Random seed
- Training, Validation, and Holdout periods
- Performance metrics & Artifact paths
- Promotion State (CANDIDATE -> CHALLENGER -> CHAMPION -> RETIRED)

Seamlessly integrates with external MLflow if available, while maintaining
an immutable local JSON/file-based tracking store.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MLOPS_DIR = Path(__file__).parent.parent.parent / "data_lake" / "mlops"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("mlflow is not installed. ExperimentTracker is using native persistent storage.")


@dataclass
class ExperimentRun:
    experiment_id: str
    run_id: str
    strategy_id: str
    strategy_dna: Dict[str, Any]
    model_name: str
    dataset_version: str
    feature_version: str
    universe_version: str
    git_commit_sha: str
    seed: int
    training_period: Dict[str, str]  # {"start": ..., "end": ...}
    validation_period: Dict[str, str]
    test_period: Dict[str, str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    promotion_state: str  # "CANDIDATE", "CHALLENGER", "CHAMPION", "REJECTED", "RETIRED"
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """
    Manages quantitative experiments and model governance.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DEFAULT_MLOPS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.storage_dir / "experiment_runs.json"
        self.runs: Dict[str, Dict[str, Any]] = self._load()

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file:///{str(self.storage_dir).replace(chr(92), '/')}/mlruns")

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.ledger_file.exists():
            try:
                return json.loads(self.ledger_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load experiment ledger: {e}")
        return {}

    def _save(self) -> None:
        try:
            self.ledger_file.write_text(json.dumps(self.runs, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save experiment ledger: {e}")

    def log_run(
        self,
        experiment_id: str,
        strategy_id: str,
        strategy_dna: Dict[str, Any],
        model_name: str,
        dataset_version: str,
        feature_version: str,
        universe_version: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_period: Dict[str, str],
        validation_period: Dict[str, str],
        test_period: Optional[Dict[str, str]] = None,
        git_commit_sha: str = "HEAD",
        seed: int = 42,
        artifacts: Optional[List[str]] = None,
        promotion_state: str = "CANDIDATE",
    ) -> ExperimentRun:
        """
        Record a complete quantitative experiment run.
        """
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        run = ExperimentRun(
            experiment_id=experiment_id,
            run_id=run_id,
            strategy_id=strategy_id,
            strategy_dna=strategy_dna,
            model_name=model_name,
            dataset_version=dataset_version,
            feature_version=feature_version,
            universe_version=universe_version,
            git_commit_sha=git_commit_sha,
            seed=seed,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period or {},
            parameters=parameters,
            metrics={k: round(float(v), 4) for k, v in metrics.items()},
            artifacts=artifacts or [],
            promotion_state=promotion_state,
            created_at=datetime.utcnow().isoformat(),
        )

        self.runs[run_id] = run.to_dict()
        self._save()

        # Sync with MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment_id)
                with mlflow.start_run(run_name=run_id):
                    mlflow.log_params(parameters)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags({
                        "strategy_id": strategy_id,
                        "model_name": model_name,
                        "dataset_version": dataset_version,
                        "feature_version": feature_version,
                        "promotion_state": promotion_state,
                    })
            except Exception as e:
                logger.warning(f"MLflow logging error: {e}")

        logger.info(f"Logged experiment run: {run_id} ({strategy_id} | Sharpe={metrics.get('sharpe', 'N/A')})")
        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Fetch a specific experiment run."""
        data = self.runs.get(run_id)
        if data:
            return ExperimentRun(**data)
        return None

    def list_runs(self, experiment_id: Optional[str] = None) -> List[ExperimentRun]:
        """List all runs, optionally filtered by experiment."""
        results = []
        for data in self.runs.values():
            if experiment_id is None or data.get("experiment_id") == experiment_id:
                results.append(ExperimentRun(**data))
        return sorted(results, key=lambda x: x.created_at, reverse=True)

    def promote_run(self, run_id: str, new_state: str) -> bool:
        """Promote or reject a candidate run (e.g. CHALLENGER -> CHAMPION)."""
        valid_states = {"CANDIDATE", "CHALLENGER", "CHAMPION", "REJECTED", "RETIRED"}
        if new_state not in valid_states:
            raise ValueError(f"Invalid promotion state {new_state}. Must be one of {valid_states}")

        if run_id not in self.runs:
            return False

        self.runs[run_id]["promotion_state"] = new_state
        self._save()
        logger.info(f"Updated run {run_id} promotion state to: {new_state}")
        return True
