"""
Dataset Registry & Manifest Engine.

Guarantees full reproducibility and point-in-time correctness of quantitative research datasets.
Every dataset slice used in training, validation, or backtesting is formally registered with
a cryptographic checksum, schema version, date range, and feature version.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent / "data_lake" / "dataset_registry.json"


@dataclass
class DatasetManifest:
    dataset_id: str
    version: str
    market: str
    source: str
    created_at: str
    date_range: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    symbols: List[str]
    schema_version: str
    feature_version: str
    checksum_sha256: str
    row_count: int
    column_names: List[str]
    storage_path: str
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetRegistry:
    """
    Maintains an immutable ledger of research datasets.
    Prevents silent dataset mutations and guarantees that experiments can reload
    the exact data environment used at research time.
    """

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read dataset registry: {e}")
                return {}
        return {}

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.datasets, f, indent=2)

    def register_dataset(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        market: str,
        source: str,
        storage_path: str,
        feature_version: str = "v1",
        schema_version: str = "1.0",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetManifest:
        """
        Register a new dataset version with automatic hash calculation and metadata recording.
        """
        # Calculate content checksum
        csv_bytes = df.to_csv(index=True).encode("utf-8")
        checksum = hashlib.sha256(csv_bytes).hexdigest()

        # Date range detection
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = str(df.index.min().date())
            end_date = str(df.index.max().date())
        elif "date" in df.columns:
            start_date = str(pd.to_datetime(df["date"]).min().date())
            end_date = str(pd.to_datetime(df["date"]).max().date())
        else:
            start_date = "UNKNOWN"
            end_date = "UNKNOWN"

        symbols = []
        if "symbol" in df.columns:
            symbols = sorted(df["symbol"].dropna().unique().tolist())
        elif isinstance(df.columns, pd.MultiIndex):
            symbols = sorted(df.columns.levels[0].tolist())

        # Versioning: if dataset_id exists, auto-increment or use semantic version
        existing_versions = [k for k in self.datasets.keys() if k.startswith(f"{dataset_id}:v")]
        version = f"v{len(existing_versions) + 1}"
        full_key = f"{dataset_id}:{version}"

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            market=market.upper(),
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
            date_range={"start": start_date, "end": end_date},
            symbols=symbols,
            schema_version=schema_version,
            feature_version=feature_version,
            checksum_sha256=checksum,
            row_count=len(df),
            column_names=[str(c) for c in df.columns],
            storage_path=str(storage_path),
            description=description,
            metadata=metadata or {},
        )

        self.datasets[full_key] = manifest.to_dict()
        self._save()
        logger.info(f"Registered research dataset: {full_key} (rows={len(df)}, hash={checksum[:8]})")
        return manifest

    def get_manifest(self, dataset_id: str, version: Optional[str] = None) -> Optional[DatasetManifest]:
        """Fetch manifest for a dataset, defaulting to latest version."""
        if version:
            key = f"{dataset_id}:{version}"
            data = self.datasets.get(key)
        else:
            matching = [k for k in self.datasets.keys() if k.startswith(f"{dataset_id}:")]
            if not matching:
                return None
            key = sorted(matching)[-1]
            data = self.datasets.get(key)

        if data:
            return DatasetManifest(**data)
        return None

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return list of all registered dataset manifests."""
        return list(self.datasets.values())

    def verify_integrity(self, dataset_id: str, version: str, df: pd.DataFrame) -> bool:
        """Verify that a loaded DataFrame matches the recorded SHA256 checksum."""
        manifest = self.get_manifest(dataset_id, version)
        if not manifest:
            logger.error(f"Cannot verify: Manifest not found for {dataset_id}:{version}")
            return False

        csv_bytes = df.to_csv(index=True).encode("utf-8")
        current_hash = hashlib.sha256(csv_bytes).hexdigest()
        is_valid = current_hash == manifest.checksum_sha256
        if not is_valid:
            logger.warning(f"Dataset integrity mismatch! Expected {manifest.checksum_sha256}, got {current_hash}")
        return is_valid
