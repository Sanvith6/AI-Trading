"""
Laya Decision Cache & Deterministic Historical Reproducibility Engine.

Keyed by: (state_hash, task, model_version, configuration_version)
Enforces exact historical determinism across repeated backtest runs (Directive 4).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .schemas import LayaDecision
from .config import laya_config

logger = logging.getLogger(__name__)


class LayaDecisionCache:
    """
    Persistent key-value cache for Laya inference decisions.
    Eliminates redundant GPU inference and guarantees deterministic replay.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or laya_config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "decision_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading decision cache: {e}")
        return {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving decision cache: {e}")

    def make_cache_key(
        self,
        state_hash: str,
        task: str,
        model_version: str = laya_config.model_version,
        config_version: str = laya_config.task_version,
    ) -> str:
        """Constructs unique cache key ensuring total parameter freezing (Directive 4)."""
        return f"{state_hash}_{task.upper()}_{model_version}_{config_version}"

    def get(
        self,
        state_hash: str,
        task: str,
        model_version: str = laya_config.model_version,
        config_version: str = laya_config.task_version,
    ) -> Optional[LayaDecision]:
        """Fetch cached decision if available."""
        if not laya_config.cache_enabled:
            return None

        key = self.make_cache_key(state_hash, task, model_version, config_version)
        data = self._cache.get(key)
        if data:
            logger.debug(f"Cache HIT for Laya decision: {key[:20]}...")
            return LayaDecision(**data)
        return None

    def put(self, decision: LayaDecision, config_version: str = laya_config.task_version) -> None:
        """Store validated decision in cache."""
        if not laya_config.cache_enabled or not decision.is_valid:
            return

        key = self.make_cache_key(decision.state_hash, decision.task, decision.model_version, config_version)
        self._cache[key] = decision.to_dict()
        # Atomic/periodic save
        if len(self._cache) % 10 == 0:
            self._save()

    def flush(self) -> None:
        """Flush in-memory cache to disk."""
        self._save()

    def size(self) -> int:
        return len(self._cache)


# Global singleton cache instance
laya_cache = LayaDecisionCache()
