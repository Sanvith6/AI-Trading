"""
Feature Registry & Formal Feature Definition Store.

Guarantees full auditability and semantic versioning of all quantitative features:
- Answers: 'What exact feature definition was used in experiment 142?'
- Eliminates silent feature changes between research and paper trading.
- Tracks feature formula, dependencies, lookback window, and timestamp semantics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_REGISTRY_PATH = Path(__file__).parent.parent.parent / "data_lake" / "feature_registry.json"


@dataclass
class FeatureDefinition:
    feature_name: str  # e.g., "ret_20d", "rsi_14", "vol_20d"
    version: str  # e.g., "v1", "v2"
    description: str
    source: str  # e.g., "price", "volume", "fundamentals", "market_context"
    lookback: int  # number of periods/bars
    timestamp_semantics: str  # "LAGGED_EOD", "REALTIME_BAR_CLOSE", "AS_OF_FILING"
    dependencies: List[str]  # e.g., ["close"]
    market: str  # "US", "IN", "GLOBAL"
    status: str  # "ACTIVE", "DEPRECATED", "EXPERIMENTAL"
    formula_expression: str = ""
    feature_hash: str = ""

    def __post_init__(self):
        if not self.feature_hash:
            # Deterministic hash of the definition to detect mutations
            raw = f"{self.feature_name}:{self.version}:{self.lookback}:{self.timestamp_semantics}:{','.join(self.dependencies)}:{self.formula_expression}"
            self.feature_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureRegistry:
    """
    Registry maintaining active and historical feature definitions.
    """

    CORE_FEATURE_CATALOG: List[FeatureDefinition] = [
        FeatureDefinition(
            feature_name="ret_1d", version="v1", description="1-day simple return",
            source="price", lookback=1, timestamp_semantics="LAGGED_EOD", dependencies=["close"],
            market="GLOBAL", status="ACTIVE", formula_expression="close[t] / close[t-1] - 1.0"
        ),
        FeatureDefinition(
            feature_name="ret_5d", version="v1", description="5-day simple return",
            source="price", lookback=5, timestamp_semantics="LAGGED_EOD", dependencies=["close"],
            market="GLOBAL", status="ACTIVE", formula_expression="close[t] / close[t-5] - 1.0"
        ),
        FeatureDefinition(
            feature_name="ret_20d", version="v1", description="20-day momentum return",
            source="price", lookback=20, timestamp_semantics="LAGGED_EOD", dependencies=["close"],
            market="GLOBAL", status="ACTIVE", formula_expression="close[t] / close[t-20] - 1.0"
        ),
        FeatureDefinition(
            feature_name="rsi_14", version="v1", description="14-day Relative Strength Index",
            source="price", lookback=14, timestamp_semantics="LAGGED_EOD", dependencies=["close"],
            market="GLOBAL", status="ACTIVE", formula_expression="100 - (100 / (1 + RS(14)))"
        ),
        FeatureDefinition(
            feature_name="vol_20d", version="v1", description="20-day annualized rolling volatility",
            source="price", lookback=20, timestamp_semantics="LAGGED_EOD", dependencies=["close"],
            market="GLOBAL", status="ACTIVE", formula_expression="std(ret_1d, 20) * sqrt(252)"
        ),
        FeatureDefinition(
            feature_name="atr_14", version="v1", description="14-day Average True Range",
            source="price", lookback=14, timestamp_semantics="LAGGED_EOD", dependencies=["high", "low", "close"],
            market="GLOBAL", status="ACTIVE", formula_expression="rolling_mean(TrueRange, 14)"
        ),
        FeatureDefinition(
            feature_name="f_quality_score", version="v1", description="Fundamental Composite Quality Score",
            source="fundamentals", lookback=90, timestamp_semantics="AS_OF_FILING", dependencies=["operating_margin", "roe", "free_cash_flow"],
            market="US", status="ACTIVE", formula_expression="clip(operating_margin*2 + roe*0.5 + fcf/revenue, 0, 1)"
        ),
        FeatureDefinition(
            feature_name="market_risk_on_off", version="v1", description="Macro Market Context Risk Regime",
            source="market_context", lookback=50, timestamp_semantics="LAGGED_EOD", dependencies=["SPY", "QQQ", "VIX", "Sector_ETFs"],
            market="US", status="ACTIVE", formula_expression="composite(tech_spread, vix_penalty, spy_trend)"
        ),
    ]

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_FEATURE_REGISTRY_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to read feature registry: {e}")

        # Initialize defaults
        catalog = {}
        for fd in self.CORE_FEATURE_CATALOG:
            key = f"{fd.feature_name}:{fd.version}"
            catalog[key] = fd.to_dict()
        self._save(catalog)
        return catalog

    def _save(self, catalog: Dict[str, Dict[str, Any]]) -> None:
        try:
            self.storage_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write feature registry: {e}")

    def register_feature(self, feature_def: FeatureDefinition) -> str:
        """
        Register a new feature definition or new version.
        Disallows silent mutations of existing feature hashes.
        """
        key = f"{feature_def.feature_name}:{feature_def.version}"
        if key in self.catalog:
            existing = self.catalog[key]
            if existing.get("feature_hash") != feature_def.feature_hash:
                raise ValueError(
                    f"Feature mutation forbidden! {key} is already registered with hash {existing.get('feature_hash')}. "
                    f"Increment version (e.g. {feature_def.feature_name}:v{int(feature_def.version.replace('v',''))+1}) instead."
                )
            return key

        self.catalog[key] = feature_def.to_dict()
        self._save(self.catalog)
        logger.info(f"Registered feature definition: {key} (hash={feature_def.feature_hash})")
        return key

    def get_feature(self, feature_name: str, version: Optional[str] = None) -> Optional[FeatureDefinition]:
        """Fetch a specific feature definition."""
        if version:
            key = f"{feature_name}:{version}"
            data = self.catalog.get(key)
        else:
            matching = [k for k in self.catalog.keys() if k.startswith(f"{feature_name}:")]
            if not matching:
                return None
            key = sorted(matching)[-1]
            data = self.catalog.get(key)

        if data:
            return FeatureDefinition(**data)
        return None

    def list_active_features(self, market: Optional[str] = None) -> List[FeatureDefinition]:
        """List active features, optionally filtered by market."""
        results = []
        for data in self.catalog.values():
            fd = FeatureDefinition(**data)
            if fd.status == "ACTIVE":
                if market is None or fd.market in ("GLOBAL", market.upper()):
                    results.append(fd)
        return results

    def compute_feature_set_fingerprint(self, feature_keys: List[str]) -> str:
        """Create a cryptographic signature for an exact set of features used in an experiment."""
        hashes = []
        for k in sorted(feature_keys):
            data = self.catalog.get(k)
            if data:
                hashes.append(data.get("feature_hash", ""))
            else:
                hashes.append(k)
        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
