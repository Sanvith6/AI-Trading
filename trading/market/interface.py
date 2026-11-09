"""
Market Data Provider Abstraction Interface.

Shared across both US_PAPER and US_LIVE to ensure 100% data pipeline consistency.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class MarketQuote:
    symbol: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_pct: float
    data_timestamp: str         # Exchange data timestamp (ISO 8601)
    received_at: float          # Unix epoch seconds when received locally
    age_seconds: float          # Quote age relative to current wall-clock
    bar_interval: str = "1m"    # Intraday cadence ("1m", "5m", "15m", "1d")
    data_mode: str = "real-time"  # "real-time", "delayed", "historical"
    source: str = "us_market_feed"
    quality_status: str = "VALID"

    provider_latency_seconds: float = 0.0
    received_age_seconds: float = 0.0

    # Backward compatibility properties
    @property
    def timestamp(self) -> str:
        return self.data_timestamp

    @property
    def retrieval_timestamp(self) -> float:
        return self.received_at

    @property
    def data_age_seconds(self) -> float:
        return self.age_seconds

    @property
    def quote_age_seconds(self) -> float:
        return self.age_seconds


    def is_fresh_for_strategy(
        self,
        strategy_cadence: str = "1m",
        max_age_cap: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Validates quote freshness against specific strategy cadence.
        A 1-minute strategy requires tighter freshness than a 5-minute or 15-minute strategy.
        """
        cadence_limits = {
            "1m": 60.0,
            "5m": 300.0,
            "15m": 900.0,
            "1h": 3600.0,
            "1d": 86400.0,
        }
        cadence_max = cadence_limits.get(strategy_cadence.lower(), 60.0)
        effective_max = min(cadence_max, max_age_cap) if max_age_cap is not None else cadence_max

        if self.age_seconds > effective_max:
            return False, f"Quote age {self.age_seconds:.1f}s exceeds limit {effective_max:.1f}s for {strategy_cadence} cadence"
        return True, "FRESH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "change_pct": self.change_pct,
            "data_timestamp": self.data_timestamp,
            "timestamp": self.data_timestamp,
            "received_at": self.received_at,
            "retrieval_timestamp": self.received_at,
            "age_seconds": round(self.age_seconds, 2),
            "data_age_seconds": round(self.age_seconds, 2),
            "bar_interval": self.bar_interval,
            "data_mode": self.data_mode,
            "source": self.source,
            "quality_status": self.quality_status,
        }


class MarketDataProvider(ABC):
    """Abstract interface for live/historical US market data ingestion."""

    @abstractmethod
    def get_quote(self, symbol: str) -> MarketQuote:
        """Fetches latest quote with freshness tracking."""
        pass

    @abstractmethod
    def get_bars(self, symbol: str, interval: str = "1d", limit: int = 120) -> pd.DataFrame:
        """Fetches OHLCV historical/intraday bars with strict point-in-time timestamping."""
        pass

    @abstractmethod
    def get_watchlist_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """Fetches quotes across full watchlist."""
        pass
