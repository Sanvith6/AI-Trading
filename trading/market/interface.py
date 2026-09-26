"""
Market Data Provider Abstraction Interface.

Shared across both US_PAPER and US_LIVE to ensure 100% data pipeline consistency.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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
    timestamp: str          # Decision/exchange timestamp (ISO 8601)
    retrieval_timestamp: float  # Unix epoch seconds when received locally
    data_age_seconds: float     # Age relative to current wall-clock
    source: str
    quality_status: str

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
            "timestamp": self.timestamp,
            "retrieval_timestamp": self.retrieval_timestamp,
            "data_age_seconds": round(self.data_age_seconds, 2),
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
