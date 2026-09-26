"""
Base Data Provider — Standard Abstract Data Interface

Every market data provider must inherit from BaseDataProvider and return records
adhering to the canonical schema:
- symbol: str (e.g. RELIANCE)
- exchange: str (e.g. NSE)
- timestamp: ISO 8601 string or datetime
- timezone: str (e.g. Asia/Kolkata or UTC)
- source: str (e.g. yahoo, angel)
- interval: canonical interval name
- retrieval_timestamp: float (UNIX epoch seconds)
- quality_status: VALID | STALE | SUSPICIOUS | MISSING
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """Abstract base class for all market data ingestion providers."""

    def __init__(self, name: str, default_exchange: str = "NSE"):
        self.name = name
        self.default_exchange = default_exchange

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the latest quote for a symbol.
        Must return a dict adhering to canonical quote schema:
        {
            'symbol': str,
            'exchange': str,
            'last_price': float,
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': int,
            'change': float,
            'change_pct': float,
            'timestamp': str,
            'timezone': str,
            'source': str,
            'retrieval_timestamp': float,
            'quality_status': str,
            'quality_flags': List[str]
        }
        """
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str, interval: str = "ONE_DAY", days: int = 120) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        Must return a pandas DataFrame with canonical columns:
        ['date', 'open', 'high', 'low', 'close', 'volume']
        where 'date' is a datetime/timestamp series and prices are positive floats.
        """
        pass

    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch quotes for multiple symbols with safe degradation."""
        results = []
        for s in symbols:
            try:
                results.append(self.get_quote(s))
            except Exception as e:
                logger.error("Failed fetching quote for %s from %s: %s", s, self.name, e)
                results.append(self._empty_quote(s, str(e)))
        return results

    def _empty_quote(self, symbol: str, error: str = "") -> Dict[str, Any]:
        """Construct a standardized error/empty quote."""
        now = time.time()
        return {
            "symbol": symbol.strip().upper(),
            "exchange": self.default_exchange,
            "last_price": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0,
            "change": 0.0,
            "change_pct": 0.0,
            "timestamp": None,
            "timezone": "Asia/Kolkata",
            "source": self.name,
            "retrieval_timestamp": now,
            "quality_status": "MISSING",
            "quality_flags": ["MISSING", f"error:{error}" if error else "no_data"],
            "error": error,
        }
