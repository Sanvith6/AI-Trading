"""
Angel One SmartAPI Data Provider — Broker Ingestion Layer with secure credential handling.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd

from trading.data.base_provider import BaseDataProvider
from trading.data.data_cache import DataCache
from trading.data.validator import validate_quote, validate_ohlcv

logger = logging.getLogger(__name__)


class AngelDataProvider(BaseDataProvider):
    """Fetches market quotes and OHLCV from Angel One SmartAPI when configured."""

    def __init__(self, cache: Optional[DataCache] = None):
        super().__init__(name="angel", default_exchange="NSE")
        self.cache = cache or DataCache()
        self._api_key = os.getenv("ANGEL_API_KEY", "")
        self._is_configured = bool(self._api_key and "your_" not in self._api_key and not self._api_key.startswith("AK"))

    def is_available(self) -> bool:
        """Check if broker credentials are configured without exposing keys."""
        return self._is_configured

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        clean_sym = symbol.strip().upper()
        if not self._is_configured:
            return self._empty_quote(clean_sym, "Angel One SmartAPI credentials not configured")

        cached = self.cache.get_quote(clean_sym)
        if cached:
            return cached

        try:
            from trading.market_data import get_quote as _angel_quote
            q = _angel_quote(clean_sym)
            if q and q.get("last_price"):
                q["source"] = "angel"
                q["exchange"] = "NSE"
                now = time.time()
                q["retrieval_timestamp"] = now
                validated = validate_quote(q, retrieval_ts=now)
                self.cache.set_quote(clean_sym, validated)
                return validated
            return self._empty_quote(clean_sym, "Empty response from Angel SmartAPI")
        except Exception as e:
            logger.warning("Angel quote failed for %s: %s", clean_sym, e)
            return self._empty_quote(clean_sym, str(e))

    def get_ohlcv(self, symbol: str, interval: str = "ONE_DAY", days: int = 120) -> pd.DataFrame:
        clean_sym = symbol.strip().upper()
        if not self._is_configured:
            return pd.DataFrame()

        cached_df = self.cache.get_ohlcv(clean_sym, interval, days)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        try:
            from trading.market_data import get_historical
            df = get_historical(clean_sym, interval=interval, days=days)
            if df is not None and not df.empty:
                df, flags = validate_ohlcv(df, clean_sym)
                self.cache.set_ohlcv(clean_sym, interval, days, df)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.warning("Angel OHLCV failed for %s: %s", clean_sym, e)
            return pd.DataFrame()
