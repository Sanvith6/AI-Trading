"""
Yahoo Finance Market Data Provider Implementation for US Equities.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from .interface import MarketDataProvider, MarketQuote
from trading.data.yahoo_provider import YahooDataProvider

logger = logging.getLogger(__name__)


class USMarketDataProvider(MarketDataProvider):
    """
    US Market data provider consuming free Yahoo Finance feed with caching & validation.
    """

    def __init__(self, raw_provider: Optional[YahooDataProvider] = None):
        self.raw = raw_provider or YahooDataProvider()

    def get_quote(self, symbol: str) -> MarketQuote:
        now = time.time()
        raw_dict = self.raw.get_quote(symbol)

        retrieval_ts = float(raw_dict.get("retrieval_timestamp", now))
        data_age = max(0.0, now - retrieval_ts)

        return MarketQuote(
            symbol=symbol.strip().upper(),
            last_price=float(raw_dict.get("last_price", 0.0)),
            open=float(raw_dict.get("open", 0.0)),
            high=float(raw_dict.get("high", 0.0)),
            low=float(raw_dict.get("low", 0.0)),
            close=float(raw_dict.get("close", 0.0)),
            volume=int(raw_dict.get("volume", 0)),
            change_pct=float(raw_dict.get("change_pct", 0.0)),
            timestamp=raw_dict.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))),
            retrieval_timestamp=retrieval_ts,
            data_age_seconds=data_age,
            source="yahoo",
            quality_status=raw_dict.get("quality_status", "VALID"),
        )

    def get_bars(self, symbol: str, interval: str = "1d", limit: int = 120) -> pd.DataFrame:
        return self.raw.get_ohlcv(symbol, interval=interval, days=limit)

    def get_watchlist_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        quotes = {}
        for s in symbols:
            quotes[s.strip().upper()] = self.get_quote(s)
        return quotes
