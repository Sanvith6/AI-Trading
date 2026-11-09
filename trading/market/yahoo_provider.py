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
        self.raw = raw_provider or YahooDataProvider(market="US")

    def get_quote(self, symbol: str) -> MarketQuote:
        t0 = time.perf_counter()
        raw_dict = self.raw.get_quote(symbol)
        provider_latency = round(time.perf_counter() - t0, 4)

        now = time.time()
        retrieval_ts = float(raw_dict.get("retrieval_timestamp", now))
        received_age = max(0.0, round(now - retrieval_ts, 2))
        data_age = received_age

        ts_str = raw_dict.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        data_mode = str(raw_dict.get("data_mode", "real-time"))

        return MarketQuote(
            symbol=symbol.strip().upper(),
            last_price=float(raw_dict.get("last_price", 0.0)),
            open=float(raw_dict.get("open", 0.0)),
            high=float(raw_dict.get("high", 0.0)),
            low=float(raw_dict.get("low", 0.0)),
            close=float(raw_dict.get("close", 0.0)),
            volume=int(raw_dict.get("volume", 0)),
            change_pct=float(raw_dict.get("change_pct", 0.0)),
            data_timestamp=ts_str,
            received_at=retrieval_ts,
            age_seconds=data_age,
            bar_interval=raw_dict.get("interval", "1m"),
            data_mode=data_mode,
            source="yahoo",
            quality_status=raw_dict.get("quality_status", "VALID"),
            provider_latency_seconds=provider_latency,
            received_age_seconds=received_age,
        )


    def get_bars(self, symbol: str, interval: str = "1d", limit: int = 120) -> pd.DataFrame:
        return self.raw.get_ohlcv(symbol, interval=interval, days=limit)

    def get_watchlist_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        quotes = {}
        for s in symbols:
            quotes[s.strip().upper()] = self.get_quote(s)
        return quotes
