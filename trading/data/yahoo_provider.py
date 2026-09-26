"""
Yahoo Finance Data Provider — Free Ingestion Layer with validation and caching.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf

from trading.data.base_provider import BaseDataProvider
from trading.data.data_cache import DataCache
from trading.data.validator import validate_quote, validate_ohlcv, QualityFlags

logger = logging.getLogger(__name__)

MARKET = os.getenv("MARKET", "IN").strip().upper()  # US | IN


class YahooDataProvider(BaseDataProvider):
    """Fetches market quotes and OHLCV from Yahoo Finance without requiring API keys."""

    INTERVAL_MAP = {
        "DAY": "1d", "DAILY": "1d", "ONE_DAY": "1d", "D": "1d",
        "ONE_HOUR": "1h", "1H": "1h", "HOUR": "1h",
        "15MINUTE": "15m", "FIFTEEN_MINUTE": "15m", "15M": "15m",
        "5MINUTE": "5m", "FIVE_MINUTE": "5m", "5M": "5m",
        "ONE_MINUTE": "1m", "1M": "1m",
        "WEEK": "1wk", "ONE_WEEK": "1wk",
    }

    def __init__(self, cache: Optional[DataCache] = None):
        exchange = "NYSE/NASDAQ" if MARKET == "US" else "NSE"
        super().__init__(name="yahoo", default_exchange=exchange)
        self.cache = cache or DataCache()

    def _format_ticker(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if MARKET == "US":
            return s  # US tickers: AAPL, NVDA, SPY — no suffix needed
        if s.endswith(".NS") or s.endswith(".BO"):
            return s
        return f"{s}.NS"

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        clean_sym = symbol.strip().upper()
        cached = self.cache.get_quote(clean_sym)
        if cached:
            return cached

        now = time.time()
        ticker_str = self._format_ticker(clean_sym)

        try:
            ticker = yf.Ticker(ticker_str)
            hist = ticker.history(period="2d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                hist = ticker.history(period="1d", interval="1m", auto_adjust=False)

            if hist is None or hist.empty:
                empty = self._empty_quote(clean_sym, "No data returned from Yahoo Finance")
                return empty

            last = hist.iloc[-1]
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else float(last["Close"])
            ltp = float(last["Close"])
            change = round(ltp - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

            ts_str = str(last.name) if hasattr(last, "name") else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

            raw_quote = {
                "symbol": clean_sym,
                "exchange": "NYSE/NASDAQ" if MARKET == "US" else "NSE",
                "last_price": round(ltp, 2),
                "open": round(float(last["Open"]), 2),
                "high": round(float(last["High"]), 2),
                "low": round(float(last["Low"]), 2),
                "close": round(float(last["Close"]), 2),
                "volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
                "change": change,
                "change_pct": change_pct,
                "timestamp": ts_str,
                "timezone": "America/New_York" if MARKET == "US" else "Asia/Kolkata",
                "source": "yahoo",
                "retrieval_timestamp": now,
            }

            validated = validate_quote(raw_quote, retrieval_ts=now)
            self.cache.set_quote(clean_sym, validated)
            return validated

        except Exception as e:
            logger.warning("Yahoo get_quote failed for %s: %s", clean_sym, e)
            return self._empty_quote(clean_sym, str(e))

    def get_ohlcv(self, symbol: str, interval: str = "ONE_DAY", days: int = 120) -> pd.DataFrame:
        clean_sym = symbol.strip().upper()
        norm_interval = self.INTERVAL_MAP.get(interval.upper(), "1d")

        cached_df = self.cache.get_ohlcv(clean_sym, norm_interval, days)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        ticker_str = self._format_ticker(clean_sym)
        if norm_interval in ("1m", "5m", "15m", "1h"):
            safe_days = min(days, 29 if norm_interval == "1h" else 7)
            period = f"{max(safe_days, 1)}d"
        else:
            period = f"{max(days, 5)}d" if days <= 730 else "2y"

        try:
            df = yf.download(ticker_str, period=period, interval=norm_interval,
                             progress=False, auto_adjust=False)
            if df is None or df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            out = pd.DataFrame({
                "date": df.index,
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].fillna(0).astype(int),
            }).reset_index(drop=True)

            out, flags = validate_ohlcv(out, clean_sym)
            if QualityFlags.MISSING not in flags:
                self.cache.set_ohlcv(clean_sym, norm_interval, days, out)

            return out
        except Exception as e:
            logger.warning("Yahoo OHLCV download failed for %s: %s", clean_sym, e)
            return pd.DataFrame()
