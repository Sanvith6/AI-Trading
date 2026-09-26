"""
Free Market Data Provider (Paper-trading default — no broker keys needed).

Priority:
  1. Yahoo Finance (FREE, no API key) for NSE symbols, e.g. RELIANCE.NS
  2. Angel One SmartAPI only when BROKER=angel AND credentials are configured.

All functions return plain dicts/DataFrames so callers never crash offline.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BROKER = os.getenv("BROKER", "paper").strip().lower()  # paper | angel | zerodha
MARKET = os.getenv("MARKET", "IN").strip().upper()      # US | IN
WATCHLIST = [s.strip().upper() for s in os.getenv("TRADE_WATCHLIST", "RELIANCE,TCS,INFY").split(",") if s.strip()]


def _to_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if MARKET == "US":
        return s  # US tickers need no suffix (AAPL, NVDA, SPY, etc.)
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


def _empty_quote(symbol: str, error: str) -> Dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "last_price": 0,
        "open": 0, "high": 0, "low": 0, "close": 0,
        "volume": 0, "change": 0.0, "change_pct": 0.0,
        "source": "none", "error": error,
    }


def get_quote(symbol: str) -> Dict[str, Any]:
    """Free live-ish quote via Yahoo Finance (1d/1m). Falls back to Angel if configured."""
    symbol = symbol.strip().upper()
    # Try Angel live first only when explicitly selected + credentials present
    if BROKER == "angel" and os.getenv("ANGEL_API_KEY", "").startswith("AK") is False and os.getenv("ANGEL_API_KEY"):
        pass  # fall through to angel attempt below
    if BROKER == "angel" and os.getenv("ANGEL_API_KEY") and "your_" not in os.getenv("ANGEL_API_KEY", ""):
        try:
            from trading.market_data import get_quote as _angel_quote
            q = _angel_quote(symbol)
            if q.get("last_price"):
                q["source"] = "angel"
                return q
        except Exception as e:
            logger.warning("Angel quote failed, falling back to Yahoo: %s", e)

    try:
        import yfinance as yf
        ticker = yf.Ticker(_to_yahoo_symbol(symbol))
        hist = ticker.history(period="2d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            # intraday fallback
            hist = ticker.history(period="1d", interval="1m", auto_adjust=False)
        if hist is None or hist.empty:
            return _empty_quote(symbol, "No data returned from Yahoo Finance (market closed or bad symbol?)")
        last = hist.iloc[-1]
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else float(last["Close"])
        ltp = float(last["Close"])
        change = round(ltp - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
        return {
            "symbol": symbol,
            "last_price": round(ltp, 2),
            "open": round(float(last["Open"]), 2),
            "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2),
            "close": round(float(last["Close"]), 2),
            "volume": int(last["Volume"]),
            "change": change,
            "change_pct": change_pct,
            "source": "yahoo",
        }
    except Exception as e:
        return _empty_quote(symbol, str(e))


def get_ohlcv(symbol: str, interval: str = "ONE_DAY", days: int = 120) -> pd.DataFrame:
    """Free OHLCV history via Yahoo Finance. Normalizes interval names."""
    symbol = symbol.strip().upper()
    iv = (interval or "ONE_DAY").upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "DAY": "1d", "DAILY": "1d", "ONE_DAY": "1d", "D": "1d",
        "ONE_HOUR": "1h", "1H": "1h", "HOUR": "1h",
        "15MINUTE": "15m", "FIFTEEN_MINUTE": "15m", "15M": "15m",
        "5MINUTE": "5m", "FIVE_MINUTE": "5m", "5M": "5m",
        "ONE_MINUTE": "1m", "1M": "1m",
        "WEEK": "1wk", "ONE_WEEK": "1wk",
    }
    yf_interval = mapping.get(iv, "1d")
    # Yahoo limits: intraday intervals only support short ranges
    if yf_interval in ("1m", "5m", "15m", "1h"):
        days = min(days, 29 if yf_interval == "1h" else 7)
        period = f"{max(days, 1)}d"
    else:
        if days <= 730:
            period = f"{max(days, 5)}d"
        else:
            years = max(2, min(10, int(round(days / 365.25))))
            period = f"{years}y"
    try:
        import yfinance as yf
        df = yf.download(_to_yahoo_symbol(symbol), period=period, interval=yf_interval,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns (yfinance >= 0.2 returns them)
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
        return out
    except Exception as e:
        logger.warning("Yahoo OHLCV failed for %s: %s", symbol, e)
        return pd.DataFrame()


def get_watchlist_quotes(watchlist: Optional[List[str]] = None, max_workers: int = 16) -> List[Dict[str, Any]]:
    """Concurrent quote fetching across watchlist symbols (optimized for 30-50+ stocks)."""
    wl = [s.strip().upper() for s in (watchlist or WATCHLIST) if s.strip()]
    if not wl:
        return []
    if len(wl) == 1:
        return [get_quote(wl[0])]
    from concurrent.futures import ThreadPoolExecutor
    workers = min(len(wl), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(get_quote, wl))


def quotes_to_map(quotes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {q.get("symbol", "").upper(): q for q in quotes}


def get_ltp(symbols: List[str]) -> Dict[str, float]:
    """Concurrent last traded price mapping for multiple symbols."""
    quotes = get_watchlist_quotes(symbols)
    return {q.get("symbol", "").upper(): (q.get("last_price", 0) or 0) for q in quotes}
