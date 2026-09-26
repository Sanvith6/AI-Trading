"""
Market Data Module — Angel One SmartAPI (FREE)
Fetches live quotes, OHLCV candles, portfolio holdings, and orders.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from trading.angel_auth import get_smartapi_client

load_dotenv(override=True)

WATCHLIST = [s.strip() for s in os.getenv("TRADE_WATCHLIST", "RELIANCE,TCS,INFY").split(",")]

# Angel One exchange codes
EXCHANGE_NSE = "NSE"
EXCHANGE_BSE = "BSE"

# Hardcoded symbol token map for common NSE stocks.
# Full list available at: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
SYMBOL_TOKENS = {
    "RELIANCE":   "2885",
    "TCS":        "11536",
    "INFY":       "1594",
    "HDFCBANK":   "1333",
    "ICICIBANK":  "4963",
    "WIPRO":      "3787",
    "SBIN":       "3045",
    "BAJFINANCE": "317",
    "ADANIENT":   "25",
    "HDFC":       "1330",
    "LT":         "11483",
    "AXISBANK":   "5900",
    "MARUTI":     "10999",
    "SUNPHARMA":  "3351",
}


def _get_token(symbol: str) -> Optional[str]:
    """Look up Angel One instrument token for a symbol (static map + searchScrip)."""
    token = SYMBOL_TOKENS.get(symbol.upper())
    if token:
        return token
    try:
        obj = _get_client()
        res = obj.searchScrip("NSE", symbol.upper())
        items = (res.get("data") or []) if isinstance(res, dict) else []
        for inst in items:
            if str(inst.get("tradingsymbol", "")).upper().startswith(symbol.upper()):
                return inst.get("symboltoken")
    except Exception:
        pass
    return None


def _get_client():
    obj, _, _ = get_smartapi_client()
    return obj


def get_quote(symbol: str) -> Dict[str, Any]:
    """Fetch live quote for a symbol via Angel (ltpData first, FULL quote fallback)."""
    try:
        obj = _get_client()
        token = _get_token(symbol)
        if not token:
            return {"error": f"Token not found for {symbol}", "symbol": symbol}

        # Fast path: LTP endpoint (most reliable)
        try:
            ltp_res = obj.ltpData(EXCHANGE_NSE, symbol.upper(), token)
            ltp_data = (ltp_res.get("data") or {})
            ltp = float(ltp_data.get("ltp", 0) or 0)
            if ltp > 0:
                close = float(ltp_data.get("close", ltp) or ltp)
                open_p = ltp_data.get("open", ltp)
                high_p = ltp_data.get("high", ltp)
                low_p = ltp_data.get("low", ltp)
                change = round(ltp - close, 2)
                change_pct = round((change / close) * 100, 2) if close else 0
                return {
                    "symbol": symbol, "last_price": ltp,
                    "open": open_p, "high": high_p, "low": low_p,
                    "close": close, "volume": ltp_data.get("tradeVolume", 0),
                    "change": change, "change_pct": change_pct, "source": "angel",
                }
        except Exception:
            pass  # fall through to FULL market data

        data = obj.getMarketData(mode="FULL", exchangeTokens={EXCHANGE_NSE: [token]})
        fetched = data.get("data", {}).get("fetched", [])
        if not fetched:
            return {"error": "No data returned", "symbol": symbol}

        q = fetched[0]
        ltp    = q.get("ltp", 0)
        close  = q.get("close", ltp)
        change = round(ltp - close, 2)
        change_pct = round((change / close) * 100, 2) if close else 0

        return {
            "symbol":     symbol,
            "last_price": ltp,
            "open":       q.get("open"),
            "high":       q.get("high"),
            "low":        q.get("low"),
            "close":      close,
            "volume":     q.get("tradingVolume") or q.get("tradedVolume"),
            "change":     change,
            "change_pct": change_pct,
            "source":     "angel",
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def get_ohlcv(symbol: str, interval: str = "ONE_DAY", days: int = 120) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles.

    interval options (normalized): ONE_DAY/DAY, ONE_HOUR, FIFTEEN_MINUTE,
        FIVE_MINUTE, ONE_MINUTE, ONE_WEEK. Also accepts 'day'/'daily'.
    """
    _norm = (interval or "ONE_DAY").upper().replace("-", "_").replace(" ", "_")
    _aliases = {"DAY": "ONE_DAY", "DAILY": "ONE_DAY", "D": "ONE_DAY",
                "1D": "ONE_DAY", "1H": "ONE_HOUR", "HOUR": "ONE_HOUR",
                "15M": "FIFTEEN_MINUTE", "5M": "FIVE_MINUTE", "1M": "ONE_MINUTE"}
    interval = _aliases.get(_norm, _norm)
    try:
        from datetime import datetime, timedelta
        obj = _get_client()
        token = _get_token(symbol)
        if not token:
            return pd.DataFrame()

        to_date   = datetime.now()
        from_date = to_date - timedelta(days=days)

        params = {
            "exchange":    EXCHANGE_NSE,
            "symboltoken": token,
            "interval":    interval,
            "fromdate":    from_date.strftime("%Y-%m-%d %H:%M"),
            "todate":      to_date.strftime("%Y-%m-%d %H:%M"),
        }
        data = obj.getCandleData(historicDataParams=params)
        candles = data.get("data", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"[market_data] OHLCV error for {symbol}: {e}")
        return pd.DataFrame()


def get_portfolio() -> Dict[str, Any]:
    """Fetch current holdings, positions and available margin."""
    try:
        obj = _get_client()
        holdings  = obj.holding()
        positions = obj.position()
        rms       = obj.rmsLimit()

        return {
            "holdings":          holdings.get("data", []),
            "positions":         positions.get("data", []),
            "available_margin":  rms.get("data", {}).get("availablecash", 0),
            "net":               rms.get("data", {}).get("net", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def get_orders() -> List[Dict[str, Any]]:
    """Fetch all of today's orders."""
    try:
        obj = _get_client()
        result = obj.orderBook()
        return result.get("data") or []
    except Exception as e:
        return [{"error": str(e)}]


def get_watchlist_quotes(watchlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch live quotes for watchlist symbols (defaults to TRADE_WATCHLIST)."""
    wl = watchlist or WATCHLIST
    return [get_quote(s) for s in wl]


def get_ltp(symbols: List[str]) -> Dict[str, float]:
    """Get last traded prices for a list of symbols."""
    result = {}
    for s in symbols:
        q = get_quote(s)
        result[s] = q.get("last_price", 0)
    return result
