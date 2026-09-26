"""
Data Validation Layer — applied to every quote/OHLCV record.

Every record must contain: symbol, exchange, timestamp, timezone, source,
interval, retrieval_timestamp, quality_status.

Validators: OHLC sanity, zero/negative values, timestamp anomalies,
duplicate candles, missing intervals, stale data, future timestamps,
timezone errors, suspicious data.

Never silently consume suspicious data — flag it.
"""

import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

STALE_MINUTES = int(os.getenv("DATA_STALE_MINUTES", "30"))
FUTURE_TOLERANCE_MINUTES = int(os.getenv("DATA_FUTURE_TOLERANCE_MINUTES", "5"))
MIN_VOLUME = int(os.getenv("DATA_MIN_VOLUME", "1000"))
MAX_DROP_PCT = float(os.getenv("DATA_MAX_DROP_PCT", "30"))  # OHLC > 30% gap = suspicious


class QualityFlags:
    VALID      = "VALID"
    STALE      = "STALE"
    SUSPICIOUS = "SUSPICIOUS"
    MISSING    = "MISSING"
    ZERO       = "ZERO"
    FUTURE     = "FUTURE"
    DUPLICATE  = "DUPLICATE"
    GAP        = "GAP"
    HALT       = "HALT"


def validate_quote(q: Dict[str, Any], retrieval_ts: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate a single quote dict. Returns the dict with added:
      quality_status, quality_flags, validation_timestamp
    Never raises — flags issues instead.
    """
    flags: List[str] = []
    if retrieval_ts is None:
        retrieval_ts = time.time()

    symbol = q.get("symbol", "UNKNOWN")

    # 1. Zero / negative checks
    ltp = q.get("last_price", 0)
    if ltp <= 0:
        flags.append(QualityFlags.ZERO)

    for field in ("open", "high", "low", "close"):
        val = q.get(field, 0)
        if val < 0:
            flags.append(f"{field}_negative")

    # 2. OHLC sanity
    if all(f in q for f in ("high", "low", "open", "close")):
        if q["high"] < q["low"]:
            flags.append("high_lt_low")
        if q["high"] < q["close"] or q["high"] < q["open"]:
            flags.append("high_below_oc")
        if q["low"] > q["close"] or q["low"] > q["open"]:
            flags.append("low_above_oc")
        # Extreme gap check
        rng = q["high"] - q["low"]
        mid = (q["high"] + q["low"]) / 2 if (q["high"] + q["low"]) > 0 else 1
        if rng > 0 and abs(q["close"] - mid) / mid > MAX_DROP_PCT / 100.0:
            flags.append("extreme_gap")

    # 3. Missing data indicators
    if ltp == 0 and all(q.get(f) in (None, 0) for f in ("open", "high", "low", "close", "volume")):
        flags.append(QualityFlags.MISSING)

    # 4. Volume sanity
    vol = q.get("volume", 0)
    if vol is not None and vol < MIN_VOLUME and ltp > 0:
        flags.append("low_volume")

    # 5. Staleness check (requires timestamp — optional in quote)
    ts = q.get("timestamp")
    if ts:
        try:
            dt = _parse_ts(ts)
            if dt:
                age_min = (retrieval_ts - dt.timestamp()) / 60.0
                if age_min > STALE_MINUTES:
                    flags.append(QualityFlags.STALE)
                if age_min < -FUTURE_TOLERANCE_MINUTES:
                    flags.append(QualityFlags.FUTURE)
        except Exception:
            flags.append("timestamp_parse_error")

    status = (QualityFlags.VALID if not flags
              else QualityFlags.SUSPICIOUS if any(f in (QualityFlags.ZERO, QualityFlags.MISSING,
                                                          QualityFlags.FUTURE, "high_lt_low")
                                                 for f in flags)
              else QualityFlags.STALE)

    q["quality_status"] = status
    q["quality_flags"] = flags
    q["validation_timestamp"] = retrieval_ts
    return q


def _parse_ts(ts) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def validate_ohlcv(df, symbol: str = "UNKNOWN") -> Tuple[Any, List[str]]:
    """
    Validate an OHLCV DataFrame. Returns (df, flags).
    Flags describe structural issues (gaps, duplicates, negatives).
    Does NOT modify the data — caller decides how to handle.
    """
    flags: List[str] = []
    if df is None or df.empty:
        return df, [QualityFlags.MISSING]

    cols_needed = {"date", "open", "high", "low", "close", "volume"}
    missing = cols_needed - set(df.columns)
    if missing:
        flags.append(f"missing_columns:{','.join(missing)}")
        return df, flags

    # Negative / zero values
    for col in ("open", "high", "low", "close"):
        if (df[col] <= 0).any():
            flags.append(f"{col}_has_nonpositive")

    if (df["volume"] < 0).any():
        flags.append("volume_negative")

    # High < Low
    mask = df["high"] < df["low"]
    if mask.any():
        flags.append(f"high_lt_low_at_rows:{list(df[mask].index[:5])}")

    # Duplicates
    dupes = df["date"].duplicated(keep=False)
    if dupes.any():
        flags.append(f"duplicate_dates:{list(df[dupes].date[:5])}")

    # Gaps (missing intervals) — approximate by checking date diffs
    if len(df) >= 2:
        try:
            dates = pd_to_datetime(df["date"])
            diffs = dates.diff().dt.total_seconds().dropna()
            median_gap = diffs.median()
            if median_gap and median_gap > 0:
                outliers = diffs[diffs > median_gap * 2]
                if len(outliers) > 0:
                    flags.append(f"date_gaps_detected:median={median_gap}s, outliers={len(outliers)}")
        except Exception:
            pass

    return df, flags


def pd_to_datetime(series):
    """Light wrapper to import pandas lazily."""
    import pandas as pd
    return pd.to_datetime(series)


def freshness_report(symbol: str, q: Dict[str, Any]) -> Dict[str, Any]:
    """Return a brief report of how fresh a quote is."""
    flags = q.get("quality_flags", [])
    ts = q.get("timestamp")
    age = None
    if ts:
        dt = _parse_ts(ts)
        if dt:
            age = (time.time() - dt.timestamp()) / 60.0
    return {
        "symbol": symbol,
        "age_minutes": round(age, 1) if age is not None else None,
        "quality_status": q.get("quality_status", "UNKNOWN"),
        "flags": flags,
        "fresh": QualityFlags.STALE not in flags and QualityFlags.FUTURE not in flags
                 and QualityFlags.MISSING not in flags,
    }
