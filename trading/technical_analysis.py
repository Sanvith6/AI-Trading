"""
Technical Analysis Module
Calculates RSI, MACD, Bollinger Bands, EMA, SMA, and volume signals.
Returns structured BUY/SELL/HOLD signals with reasons and confidence scores.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

_MARKET = os.getenv("MARKET", "IN").strip().upper()
_CURRENCY = "$" if _MARKET == "US" else "\u20b9"

_BROKER = os.getenv("BROKER", "paper").strip().lower()
if _BROKER == "angel":
    from trading.market_data import get_ohlcv  # live broker candles
else:
    from trading.data_provider import get_ohlcv  # FREE Yahoo Finance (default)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — the volatility ruler for ATR-based stops."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — trend-strength gauge (regime filter)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.rolling(period).mean().fillna(0)


def analyze(symbol: str, interval: str = "day", days: int = 120) -> Dict[str, Any]:
    """
    Run full technical analysis on a stock symbol.

    Returns a dictionary with all indicator values and an overall signal.
    """
    df = get_ohlcv(symbol, interval=interval, days=days)
    if df.empty or len(df) < 30:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "reason": "Insufficient data to analyze.",
            "indicators": {},
        }

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    last_close = float(close.iloc[-1])

    # RSI
    rsi_series = _rsi(close)
    rsi = float(rsi_series.iloc[-1])

    # MACD
    macd_line, signal_line, histogram = _macd(close)
    macd_val = float(macd_line.iloc[-1])
    signal_val = float(signal_line.iloc[-1])
    hist_val = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    bb_u = float(bb_upper.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])

    # EMA crossover (9 & 21)
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema9_val = float(ema9.iloc[-1])
    ema21_val = float(ema21.iloc[-1])
    prev_ema9 = float(ema9.iloc[-2]) if len(ema9) > 1 else ema9_val
    prev_ema21 = float(ema21.iloc[-2]) if len(ema21) > 1 else ema21_val

    # SMA 50 & 200
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # Volume spike (is today's volume > 1.5x 20-day avg?)
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_today = float(volume.iloc[-1])
    vol_spike = vol_today > 1.5 * vol_avg if vol_avg > 0 else False

    # ATR (volatility ruler) + ADX (regime filter)
    atr_val = float(_atr(high, low, close).iloc[-1])
    atr_pct = round(atr_val / last_close * 100, 2) if last_close else 0.0
    adx_val = float(_adx(high, low, close).iloc[-1])
    trending = adx_val >= 20.0
    choppy = adx_val < 15.0

    # ── Signal Scoring ──
    buy_score = 0
    sell_score = 0
    reasons = []

    # RSI
    if rsi < 35:
        buy_score += 2
        reasons.append(f"RSI={rsi:.1f} — Oversold (bullish)")
    elif rsi < 45:
        buy_score += 1
        reasons.append(f"RSI={rsi:.1f} — Approaching oversold")
    elif rsi > 70:
        sell_score += 2
        reasons.append(f"RSI={rsi:.1f} — Overbought (bearish)")
    elif rsi > 60:
        sell_score += 1
        reasons.append(f"RSI={rsi:.1f} — Approaching overbought")
    else:
        reasons.append(f"RSI={rsi:.1f} — Neutral")

    # MACD crossover
    if macd_val > signal_val and prev_hist <= 0 and hist_val > 0:
        buy_score += 2
        reasons.append("MACD crossed above signal line (bullish crossover)")
    elif macd_val < signal_val and prev_hist >= 0 and hist_val < 0:
        sell_score += 2
        reasons.append("MACD crossed below signal line (bearish crossover)")
    elif macd_val > signal_val:
        buy_score += 1
        reasons.append(f"MACD above signal ({macd_val:.2f} > {signal_val:.2f})")
    else:
        sell_score += 1
        reasons.append(f"MACD below signal ({macd_val:.2f} < {signal_val:.2f})")

    # Bollinger Bands
    if last_close < bb_l:
        buy_score += 2
        reasons.append(f"Price below lower Bollinger Band ({_CURRENCY}{last_close:.2f} < {_CURRENCY}{bb_l:.2f})")
    elif last_close > bb_u:
        sell_score += 2
        reasons.append(f"Price above upper Bollinger Band ({_CURRENCY}{last_close:.2f} > {_CURRENCY}{bb_u:.2f})")
    else:
        bb_pct = (last_close - bb_l) / (bb_u - bb_l) * 100 if (bb_u - bb_l) > 0 else 50
        reasons.append(f"Price within Bollinger Bands ({bb_pct:.0f}% position)")

    # EMA crossover
    if prev_ema9 <= prev_ema21 and ema9_val > ema21_val:
        buy_score += 2
        reasons.append("EMA9 crossed above EMA21 (golden cross)")
    elif prev_ema9 >= prev_ema21 and ema9_val < ema21_val:
        sell_score += 2
        reasons.append("EMA9 crossed below EMA21 (death cross)")
    elif ema9_val > ema21_val:
        buy_score += 1
        reasons.append(f"EMA9 ({ema9_val:.2f}) > EMA21 ({ema21_val:.2f}) — Uptrend")
    else:
        sell_score += 1
        reasons.append(f"EMA9 ({ema9_val:.2f}) < EMA21 ({ema21_val:.2f}) — Downtrend")

    # SMA trend
    if sma50 and last_close > sma50:
        buy_score += 1
        reasons.append(f"Price above 50-SMA ({_CURRENCY}{sma50:.2f})")
    elif sma50 and last_close < sma50:
        sell_score += 1
        reasons.append(f"Price below 50-SMA ({_CURRENCY}{sma50:.2f})")

    # Volume confirmation
    if vol_spike:
        if buy_score > sell_score:
            buy_score += 1
            reasons.append(f"High volume spike ({vol_today:,.0f} vs avg {vol_avg:,.0f}) confirms bullish move")
        else:
            sell_score += 1
            reasons.append(f"High volume spike ({vol_today:,.0f} vs avg {vol_avg:,.0f}) confirms bearish move")

    # ── Final Signal ──
    total = buy_score + sell_score
    if total == 0:
        signal = "HOLD"
        confidence = 0
    elif buy_score > sell_score:
        signal = "BUY"
        confidence = round((buy_score / total) * 100)
    elif sell_score > buy_score:
        signal = "SELL"
        confidence = round((sell_score / total) * 100)
    else:
        signal = "HOLD"
        confidence = 50

    # Regime filter: in chop (ADX<15) trend-following edges die — demand
    # stronger evidence before acting.
    if signal != "HOLD" and choppy and confidence < 75:
        reasons.append(f"Choppy regime (ADX={adx_val:.1f}<15) — downgraded to HOLD")
        signal = "HOLD"

    # Minimum confidence threshold — only BUY/SELL if > 60% confident
    if signal != "HOLD" and confidence < 60:
        reasons.append(f"Confidence too low ({confidence}%) — downgraded to HOLD")
        signal = "HOLD"

    # ATR-based risk guide: 2×ATR stop, 3×ATR target from last price
    atr_stop = round(last_close - 2 * atr_val, 2)
    atr_target = round(last_close + 3 * atr_val, 2)

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "last_price": last_close,
        "indicators": {
            "rsi": round(rsi, 2),
            "macd": round(macd_val, 4),
            "macd_signal": round(signal_val, 4),
            "macd_histogram": round(hist_val, 4),
            "bb_upper": round(bb_u, 2),
            "bb_mid": round(bb_m, 2),
            "bb_lower": round(bb_l, 2),
            "ema9": round(ema9_val, 2),
            "ema21": round(ema21_val, 2),
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "volume": int(vol_today),
            "volume_avg_20d": int(vol_avg),
            "volume_spike": vol_spike,
            "atr": round(atr_val, 2),
            "atr_pct": atr_pct,
            "adx": round(adx_val, 1),
            "regime": "TRENDING" if trending else ("CHOPPY" if choppy else "WEAK_TREND"),
            "atr_stop_guide": atr_stop,
            "atr_target_guide": atr_target,
        },
        "reasons": reasons,
    }


def generate_signal(symbol: str) -> str:
    """
    Convenience wrapper — returns formatted signal string for use in agent tools.
    """
    result = analyze(symbol)
    sig = result["signal"]
    conf = result["confidence"]
    price = result.get("last_price", "N/A")
    reasons = "\n  - ".join(result.get("reasons", []))
    indicators = result.get("indicators", {})

    return (
        f"Signal for {symbol}: {sig} (Confidence: {conf}%)\n"
        f"Last Price: {_CURRENCY}{price}\n"
        f"Regime: {result.get('indicators', {}).get('regime')} (ADX {indicators.get('adx')}) | ATR {indicators.get('atr_pct')}% → guide stop {_CURRENCY}{indicators.get('atr_stop_guide')}, target {_CURRENCY}{indicators.get('atr_target_guide')}\n"
        f"Key Indicators:\n"
        f"  RSI: {indicators.get('rsi')}\n"
        f"  MACD: {indicators.get('macd')} | Signal: {indicators.get('macd_signal')}\n"
        f"  Bollinger: Upper {_CURRENCY}{indicators.get('bb_upper')} | Mid {_CURRENCY}{indicators.get('bb_mid')} | Lower {_CURRENCY}{indicators.get('bb_lower')}\n"
        f"  EMA9: {indicators.get('ema9')} | EMA21: {indicators.get('ema21')}\n"
        f"  Volume Spike: {indicators.get('volume_spike')}\n"
        f"Reasons:\n  - {reasons}"
    )


def batch_analyze(symbols: List[str], interval: str = "day", days: int = 120, max_workers: int = 16) -> List[Dict[str, Any]]:
    """
    Run full technical analysis concurrently on a batch of stock symbols (e.g. 30-50+ stocks).
    Returns list of analysis dicts in original order.
    """
    clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not clean_symbols:
        return []
    if len(clean_symbols) == 1:
        return [analyze(clean_symbols[0], interval=interval, days=days)]

    workers = min(len(clean_symbols), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda s: analyze(s, interval=interval, days=days), clean_symbols))

