"""
Feature Engine — Time-safe features for quantitative strategies.

Every feature computed here uses ONLY data available at decision time.
No feature uses information unavailable at the moment of signal generation.

Categories:
- PRICE: returns, ATR, realized vol, drawdown, distance from highs/lows
- TREND: EMA, SMA, ADX, trend slope
- MOMENTUM: RSI, ROC, multi-horizon momentum, relative momentum, rank
- MEAN_REVERSION: z-score, distance from MA, Bollinger deviation
- VOLUME: relative volume, acceleration, surprise, price-volume interaction
- VOLATILITY: realized vol, ATR ratio, compression, expansion
- MARKET: index return, regime
- CROSS_SECTIONAL: rank, percentile, relative strength

All features are computed on expanding/past-only windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


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
    return macd_line, signal_line, macd_line - signal_line


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _atr(high, low, close, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high, low, close, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    tr = pd.concat([
        (high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.rolling(period).mean().fillna(0)


def _bollinger(close, period: int = 20, std_dev: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features on a price DataFrame with columns:
    date, open, high, low, close, volume.

    Returns the same DataFrame augmented with feature columns.
    All columns use only past/current data (time-safe via .shift() where needed).
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── PRICE ──
    df["ret_1"] = c.pct_change(1)
    df["ret_5"] = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)
    df["log_ret_1"] = np.log(c / c.shift(1))
    df["atr_14"] = _atr(h, l, c, 14)
    df["atr_pct"] = df["atr_14"] / c * 100
    df["realized_vol_20"] = df["ret_1"].rolling(20).std() * np.sqrt(252) * 100
    df["drawdown"] = (c / c.cummax() - 1) * 100
    df["dist_from_high_20"] = (c - c.rolling(20).max()) / c.rolling(20).max() * 100
    df["dist_from_low_20"] = (c - c.rolling(20).min()) / c.rolling(20).min() * 100
    df["range_10"] = (h.rolling(10).max() - l.rolling(10).min()) / c * 100

    # ── TREND ──
    df["ema_9"] = _ema(c, 9)
    df["ema_21"] = _ema(c, 21)
    df["sma_20"] = _sma(c, 20)
    df["sma_50"] = _sma(c, 50)
    df["sma_200"] = _sma(c, 200)
    df["adx_14"] = _adx(h, l, c, 14)
    df["trend_slope_20"] = (c - c.shift(20)) / 20
    df["trend_persistence"] = (
        (c > c.shift(1)).astype(int).rolling(20).mean()
    )

    # ── MOMENTUM ──
    df["rsi_14"] = _rsi(c, 14)
    macd_line, signal_line, hist = _macd(c)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["roc_1"] = c.pct_change(1) * 100
    df["roc_10"] = (c / c.shift(10) - 1) * 100
    df["roc_20"] = (c / c.shift(20) - 1) * 100
    df["momentum_5_20"] = df["ret_5"] - df["ret_20"]
    df["dist_from_sma_20"] = (c - df["sma_20"]) / df["sma_20"] * 100

    # ── MEAN REVERSION ──
    df["zscore_20"] = (c - df["sma_20"]) / df["sma_20"].rolling(20).std()
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(c)
    df["bb_deviation"] = (c - df["bb_mid"]) / (df["bb_upper"] - df["bb_lower"]) * 100
    df["reversal_signal"] = np.where(df["rsi_14"] < 35, 1,
                              np.where(df["rsi_14"] > 65, -1, 0))

    # ── VOLUME ──
    df["volume_20_avg"] = v.rolling(20).mean()
    df["relative_volume"] = v / df["volume_20_avg"]
    df["volume_acceleration"] = v / v.shift(5)
    df["vol_spike"] = df["relative_volume"] > 1.5
    df["price_vol_interaction"] = df["ret_1"] * df["relative_volume"]

    # ── VOLATILITY ──
    df["atr_ratio_10_20"] = df["atr_14"] / df["atr_14"].rolling(20).mean()
    df["vol_compression"] = df["realized_vol_20"].rolling(20).mean() / df["realized_vol_20"]
    df["vol_expansion"] = df["realized_vol_20"] / df["realized_vol_20"].shift(10)

    # ── CROSS-SECTIONAL (time-safe rolling rank within historical window) ──
    df["return_rank"] = df["ret_20"].rolling(window=60, min_periods=5).rank(pct=True).fillna(0.5)
    df["vol_rank"] = df["realized_vol_20"].rolling(window=60, min_periods=5).rank(pct=True).fillna(0.5)

    # Tag derived columns as time-safe (no future data)
    feature_cols = [
        "ret_1", "ret_5", "ret_20", "log_ret_1", "atr_14", "atr_pct",
        "realized_vol_20", "drawdown", "dist_from_high_20", "dist_from_low_20",
        "range_10", "ema_9", "ema_21", "sma_20", "sma_50", "sma_200",
        "adx_14", "trend_slope_20", "trend_persistence", "rsi_14", "macd",
        "macd_signal", "macd_hist", "roc_1", "roc_10", "roc_20",
        "momentum_5_20", "dist_from_sma_20", "zscore_20", "bb_upper",
        "bb_mid", "bb_lower", "bb_deviation", "reversal_signal",
        "volume_20_avg", "relative_volume", "volume_acceleration", "vol_spike",
        "price_vol_interaction", "atr_ratio_10_20", "vol_compression",
        "vol_expansion", "return_rank", "vol_rank",
    ]

    df["feature_count"] = len([c for c in feature_cols if c in df.columns])

    return df


def get_feature_names() -> List[str]:
    """Return list of all available feature column names."""
    return [
        "ret_1", "ret_5", "ret_20", "log_ret_1", "atr_14", "atr_pct",
        "realized_vol_20", "drawdown", "dist_from_high_20", "dist_from_low_20",
        "range_10", "ema_9", "ema_21", "sma_20", "sma_50", "sma_200",
        "adx_14", "trend_slope_20", "trend_persistence", "rsi_14", "macd",
        "macd_signal", "macd_hist", "roc_1", "roc_10", "roc_20",
        "momentum_5_20", "dist_from_sma_20", "zscore_20", "bb_upper",
        "bb_mid", "bb_lower", "bb_deviation", "reversal_signal",
        "relative_volume", "vol_spike", "price_vol_interaction",
        "atr_ratio_10_20", "vol_compression", "vol_expansion",
        "return_rank", "vol_rank",
    ]
