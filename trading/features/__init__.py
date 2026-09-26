"""
Feature Engineering Module

Time-safe modular features for alpha generation.
Every feature must be time-safe: no information unavailable at decision time.

Follow the structure:
- PRICE: returns, log returns, rolling returns, gap returns, range, ATR, realized volatility, drawdown, distance from highs/lows
- TREND: EMA, SMA, ADX, trend slope, trend persistence
- MOMENTUM: ROC, RSI, multi-horizon momentum, relative momentum, cross-sectional rank
- MEAN REVERSION: z-score, distance from moving average, Bollinger deviation, short-term reversal, residual reversion
- VOLUME: relative volume, volume acceleration, volume surprise, price-volume interaction
- VOLATILITY: realized volatility, ATR ratio, volatility compression, volatility expansion
- MARKET: index return, index volatility, breadth, sector momentum, market regime
- CROSS-SECTIONAL: rank, percentile, relative strength, sector-adjusted momentum, market-adjusted momentum
- STATISTICAL: beta, rolling correlation, residual return, correlation regime, volatility ratio
"""

from trading.features.price_features import *
from trading.features.trend_features import *
from trading.features.momentum_features import *
from trading.features.mean_reversion_features import *
from trading.features.volume_features import *
from trading.features.volatility_features import *
from trading.features.cross_sectional_features import *
from trading.features.statistical_features import *

__all__ = [
    "price_features",
    "trend_features", 
    "momentum_features",
    "mean_reversion_features",
    "volume_features",
    "volatility_features",
    "cross_sectional_features",
    "statistical_features",
]