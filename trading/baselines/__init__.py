"""
Baseline Strategies Package — Standard institutional benchmarks against which all AI alpha is judged.
"""

from trading.baselines.baselines import (
    BaseStrategy,
    BuyAndHold,
    SmaTrend,
    EmaTrend,
    MomentumBaseline,
    MeanReversionBaseline,
    BreakoutBaseline,
    VolatilityTargetBaseline,
    get_all_baselines,
)

__all__ = [
    "BaseStrategy",
    "BuyAndHold",
    "SmaTrend",
    "EmaTrend",
    "MomentumBaseline",
    "MeanReversionBaseline",
    "BreakoutBaseline",
    "VolatilityTargetBaseline",
    "get_all_baselines",
]
