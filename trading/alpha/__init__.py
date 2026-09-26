"""
Alpha Library — 12 Independent Alpha Families

Every alpha implements the common BaseAlpha interface:
- generate_signal() -> signal in [-1, +1]
- score() -> confidence in [0, 1]
- explain() -> reason codes, horizon, risk
- required_features() -> list of features
- parameters() -> parameter dict
- validate() -> parameter validation

Output: signal in [-1, +1], HOLD is always a valid and preferred decision.
"""

from trading.alpha.base_alpha import BaseAlpha
from trading.alpha.trend_alpha import TrendAlpha
from trading.alpha.momentum_alpha import MomentumAlpha
from trading.alpha.mean_reversion_alpha import MeanReversionAlpha
from trading.alpha.breakout_alpha import BreakoutAlpha
from trading.alpha.volatility_alpha import VolatilityAlpha
from trading.alpha.volume_flow_alpha import VolumeFlowAlpha
from trading.alpha.cross_sectional_alpha import CrossSectionalAlpha
from trading.alpha.statistical_alpha import StatisticalAlpha
from trading.alpha.relative_strength_alpha import RelativeStrengthAlpha
from trading.alpha.defensive_alpha import DefensiveAlpha
from trading.alpha.regime_alpha import RegimeAlpha
from trading.alpha.ml_hybrid_alpha import MLHybridAlpha

__all__ = [
    "BaseAlpha",
    "TrendAlpha",
    "MomentumAlpha",
    "MeanReversionAlpha",
    "BreakoutAlpha",
    "VolatilityAlpha",
    "VolumeFlowAlpha",
    "CrossSectionalAlpha",
    "StatisticalAlpha",
    "RelativeStrengthAlpha",
    "DefensiveAlpha",
    "RegimeAlpha",
    "MLHybridAlpha",
]


def get_all_alphas() -> list[BaseAlpha]:
    """Factory helper returning an instantiated list of all 12 alpha families."""
    return [
        TrendAlpha(),
        MomentumAlpha(),
        MeanReversionAlpha(),
        BreakoutAlpha(),
        VolatilityAlpha(),
        VolumeFlowAlpha(),
        CrossSectionalAlpha(),
        StatisticalAlpha(),
        RelativeStrengthAlpha(),
        DefensiveAlpha(),
        RegimeAlpha(),
        MLHybridAlpha(),
    ]