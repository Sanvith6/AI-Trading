"""
Institutional Execution Capacity & Market Impact Foundation.

Models scalable execution dynamics across configurable AUM tiers:
$10K, $50K, $100K, $500K, $1M, $5M, $10M

Uses the institutional Square-Root Market Impact Law (Barra / Almgren-Chriss formulation):
    Impact = Y * sigma * sqrt(OrderSize / ADV)
Where:
- Y: Market impact coefficient (~0.5 - 0.7 for liquid equities)
- sigma: Daily volatility
- ADV: 20-day Average Daily Dollar Volume

Calculates participation rates, slippage escalation, transaction drag, and net Sharpe decay.
Explicitly labels outputs as 'MODEL-BASED ESTIMATES'.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapacityTierResult:
    aum: float
    participation_rate_pct: float
    market_impact_bps: float
    slippage_bps: float
    annual_cost_drag_pct: float
    gross_cagr: float
    net_cagr: float
    gross_sharpe: float
    net_sharpe: float
    is_capacity_constrained: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CapacityEngine:
    """
    Simulates portfolio scalability, market-impact decay, and execution limits.
    """

    DEFAULT_AUM_TIERS = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    def __init__(
        self,
        impact_coefficient: float = 0.6,
        base_fee_bps: float = 5.0,  # 5 bps broker/exchange fee
        max_safe_participation: float = 0.02,  # max 2% of ADV before critical market disruption
    ):
        self.impact_coefficient = impact_coefficient
        self.base_fee_bps = base_fee_bps
        self.max_safe_participation = max_safe_participation

    def evaluate_capacity_curve(
        self,
        gross_cagr: float = 0.28,
        gross_sharpe: float = 1.65,
        annual_turnover: float = 8.0,  # 800% portfolio turnover per year
        avg_symbol_adv: float = 50_000_000.0,  # $50M ADV for large-cap US equities
        num_positions: int = 10,
        daily_volatility: float = 0.018,
        aum_tiers: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates execution capacity decay across multiple AUM tiers.
        """
        tiers = aum_tiers or self.DEFAULT_AUM_TIERS
        results = []
        bottleneck_aum = None

        for aum in tiers:
            position_size = aum / max(num_positions, 1)
            # Daily trading volume per position based on turnover
            avg_daily_trade = (position_size * annual_turnover) / 252.0
            participation = avg_daily_trade / max(avg_symbol_adv, 1000.0)

            # Square-Root Market Impact Law: Y * sigma * sqrt(trade / ADV)
            impact_bps = self.impact_coefficient * daily_volatility * np.sqrt(max(participation, 1e-6)) * 10000.0
            total_slippage_bps = impact_bps + self.base_fee_bps

            # Annual cost drag from fees + slippage = annual_turnover * slippage_bps / 10000
            annual_cost_drag = annual_turnover * (total_slippage_bps / 10000.0)
            net_cagr = gross_cagr - annual_cost_drag

            # Sharpe decay: net_sharpe = gross_sharpe * (net_cagr / gross_cagr)
            sharpe_decay_ratio = max(0.0, net_cagr / max(gross_cagr, 0.001))
            net_sharpe = gross_sharpe * sharpe_decay_ratio

            is_constrained = (participation > self.max_safe_participation) or (net_sharpe < 0.8)

            if is_constrained and bottleneck_aum is None:
                bottleneck_aum = aum

            results.append(CapacityTierResult(
                aum=float(aum),
                participation_rate_pct=round(float(participation * 100.0), 3),
                market_impact_bps=round(float(impact_bps), 2),
                slippage_bps=round(float(total_slippage_bps), 2),
                annual_cost_drag_pct=round(float(annual_cost_drag * 100.0), 3),
                gross_cagr=round(float(gross_cagr), 4),
                net_cagr=round(float(net_cagr), 4),
                gross_sharpe=round(float(gross_sharpe), 3),
                net_sharpe=round(float(net_sharpe), 3),
                is_capacity_constrained=bool(is_constrained),
            ))

        return {
            "model_type": "Almgren-Chriss Square-Root Market Impact Model",
            "status": "MODEL_BASED_ESTIMATE",
            "caveat": "Realized execution capacity requires live order book tick data and actual fill logs. These curves are theoretical benchmarks.",
            "estimated_bottleneck_aum": bottleneck_aum or "> $10M",
            "tier_results": [r.to_dict() for r in results]
        }
