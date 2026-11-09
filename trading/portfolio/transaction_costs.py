"""
Transaction Costs and Turnover Penalty Module.

Implements realistic institutional execution drag models:
1. Bid-ask spread cost
2. Market impact & expected slippage
3. Brokerage / exchange commission
4. Turnover friction penalty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """Estimates realistic transaction costs and turnover penalties for portfolio rebalancing."""

    def __init__(
        self,
        half_spread_bps: float = 3.0,
        slippage_bps: float = 4.0,
        commission_bps: float = 1.0,
        turnover_penalty_multiplier: float = 0.002,
        cost_penalty_multiplier: float = 1.0,
    ):
        self.half_spread_bps = half_spread_bps
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.turnover_penalty_multiplier = turnover_penalty_multiplier
        self.cost_penalty_multiplier = cost_penalty_multiplier

    def compute_turnover(
        self,
        new_weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculates L1 turnover: sum_i |w_new,i - w_current,i|.
        Range: [0.0, 2.0].
        """
        current_weights = current_weights or {}
        all_symbols = set(new_weights.keys()).union(set(current_weights.keys()))
        turnover = sum(
            abs(new_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            for s in all_symbols
        )
        return float(turnover)

    def compute_cost_drag(
        self,
        new_weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]] = None,
        custom_spread_bps: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Computes the expected transaction cost drag and turnover penalty.
        
        Returns:
            Tuple of (estimated_cost_drag, turnover_friction, total_penalty)
        """
        current_weights = current_weights or {}
        custom_spread_bps = custom_spread_bps or {}
        all_symbols = set(new_weights.keys()).union(set(current_weights.keys()))

        total_cost_drag = 0.0
        total_turnover = 0.0

        for s in all_symbols:
            delta_w = abs(new_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            if delta_w <= 1e-6:
                continue

            total_turnover += delta_w
            spread_bps = custom_spread_bps.get(s, self.half_spread_bps)
            bps_total = spread_bps + self.slippage_bps + self.commission_bps
            # Convert bps to decimal fraction (1 bps = 1e-4)
            cost_fraction = bps_total * 1e-4
            total_cost_drag += delta_w * cost_fraction

        turnover_friction = self.turnover_penalty_multiplier * total_turnover
        total_penalty = (self.cost_penalty_multiplier * total_cost_drag) + turnover_friction

        return float(total_cost_drag), float(turnover_friction), float(total_penalty)

    def evaluate_vectorized(
        self,
        w_new: np.ndarray,
        w_current: np.ndarray,
    ) -> float:
        """
        Vectorized penalty calculation for optimization objective.
        
        cost_drag = sum |w_new - w_current| * total_bps * 1e-4
        penalty = cost_multiplier * cost_drag + turnover_multiplier * turnover
        """
        delta = np.abs(w_new - w_current)
        turnover = np.sum(delta)
        total_bps = self.half_spread_bps + self.slippage_bps + self.commission_bps
        cost_drag = np.sum(delta) * (total_bps * 1e-4)
        return float((self.cost_penalty_multiplier * cost_drag) + (self.turnover_penalty_multiplier * turnover))
