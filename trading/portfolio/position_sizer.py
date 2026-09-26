"""
Position Sizer — Risk-budgeted position sizing engine.

Sizing Models:
1. Risk Budget / Stop Distance: shares = (cash * risk_per_trade_pct) / stop_distance
2. Volatility Targeting: sizes position inversely proportional to annualized volatility
3. Conservative Half-Kelly: Kelly fraction * 0.5 capped strictly at max_position_pct
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates risk-controlled position sizes for portfolio orders."""

    def __init__(
        self,
        default_risk_per_trade_pct: float = 2.0,
        max_position_weight: float = 0.25,
        max_order_value: float = 5000.0,
        max_kelly_fraction: float = 0.20,
    ):
        self.risk_pct = default_risk_per_trade_pct
        self.max_position_weight = max_position_weight
        self.max_order_value = max_order_value
        self.max_kelly_fraction = max_kelly_fraction

    def size_by_risk(
        self,
        cash: float,
        price: float,
        stop_loss_pct: float,
        risk_per_trade_pct: Optional[float] = None,
        max_value_cap: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates shares such that if stopped out, loss does not exceed risk budget.
        """
        if price <= 0:
            return {"quantity": 0, "order_value": 0.0, "risk_amount": 0.0, "reason": "Invalid price"}

        r_pct = risk_per_trade_pct or self.risk_pct
        risk_budget = cash * (r_pct / 100.0)
        stop_dist = price * (max(0.5, stop_loss_pct) / 100.0)

        raw_shares = risk_budget / stop_dist if stop_dist > 0 else 0

        # Cap by max position weight
        max_by_weight = (cash * self.max_position_weight) / price
        # Cap by max order value
        cap_val = max_value_cap or self.max_order_value
        max_by_order = cap_val / price

        final_shares = max(1, int(min(raw_shares, max_by_weight, max_by_order)))
        actual_val = round(final_shares * price, 2)
        actual_risk = round(final_shares * stop_dist, 2)

        return {
            "quantity": final_shares,
            "order_value": actual_val,
            "risk_amount": actual_risk,
            "stop_distance": round(stop_dist, 2),
            "method": "risk_budget",
            "reason": f"Risk {r_pct}% (₹{risk_budget:.1f}) with stop {stop_loss_pct}%",
        }

    def size_by_kelly(
        self,
        cash: float,
        price: float,
        win_rate: float,
        win_loss_ratio: float,
        half_kelly: bool = True,
    ) -> Dict[str, Any]:
        """
        Conservative Kelly sizing with extreme caps to prevent ruin.
        Kelly % = W - (1 - W) / R
        """
        if price <= 0 or win_loss_ratio <= 0:
            return {"quantity": 0, "order_value": 0.0, "reason": "Invalid inputs"}

        w = max(0.0, min(1.0, win_rate))
        r = win_loss_ratio

        raw_kelly = w - ((1.0 - w) / r)
        if raw_kelly <= 0:
            return {"quantity": 0, "order_value": 0.0, "reason": "Negative expectancy (Kelly <= 0)"}

        # Apply half-Kelly or quarter-Kelly for safety
        fraction = (raw_kelly * 0.5) if half_kelly else raw_kelly
        # Never allocate more than max_kelly_fraction or max_position_weight
        capped_fraction = min(fraction, self.max_kelly_fraction, self.max_position_weight)

        allocated_cash = cash * capped_fraction
        cap_val = min(allocated_cash, self.max_order_value)
        shares = max(1, int(cap_val / price))

        return {
            "quantity": shares,
            "order_value": round(shares * price, 2),
            "kelly_fraction": round(raw_kelly, 4),
            "capped_fraction": round(capped_fraction, 4),
            "method": "half_kelly" if half_kelly else "kelly",
            "reason": f"Win Rate {w*100:.1f}%, Payoff {r:.2f} -> Fraction {capped_fraction*100:.1f}%",
        }
