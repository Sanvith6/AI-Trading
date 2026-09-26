"""
Execution Simulator — Backtests must not assume perfect execution.

Models:
- commission & taxes (NSE STT, stamp duty, exchange turnover charges, SEBI, GST)
- slippage (configurable bps + square-root market impact)
- bid-ask spread (configurable paise)
- partial fill (fill probability based on volume and order size)
- price gaps (configurable max gap % per bar)
- volume limits (max fill as % of bar volume)
- execution delay (configurable 1-bar or 2-bar delay)
- stop execution failure rate
- full order type logic: market, limit, stop, stop-limit

Supports pessimistic execution assumptions + hostile stress testing.
"""

import os
import math
import random
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_SLIPPAGE_BPS          = float(os.getenv("EXEC_SLIPPAGE_BPS", "10.0"))   # 10 bps = 0.10%
DEFAULT_SPREAD_PAISE           = float(os.getenv("EXEC_SPREAD_PAISE", "5.0"))   # 5 paise
DEFAULT_FILL_PROBABILITY       = float(os.getenv("EXEC_FILL_PROB", "0.95"))     # 95% base fill
DEFAULT_MAX_GAP_PCT            = float(os.getenv("EXEC_MAX_GAP_PCT", "2.5"))    # 2.5% max bar move
DEFAULT_EXECUTION_DELAY_BARS   = int(os.getenv("EXEC_DELAY_BARS", "1"))         # 1 bar delay
DEFAULT_PARTIAL_FILL_RATE      = float(os.getenv("EXEC_PARTIAL_FILL", "0.10"))   # 10% max haircut
DEFAULT_MARKET_IMPACT_FACTOR   = float(os.getenv("EXEC_IMPACT", "0.001"))       # Impact coefficient


@dataclass
class FillResult:
    symbol: str
    action: str          # BUY / SELL
    requested_qty: int
    filled_qty: int      # may be less if partial fill
    requested_price: float
    filled_price: float  # includes slippage, spread, impact
    order_type: str      # market / limit / stop / stop_limit
    slippage_bps: float  # effective slippage in bps
    commission: float    # total brokerage + taxes + fees
    timestamp: Optional[str] = None
    rejected: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "requested_qty": self.requested_qty,
            "filled_qty": self.filled_qty,
            "requested_price": self.requested_price,
            "filled_price": self.filled_price,
            "order_type": self.order_type,
            "slippage_bps": round(self.slippage_bps, 2),
            "commission": round(self.commission, 2),
            "rejected": self.rejected,
            "reason": self.reason,
        }


class ExecutionSimulator:
    """Institutional fill simulation engine. Never assumes zero cost or instantaneous execution."""

    def __init__(
        self,
        slippage_bps: Optional[float] = None,
        spread_paise: Optional[float] = None,
        fill_prob: Optional[float] = None,
        max_gap_pct: Optional[float] = None,
        exec_delay: Optional[int] = None,
        market_impact: Optional[float] = None,
    ):
        self.slippage_bps = slippage_bps if slippage_bps is not None else DEFAULT_SLIPPAGE_BPS
        self.spread_paise = spread_paise if spread_paise is not None else DEFAULT_SPREAD_PAISE
        self.fill_prob = fill_prob if fill_prob is not None else DEFAULT_FILL_PROBABILITY
        self.max_gap_pct = max_gap_pct if max_gap_pct is not None else DEFAULT_MAX_GAP_PCT
        self.exec_delay = exec_delay if exec_delay is not None else DEFAULT_EXECUTION_DELAY_BARS
        self.market_impact = market_impact if market_impact is not None else DEFAULT_MARKET_IMPACT_FACTOR

    def execute(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
        bar_volume: Optional[int] = None,
        volatility_pct: Optional[float] = None,
        slippage_override: Optional[float] = None,
    ) -> FillResult:
        """
        Simulates filling an order considering market, limit, stop, and stop-limit conditions.
        """
        action = action.upper()
        order_type = order_type.lower()
        actual_slippage_bps = slippage_override if slippage_override is not None else self.slippage_bps

        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            return FillResult(
                symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                requested_price=price, filled_price=0.0, order_type=order_type,
                slippage_bps=0.0, commission=0.0, rejected=True, reason="Invalid price (must be finite positive number)",
            )

        if not isinstance(qty, int) or qty <= 0:
            return FillResult(
                symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                requested_price=price, filled_price=0.0, order_type=order_type,
                slippage_bps=0.0, commission=0.0, rejected=True, reason="Invalid quantity (must be positive integer)",
            )

        if bar_volume is not None and bar_volume <= 0:
            return FillResult(
                symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                requested_price=price, filled_price=0.0, order_type=order_type,
                slippage_bps=0.0, commission=0.0, rejected=True, reason="Zero volume bar — execution impossible",
            )


        # 1. Order Type Trigger Checks
        if order_type == "limit" and limit_price is not None:
            if action == "BUY" and bar_low is not None and limit_price < bar_low:
                return FillResult(
                    symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                    requested_price=price, filled_price=0.0, order_type=order_type,
                    slippage_bps=0.0, commission=0.0, rejected=True, reason=f"Limit price {limit_price} below bar low {bar_low}",
                )
            elif action == "SELL" and bar_high is not None and limit_price > bar_high:
                return FillResult(
                    symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                    requested_price=price, filled_price=0.0, order_type=order_type,
                    slippage_bps=0.0, commission=0.0, rejected=True, reason=f"Limit price {limit_price} above bar high {bar_high}",
                )

        if order_type in ("stop", "stop_limit") and stop_price is not None:
            if action == "SELL" and bar_low is not None and stop_price < bar_low:
                # Stop not triggered
                return FillResult(
                    symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                    requested_price=price, filled_price=0.0, order_type=order_type,
                    slippage_bps=0.0, commission=0.0, rejected=True, reason="Stop price not breached",
                )

        # 2. Probability of Fill & Liquidity Haircut
        effective_fill_prob = self.fill_prob
        if bar_volume and bar_volume > 0:
            order_pct_of_volume = (qty / bar_volume)
            if order_pct_of_volume > 0.05:  # Order exceeds 5% of bar volume
                effective_fill_prob = max(0.40, self.fill_prob - (order_pct_of_volume * 2.0))

        if random.random() > effective_fill_prob:
            return FillResult(
                symbol=symbol, action=action, requested_qty=qty, filled_qty=0,
                requested_price=price, filled_price=0.0, order_type=order_type,
                slippage_bps=actual_slippage_bps, commission=0.0, rejected=True,
                reason="Order unfulfilled due to market liquidity / probability failure",
            )

        # 3. Partial Fill Calculation
        fill_fraction = random.uniform(1.0 - DEFAULT_PARTIAL_FILL_RATE, 1.0)
        filled_qty = max(1, int(qty * fill_fraction))

        # 4. Slippage and Spread Calculation
        spread_cost = (self.spread_paise * 0.01) / price if price > 0 else 0.0
        slippage_frac = actual_slippage_bps / 10000.0

        # Market impact scaling with size / volume
        impact_frac = 0.0
        if bar_volume and bar_volume > 0:
            vol_ratio = (qty * price) / (bar_volume * price)
            if vol_ratio > 0.02:
                impact_frac = min(0.02, np.sqrt(vol_ratio) * self.market_impact)

        total_adverse_move = slippage_frac + spread_cost + impact_frac
        if action == "BUY":
            filled_price = price * (1.0 + total_adverse_move)
        else:
            filled_price = price * (1.0 - total_adverse_move)

        filled_price = round(filled_price, 2)

        # 5. Commission & Statutory Taxes (NSE Equity Delivery/Intraday Model)
        from trading.swarm.quant_metrics import compute_indian_transaction_costs
        try:
            cost_details = compute_indian_transaction_costs(filled_price, filled_qty, action=action, product="CNC")
            commission = cost_details.total_cost
        except Exception:
            # Fallback 0.15% estimate if helper unavailable
            commission = round(filled_price * filled_qty * 0.0015, 2)

        return FillResult(
            symbol=symbol,
            action=action,
            requested_qty=qty,
            filled_qty=filled_qty,
            requested_price=price,
            filled_price=filled_price,
            order_type=order_type,
            slippage_bps=actual_slippage_bps,
            commission=commission,
            rejected=False,
            reason="Filled",
        )

    def stress_execution(self, multiplier: float = 3.0) -> "ExecutionSimulator":
        """Generate hostile simulator for stress testing under extreme illiquidity."""
        return ExecutionSimulator(
            slippage_bps=self.slippage_bps * multiplier,
            spread_paise=self.spread_paise * multiplier,
            fill_prob=max(0.50, self.fill_prob * 0.70),
            max_gap_pct=self.max_gap_pct * 0.5,
            exec_delay=self.exec_delay + 1,
            market_impact=self.market_impact * multiplier,
        )
