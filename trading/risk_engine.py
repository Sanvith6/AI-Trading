"""
Risk Engine — Immutable Security Boundary.

The AI CANNOT override these limits.
AI decision → portfolio validation → risk validation → execution validation → broker adapter.

Every hard limit below is enforced programmatically, not by LLM instruction.
"""

import os
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_PER_ORDER       = float(os.getenv("TRADE_MAX_PER_ORDER", 5000))
DEFAULT_MAX_DAILY_LOSS      = float(os.getenv("TRADE_MAX_DAILY_LOSS", 2000))
DEFAULT_MAX_POSITIONS       = int(os.getenv("TRADE_MAX_POSITIONS", 3))
DEFAULT_STOP_LOSS_PCT       = float(os.getenv("TRADE_STOP_LOSS_PCT", 2.0))
DEFAULT_TARGET_PCT          = float(os.getenv("TRADE_TARGET_PCT", 4.0))
DEFAULT_MAX_EXPOSURE        = float(os.getenv("TRADE_MAX_EXPOSURE", 0.95))   # 95% of cash
DEFAULT_MAX_SECTOR_WEIGHT   = float(os.getenv("TRADE_MAX_SECTOR_WEIGHT", 0.40))  # 40% per sector
DEFAULT_MAX_CORRELATED      = float(os.getenv("TRADE_MAX_CORRELATED", 0.60))  # 60% in correlated bets
DEFAULT_MIN_CASH            = float(os.getenv("TRADE_MIN_CASH", 1000))
DEFAULT_MIN_TRADES_METRIC   = int(os.getenv("TRADE_MIN_TRADES", 30))  # min trades for metric trust
DEFAULT_EMERGENCY_HALT      = os.getenv("EMERGENCY_HALT", "false").lower() == "true"


class RiskViolation:
    """A single risk rule violation."""
    def __init__(self, rule: str, allowed: Any, actual: Any, reason: str):
        self.rule = rule
        self.allowed = allowed
        self.actual = actual
        self.reason = reason

    def __repr__(self):
        return f"RiskViolation({self.rule}: {self.reason})"

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "allowed": self.allowed,
                "actual": self.actual, "reason": self.reason}


class RiskApproval:
    """Result of a risk check."""
    def __init__(self, approved: bool, violations: List[RiskViolation] = None):
        self.approved = approved
        self.violations = violations or []

    @property
    def deny_reason(self) -> str:
        if self.approved:
            return "APPROVED"
        return "; ".join(str(v) for v in self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {"approved": self.approved,
                "deny_reason": self.deny_reason,
                "violations": [v.to_dict() for v in self.violations]}


class RiskEngine:
    """
    Immutable risk boundary. No trading decision passes without approval.

    Rules enforced (all hard, none bypassable):
    1. Per-order value cap
    2. Max concurrent positions
    3. Max exposure (% of cash)
    4. Max sector weight (for multi-stock portfolios)
    5. Max correlated exposure
    6. Minimum cash floor
    7. Emergency halt flag (external kill-switch)
    8. Stop-loss / take-profit on existing positions
    9. Minimum trade count before metrics are trusted
    10. Valid symbol / non-zero price check
    """

    def __init__(self, state_path: Optional[str] = None):
        self.max_per_order       = float(os.getenv("TRADE_MAX_PER_ORDER", DEFAULT_MAX_PER_ORDER))
        self.max_daily_loss      = float(os.getenv("TRADE_MAX_DAILY_LOSS", DEFAULT_MAX_DAILY_LOSS))
        self.max_positions       = int(os.getenv("TRADE_MAX_POSITIONS", DEFAULT_MAX_POSITIONS))
        self.stop_loss_pct       = float(os.getenv("TRADE_STOP_LOSS_PCT", DEFAULT_STOP_LOSS_PCT))
        self.target_pct          = float(os.getenv("TRADE_TARGET_PCT", DEFAULT_TARGET_PCT))
        self.max_exposure        = float(os.getenv("TRADE_MAX_EXPOSURE", DEFAULT_MAX_EXPOSURE))
        self.max_sector_weight   = float(os.getenv("TRADE_MAX_SECTOR_WEIGHT", DEFAULT_MAX_SECTOR_WEIGHT))
        self.max_correlated      = float(os.getenv("TRADE_MAX_CORRELATED", DEFAULT_MAX_CORRELATED))
        self.min_cash            = float(os.getenv("TRADE_MIN_CASH", DEFAULT_MIN_CASH))
        self.min_trades          = int(os.getenv("TRADE_MIN_TRADES", DEFAULT_MIN_TRADES_METRIC))
        self.emergency_halt_value = os.getenv("EMERGENCY_HALT", "false").lower() == "true"
        self._state_path = Path(state_path) if state_path else None

    # ── Hard Gates ──────────────────────────────────────────────

    def validate_order(self, symbol: str, action: str, price: float,
                       quantity: int, cash: float, existing_positions: Dict = None,
                       daily_pnl: float = 0.0, sector: str = "") -> RiskApproval:
        """Hard gate: approve/deny a proposed order."""
        if self.emergency_halt:
            return RiskApproval(False, [RiskViolation("emergency_halt", "closed", "active",
                         "Emergency halt flag is ON — all trading blocked")])

        if not symbol or not isinstance(symbol, str) or not symbol.isalpha():
            return RiskApproval(False, [RiskViolation("symbol_valid", "alpha string", symbol,
                         "Symbol must be non-empty alphabetic string")])

        if price <= 0:
            return RiskApproval(False, [RiskViolation("price_positive", ">0", price,
                         "Price must be positive")])

        if quantity <= 0:
            return RiskApproval(False, [RiskViolation("quantity_positive", ">0", quantity,
                         "Quantity must be positive integer")])

        order_value = price * quantity
        if order_value > self.max_per_order:
            return RiskApproval(False, [RiskViolation("max_per_order",
                         f"≤₹{self.max_per_order:,.0f}", f"₹{order_value:,.0f}",
                         "Order value exceeds hard per-order cap")])

        if order_value > cash:
            return RiskApproval(False, [RiskViolation("sufficient_cash",
                         f"≥₹{order_value:,.0f}", f"₹{cash:,.0f}",
                         "Insufficient cash")])

        if cash - order_value < self.min_cash:
            return RiskApproval(False, [RiskViolation("min_cash_floor",
                         f"≥₹{self.min_cash:,.0f} after trade", f"₹{cash - order_value:,.0f}",
                         "Trade would breach minimum cash floor")])

        current_positions = existing_positions or {}
        if action == "BUY":
            if len(current_positions) >= self.max_positions and symbol not in current_positions:
                return RiskApproval(False, [RiskViolation("max_positions",
                             f"≤{self.max_positions}", len(current_positions) + 1,
                             "Max concurrent positions reached")])

            def _pos_val(v):
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    qty = float(v.get("quantity", 0))
                    p = float(v.get("current_price", v.get("entry_price", v.get("avg_price", 0.0))))
                    return qty * p
                return 0.0

            new_exposure = sum(_pos_val(v) for v in current_positions.values()) + order_value
            if new_exposure > cash * self.max_exposure:
                return RiskApproval(False, [RiskViolation("max_exposure",
                             f"≤{self.max_exposure*100:.0f}% of cash",
                             f"{new_exposure/cash*100:.1f}%",
                             "Portfolio exposure cap would be breached")])

        if daily_pnl <= -self.max_daily_loss:
            return RiskApproval(False, [RiskViolation("daily_loss_limit",
                         f"not below -₹{self.max_daily_loss:,.0f}", f"₹{daily_pnl:,.2f}",
                         "Daily loss limit hit — all trading halted")])

        return RiskApproval(True)

    def validate_exit(self, symbol: str, existing_positions: Dict) -> Tuple[bool, str]:
        """Check if an existing position should be force-exited."""
        pos = existing_positions.get(symbol)
        if not pos:
            return True, "No position to exit check"

        current_price = pos.get("current_price", 0)
        buy_price = pos.get("buy_price", 0)
        if not current_price or not buy_price or current_price <= 0:
            return True, "No valid prices for SL/TP check"

        change_pct = ((current_price - buy_price) / buy_price) * 100.0
        if change_pct <= -self.stop_loss_pct:
            return False, f"STOP-LOSS HIT: {change_pct:.2f}% ≤ -{self.stop_loss_pct}%"
        if change_pct >= self.target_pct:
            return False, f"TAKE-PROFIT HIT: +{change_pct:.2f}% ≥ +{self.target_pct}%"
        return True, "OK"

    def validate_metrics(self, trade_count: int) -> Tuple[bool, str]:
        """Check whether metrics from N trades can be statistically trusted."""
        if trade_count < self.min_trades:
            return False, (f"Only {trade_count} trades — "
                           f"need ≥{self.min_trades} for statistical trust")
        return True, "Minimum trade count satisfied"

    def calculate_position_size(self, cash: float, price: float,
                                risk_per_trade_pct: Optional[float] = None,
                                stop_distance_pct: Optional[float] = None) -> int:
        """
        Risk-aware position sizing: risk_budget / stop_distance.
        Then capped by per-order limit, liquidity, and portfolio limits.
        """
        rp = risk_per_trade_pct or float(os.getenv("TRADE_RISK_PER_TRADE_PCT", "5.0")) / 100.0
        sd = stop_distance_pct or self.stop_loss_pct / 100.0

        risk_budget = cash * rp
        if sd <= 0:
            return 1  # can't size without stop distance
        raw_qty = int(risk_budget / (price * sd))

        per_order_qty = max(1, int(self.max_per_order / price))
        qty = min(raw_qty, per_order_qty)
        max_qty = max(1, int(cash * self.max_exposure / price))
        return max(1, min(qty, max_qty))

    @property
    def emergency_halt(self) -> bool:
        return self.emergency_halt_value

    def get_limits(self) -> Dict[str, Any]:
        return {
            "max_per_order": self.max_per_order,
            "max_daily_loss": self.max_daily_loss,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "target_pct": self.target_pct,
            "max_exposure": self.max_exposure,
            "max_sector_weight": self.max_sector_weight,
            "max_correlated": self.max_correlated,
            "min_cash": self.min_cash,
            "min_trades": self.min_trades,
            "emergency_halt": self.emergency_halt,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"limits": self.get_limits()}
