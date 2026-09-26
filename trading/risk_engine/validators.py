"""
Risk Engine Validators — Immutable Hard Gates

These are the non-bypassable risk constraints that every order must pass.
The LLM never sees these directly — it proposes, the validator disposes.
"""

import math
from trading.risk_engine.risk_state import RiskState
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of an order validation check."""
    approved: bool
    reason: Optional[str] = None
    deny_category: Optional[str] = None  # e.g., "daily_loss", "max_exposure", etc.


class RiskValidator:
    """
    Immutable risk validation engine.
    
    The AI CANNOT override these limits. All orders must pass validation.
    """
    
    # Hard limits from environment (configurable but non-bypassable at runtime)
    def __init__(self, risk_state: Optional[RiskState] = None):
        self.risk_state = risk_state or RiskState()
        self._initialized = False
        self._seen_order_ids = set()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize from configuration dict. Called once at startup."""
        self._apply_config(config)
        self._initialized = True

    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply config values. Override from .env or explicit config."""
        # Per-order limit
        self.max_per_order = float(config.get("TRADE_MAX_PER_ORDER", 5000))
        
        # Daily loss limit
        self.max_daily_loss = float(config.get("TRADE_MAX_DAILY_LOSS", 2000))
        
        # Max positions
        self.max_positions = int(config.get("TRADE_MAX_POSITIONS", 3))
        
        # Max portfolio exposure (% of cash)
        self.max_exposure = float(config.get("TRADE_MAX_EXPOSURE", 0.95))
        
        # Max sector weight (%)
        self.max_sector_weight = float(config.get("TRADE_MAX_SECTOR_WEIGHT", 0.40))
        
        # Max correlated exposure (%)
        self.max_correlated = float(config.get("TRADE_MAX_CORRELATED", 0.60))
        
        # Min cash floor
        self.min_cash = float(config.get("TRADE_MIN_CASH", 1000))
        
        # Per-trade risk % of cash
        self.risk_per_trade_pct = float(config.get("TRADE_RISK_PER_TRADE_PCT", 5.0))
        
        # Stop loss % 
        self.stop_loss_pct = float(config.get("TRADE_STOP_LOSS_PCT", 2.0))
        
        # Take profit %
        self.target_pct = float(config.get("TRADE_TARGET_PCT", 4.0))
        
        # Min trades for metric trust
        self.min_trades = int(config.get("TRADE_MIN_TRADES", 30))
    
    def validate_order(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: int,
        cash: float,
        existing_positions: Dict[str, Any],
        daily_pnl: float = 0.0,
        sector: str = "UNKNOWN",
        portfolio_exposure: float = 0.0,
        order_id: Optional[str] = None,
        quote_age_seconds: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate a trading order against ALL hard risk gates.
        
        Returns ValidationResult(approved=True) if all gates pass,
        or ValidationResult(approved=False, reason=...) if any gate fails.
        """
        if not self._initialized:
            # Default config if not initialized
            self._apply_config({})
            self._initialized = True

        # Gate 0: Duplicate order ID protection
        if order_id is not None:
            if order_id in self._seen_order_ids:
                return ValidationResult(
                    approved=False,
                    reason=f"Duplicate order ID rejected: {order_id}",
                    deny_category="duplicate_order"
                )

        # Gate 0b: Stale quote check
        if quote_age_seconds is not None and quote_age_seconds > 300.0:
            return ValidationResult(
                approved=False,
                reason=f"Stale quote rejected: quote age {quote_age_seconds:.1f}s exceeds threshold 300.0s",
                deny_category="stale_quote"
            )

        # Gate 1: Emergency halt
        if self.risk_state.emergency_halt:
            return ValidationResult(
                approved=False,
                reason="Emergency halt active — all trading blocked",
                deny_category="emergency_halt"
            )
        
        # Gate 2: Finite and positive price check
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            return ValidationResult(
                approved=False,
                reason=f"Invalid price: {price} — must be finite positive number",
                deny_category="invalid_price"
            )

        # Gate 2b: Finite cash balance check
        if not isinstance(cash, (int, float)) or not math.isfinite(cash) or cash < 0:
            return ValidationResult(
                approved=False,
                reason=f"Invalid cash balance: {cash} — must be non-negative finite number",
                deny_category="invalid_cash"
            )

        
        # Gate 3: Order value per-order limit
        order_value = price * quantity
        if order_value > self.max_per_order:
            return ValidationResult(
                approved=False,
                reason=f"Order value ₹{order_value:.2f} exceeds max per-order ₹{self.max_per_order:.2f}",
                deny_category="max_per_order"
            )
        
        # Gate 4: Daily loss limit
        if daily_pnl <= -self.max_daily_loss:
            return ValidationResult(
                approved=False,
                reason=f"Daily loss limit hit (₹{daily_pnl:.2f}). No more trades today.",
                deny_category="daily_loss"
            )
        
        # Gate 5: Max positions check (new positions only)
        if action == "BUY":
            open_count = len(existing_positions)
            # Count positions that are not this symbol (adding 1 if new)
            if symbol not in existing_positions:
                if open_count >= self.max_positions:
                    return ValidationResult(
                        approved=False,
                        reason=f"Max positions limit reached ({self.max_positions}). Close a position first.",
                        deny_category="max_positions"
                    )
            elif existing_positions[symbol]["quantity"] > 0:
                # Already holding this, adding more counts as new position if we'd exceed max
                if open_count > self.max_positions:
                    return ValidationResult(
                        approved=False,
                        reason=f"Would exceed max positions ({self.max_positions}).",
                        deny_category="max_positions"
                    )
        
        # Gate 6: Min cash floor — ensure we don't drain cash below minimum
        remaining_cash = cash - order_value
        if remaining_cash < self.min_cash:
            return ValidationResult(
                approved=False,
                reason=f"Trade would leave cash ₹{remaining_cash:.2f} below minimum ₹{self.min_cash:.2f}",
                deny_category="min_cash"
            )
        
        # Gate 7: Portfolio exposure limit
        # After this trade, total exposure should not exceed max_exposure
        # This is a simplified check — full correlation check is separate
        new_total_exposure = portfolio_exposure + order_value
        if new_total_exposure > cash * self.max_exposure:
            return ValidationResult(
                approved=False,
                reason=f"Portfolio exposure ₹{new_total_exposure:.2f} would exceed max {self.max_exposure*100:.1f}% of cash ₹{cash:.2f}",
                deny_category="max_exposure"
            )
        
        # Gate 8: Sector weight limit
        sector_exposure = order_value  # Simplified — would need sector tracking
        new_sector_exposure = sector_exposure  # Track per-sector in full implementation
        if new_sector_exposure > cash * self.max_sector_weight and sector != "UNKNOWN":
            # Only enforce if we have sector tracking
            pass  # Placeholder — full implementation tracks sector exposure
        
        # Gate 9: Min trade size — at least 1 share
        if not isinstance(quantity, int) or quantity < 1:
            return ValidationResult(
                approved=False,
                reason=f"Quantity must be a positive integer >= 1, got {quantity}",
                deny_category="min_quantity"
            )
        
        # Gate 10: Valid symbol — alphabetic check
        if not symbol.isalpha():
            return ValidationResult(
                approved=False,
                reason=f"Invalid symbol: {symbol} — must be alphabetic ticker",
                deny_category="invalid_symbol"
            )
        
        # Gate 11: Position sizing — risk-aware sizing check
        # Risk budget = cash * risk_per_trade_pct / 100
        # Stop distance = price * stop_loss_pct / 100
        # Max shares = risk_budget / stop_distance
        risk_budget = cash * self.risk_per_trade_pct / 100.0
        stop_distance = price * self.stop_loss_pct / 100.0
        if stop_distance > 0:
            max_by_risk = int(risk_budget / stop_distance)
            if quantity > max_by_risk:
                actual_qty = max(1, max_by_risk)
                return ValidationResult(
                    approved=False,
                    reason=f"Position size {quantity} exceeds risk-based limit {max_by_risk}. "
                           f"Reducing to {actual_qty} shares (risk {self.risk_per_trade_pct}% of cash).",
                    deny_category="risk_sizing"
                )
        
        # Record approved order ID to prevent replay
        if order_id is not None:
            self._seen_order_ids.add(order_id)

        # All gates passed
        return ValidationResult(approved=True, reason="OK")


class PositionSizer:
    """
    Risk-aware position sizing — does NOT simply use cash / number_of_stocks.
    
    Uses risk-based sizing: position_size = risk_budget / stop_distance
    Then enforces all hard caps.
    """
    
    def __init__(self, risk_validator: RiskValidator):
        self.risk_validator = risk_validator
    
    def calculate_position_size(
        self,
        cash: float,
        price: float,
        stop_loss_pct: float,
        risk_per_trade_pct: float,
        existing_exposure: float = 0.0,
        max_position_weight: float = None,
    ) -> Dict[str, Any]:
        """
        Calculate the recommended position size using risk-aware model.
        
        Returns dict with:
        - raw_size: theoretical max shares by risk
        - capped_size: size after all caps applied
        - risk_amount: ₹ amount at risk
        - stop_distance: ₹ distance to stop
        """
        risk_budget = cash * risk_per_trade_pct / 100.0
        stop_distance = price * stop_loss_pct / 100.0
        
        if stop_distance <= 0:
            return {
                "raw_size": 0,
                "capped_size": 0,
                "risk_amount": 0,
                "stop_distance": 0,
                "note": "Stop distance too small; cannot size position."
            }
        
        # Raw size by risk
        raw_size = risk_budget / stop_distance
        
        # Apply per-order value cap
        max_order_value = self.risk_validator.max_per_order
        max_by_order = max_order_value / price if price > 0 else 0
        capped_by_order = min(raw_size, max_by_order)
        
        # Apply max position weight if specified
        if max_position_weight:
            max_by_weight = (cash * max_position_weight) / price if price > 0 else 0
            capped_by_weight = min(capped_by_order, max_by_weight)
        else:
            capped_by_weight = capped_by_order
        
        # Apply liquidity/portfolio constraints
        capped_size = max(1, int(capped_by_weight))  # At least 1 share, floor at 1
        
        actual_risk = capped_size * stop_distance
        
        return {
            "raw_size": round(raw_size, 2),
            "capped_size": capped_size,
            "risk_amount": round(actual_risk, 2),
            "stop_distance": round(stop_distance, 2),
            "note": f"Risk {risk_per_trade_pct}% of cash, stop at {stop_loss_pct}%"
        }