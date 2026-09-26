"""
Risk Manager Module
Enforces position limits, stop-loss, daily loss circuit breakers before every trade.
"""

import os
import csv
import json
from datetime import date
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv(override=True)

MAX_PER_ORDER    = float(os.getenv("TRADE_MAX_PER_ORDER", 5000))
MAX_DAILY_LOSS   = float(os.getenv("TRADE_MAX_DAILY_LOSS", 2000))
MAX_POSITIONS    = int(os.getenv("TRADE_MAX_POSITIONS", 3))
STOP_LOSS_PCT    = float(os.getenv("TRADE_STOP_LOSS_PCT", 2.0))
TARGET_PCT       = float(os.getenv("TRADE_TARGET_PCT", 4.0))
TRADE_LOG        = Path(__file__).parent / "trade_log.csv"
STATE_FILE       = Path(__file__).parent / "risk_state.json"


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        if state.get("date") == str(date.today()):
            return state
    # Reset for new day
    state = {"date": str(date.today()), "daily_pnl": 0.0, "open_positions": []}
    _save_state(state)
    return state


def _save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


class RiskManager:
    """
    Tracks P&L and position count for the current trading day.
    Enforces hard limits before any trade is placed.
    """

    def __init__(self):
        self.state = _load_state()

    def refresh(self):
        """Reload state from disk (call this before each check)."""
        self.state = _load_state()

    # ── Core Guard ──────────────────────────────────────────────

    def is_trade_allowed(self, symbol: str, order_value: float, action: str = "BUY") -> Tuple[bool, str]:
        """
        Check if a trade is permitted given current risk state.

        Returns:
            (True, "OK") if trade is allowed
            (False, reason) if trade is blocked
        """
        self.refresh()

        # 1. Daily loss limit
        if self.state["daily_pnl"] <= -MAX_DAILY_LOSS:
            return False, f"Daily loss limit hit (₹{self.state['daily_pnl']:.2f}). No more trades today."

        # 2. Max position count (only check for new BUY entries)
        if action == "BUY":
            open_pos = self.state.get("open_positions", [])
            if symbol not in open_pos and len(open_pos) >= MAX_POSITIONS:
                return False, f"Max positions limit reached ({MAX_POSITIONS}). Close a position first."

        # 3. Order size limit
        if order_value > MAX_PER_ORDER:
            return False, f"Order value ₹{order_value:.2f} exceeds max per-order limit ₹{MAX_PER_ORDER:.2f}."

        return True, "OK"

    # ── Position Tracking ────────────────────────────────────────

    def on_buy(self, symbol: str, buy_price: float, quantity: int) -> None:
        """Record a new BUY position."""
        self.refresh()
        positions = self.state.get("open_positions", [])
        if symbol not in positions:
            positions.append(symbol)
        self.state["open_positions"] = positions
        _save_state(self.state)

    def on_sell(self, symbol: str, buy_price: float, sell_price: float, quantity: int) -> None:
        """Record a SELL and update realized P&L."""
        self.refresh()
        pnl = (sell_price - buy_price) * quantity
        self.state["daily_pnl"] = round(self.state.get("daily_pnl", 0) + pnl, 2)
        positions = self.state.get("open_positions", [])
        if symbol in positions:
            positions.remove(symbol)
        self.state["open_positions"] = positions
        _save_state(self.state)

    # ── Stop-Loss / Target Checks ────────────────────────────────

    def should_stop_loss(self, buy_price: float, current_price: float) -> bool:
        """Return True if current price has breached the stop-loss threshold."""
        drop_pct = ((buy_price - current_price) / buy_price) * 100
        return drop_pct >= STOP_LOSS_PCT

    def should_take_profit(self, buy_price: float, current_price: float) -> bool:
        """Return True if current price has hit the take-profit target."""
        gain_pct = ((current_price - buy_price) / buy_price) * 100
        return gain_pct >= TARGET_PCT

    # ── Status Summary ───────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        self.refresh()
        return {
            "date": self.state["date"],
            "daily_pnl": self.state["daily_pnl"],
            "open_positions": self.state["open_positions"],
            "positions_count": len(self.state["open_positions"]),
            "max_positions": MAX_POSITIONS,
            "max_per_order": MAX_PER_ORDER,
            "daily_loss_limit": MAX_DAILY_LOSS,
            "stop_loss_pct": STOP_LOSS_PCT,
            "target_pct": TARGET_PCT,
            "trading_halted": self.state["daily_pnl"] <= -MAX_DAILY_LOSS,
        }
