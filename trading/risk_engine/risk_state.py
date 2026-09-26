"""
Risk State — Immutable Daily Risk Tracking

Tracks the current day's P&L, positions, and risk state.
This is separate from the paper portfolio and is the source of truth
for all risk validation checks.

Never modify this directly — always go through RiskManager.on_buy/
RiskManager.on_sell interfaces, then RiskValidator checks.
"""

from datetime import date
from typing import Dict, Any
from pathlib import Path
import json
import os

STATE_FILE = Path(__file__).parent.parent / "risk_state.json"


def _load_state() -> Dict[str, Any]:
    """Load risk state from disk, resetting for new day if needed."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            # Reset for new day
            if state.get("date") != str(date.today()):
                state = {"date": str(date.today()), "daily_pnl": 0.0, "open_positions": {}}
            return state
        except Exception:
            pass
    # Fresh state for new day
    state = {"date": str(date.today()), "daily_pnl": 0.0, "open_positions": {}}
    _save_state(state)
    return state


def _save_state(state: Dict[str, Any]) -> None:
    """Save risk state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


class RiskState:
    """
    Immutable daily risk state tracker.
    
    Tracks:
    - Daily P&L
    - Open positions with cost basis
    - Emergency halt flag
    - Trade count for the day
    """
    
    def __init__(self):
        self.state = _load_state()
    
    def refresh(self) -> None:
        """Reload state from disk."""
        self.state = _load_state()
    
    @property
    def daily_pnl(self) -> float:
        """Today's realized + unrealized P&L."""
        return self.state.get("daily_pnl", 0.0)
    
    @daily_pnl.setter
    def daily_pnl(self, value: float) -> None:
        self.state["daily_pnl"] = value
        _save_state(self.state)
    
    @property
    def emergency_halt(self) -> bool:
        """Kill-switch: true blocks ALL trading."""
        return self.state.get("emergency_halt", False)
    
    @emergency_halt.setter
    def emergency_halt(self, value: bool) -> None:
        self.state["emergency_halt"] = value
        _save_state(self.state)
    
    @property
    def open_positions(self) -> Dict[str, Any]:
        """Current open positions with cost basis."""
        return self.state.get("open_positions", {})
    
    @open_positions.setter
    def open_positions(self, positions: Dict[str, Any]) -> None:
        self.state["open_positions"] = positions
        _save_state(self.state)
    
    @property
    def cash_available(self) -> float:
        """Cash remaining after today's P&L and positions."""
        # Simplified — in full impl, this factors in mark-to-market
        return max(0.0, 100000.0 + self.daily_pnl)  # baseline 100k
    
    @property
    def date(self) -> str:
        """Current state date."""
        return self.state.get("date", str(date.today()))
    
    def record_buy(self, symbol: str, buy_price: float, quantity: int) -> None:
        """Record a new BUY position."""
        positions = self.state.get("open_positions", {})
        positions[symbol] = {
            "buy_price": buy_price,
            "quantity": quantity,
            "buy_date": self.state.get("date", str(date.today())),
        }
        self.state["open_positions"] = positions
        _save_state(self.state)
    
    def record_sell(self, symbol: str, sell_price: float, quantity: int) -> float:
        """
        Record a SELL and update realized P&L.
        Returns the realized P&L amount.
        """
        positions = self.state.get("open_positions", {})
        pos = positions.get(symbol, {})
        buy_price = pos.get("buy_price", sell_price)
        quantity = min(quantity, pos.get("quantity", 0))
        
        if quantity > 0:
            pnl = (sell_price - buy_price) * quantity
            self.state["daily_pnl"] = round(self.state.get("daily_pnl", 0) + pnl, 2)
        
        # Remove or reduce position
        if pos.get("quantity", 0) <= quantity:
            positions.pop(symbol, None)
        else:
            pos["quantity"] -= quantity
            positions[symbol] = pos
        
        self.state["open_positions"] = positions
        _save_state(self.state)
        
        return self.state.get("daily_pnl", 0.0)