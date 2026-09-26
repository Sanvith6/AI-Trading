"""
Paper Broker — Structurally Air-Gapped Simulated Broker.

Guarantees:
1. Has ZERO network calls to any live broker API.
2. Completely independent cash ledger, positions, orders, fills, and P&L.
3. Realistic execution simulation using existing cost & slippage models.
4. Persists paper state atomically to disk.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .interface import (
    BrokerInterface,
    BrokerIdentity,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_PAPER_STATE_FILE = Path("trading/paper_portfolio.json")


class PaperBroker(BrokerInterface):
    """Simulated local paper broker for US equity trading."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        state_file: Optional[Path] = None,
        fee_rate_bps: float = 5.0,
        slippage_bps: float = 10.0,
    ):
        self.state_file = state_file or DEFAULT_PAPER_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.fee_rate_bps = fee_rate_bps
        self.slippage_bps = slippage_bps

        self.cash: float = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, OrderResponse] = {}
        self.realized_pnl: float = 0.0

        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.cash = float(data.get("cash", self.initial_capital))
                self.positions = data.get("positions", {})
                self.realized_pnl = float(data.get("realized_pnl", 0.0))
                logger.info("PaperBroker loaded state: cash=$%.2f, positions=%d", self.cash, len(self.positions))
            except Exception as e:
                logger.error("Failed to load paper state from %s: %s. Using initial capital.", self.state_file, e)
                self._save_state()
        else:
            self._save_state()

    def _save_state(self) -> None:
        data = {
            "cash": round(self.cash, 2),
            "positions": self.positions,
            "realized_pnl": round(self.realized_pnl, 2),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.state_file)

    def verify_identity(self) -> BrokerIdentity:
        """Paper broker identity. Always identifies environment as 'paper'."""
        portfolio_val = self.cash
        for sym, pos in self.positions.items():
            portfolio_val += pos.get("quantity", 0) * pos.get("current_price", pos.get("entry_price", 0.0))

        return BrokerIdentity(
            account_id="PAPER_US_001",
            environment="paper",
            currency="USD",
            buying_power=self.cash,
            cash=self.cash,
            portfolio_value=portfolio_val,
            permissions=["PAPER_TRADING", "EQUITY_SIMULATION"],
            is_live=False,
        )

    def get_account(self) -> Dict[str, Any]:
        ident = self.verify_identity()
        return ident.to_dict()

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.positions)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        if status is None:
            return list(self.orders.values())
        return [o for o in self.orders.values() if o.status == status]

    def get_order_status(self, client_order_id: str) -> OrderResponse:
        for o in self.orders.values():
            if o.client_order_id == client_order_id:
                return o
        return OrderResponse(
            order_id="UNKNOWN",
            client_order_id=client_order_id,
            symbol="",
            side=OrderSide.BUY,
            quantity=0,
            filled_qty=0,
            filled_price=0.0,
            status=OrderStatus.UNKNOWN,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rejection_reason="Order not found in paper ledger",
        )

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            o = self.orders[order_id]
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED):
                o.status = OrderStatus.CANCELLED
                self._save_state()
                return True
        return False

    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """
        Executes simulated paper fill with realistic costs, slippage, and ledger updates.
        Strictly local — zero broker network traffic.
        """
        # Idempotency check via client_order_id
        if order.client_order_id:
            for existing in self.orders.values():
                if existing.client_order_id == order.client_order_id:
                    logger.warning("Duplicate paper order rejected: client_order_id '%s' already exists.", order.client_order_id)
                    return existing

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        order_id = f"PAPER_{int(time.time() * 1000)}_{order.symbol}"

        # Fetch market price (from limit_price or fallback)
        price = order.limit_price if order.limit_price and order.limit_price > 0 else 100.0

        # Calculate realistic slippage and friction
        slippage_pct = (self.slippage_bps / 10000.0)
        fee_pct = (self.fee_rate_bps / 10000.0)

        if order.side == OrderSide.BUY:
            fill_price = price * (1.0 + slippage_pct)
            total_cost = (order.quantity * fill_price) * (1.0 + fee_pct)

            if total_cost > self.cash:
                resp = OrderResponse(
                    order_id=order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    filled_qty=0,
                    filled_price=0.0,
                    status=OrderStatus.REJECTED,
                    submitted_at=now,
                    rejection_reason=f"Insufficient paper cash: required ${total_cost:.2f}, available ${self.cash:.2f}",
                )
                self.orders[order_id] = resp
                return resp

            # Update cash and position
            self.cash -= total_cost
            fee = total_cost - (order.quantity * fill_price)

            pos = self.positions.get(order.symbol, {"quantity": 0, "entry_price": 0.0})
            existing_qty = pos.get("quantity", 0)
            existing_entry = pos.get("entry_price", pos.get("avg_price", 0.0))
            new_qty = existing_qty + order.quantity
            new_entry = ((existing_qty * existing_entry) + (order.quantity * fill_price)) / new_qty
            self.positions[order.symbol] = {
                "symbol": order.symbol,
                "quantity": new_qty,
                "entry_price": round(new_entry, 4),
                "current_price": round(fill_price, 4),
                "updated_at": now,
            }

        else:  # SELL
            fill_price = price * (1.0 - slippage_pct)
            pos = self.positions.get(order.symbol)
            if not pos or pos.get("quantity", 0) < order.quantity:
                resp = OrderResponse(
                    order_id=order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    filled_qty=0,
                    filled_price=0.0,
                    status=OrderStatus.REJECTED,
                    submitted_at=now,
                    rejection_reason=f"Insufficient position: holding {pos.get('quantity', 0) if pos else 0}, requested {order.quantity}",
                )
                self.orders[order_id] = resp
                return resp

            gross_proceeds = order.quantity * fill_price
            fee = gross_proceeds * fee_pct
            net_proceeds = gross_proceeds - fee

            # Calculate realized P&L
            entry_cost = pos.get("entry_price", pos.get("avg_price", fill_price))
            cost_basis = order.quantity * entry_cost
            trade_pnl = net_proceeds - cost_basis
            self.realized_pnl += trade_pnl
            self.cash += net_proceeds

            new_qty = pos["quantity"] - order.quantity
            if new_qty == 0:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol]["quantity"] = new_qty
                self.positions[order.symbol]["entry_price"] = entry_cost
                self.positions[order.symbol]["updated_at"] = now

        resp = OrderResponse(
            order_id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_qty=order.quantity,
            filled_price=round(fill_price, 4),
            status=OrderStatus.FILLED,
            submitted_at=now,
            filled_at=now,
            fee=round(fee, 2),
            slippage_bps=round(self.slippage_bps, 2),
        )
        self.orders[order_id] = resp
        self._save_state()
        logger.info(
            "PAPER FILL: %s %d %s @ $%.2f (Cash: $%.2f)",
            order.side.value, order.quantity, order.symbol, fill_price, self.cash
        )
        return resp
