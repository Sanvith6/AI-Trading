"""
Broker Abstraction Layer for US Equity Trading.

Provides uniform interface across:
- PaperBroker (simulated, air-gapped, zero external network access)
- LiveUSBrokerAdapter (real US broker e.g., Alpaca / IBKR)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class BrokerIdentity:
    account_id: str
    environment: str  # "live" or "paper"
    currency: str     # "USD"
    buying_power: float
    cash: float
    portfolio_value: float
    permissions: List[str]
    is_live: bool
    verified_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "environment": self.environment,
            "currency": self.currency,
            "buying_power": self.buying_power,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "permissions": self.permissions,
            "is_live": self.is_live,
            "verified_at": self.verified_at,
        }


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: str = ""
    strategy_id: str = "laya_us_v2"
    decision_timestamp: str = ""
    state_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "client_order_id": self.client_order_id,
            "strategy_id": self.strategy_id,
            "decision_timestamp": self.decision_timestamp,
            "state_hash": self.state_hash,
        }


@dataclass
class OrderResponse:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_qty: int
    filled_price: float
    status: OrderStatus
    submitted_at: str
    filled_at: Optional[str] = None
    fee: float = 0.0
    slippage_bps: float = 0.0
    rejection_reason: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "filled_at": self.filled_at,
            "fee": self.fee,
            "slippage_bps": self.slippage_bps,
            "rejection_reason": self.rejection_reason,
        }


class BrokerInterface(ABC):
    """Abstract interface that all broker adapters must implement."""

    @abstractmethod
    def verify_identity(self) -> BrokerIdentity:
        """Queries broker account environment, permissions, and balances."""
        pass

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Returns cash, buying power, portfolio equity."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Returns map of symbol -> position details."""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        """Returns list of open or recent orders."""
        pass

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submits order to broker. Must enforce idempotency via client_order_id."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels open order."""
        pass

    @abstractmethod
    def get_order_status(self, client_order_id: str) -> OrderResponse:
        """Reconciles ambiguous or pending order status."""
        pass
