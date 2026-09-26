"""
Base Execution Adapter Abstract Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from trading.brokers.interface import OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus


@dataclass
class ExecutionResult:
    success: bool
    order_response: OrderResponse
    execution_type: str  # "PAPER" or "LIVE"
    execution_latency_ms: float
    reconciled: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order_response": self.order_response.to_dict(),
            "execution_type": self.execution_type,
            "execution_latency_ms": self.execution_latency_ms,
            "reconciled": self.reconciled,
            "error_message": self.error_message,
        }


class ExecutionAdapter(ABC):
    """Abstract execution adapter."""

    @abstractmethod
    def execute_order(self, order_request: OrderRequest) -> ExecutionResult:
        """Executes trade order."""
        pass

    @abstractmethod
    def get_portfolio(self) -> Dict[str, Any]:
        """Returns current portfolio ledger/holdings."""
        pass

    @abstractmethod
    def reconcile(self) -> Dict[str, Any]:
        """Reconciles internal state with underlying ledger/broker."""
        pass
