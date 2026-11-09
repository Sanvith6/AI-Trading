"""
Base Execution Adapter Abstract Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from trading.brokers.interface import OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus, OrderIntent
from pathlib import Path


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

    def execute_intent(self, intent: OrderIntent, **kwargs) -> ExecutionResult:
        """Executes order directly from formal OrderIntent."""
        return self.execute_order(intent.to_order_request(), **kwargs)

    def execute_sequential_liquidation(
        self,
        symbol: str,
        total_quantity: int,
        price: float,
        max_order_value: float = 5000.0,
        strategy_id: str = "US_QUANT_LIQUIDATION",
        order_type: OrderType = OrderType.MARKET,
        state_hash: str = "",
    ):
        """Executes large liquidation using sequential fill-aware chunking."""
        from .sequential_liquidation import SequentialChunkLiquidator
        liquidator = SequentialChunkLiquidator(max_order_value=max_order_value)
        return liquidator.liquidate(
            adapter=self,
            symbol=symbol,
            total_quantity=total_quantity,
            price=price,
            strategy_id=strategy_id,
            order_type=order_type,
            state_hash=state_hash,
        )


    @abstractmethod
    def get_portfolio(self) -> Dict[str, Any]:
        """Returns current portfolio ledger/holdings."""
        pass

    @abstractmethod
    def reconcile(self) -> Dict[str, Any]:
        """Reconciles internal state with underlying ledger/broker."""
        pass

    @abstractmethod
    def get_execution_type(self) -> str:
        """Returns execution type (e.g. 'PAPER' or 'LIVE')."""
        pass

    @property
    @abstractmethod
    def journal_path(self) -> Path:
        """Returns path to the audit journal parquet file."""
        pass
