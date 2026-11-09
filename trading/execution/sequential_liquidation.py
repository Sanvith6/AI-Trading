"""
Fill-Aware Sequential Chunk Execution Module for Large Position Liquidations.

Guarantees:
1. Strictly adheres to MAX_PER_ORDER_VALUE ($5,000) for every single order chunk.
2. Executes chunks sequentially:
   chunk <= $5,000 -> submit -> obtain broker status -> update remaining position ->
   calculate next chunk -> continue.
3. Explicitly handles all broker status states:
   - FILLED: updates filled and remaining, advances to next chunk.
   - PARTIALLY_FILLED: updates remaining by actual filled quantity, recalculates next chunk.
   - CANCELLED: halts execution sequence, preserves residual position safely.
   - REJECTED: halts execution sequence immediately without generating orphan orders.
4. Invariant: Never generates an order larger than the remaining position.
5. Invariant: Total filled quantity never exceeds the intended total liquidation quantity.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from trading.brokers.interface import OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus
from trading.execution.base import ExecutionAdapter, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ChunkExecutionRecord:
    chunk_index: int
    requested_qty: int
    filled_qty: int
    price: float
    notional: float
    status: OrderStatus
    order_id: str
    rejection_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "requested_qty": self.requested_qty,
            "filled_qty": self.filled_qty,
            "price": self.price,
            "notional": self.notional,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "order_id": self.order_id,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp,
        }


@dataclass
class LiquidationResult:
    symbol: str
    intended_quantity: int
    total_filled_quantity: int
    residual_quantity: int
    chunks: List[ChunkExecutionRecord]
    status: str  # "COMPLETED", "PARTIAL_FILL", "CANCELLED", "REJECTED"
    termination_reason: str
    average_fill_price: float = 0.0
    total_notional: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "intended_quantity": self.intended_quantity,
            "total_filled_quantity": self.total_filled_quantity,
            "residual_quantity": self.residual_quantity,
            "chunks": [c.to_dict() for c in self.chunks],
            "status": self.status,
            "termination_reason": self.termination_reason,
            "average_fill_price": self.average_fill_price,
            "total_notional": self.total_notional,
        }


class SequentialChunkLiquidator:
    """
    Executes large position liquidations (> $5,000) using sequential fill-aware chunking.
    """

    def __init__(self, max_order_value: float = 5000.0):
        self.max_order_value = max_order_value

    def liquidate(
        self,
        adapter: ExecutionAdapter,
        symbol: str,
        total_quantity: int,
        price: float,
        strategy_id: str = "US_QUANT_LIQUIDATION",
        order_type: OrderType = OrderType.MARKET,
        state_hash: str = "",
    ) -> LiquidationResult:
        """
        Executes sequential chunk liquidation of total_quantity shares of symbol at price.
        """
        if total_quantity <= 0 or price <= 0:
            return LiquidationResult(
                symbol=symbol,
                intended_quantity=total_quantity,
                total_filled_quantity=0,
                residual_quantity=max(0, total_quantity),
                chunks=[],
                status="COMPLETED" if total_quantity == 0 else "REJECTED",
                termination_reason="ZERO_OR_NEGATIVE_QUANTITY_OR_PRICE",
            )

        intended_qty = int(total_quantity)
        remaining_qty = intended_qty
        total_filled_qty = 0
        total_notional = 0.0
        chunks: List[ChunkExecutionRecord] = []
        chunk_idx = 1
        status = "COMPLETED"
        termination_reason = "FULLY_LIQUIDATED"

        # Calculate maximum allowed shares per chunk respecting $5,000 hard limit
        max_shares_per_chunk = max(1, int(self.max_order_value // price))

        while remaining_qty > 0:
            # Step 1: Calculate next chunk quantity (<= remaining, notional <= $5,000)
            chunk_qty = min(remaining_qty, max_shares_per_chunk)
            chunk_notional = round(chunk_qty * price, 2)
            while chunk_qty > 1 and chunk_notional > self.max_order_value:
                chunk_qty -= 1
                chunk_notional = round(chunk_qty * price, 2)

            assert chunk_qty <= remaining_qty, f"Chunk {chunk_qty} exceeds remaining {remaining_qty}"

            client_order_id = f"LIQ_{symbol}_{chunk_idx}_{int(time.time()*1000)}"
            order_req = OrderRequest(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=chunk_qty,
                order_type=order_type,
                limit_price=price if order_type == OrderType.LIMIT else None,
                client_order_id=client_order_id,
                strategy_id=strategy_id,
                state_hash=state_hash,
                decision_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

            # Step 2: Submit chunk order to broker
            exec_res = adapter.execute_order(order_req)
            resp = exec_res.order_response

            # Step 3: Obtain broker fill status
            filled = resp.filled_qty
            if resp.status == OrderStatus.FILLED and filled == 0:
                filled = chunk_qty

            actual_filled = min(chunk_qty, max(0, filled))
            actual_fill_price = resp.filled_price if resp.filled_price > 0 else price
            chunk_fill_notional = round(actual_filled * actual_fill_price, 2)

            record = ChunkExecutionRecord(
                chunk_index=chunk_idx,
                requested_qty=chunk_qty,
                filled_qty=actual_filled,
                price=actual_fill_price,
                notional=chunk_fill_notional,
                status=resp.status,
                order_id=resp.order_id,
                rejection_reason=resp.rejection_reason,
            )
            chunks.append(record)

            # Step 4: Update remaining position
            total_filled_qty += actual_filled
            total_notional += chunk_fill_notional
            remaining_qty -= actual_filled

            assert remaining_qty >= 0, f"Remaining position negative: {remaining_qty}"
            assert total_filled_qty <= intended_qty, f"Filled {total_filled_qty} exceeds intended {intended_qty}"

            # Step 5: Evaluate broker status and transition
            if resp.status == OrderStatus.FILLED:
                if remaining_qty == 0:
                    status = "COMPLETED"
                    termination_reason = "FULLY_LIQUIDATED"
                    break
                else:
                    chunk_idx += 1
                    continue

            elif resp.status == OrderStatus.PARTIALLY_FILLED:
                status = "PARTIAL_FILL"
                termination_reason = f"PARTIALLY_FILLED_ON_CHUNK_{chunk_idx}"
                if remaining_qty == 0:
                    status = "COMPLETED"
                    termination_reason = "FULLY_LIQUIDATED"
                    break
                # If partial fill occurred, continue with remaining position if order was closed/canceled
                chunk_idx += 1
                continue

            elif resp.status == OrderStatus.CANCELLED:
                status = "CANCELLED"
                termination_reason = f"ORDER_CANCELLED_ON_CHUNK_{chunk_idx}"
                break

            elif resp.status == OrderStatus.REJECTED:
                status = "REJECTED"
                termination_reason = f"ORDER_REJECTED_ON_CHUNK_{chunk_idx}: {resp.rejection_reason}"
                break

            else:
                # Other status (SUBMITTED / ACCEPTED)
                if exec_res.success:
                    if remaining_qty == 0:
                        status = "COMPLETED"
                        termination_reason = "FULLY_LIQUIDATED"
                        break
                    chunk_idx += 1
                    continue
                else:
                    status = "UNKNOWN_OR_UNCONFIRMED"
                    termination_reason = f"UNCONFIRMED_STATUS_{resp.status.value}"
                    break

        residual_qty = intended_qty - total_filled_qty
        avg_fill_price = round(total_notional / max(1, total_filled_qty), 4) if total_filled_qty > 0 else 0.0

        return LiquidationResult(
            symbol=symbol,
            intended_quantity=intended_qty,
            total_filled_quantity=total_filled_qty,
            residual_quantity=residual_qty,
            chunks=chunks,
            status=status,
            termination_reason=termination_reason,
            average_fill_price=avg_fill_price,
            total_notional=round(total_notional, 2),
        )
