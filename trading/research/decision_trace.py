"""
Trade Decision Trace & Explainability Engine.

Captures an immutable, comprehensive snapshot of why a trade was executed:
- Timestamp & Symbol & Market
- Universe eligibility & rank
- Individual Alpha scores from all active alpha families
- ML Meta-Model probability and calibrated confidence
- Market Context (SPY trend, VIX regime, leading sector)
- Fundamental metrics as-of execution
- Portfolio state and concentration
- RiskEngine pre-trade check validation
- Position sizing calculation
- Execution assumptions (slippage & fees)
- Final decision reason

Enables institutional post-mortem:
'Why did the algorithm buy/sell this symbol at this exact bar?'
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DECISION_TRACE_PATH = Path(__file__).parent.parent.parent / "data_lake" / "decision_traces.json"


@dataclass
class TradeDecisionTrace:
    trade_id: str
    timestamp: str  # "YYYY-MM-DDTHH:MM:SSZ"
    market: str  # "US" or "IN"
    symbol: str
    action: str  # "BUY", "SELL", "REBALANCE"
    universe_rank: int
    alpha_scores: Dict[str, float]  # individual alpha signals
    ml_probability: float
    ml_confidence: float
    regime: str
    market_context: Dict[str, Any]
    fundamentals: Dict[str, float]
    portfolio_state_before: Dict[str, Any]
    risk_validation: Dict[str, Any]  # e.g., {"passed": True, "checks": ...}
    position_size: float
    target_weight: float
    execution_assumption: Dict[str, float]  # slippage, fee
    final_decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionTraceEngine:
    """
    Manages structured trade decision records for auditability and replay.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_DECISION_TRACE_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.traces: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load decision traces: {e}")
        return {}

    def _save(self) -> None:
        try:
            self.storage_path.write_text(json.dumps(self.traces, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save decision traces: {e}")

    def record_trace(self, trace: TradeDecisionTrace) -> None:
        """Record an immutable trade decision trace."""
        self.traces[trace.trade_id] = trace.to_dict()
        self._save()
        logger.info(f"Recorded decision trace for trade {trace.trade_id} ({trace.symbol} {trace.action})")

    def get_trace(self, trade_id: str) -> Optional[TradeDecisionTrace]:
        """Fetch a specific decision trace."""
        data = self.traces.get(trade_id)
        if data:
            return TradeDecisionTrace(**data)
        return None

    def list_traces(self, symbol: Optional[str] = None, limit: int = 50) -> List[TradeDecisionTrace]:
        """List recent decision traces."""
        res = []
        for data in self.traces.values():
            if symbol is None or data.get("symbol") == symbol.upper():
                res.append(TradeDecisionTrace(**data))
        return sorted(res, key=lambda x: x.timestamp, reverse=True)[:limit]
