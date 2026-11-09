"""
Prometheus Metrics Exporter for US Market Trading Runtime.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingMetrics:
    """In-memory Prometheus-compatible metrics registry."""

    _instance: Optional["TradingMetrics"] = None

    def __init__(self):
        self.counters = {
            "paper_orders_total": 0,
            "live_orders_total": 0,
            "paper_fills_total": 0,
            "live_fills_total": 0,
            "laya_queries_total": 0,
            "laya_errors_total": 0,
            "risk_blocks_total": 0,
            "risk_blocks": 0,
            "broker_errors_total": 0,
            "reconciliation_errors_total": 0,
            "reconciliation_errors": 0,
            "duplicate_order_attempts_total": 0,
            "laya_shadow_count": 0,
            "laya_shadow_errors": 0,
            "portfolio_decisions": 0,
            "execution_intents": 0,
            "execution_fills": 0,
            "execution_unknown": 0,
        }
        self.gauges = {
            "trading_mode_info": 1.0,
            "market_session_status": 1.0,
            "universe_size": 100.0,
            "candidate_count": 0.0,
            "laya_shadow_latency_ms": 0.0,
            "data_freshness_seconds": 0.0,
            "market_data_freshness_seconds": 0.0,
            "kill_switch_status": 0.0,
            "kill_switch_active": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_usd": 0.0,
            "alpha_latency_ms": 0.0,
            "ml_latency_ms": 0.0,
            "regime_latency_ms": 0.0,
            "laya_latency_ms": 0.0,
            "broker_latency_ms": 0.0,
            "current_drawdown_pct": 0.0,
            "open_positions_count": 0.0,
            "portfolio_cash_usd": 100000.0,
            "portfolio_value_usd": 100000.0,
        }

    @classmethod
    def get_instance(cls) -> "TradingMetrics":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def inc_counter(self, name: str, value: int = 1) -> None:
        if name in self.counters:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        if name in self.gauges:
            self.gauges[name] = float(value)

    def export_prometheus_text(self) -> str:
        """Renders metrics in official Prometheus plaintext exposition format."""
        lines = []
        for name, val in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {val}")
        for name, val in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {val}")
        return "\n".join(lines) + "\n"

    def get_summary(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
