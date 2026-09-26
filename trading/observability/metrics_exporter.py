"""
Quantitative Observability & Prometheus Telemetry Exporter.

Exposes low-overhead real-time metrics for quantitative pipelines:
- Data & Ingestion Latency / Failures
- Feature Engineering & Model Inference Latency
- Signal Generation & Risk Engine Rejections
- Orders, Fills, & Slippage
- Realized Portfolio PnL, Drawdown, Exposure, Turnover
- Statistical Drift State (0=NORMAL, 1=WARNING, 2=CRITICAL)
- Strategy Health & Degradation Scores

Graceful fallback when prometheus server is not active.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available; MetricsExporter will log metrics in-memory.")


class QuantitativeMetricsExporter:
    """
    Prometheus metrics collector and exporter for trading engine telemetry.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QuantitativeMetricsExporter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, port: Optional[int] = None):
        if self._initialized:
            return
        self._initialized = True
        self.port = port
        self._server_started = False
        self.local_metrics: Dict[str, Any] = {}

        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

        if port and PROMETHEUS_AVAILABLE:
            self.start_server(port)

    def _init_prometheus_metrics(self):
        try:
            self.data_latency = Histogram("quant_data_latency_seconds", "Market data fetch latency", ["source", "market"])
            self.data_failures = Counter("quant_data_failures_total", "Market data fetch errors", ["source", "market"])
            self.feature_latency = Histogram("quant_feature_latency_seconds", "Feature computation latency")
            self.model_latency = Histogram("quant_model_latency_seconds", "ML model inference latency")
            self.signals_count = Counter("quant_signals_total", "Alpha signals emitted", ["strategy", "signal_type"])
            self.risk_rejects = Counter("quant_risk_rejects_total", "RiskEngine rejected trades", ["rule_violated"])
            self.orders_placed = Counter("quant_orders_total", "Executed orders", ["action", "market"])
            self.portfolio_pnl = Gauge("quant_portfolio_pnl", "Current net portfolio PnL")
            self.portfolio_drawdown = Gauge("quant_portfolio_drawdown_pct", "Current portfolio drawdown percentage")
            self.portfolio_exposure = Gauge("quant_portfolio_exposure_pct", "Total capital exposure percentage")
            self.drift_state = Gauge("quant_drift_alert_level", "Drift status: 0=NORMAL, 1=WARNING, 2=CRITICAL")
            self.strategy_health = Gauge("quant_strategy_health_score", "Strategy health score 0.0 - 1.0", ["strategy_id"])
        except ValueError:
            # Metrics already registered in registry
            pass

    def start_server(self, port: int = 9100) -> None:
        """Start Prometheus metric HTTP endpoint."""
        if not PROMETHEUS_AVAILABLE or self._server_started:
            return
        try:
            start_http_server(port)
            self._server_started = True
            logger.info(f"Started Prometheus metrics exporter on port {port}")
        except Exception as e:
            logger.warning(f"Could not bind Prometheus metrics port {port}: {e}")

    def record_data_fetch(self, latency_seconds: float, success: bool, source: str = "yahoo", market: str = "US"):
        self.local_metrics["last_data_latency"] = latency_seconds
        if PROMETHEUS_AVAILABLE:
            try:
                self.data_latency.labels(source=source, market=market).observe(latency_seconds)
                if not success:
                    self.data_failures.labels(source=source, market=market).inc()
            except Exception:
                pass

    def record_feature_computation(self, latency_seconds: float):
        self.local_metrics["last_feature_latency"] = latency_seconds
        if PROMETHEUS_AVAILABLE:
            try:
                self.feature_latency.observe(latency_seconds)
            except Exception:
                pass

    def record_model_inference(self, latency_seconds: float):
        self.local_metrics["last_model_latency"] = latency_seconds
        if PROMETHEUS_AVAILABLE:
            try:
                self.model_latency.observe(latency_seconds)
            except Exception:
                pass

    def record_signal(self, strategy: str, signal_type: str):
        key = f"signal_{strategy}_{signal_type}"
        self.local_metrics[key] = self.local_metrics.get(key, 0) + 1
        if PROMETHEUS_AVAILABLE:
            try:
                self.signals_count.labels(strategy=strategy, signal_type=signal_type).inc()
            except Exception:
                pass

    def record_risk_reject(self, rule_violated: str):
        key = f"risk_reject_{rule_violated}"
        self.local_metrics[key] = self.local_metrics.get(key, 0) + 1
        if PROMETHEUS_AVAILABLE:
            try:
                self.risk_rejects.labels(rule_violated=rule_violated).inc()
            except Exception:
                pass

    def record_order(self, action: str, market: str = "US"):
        key = f"order_{action}_{market}"
        self.local_metrics[key] = self.local_metrics.get(key, 0) + 1
        if PROMETHEUS_AVAILABLE:
            try:
                self.orders_placed.labels(action=action, market=market).inc()
            except Exception:
                pass

    def update_portfolio_state(self, pnl: float, drawdown_pct: float, exposure_pct: float):
        self.local_metrics["portfolio_pnl"] = pnl
        self.local_metrics["drawdown_pct"] = drawdown_pct
        self.local_metrics["exposure_pct"] = exposure_pct
        if PROMETHEUS_AVAILABLE:
            try:
                self.portfolio_pnl.set(pnl)
                self.portfolio_drawdown.set(drawdown_pct)
                self.portfolio_exposure.set(exposure_pct)
            except Exception:
                pass

    def update_drift_status(self, alert_level: str):
        # 0=NORMAL, 1=WARNING, 2=CRITICAL
        level_map = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        code = level_map.get(alert_level.upper(), 0)
        self.local_metrics["drift_status"] = alert_level
        if PROMETHEUS_AVAILABLE:
            try:
                self.drift_state.set(code)
            except Exception:
                pass

    def update_strategy_health(self, strategy_id: str, health_score: float):
        self.local_metrics[f"health_{strategy_id}"] = health_score
        if PROMETHEUS_AVAILABLE:
            try:
                self.strategy_health.labels(strategy_id=strategy_id).set(health_score)
            except Exception:
                pass

    def get_snapshot(self) -> Dict[str, Any]:
        """Return memory snapshot of latest recorded metrics."""
        return dict(self.local_metrics)


# Global singleton instance
metrics_exporter = QuantitativeMetricsExporter()
