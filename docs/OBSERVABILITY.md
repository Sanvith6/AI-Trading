# Quantitative Telemetry & Observability Guide

## Prometheus Metrics Export
`QuantitativeMetricsExporter` exposes low-overhead real-time metrics (compatible with Prometheus & Grafana):

| Metric | Type | Labels | Description |
|---|---|---|---|
| `quant_data_latency_seconds` | Histogram | `source`, `market` | Ingestion latency for market quotes |
| `quant_data_failures_total` | Counter | `source`, `market` | Provider network/parsing failures |
| `quant_feature_latency_seconds` | Histogram | - | Feature computation pipeline time |
| `quant_model_latency_seconds` | Histogram | - | ML model inference duration |
| `quant_signals_total` | Counter | `strategy`, `signal_type` | Alpha consensus signals emitted |
| `quant_risk_rejects_total` | Counter | `rule_violated` | Trades blocked by RiskEngine gates |
| `quant_portfolio_pnl` | Gauge | - | Mark-to-market net PnL |
| `quant_portfolio_drawdown_pct`| Gauge | - | Peak-to-trough portfolio drawdown |
| `quant_drift_alert_level` | Gauge | - | 0=NORMAL, 1=WARNING, 2=CRITICAL |
| `quant_strategy_health_score` | Gauge | `strategy_id` | Rolling health score (0.0 to 1.0) |

## Operational Decoupling of Laya
Laya and third-party orchestration tools are strictly decoupled from order execution. They receive read-only notifications for risk breaches or drift alerts; trading logic never stalls if Laya is unreachable.
