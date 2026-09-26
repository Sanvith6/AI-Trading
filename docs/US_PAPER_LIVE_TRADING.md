# US Paper & Live Trading Architecture Specification

## 1. System Overview

The **US Market Trading Runtime** is a unified, institutional-grade equity decision and execution system. It shares a single, deterministic intelligence pipeline across both **Paper Trading (`US_PAPER`)** and **Live Trading (`US_LIVE`)**.

Only the **Execution Adapter** changes between modes:
- **`US_PAPER`**: Consumes real-time US market data, runs the shared quantitative decision pipeline, simulates fills with institutional friction, and updates an isolated local ledger (`trading/paper_journal/`). **Structurally air-gapped from live broker networks.**
- **`US_LIVE`**: Consumes the identical real-time US market data, runs the identical decision pipeline, enforces a 15-condition security gate, and routes orders to verified live US broker endpoints (`trading/live_journal/`).

```
                              Real US Market Data (NYSE/NASDAQ)
                                             │
                                             ▼
                             Point-in-Time Feature Engine (Zero Lookahead)
                                             │
                                             ▼
                             12 Quantitative Alpha Families ([-1.0, +1.0])
                                             │
                                             ▼
                             ML Meta-Model & Disagreement Gate
                                             │
                                             ▼
                             Authoritative RegimeEngine Multi-State Detector
                                             │
                                             ▼
                             Candidate Ranking & Top-N Selection
                                             │
                                             ▼
                        Laya Financial Specialist V2 (Secondary Intelligence)
                          (SHADOW: Observes & Logs | ADVISORY: Gating)
                                             │
                                             ▼
                             Portfolio Construction & Capacity Engine
                                             │
                                             ▼
                             Immutable RiskEngine (Hard Gating Boundary)
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         [TRADING_MODE=paper]                            [TRADING_MODE=live]
        PaperExecutionAdapter                           LiveExecutionAdapter
                 │                                               │
     Local Institutional Simulator                     15-Condition Security Gate
                 │                                               │
         Paper Cash Ledger                             Verified Live Broker API
                 │                                               │
     trading/paper_journal/                           trading/live_journal/
```

---

## 2. Configuration Model & Hard Air Gap

Configuration is managed via explicit, strongly typed enums in `trading/runtime/config.py`:

| Parameter | Type / Enum | Default | Description |
| :--- | :--- | :--- | :--- |
| `MARKET` | `MarketType.US` | `US` | Strict market boundary. |
| `TRADING_MODE` | `TradingMode` | `paper` | `paper` or `live`. Never defaults to live. |
| `DATA_MODE` | `DataMode` | `live` | `live`, `historical`, or `replay`. |
| `LAYA_MODEL` | `str` | `laya-financial-us-v2` | Frozen specialist checkpoint. |
| `LAYA_MODE` | `LayaMode` | `shadow` | `shadow` (observe/log) or `advisory` (filtering). |
| `BROKER_PROVIDER` | `BrokerProvider` | `paper` | `paper`, `alpaca`, `ibkr`. |
| `LIVE_TRADING_ENABLED`| `bool` | `false` | Must be explicitly unlocked for live. |
| `LIVE_CONFIRMATION_REQUIRED` | `bool` | `true` | Requires confirmation token at runtime. |

### Hard Paper/Live Air Gap
- `PaperExecutionAdapter` connects solely to `PaperBroker`. It has **zero broker client code or live network endpoints**.
- `LiveExecutionAdapter` is guarded by `LiveTradingGuard.verify_live_eligibility()`. If `LIVE_TRADING_ENABLED` is false, or if broker credentials point to paper/sandbox, execution is unconditionally blocked.

---

## 3. Decision Invariance Guarantee

The decision engine is 100% shared:
Given the same market quote, feature snapshot, and random seed:
1. Candidate list is identical.
2. Alpha scores are identical.
3. ML Meta-model predictions are identical.
4. RegimeEngine state is identical.
5. Laya Specialist V2 probabilities are identical.
6. Portfolio position sizing is identical.
7. RiskEngine approval status is identical.

Only the execution adapter differs.

---

## 4. Multi-Condition Live Safety Gate

Before any live order can be created, `LiveTradingGuard` enforces all 15 conditions:
1. `MARKET == US`
2. `TRADING_MODE == live`
3. `DATA_MODE == live`
4. `LIVE_TRADING_ENABLED == true`
5. `LIVE_CONFIRMATION_TOKEN` verified
6. Global Kill Switch inactive
7. Market session open (`SessionStatus.REGULAR`)
8. Broker account environment verified as genuinely `live`
9. RiskEngine healthy and daily loss limit not breached
10. Data freshness verified (`data_age_seconds <= 60`)
11. Point-in-Time feature causality verified
12. Laya model checkpoint healthy
13. Portfolio cash floor respected
14. Max concurrent positions respected
15. Session manifest matched

If **ANY** check fails, live trading fails closed.

---

## 5. Idempotent Order Execution & Duplicate Protection

Every live order is assigned a deterministic `client_order_id`:
$$\text{client\_order\_id} = \text{SHA256}(\text{strategy} \parallel \text{symbol} \parallel \text{decision\_timestamp} \parallel \text{state\_hash})[:16]$$

- Prevents duplicate order submission across restarts, network retries, or broker timeouts.
- If a broker response is ambiguous (HTTP timeout or socket drop), the adapter enters `OrderStatus.UNKNOWN` and executes an immediate reconciliation query by `client_order_id` rather than blindly resubmitting.
