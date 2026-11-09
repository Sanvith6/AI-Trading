# US Paper & Live Trading Architecture Specification

## 1. System Overview

The **US Market Trading Runtime** is a unified, institutional-grade equity decision and execution system. It shares a single, deterministic intelligence pipeline across three distinct operational modes:
1. **`US_PAPER_LOCAL` (`TRADING_MODE=paper_local`)**: Fully air-gapped simulated execution on an in-process local ledger (`trading/paper_journal/`). Zero external network calls.
2. **`US_PAPER_BROKER` (`TRADING_MODE=paper_broker`)**: Official Alpaca Paper Broker API integration (`https://paper-api.alpaca.markets`) with isolated journal (`trading/paper_broker_journal/`), complete order lifecycle management, and offline simulation fallback.
3. **`US_LIVE` (`TRADING_MODE=live`)**: Production Alpaca/IBKR live broker execution (`https://api.alpaca.markets`) guarded by a 15-condition fail-closed safety gate and interactive operator authorization (`trading/live_journal/`).

Only the **Execution Adapter** changes between modes:

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
                                              ▼
                              Formal OrderIntent Abstraction
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
           [TRADING_MODE=paper_local] [TRADING_MODE=paper_broker] [TRADING_MODE=live]
              PaperExecutionAdapter     BrokerPaperExecutionAdapter  LiveExecutionAdapter
                      │                       │                       │
           In-Process Air-Gapped        Alpaca Paper API         15-Condition Safety Gate
              Local Ledger           (paper-api.alpaca.markets)               │
                      │                       │                  Verified Live Broker API
                      │                       │                   (api.alpaca.markets)
                      ▼                       ▼                       ▼
            trading/paper_journal/      trading/paper_broker_journal/  trading/live_journal/
```

---

## 2. Configuration Model & Hard Air Gap

Configuration is managed via explicit, strongly typed enums in `trading/runtime/config.py`:

| Parameter | Type / Enum | Default | Description |
| :--- | :--- | :--- | :--- |
| `MARKET` | `MarketType.US` | `US` | Strict market boundary. |
| `TRADING_MODE` | `TradingMode` | `paper_local` | `paper_local`, `paper_broker`, or `live`. Never defaults to live. |
| `DATA_MODE` | `DataMode` | `live` | `live`, `historical`, or `replay`. |
| `LAYA_MODEL` | `str` | `laya-financial-us-v2` | Frozen specialist checkpoint. |
| `LAYA_MODE` | `LayaMode` | `shadow` | `shadow` (observe/log) or `advisory` (filtering). |
| `BROKER_PROVIDER` | `BrokerProvider` | `local` | `local`, `alpaca`, `ibkr`, `mock_live`. |
| `ALPACA_ENV` | `AlpacaEnv` | `paper` | `paper` or `live`. Strictly matched against mode. |
| `LIVE_TRADING_ENABLED`| `bool` | `false` | Must be explicitly unlocked for live. |
| `LIVE_CONFIRMATION_REQUIRED` | `bool` | `true` | Requires confirmation token at runtime. |

### Strict Endpoint & Credential Separation
- `PaperExecutionAdapter` connects solely to `PaperBroker`. It has **zero broker client code or live network endpoints**.
- `BrokerPaperExecutionAdapter` verifies that `base_url` contains `paper` or `sandbox`. Any live URL or credential injection fails closed immediately.
- `LiveExecutionAdapter` is guarded by `LiveTradingGuard.verify_live_eligibility()`. If `LIVE_TRADING_ENABLED` is false, or if broker credentials point to paper/sandbox, execution is unconditionally blocked.

---

## 3. OrderIntent Abstraction & Parity

Before any order is sent to an execution adapter, the decision pipeline constructs an immutable, deterministic `OrderIntent`:

```python
@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    quantity: int
    limit_price: Optional[float]
    order_type: OrderType
    strategy_id: str
    decision_timestamp: str
    state_hash: str
    client_order_id: str = field(default_factory=generate_client_order_id)
    target_mode: str
    risk_approved: bool
```

### Deterministic Invariance
Given identical market state:
1. `CandidateDecision` candidate ordering is identical across all 3 modes.
2. Alpha scores and ML confidence are identical.
3. RegimeEngine state is identical.
4. Top-N selection is identical.
5. RiskEngine validations are identical.
6. `OrderIntent` parameters (`symbol`, `side`, `quantity`, `limit_price`, `state_hash`) are identical.

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
9. Broker account status is `ACTIVE`
10. Broker URL does not point to paper/sandbox
11. Point-in-time causality verified (zero future leakage)
12. Data freshness verified (within configured freshness window)
13. RiskEngine healthy and emergency halt inactive
14. Laya operating in shadow isolation (no broker access)
15. Internal positions reconciled with broker

---

## 5. CLI Verification Commands

```bash
# 1. Full 15-subsystem startup preflight:
python -m trading.cli preflight

# 2. Pre-Monday operational readiness check:
python -m trading.cli monday-readiness

# 3. Read-only live broker connectivity & permissions probe:
python -m trading.cli broker verify-live

# 4. Dry-run execution cycle (zero broker orders transmitted):
python -m trading.cli run --mode paper_broker --dry-run
python -m trading.cli run --mode live --dry-run

# 5. Automated 10-dimension safety health audit:
python -m trading.cli health
```
