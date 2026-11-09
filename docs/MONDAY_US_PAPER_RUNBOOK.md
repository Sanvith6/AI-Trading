# Monday US Paper Trading Operator Runbook
**Target Session:** Monday, September 28, 2026  
**Market:** US Equities (NYSE / NASDAQ)  
**Execution Environment:** `US_PAPER_LOCAL` & `US_PAPER_BROKER`  
**Intelligence Mode:** `laya-financial-us-v2` in `SHADOW`  

---

## 1. Timeline & Schedule (US Eastern Time)

| Time (EDT) | Phase | Operator Actions | Automated System State |
|---|---|---|---|
| **08:30 - 09:00** | System Startup & Health Audit | Run Preflight Audit & 10-dimension Health Check | All 15 subsystems validated; Laya checksum verified |
| **09:00 - 09:15** | Data & Universe Refresh | Verify Yahoo real-time feed; resolve dynamic top-50 liquid universe | Market quotes streaming; cache initialized |
| **09:15 - 09:25** | Pre-Market Dry-Run | Execute dry-run cycle across watchlist | 12 alphas generated; ML meta-model scored; OrderIntents verified |
| **09:30** | Market Open | Regular trading session opens (`is_regular_trading_open=True`) | Trading calendar state transitions from CLOSED to REGULAR |
| **09:30 - 15:55** | Intraday Trading Cycles | Autonomous or scheduled 5-minute trading cycles | Alpha -> ML -> Regime -> Top-N -> RiskEngine -> Execution |
| **12:00 & 14:00** | Midday Checkpoints | Run portfolio reconciliation and drift detection checks | Ensure ledger matches broker state; monitor Laya shadow |
| **16:00** | Market Close | Regular trading session closes | Order intake paused; pending orders expired |
| **16:05 - 16:30** | Post-Market Reconciliation | Generate end-of-day audit report, archive forward journals | Reconcile fills, commissions, slippage, P&L |

---

## 2. Pre-Market Startup Checklist (08:30 - 09:15 EDT)

### Step 1: Initialize Environment
Choose either local air-gapped paper or broker-backed paper:

**Option A: Local Paper Trading (Air-gapped)**
```powershell
$env:TRADING_MODE="paper_local"
$env:DATA_MODE="live"
$env:LAYA_MODE="shadow"
$env:BROKER_PROVIDER="local"
$env:LIVE_TRADING_ENABLED="false"
```

**Option B: Alpaca Broker Paper Trading (Real Broker Paper)**
```powershell
$env:TRADING_MODE="paper_broker"
$env:DATA_MODE="live"
$env:LAYA_MODE="shadow"
$env:BROKER_PROVIDER="alpaca"
$env:ALPACA_ENV="paper"
$env:ALPACA_PAPER_API_KEY="PK..."
$env:ALPACA_PAPER_SECRET_KEY="..."
$env:LIVE_TRADING_ENABLED="false"
```

### Step 2: Execute Preflight Audit (CUDA GPU Environment)
Verify all 15 operational subsystems. Execution must be run in the CUDA environment (e.g. using `run_monday_cuda.bat` or `./run_monday_cuda.sh`) to ensure GPU acceleration for Laya V2:
```powershell
.\run_monday_cuda.bat preflight --mode paper_broker
```
*Criteria: ALL 15 CHECKS MUST REPORT `[PASS]`. Confirms AlpacaMarketDataProvider transport, feed, connection state, and fails closed if PyTorch is CPU-only.*

### Step 3: Test Alpaca Paper Order Lifecycle & Reconciliation
Perform an end-to-end roundtrip test:
```powershell
# If ALPACA_PAPER_API_KEY is configured (Genuine Online Broker Verification):
.\run_monday_cuda.bat broker test-paper

# If testing software path without broker credentials (Isolated Simulation):
.\run_monday_cuda.bat broker test-paper --offline-sim
```
*Validates: Credentials, endpoint, order submission, status query, fill/cancel, portfolio reconciliation, and forward journal.*

### Step 4: Verify Pre-Monday Readiness Verdict
```powershell
.\run_monday_cuda.bat monday-readiness
```
*Expected Output:*
```
============================================================
PRE-MONDAY US TRADING OPERATIONAL READINESS VERDICTS
============================================================
READY_FOR_US_PAPER_LOCAL:       YES
READY_FOR_ALPACA_PAPER:         YES (GENUINE_BROKER)
LIVE_CONNECTIVITY:              NOT_VERIFIED (NO_CREDENTIALS)
LIVE_EXECUTION:                 DISABLED (LOCKED)
============================================================
```
*(Note: As soon as real paper credentials are provided, `READY_FOR_ALPACA_PAPER` transitions to `YES (GENUINE_BROKER)`).*

### Step 5: Run 10-Dimension Safety Health Audit
```powershell
.\run_monday_cuda.bat health
```
*Criteria: Overall Status must be `HEALTHY` or `WARNING` with 0 blocking issues.*

### Step 6: Execute Pre-Market Dry-Run
```powershell
.\run_monday_cuda.bat run --mode paper_broker --dry-run
```
*Criteria: Verifies decision pipeline, Alpha scores, and OrderIntent creation without sending orders.*

---

## 3. Market Hours Execution (09:30 - 16:00 EDT)

### Executing a Trading Cycle
To run a discrete cycle in the CUDA GPU environment:
```powershell
.\run_monday_cuda.bat run --mode paper_broker
```

### Continuous Automated Loop
To run cycles every 60 seconds during regular trading hours:
```powershell
while ($true) {
    .\run_monday_cuda.bat run --mode paper_broker
    Start-Sleep -Seconds 60
}
```

### Intraday Monitoring Commands
Open a separate monitoring terminal:
```powershell
# 1. View runtime status & kill switch state:
.\run_monday_cuda.bat status

# 2. View current cash, positions, and unrealized P&L:
.\run_monday_cuda.bat portfolio

# 3. Reconcile executed orders with underlying broker:
.\run_monday_cuda.bat reconcile
```

---

## 4. Emergency Procedures & Incident Response

### Scenario A: Sudden Market Disruption / Volatility Spike
Engage the emergency kill switch immediately:
```powershell
python -m trading.cli kill-switch on
```
- Blocks all subsequent trading cycles instantly.
- Existing open orders are preserved or cancelled safely.

To resume trading once conditions normalize:
```powershell
python -m trading.cli kill-switch off
```

### Scenario B: Market Data Feed Latency / Stale Quotes
If market data latency exceeds the configured freshness window:
1. Strategy-aware freshness gate automatically skips stale symbols.
2. If all symbols are stale, cycle pauses gracefully without executing orders.
3. Check internet connectivity and Yahoo Finance endpoint reachability.

### Scenario C: Unreconciled Order or Network Disconnect
If an order response returns status `UNKNOWN`:
1. The orchestrator records the order as pending reconciliation.
2. In the next cycle, `orchestrator.reconcile()` queries the broker by `client_order_id`.
3. Due to idempotency, no duplicate orders will ever be placed.

---

## 5. Post-Market Close Procedure (16:00 - 16:30 EDT)

1. Verify all daily orders have settled:
   ```powershell
   python -m trading.cli reconcile
   ```
2. Print final portfolio summary:
   ```powershell
   python -m trading.cli portfolio
   ```
3. Verify that Laya V2 shadow records are properly journaled:
   - Check `trading/laya_shadow_journal/laya_shadow_records.parquet`.
4. Ensure all forward journal parquets are intact:
   - `trading/paper_broker_journal/broker_paper_orders_journal.parquet`
   - `trading/paper_journal/paper_orders_journal.parquet`
