# US Trading Runtime: Operational Runbook

This runbook defines the operational procedures for managing the US Equity AI Trading Runtime across Local Paper, Broker Paper, and Live modes.

---

## 1. Execution Modes & Quick Start

### 1.1 Local Air-Gapped Paper Trading (`US_PAPER_LOCAL`)
```bash
python -m trading.cli run --mode paper_local
```
- Real market data, in-process ledger simulation, zero outbound network calls to brokers.
- Forward journal stored in `trading/paper_journal/paper_orders_journal.parquet`.

### 1.2 Broker-Backed Paper Trading (`US_PAPER_BROKER`)
```bash
python -m trading.cli run --mode paper_broker
```
- Official Alpaca Paper Broker endpoint (`https://paper-api.alpaca.markets`).
- Full order lifecycle, realistic fills, order management.
- Forward journal stored in `trading/paper_broker_journal/broker_paper_orders_journal.parquet`.

### 1.3 Safe Dry-Run Simulation (Any Mode)
```bash
python -m trading.cli run --mode paper_broker --dry-run
python -m trading.cli run --mode live --dry-run
```
- Full decision pipeline, alpha computation, ML scoring, RiskEngine validation, and `OrderIntent` generation.
- Order transmission suppressed; zero orders reach any broker.

---

## 2. Startup Verification & Diagnostics

### 2.1 Run Preflight Audit
Verifies 15 operational subsystems (Python, CUDA, Laya checkpoint, data feeds, calendar, universe, alphas, ML, Regime, Portfolio, RiskEngine, adapters, journals, metrics, kill switch):
```bash
python -m trading.cli preflight
```

### 2.2 Evaluate Pre-Monday Readiness
```bash
python -m trading.cli monday-readiness
```

### 2.3 Read-Only Live Broker Probe
Verify live broker account status, cash, buying power, and trading permissions with ZERO orders placed:
```bash
python -m trading.cli broker verify-live
```

### 2.4 Run 10-Dimension Safety Health Audit
```bash
python -m trading.cli health
```

---

## 3. Emergency Kill Switch Operations

### 3.1 Engage Global Kill Switch (Immediate Halt)
To immediately block all new order generation:
```bash
python -m trading.cli kill-switch on
```
Or set `EMERGENCY_HALT=true` in `.env`.

*Effect*: Prevents new orders immediately across local paper, broker paper, and live modes. Read-only market data and reconciliation remain operational.

### 3.2 Disengage Kill Switch (Resume Normal Operations)
```bash
python -m trading.cli kill-switch off
```

---

## 4. Live Trading Operations (Strict Protocol)

### 4.1 Prerequisites for Live Mode
1. Active, funded US broker account (Alpaca / IBKR).
2. Live credentials configured in `.env`:
   ```bash
   ALPACA_LIVE_API_KEY=your_key
   ALPACA_LIVE_SECRET_KEY=your_secret
   US_LIVE_BROKER_URL=https://api.alpaca.markets
   ```
3. Verified market session is open (`REGULAR`).
4. Risk limits verified and approved.

### 4.2 Request US Live Trading Mode
```bash
python -m trading.cli run --mode live
```
The CLI will display the live warning banner:
```
================================================================================
🚨 LIVE TRADING MODE — REAL BROKER ORDERS MAY BE SUBMITTED 🚨
================================================================================
Type 'YES_EXECUTE_LIVE_TRADES' to confirm live order submission:
```
Type `YES_EXECUTE_LIVE_TRADES` to proceed. If any guard fails, execution halts automatically.

> [!CAUTION]
> `--force-open` is strictly rejected in LIVE mode and will raise an immediate fatal safety exception (`FORCE_OPEN_VIOLATION`). In PAPER mode, `--force-open` is only permitted when running in a designated testing environment (`TESTING=true` or `ALLOW_FORCE_OPEN=true`).

---

## 5. Portfolio & Reconciliation

### 5.1 Inspect Current Portfolio
```bash
python -m trading.cli portfolio
```

### 5.2 Reconcile State with Broker
```bash
python -m trading.cli reconcile
```
