# US Trading Runtime: Operational Runbook

This runbook defines the operational procedures for managing the US Equity AI Trading Runtime in Paper and Live modes.

---

## 1. Paper Trading Operations

### 1.1 Start US Paper Trading
```bash
python -m trading.cli run --market US --mode paper
```
Expected output banner:
```
================================================================================
🟢 US PAPER TRADING — REAL MARKET DATA | SIMULATED EXECUTION | NO REAL ORDERS 🟢
================================================================================
```

### 1.2 Check System Status
```bash
python -m trading.cli status
```
Displays market session status, kill-switch state, data freshness, and cash balances.

### 1.3 Inspect Current Paper Portfolio
```bash
python -m trading.cli portfolio
```
Displays cash balance, open positions, average entry prices, and current market values.

### 1.4 Inspect Forward Journal
Paper orders and fills are stored in `trading/paper_journal/paper_orders_journal.parquet`:
```python
import pandas as pd
df = pd.read_parquet("trading/paper_journal/paper_orders_journal.parquet")
print(df.tail(10))
```

---

## 2. Emergency Kill Switch Operations

### 2.1 Engage Global Kill Switch (Immediate Halt)
To immediately block all new order generation:
```bash
python -m trading.cli kill-switch on
```
Or set `EMERGENCY_HALT=true` in `.env`.

*Effect*: Prevents new orders immediately across both paper and live modes. Read-only market data and reconciliation remain operational.

### 2.2 Disengage Kill Switch (Resume Normal Operations)
```bash
python -m trading.cli kill-switch off
```

---

## 3. Live Trading Operations (Strict Protocol)

### 3.1 Prerequisites for Live Mode
1. Active, funded US broker account (Alpaca / IBKR).
2. Live credentials configured in `.env`:
   ```bash
   ALPACA_LIVE_API_KEY=your_key
   ALPACA_LIVE_SECRET_KEY=your_secret
   US_LIVE_BROKER_URL=https://api.alpaca.markets
   ```
3. Verified market session is open (`REGULAR`).
4. Risk limits verified and approved.

### 3.2 Request US Live Trading Mode
```bash
python -m trading.cli run --market US --mode live
```
The CLI will display the live warning banner:
```
================================================================================
🚨 LIVE TRADING MODE — REAL BROKER ORDERS MAY BE SUBMITTED 🚨
================================================================================
Type 'YES_EXECUTE_LIVE_TRADES' to confirm live order submission:
```
Type `YES_EXECUTE_LIVE_TRADES` to proceed. If any guard fails, execution halts automatically.

---

## 4. Reconciliation & Failure Recovery

### 4.1 Run Startup Reconciliation
```bash
python -m trading.cli reconcile
```
Compares local order state with the broker ledger, verifying cash, positions, and open orders.

### 4.2 Recovery from Network Interruption / Ambiguous Responses
1. Do NOT manually force repeat orders.
2. The runtime tracks orders by deterministic `client_order_id`.
3. Run `python -m trading.cli reconcile` to pull the latest order status from the broker.
4. If discrepancies persist, engage the kill switch: `python -m trading.cli kill-switch on`.
