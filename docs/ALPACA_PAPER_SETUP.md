# Alpaca Paper Trading Setup & Configuration Guide

## Overview

The US Trading Runtime provides first-class support for broker-backed paper trading via **Alpaca Paper Trading API**. This mode routes all orders to Alpaca's dedicated paper trading endpoint (`https://paper-api.alpaca.markets`), providing realistic execution fills, order management, and portfolio accounting without risking live capital.

---

## Architecture & Air-Gap Guarantees

1. **Endpoint Isolation**:
   - Paper trading is hardcoded to validate that the broker URL contains `paper` or `sandbox`.
   - Any attempt to configure the paper adapter with a live endpoint (`api.alpaca.markets`) or live API keys fails closed with a `ValueError`.
2. **Credential Segregation**:
   - Paper trading strictly consumes `ALPACA_PAPER_API_KEY` and `ALPACA_PAPER_SECRET_KEY` (or `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`).
   - Live credentials (`ALPACA_LIVE_API_KEY`) are ignored in paper broker mode.
3. **Journal Isolation**:
   - All broker paper executions are logged exclusively to `trading/paper_broker_journal/broker_paper_orders_journal.parquet`.
   - It is physically isolated from local paper (`trading/paper_journal/`) and live (`trading/live_journal/`).
4. **Offline Simulation Fallback**:
   - If API keys are omitted or set to placeholders, `AlpacaPaperBrokerAdapter` automatically activates a high-fidelity local simulation mode. This allows CI, unit tests, and air-gapped development environments to run without an active internet connection or broker account.

---

## Step-by-Step Setup

### Step 1: Create an Alpaca Account
1. Visit [https://alpaca.markets](https://alpaca.markets) and sign up for an individual account.
2. In the Alpaca Dashboard, toggle the switch from **Live** to **Paper Trading**.

### Step 2: Generate Paper API Keys
1. In the Paper Trading dashboard, locate the **API Keys** box on the right.
2. Click **Generate New Key**.
3. Copy the **Key ID** (starts with `PK...`) and **Secret Key**. Store them securely.

### Step 3: Configure Environment Variables
Set the following environment variables in your shell, `.env` file, or CI environment:

```bash
# Windows PowerShell:
$env:TRADING_MODE="paper_broker"
$env:BROKER_PROVIDER="alpaca"
$env:ALPACA_ENV="paper"
$env:ALPACA_PAPER_API_KEY="PK********************"
$env:ALPACA_PAPER_SECRET_KEY="****************************************"
$env:ALPACA_PAPER_BROKER_URL="https://paper-api.alpaca.markets"

# Linux / macOS Bash:
export TRADING_MODE="paper_broker"
export BROKER_PROVIDER="alpaca"
export ALPACA_ENV="paper"
export ALPACA_PAPER_API_KEY="PK********************"
export ALPACA_PAPER_SECRET_KEY="****************************************"
export ALPACA_PAPER_BROKER_URL="https://paper-api.alpaca.markets"
```

---

## Verification Commands

### 1. Run Preflight Audit
Verify that the runtime detects the broker paper adapter and all 15 operational subsystems:

```bash
python -m trading.cli preflight
```

### 2. Test Paper Order Lifecycle & Reconciliation (Roundtrip)
Submit a single tiny paper order, verify order ID, query status update, fill/cancel, and reconcile portfolio state against the broker:

```bash
python -m trading.cli broker test-paper
# Or in Windows CUDA:
.\run_monday_cuda.bat broker test-paper
```

### 3. Verify Operational Readiness
Check readiness for the next market session:

```bash
python -m trading.cli monday-readiness
```

### 3. Run a Dry-Run Trading Cycle
Execute the full decision pipeline with live market data, evaluate candidates through RiskEngine, construct formal `OrderIntent` objects, but suppress order transmission:

```bash
python -m trading.cli run --mode paper_broker --dry-run
```

### 4. Execute Full Broker Paper Trading
During regular market hours (09:30 - 16:00 EDT), execute a live cycle on Alpaca paper:

```bash
python -m trading.cli run --mode paper_broker
```

*(Note: During weekends or test environments, you can supply `--force-open` if `ALLOW_FORCE_OPEN=1` is set in the environment).*

---

## Reconciling Paper Orders & Positions

To inspect paper positions, cash balance, and reconcile against Alpaca:

```bash
# View portfolio cash & positions:
python -m trading.cli portfolio

# Reconcile orders with broker:
python -m trading.cli reconcile
```

---

## Safety Features

- **Global Kill Switch**: Stop all order placement instantly:
  ```bash
  python -m trading.cli kill-switch on
  ```
- **RiskEngine Guard**: Even in paper broker mode, all orders must pass the immutable RiskEngine caps (`max_per_order`, `max_daily_loss`, `max_exposure`, and sector limits).
- **Laya Specialist Shadow Mode**: Laya V2 runs in parallel, recording counterfactual opinions to `trading/laya_shadow_journal/` with zero broker access.
