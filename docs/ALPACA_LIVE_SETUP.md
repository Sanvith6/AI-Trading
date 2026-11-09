# Alpaca Live Trading Setup & Safety Gating Guide

## Overview

The US Trading Runtime implements **fail-closed multi-gate safety authorization** before any live capital can be deployed. Live trading submits real orders with real money to the Alpaca Live Brokerage API (`https://api.alpaca.markets`).

---

## Hard Safety Principles & Invariants

1. **Default Fail-Closed**:
   - `TRADING_MODE` defaults to `paper_local`.
   - `LIVE_TRADING_ENABLED` defaults to `false`.
   - Any attempt to run live without explicitly fulfilling all 15 independent safety conditions raises an uncatchable block or aborts immediately.
2. **Explicit Operator Authorization**:
   - Live execution requires explicit interactive confirmation: `YES_EXECUTE_LIVE_TRADES` (or an authenticated token in programmatic deployments).
3. **Endpoint Validation**:
   - Live broker adapter strictly enforces `https://api.alpaca.markets`.
   - Pointing the live adapter to sandbox/paper URLs is strictly blocked.
4. **No Force-Open in Live Mode**:
   - `--force-open` is **strictly prohibited** in LIVE trading mode.
   - Attempting `--force-open` in live mode immediately raises a hard `ValueError` and terminates the process. Real capital can only be traded during authentic US market sessions.
5. **Zero Broker Access for Laya Specialist**:
   - `laya-financial-us-v2` is completely frozen and immutable.
   - In `LAYA_MODE=shadow`, Laya runs strictly in parallel as a non-influencing logger. It has zero network access, zero broker keys, and zero ability to modify orders.
6. **Immutable RiskEngine Authority**:
   - Hard limits on max order size (`TRADE_MAX_PER_ORDER`), max daily loss (`TRADE_MAX_DAILY_LOSS`), max positions, max leverage, and max sector exposure are evaluated in pure deterministic code.

---

## 15 Pre-Execution Live Guard Checks

Before each live trading cycle or order execution, `LiveTradingGuard.verify_live_eligibility()` verifies all 15 conditions:

| # | Guard Check | Pass Condition |
|---|---|---|
| 1 | `live_trading_enabled` | `LIVE_TRADING_ENABLED=true` |
| 2 | `trading_mode_is_live` | `TRADING_MODE=live` |
| 3 | `data_mode_is_live` | `DATA_MODE=live` |
| 4 | `market_session_open` | `USMarketCalendar.can_execute_orders == True` (Regular session) |
| 5 | `no_force_session_open` | `force_session_open == False` |
| 6 | `broker_identity_verified` | Live adapter returns `is_live == True`, `environment == 'live'` |
| 7 | `broker_account_active` | Account status is `ACTIVE` |
| 8 | `broker_not_paper` | Account number does not have paper prefix, URL is live |
| 9 | `risk_engine_healthy` | RiskEngine initialized, emergency halt not active |
| 10 | `data_freshness_verified` | Quotes within freshness window (< 60s for real-time) |
| 11 | `feature_pit_valid` | No future data leakage, PIT causality guaranteed |
| 12 | `laya_shadow_safe` | Laya operating in shadow or advisory mode without broker access |
| 13 | `portfolio_reconciled` | Internal position map matches broker records |
| 14 | `kill_switch_inactive` | Global Kill Switch is disengaged (`active == False`) |
| 15 | `operator_confirmed` | Confirmation token or interactive prompt verified |

---

## Configuration & Setup

### Environment Variables

```bash
# Windows PowerShell:
$env:TRADING_MODE="live"
$env:BROKER_PROVIDER="alpaca"
$env:ALPACA_ENV="live"
$env:LIVE_TRADING_ENABLED="true"
$env:ALPACA_LIVE_API_KEY="AK********************"
$env:ALPACA_LIVE_SECRET_KEY="****************************************"
$env:US_LIVE_BROKER_URL="https://api.alpaca.markets"

# Linux / macOS Bash:
export TRADING_MODE="live"
export BROKER_PROVIDER="alpaca"
export ALPACA_ENV="live"
export LIVE_TRADING_ENABLED="true"
export ALPACA_LIVE_API_KEY="AK********************"
export ALPACA_LIVE_SECRET_KEY="****************************************"
export US_LIVE_BROKER_URL="https://api.alpaca.markets"
```

---

## Safe Verification Procedures (Zero Capital Risk)

### 1. Read-Only Live Account Verification
Probe the live broker connection without placing any orders:

```bash
python -m trading.cli broker verify-live
# Or in Windows CUDA:
.\run_monday_cuda.bat broker verify-live
```

**If real live credentials are unset (Mock Diagnostic):**
```
[MOCK DIAGNOSTIC ONLY] No real ALPACA_LIVE_API_KEY configured in environment.
Status:              MOCK_DIAGNOSTIC (NOT GENUINE LIVE)
Account ID:          US_LIVE_MOCK_ACC_9981 [MOCK SIMULATION]
Environment:         live (mock)
Live Attestation:    NO LIVE CREDENTIALS SUPPLIED
Orders Submitted:    ZERO (100% read-only diagnostic)
```

**If real live credentials are configured (Genuine Alpaca Live):**
```
Status:              GENUINE LIVE VERIFIED
Broker Endpoint:     https://api.alpaca.markets (Official Alpaca Live API)
Account ID:          AK_L***9981
Environment:         LIVE (is_live=True)
Currency:            USD
Cash Balance:        $100,000.00
Buying Power:        $250,000.00
Portfolio Value:     $100,000.00
Permissions:         LIVE_TRADING, US_EQUITIES, MARGIN
Multi-Asset Stock:   YES (US Equities & ETFs enabled)
Orders Submitted:    ZERO (100% read-only verification)
```

### 2. Live Dry-Run Trading Cycle
Run the entire production pipeline with live market data, calculate alpha signals, run ML meta-model, classify regime, evaluate portfolio sizing, validate against RiskEngine, and construct formal `OrderIntent` objects, while **guaranteeing zero orders reach the broker**:

```bash
python -m trading.cli run --mode live --dry-run
```

---

## Live Execution Mode

To run live trading during regular market hours:

```bash
python -m trading.cli run --mode live
```
The operator will be prompted:
```
Type 'YES_EXECUTE_LIVE_TRADES' to confirm live order submission:
```
Only after typing the exact confirmation phrase will the execution adapter be unlocked.
