# RISK MODEL — Immutable Security Boundary

> The Risk Engine is stronger than the AI. The AI CANNOT override these limits.

## 1. Philosophy

The LLM proposes. The mathematics validates. The risk engine constrains.
**Every trading decision passes through this chain:**

```
AI decision
  → Portfolio construction validation
    → Risk validation (RiskEngine.validate_order)
      → Execution validation (simulator)
        → Broker adapter (paper or live gateway)
```

The LLM is **never** allowed to skip this chain.
The LLM is never allowed to directly invoke broker endpoints.

## 2. Hard Rules (non-bypassable)

| # | Rule | Default | Enforcement |
|---|---|---|---|
| R1 | Emergency halt | `EMERGENCY_HALT=false` | Blocks ALL trading if true |
| R2 | Max per order | `TRADE_MAX_PER_ORDER=₹5000` | Hard reject if exceeded |
| R3 | Max positions | `TRADE_MAX_POSITIONS=3` | Reject new BUY if full |
| R4 | Max exposure | `TRADE_MAX_EXPOSURE=95%` | Reject if would exceed |
| R5 | Max sector weight | `TRADE_MAX_SECTOR_WEIGHT=40%` | Reject if would exceed |
| R6 | Max correlated exposure | `TRADE_MAX_CORRELATED=60%` | Reject if would exceed |
| R7 | Daily loss limit | `TRADE_MAX_DAILY_LOSS=₹2000` | Halt ALL trading |
| R8 | Min cash floor | `TRADE_MIN_CASH=₹1000` | Reject if trade breaches |
| R9 | Stop-loss per trade | `TRADE_STOP_LOSS_PCT=2.0%` | Force-exit below |
| R10 | Take-profit per trade | `TRADE_TARGET_PCT=4.0%` | Force-exit above |
| R11 | Per-trade risk cap | `TRADE_RISK_PER_TRADE_PCT=5%` | Auto-shrinks position |
| R12 | Min trades for metrics | `TRADE_MIN_TRADES=30` | Flag metrics as unreliable |
| R13 | Valid symbol required | alphabetic only | Reject non-alphabetic |
| R14 | Positive price required | >0 | Reject zero/negative |

## 3. Validation Chain

```python
from trading.risk_engine import RiskEngine

risk = RiskEngine()
approval = risk.validate_order(
    symbol="RELIANCE",
    action="BUY",
    price=1226.40,
    quantity=4,
    cash=100000.0,
    existing_positions={"TCS": {"buy_price": 2100, "quantity": 2, "current_price": 2150}},
    daily_pnl=-500.0,
)
if not approval.approved:
    print("REJECTED:", approval.deny_reason)
else:
    print("APPROVED — execute through simulator")
```

## 4. Position Sizing

Risk-aware sizing, NOT `cash / number_of_stocks`:

```
raw_size = risk_budget / stop_distance
risk_budget = cash × RISK_PER_TRADE_PCT (5% default)
stop_distance = price × STOP_LOSS_PCT (2% default)
```

Then capped by:
- Per-order limit (`max_per_order / price`)
- Portfolio exposure limit (`cash × max_exposure / price`)
- Liquidity (not implemented — placeholder for broker API)

```python
qty = risk.calculate_position_size(cash=100000, price=1226.40)
print("Position size:", qty)  # Conservative int
```

## 5. Portfolio-Level Checks

For multi-stock portfolios, the RiskEngine additionally enforces:

- **Total exposure**: sum of all positions ≤ 95% of cash
- **Sector concentration**: no sector > 40% of portfolio
- **Correlation**: no >60% of capital in correlated positions
- **Position count**: no more than 3 concurrent positions (configurable)
- **Daily loss**: auto-halt at ₹2000 daily loss

## 6. Exit Enforcement

```python
allowed, reason = risk.validate_exit("RELIANCE", positions)
if not allowed:
    # Force-exit: stop-loss or take-profit HIT
    print("FORCE EXIT:", reason)
```

Stop-loss and take-profit checks happen at EVERY arena round,
plus continuously during monitoring cycles. No position survives
a breach — even if it means a worse fill price.

## 7. Minimum Trade Count for Metric Trust

Before any strategy metric is trusted:
- **Sharpe/Sortino/Calmar**: requires ≥30 trades
- **Profit factor**: requires ≥10 trades
- **Win rate**: requires ≥30 trades
- **Any metric with <30 trades**: explicitly flagged as unreliable

```python
trusted, message = risk.validate_metrics(trade_count=15)
# → False, "Only 15 trades — need ≥30 for statistical trust"
```

## 8. Risk Limits Configuration (from `.env`)

```env
TRADE_MAX_PER_ORDER=5000          # Max ₹ per single trade
TRADE_MAX_DAILY_LOSS=2000         # Halt ALL trading if day loss exceeds this
TRADE_MAX_POSITIONS=3             # Max concurrent open positions
TRADE_STOP_LOSS_PCT=2.0           # Force exit if drops this % below buy
TRADE_TARGET_PCT=4.0               # Force exit if gains this % above buy
TRADE_RISK_PER_TRADE_PCT=5.0      # Max % of cash risked per trade
TRADE_MIN_TRADES=30               # Min trades before trusting metrics
TRADE_MAX_EXPOSURE=0.95           # Max % of cash deployed
TRADE_MAX_SECTOR_WEIGHT=0.40      # Max % in any sector
TRADE_MAX_CORRELATED=0.60         # Max % in correlated bets
TRADE_MIN_CASH=1000               # Min cash floor
EMERGENCY_HALT=false              # Kill-switch: true blocks ALL trading
```

## 9. What the AI Cannot Do

| AI CAN | AI CANNOT |
|---|---|
| Propose a trade | Override risk limits |
| Suggest a position size | Bypass position sizing calc |
| Recommend BUY/SELL | Execute without risk validation |
| Analyze portfolio | Change daily loss limit |
| Identify opportunities | Unlock live trading |
| Analyze risk status | Access broker credentials |
| Read market data | Skip emergency halt |
| Propose strategies | Modify risk engine code |

## 10. Live Trading Safety (NOT YET ENABLED)

When `BROKER=angel` AND `TRADE_MODE=live` AND human promotion is complete,
orders pass through an additional `Gateway`:

1. Schema validation (all required fields present)
2. Risk validation (RiskEngine.validate_order)
3. Position validation (position size, exposure)
4. Capital validation (sufficient cash)
5. Market-open validation (no off-hours trading)
6. Duplicate-order validation (same symbol/action/price)
7. Rate-limit validation (max orders per minute)
8. Emergency-stop validation (halt flag)

**Live trading requires human activation.** This implementation does NOT
enable live trading by default.

---

*Every risk limit is configurable via `.env`. Changes require code restart.
No in-flight trade is interrupted by a limit change — limits apply to new orders only.*
