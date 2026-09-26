# BACKTEST METHODOLOGY

> No backtest is trusted unless it survives hostile evaluation.
> This document defines how the platform backtests.

## 1. Core Principle

**The backtest must be WORSE than reality would be for a naive implementation.**
Every assumption that would inflate returns is considered a bug.

## 2. Walk-Forward Design (No Look-Ahead)

```
Signal at bar i close → EXECUTE at bar i+1 open
```

This is the ONLY realistic execution model for daily data. A close-priced
signal at time T is actionable at T+1 open, never at T close.

```python
# Correct: 1-bar delay
for i in range(50, len(df) - 1, 1):
    signal = analyze(df.iloc[:i+1])  # past-only analysis
    exec_price = df.iloc[i+1]["open"]  # next bar's open
```

**What we AVOID** (and why):
- Using `df.iloc[-1]` for both signal and execution → look-ahead bias
- Random shuffle of time series → destroys autocorrelation
- In-sample parameter tuning on test data → overfitting

## 3. Data Splits

```
RAW DATA (Yahoo) → TRAIN (first 50%) → VALIDATE (next 25%) → TEST (next 25%) → HOLDOUT (protected)
```

- **Train**: Parameter fitting, strategy discovery
- **Validate**: Signal threshold tuning, feature selection
- **Test**: Final performance estimate
- **Holdout**: Protected permanently; only finalists tested against it

Walk-forward is used within TRAIN and VALIDATE. TEST uses strict chronological out-of-sample.

## 4. Transaction Cost Model

Every backtest uses the Indian NSE cost model:

| Cost | Rate | Applies |
|---|---|---|
| STT (CNC delivery) | 0.1% both legs | Buy + Sell |
| Stamp duty | 0.015% Buy only | Buy only |
| Exchange charges (NSE) | 0.00345% both legs | Buy + Sell |
| SEBI charges | 0.0001% (₹10/crore) both legs | Buy + Sell |
| GST | 18% on brokerage+exchange+SEBI | Both legs |
| Slippage | Configurable (10 bps default) | Both legs |
| Spread | Configurable (5 paise default) | Both legs |
| Market impact | 0.1% × (size/volume ratio) | Both legs |

Total cost per round-trip trade ≈ 0.25-0.45% depending on size.

**Cost-aware execution**: All fills in the backtest use `ExecutionSimulator`,
NOT raw price execution. Backtests without cost modeling are invalid.

## 5. Execution Realism

```python
sim = ExecutionSimulator()  # Realistic defaults
stress_sim = sim.stress_execution()  # 3× worse parameters

# Every trade is fill()ed through the simulator, not at raw price
fill = sim.execute(symbol, "BUY", qty, price,
                    bar_volume=int(bar["volume"]),
                    volatility_pct=vol)
```

What's modeled:
- Partial fills (default 10% partial rate)
- Slippage (10 bps default, escalates in stress)
- Spread (5 paise default)
- Market impact (scales with order size vs. volume)
- Price gaps (warned when volatility exceeds threshold)
- Execution delay (1 bar default)

What's NOT modeled (would need broker data):
- Order book depth
- Time-weighted execution
- Exchange-specific routing

## 6. Stress Testing (Adversarial)

Every finalist strategy runs adversarial tests:

| Stress | Parameter Change |
|---|---|
| Cost stress | 3× slippage, 3× spread |
| Partial fill stress | Fill probability 0.75 |
| Delay stress | +1 bar execution delay |
| Gap stress | Max gap halved |
| Noise injection | ±1% price noise |
| Parameter perturbation | ±20% on key parameters |
| Universe perturbation | 10% random symbol swaps |
| Missing data | Random bar removal |
| Monte Carlo | 1000 shuffled trade sequences |
| Regime isolation | Run separately in bull/bear/sideways |
| Crash period | 2020 COVID, 2022 bear |

**Passing criterion**: Strategy must remain net positive under cost stress
AND must not exceed drawdown limits under any single stress test.

## 7. Parameter Sensitivity

After a strategy is identified, run parameter sweep:
- Vary each parameter ±20%
- Measure return change
- Compute sensitivity score: `(best_return - worst_return) / best_return`
- **Reject if sensitivity > 50%** (fragile strategy)

A robust strategy survives ±20% parameter changes with <50% return variance.

## 8. Minimum Trade Count

**No metric is trusted with fewer than 30 trades.**

With fewer trades, the backtest returns are point estimates with unknown
distribution. The backtest output explicitly flags:

```
"metrics_reliable": false,
"minimum_trades_required": 30,
"actual_trades": 4,
"note": "Metrics unreliable — insufficient sample"
```

## 9. What IS Reported

Every backtest returns:
- Total return, CAGR, annualized volatility
- Sharpe, Sortino, Calmar (with CI notes if n<30)
- Max drawdown, drawdown duration
- Profit factor, win rate, expectancy
- Average win, average loss, payoff ratio
- Turnover, total cost, cost ratio
- Slippage burden, market impact burden
- Monthly/weekly/daily return breakdown
- Parameter sensitivity score
- Stress test results summary
- Walk-forward stability metrics

## 10. What IS NOT Reported (would be misleading)

- "Sharpe ratio with no confidence interval" → reported as UNRELIABLE
- Returns without cost analysis → rejected
- Single-stock results without portfolio context → flagged
- In-sample returns without OOS validation → rejected

## 11. Running a Backtest

```bash
# Standard backtest (paper costs, realistic execution)
python -m trading.backtest RELIANCE --days 365

# Stress test (3× worse execution)
EXEC_SLIPPAGE_BPS=30 python -m trading.backtest RELIANCE --days 365

# Via dashboard
# → Refresh Signals → backtest results appear
```

## 12. Reproducibility Requirements

Every backtest experiment records:
- `dataset_version` (Yahoo data hash)
- `universe_version` (which symbols, when)
- `feature_version` (which features computed)
- `strategy_version` (which parameters)
- `code_version` (git commit hash)
- `random_seed` (seed used)
- `execution_model` (simulator config)
- `look_ahead_bias` (should always be "none (1-bar delay)")

Without all fields, the experiment is incomplete and flagged.

---

*When in doubt: add more costs, add a stress test, add a minimum trade check.
Never optimize a backtest without simultaneously checking if it survives adversarial evaluation.*
