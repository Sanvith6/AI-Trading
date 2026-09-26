# LIVE READINESS — Pre-Production Assessment

> Live trading is NOT enabled in this implementation.
> This document outlines the status of all institutional controls and what must be true before anyone considers real capital deployment.

## 1. Live Trading Requirements

**NEVER automatic.** Requires ALL of:
- `BROKER=angel` (paid broker, real money)
- `TRADE_MODE=live` (NOT paper)
- Human promotion after passing all 14 hard promotion gates
- Emergency stop circuit breaker tested
- All risk gates proven over ≥3 months paper trading

**Default is and remains paper mode forever.**

---

## 2. Technical Readiness Checklist

| Item | Status | Implemented Module |
|---|---|---|
| Ingestion & Data validation | ✅ COMPLETED | `trading/data/base_provider.py`, `yahoo_provider.py`, `validator.py` |
| Local caching & Offline resilience | ✅ COMPLETED | `trading/data/data_cache.py` |
| Immutable experiment versioning | ✅ COMPLETED | `trading/data/versioning.py` |
| Dynamic universe discovery | ✅ COMPLETED | `trading/universe/universe_manager.py` |
| Time-safe feature engine (40+ features) | ✅ COMPLETED | `trading/features/engine.py` |
| 12 Standardized Alpha Families | ✅ COMPLETED | `trading/alpha/` (12 families in `[-1, +1]`) |
| Market Regime Detection | ✅ COMPLETED | `trading/regime/regime_detector.py` |
| 9 Institutional Baseline Strategies | ✅ COMPLETED | `trading/baselines/baselines.py` |
| Tabular ML & Model Calibration | ✅ COMPLETED | `trading/ml/tabular_models.py`, `calibration.py` |
| Meta Model & Disagreement Gate | ✅ COMPLETED | `trading/ml/meta_model.py` |
| Portfolio Construction & Correlation Clustering | ✅ COMPLETED | `trading/portfolio/portfolio_constructor.py` |
| Risk-Aware Position Sizing | ✅ COMPLETED | `trading/portfolio/position_sizer.py` |
| Hard Risk Engine (13+ non-bypassable gates) | ✅ COMPLETED | `trading/risk_engine/validators.py` |
| Execution Simulator & Indian Statutory Taxes | ✅ COMPLETED | `trading/execution/simulator.py` |
| Walk-Forward Splits (Purge + Embargo) | ✅ COMPLETED | `trading/validation/walk_forward.py` |
| Permanent Protected Holdout Guard | ✅ COMPLETED | `trading/validation/holdout.py` |
| Adversarial Stress Testing & Sensitivity | ✅ COMPLETED | `trading/validation/adversarial.py` |
| Darwinian Strategy DNA (15+ fields) | ✅ COMPLETED | `trading/swarm/strategy_dna.py` |
| Multi-Objective Fitness & Novelty Scoring | ✅ COMPLETED | `trading/swarm/multi_objective_fitness.py` |
| Machine-Readable Research Memory & Failure DB | ✅ COMPLETED | `trading/swarm/research_memory.py` |
| Strategy Correlation & Return Attribution | ✅ COMPLETED | `trading/portfolio/correlation_matrix.py`, `attribution.py` |
| Drift Detection (PSI & Kolmogorov-Smirnov) | ✅ COMPLETED | `trading/monitoring/drift_detector.py` |
| Champion / Challenger & Shadow Mode | ✅ COMPLETED | `trading/research/champion_challenger.py` |
| Automated Research Cadence (Daily, Weekly, Monthly) | ✅ COMPLETED | `trading/research/cycles.py` |
| Live paper trading proven for ≥3 months | ⏳ IN PROGRESS | Operational in paper mode |

---

## 3. Data & Risk Readiness

All 13 hard risk gates are enforced in Python code and CANNOT be bypassed by the LLM or any external caller:
1. Emergency halt check
2. Non-positive price rejection
3. Max order value limit (₹5,000 default)
4. Daily loss limit circuit breaker (₹2,000 default)
5. Max positions limit (3-8 positions)
6. Min cash floor retention (₹1,000 minimum)
7. Portfolio exposure cap (95% maximum)
8. Sector exposure cap (40% maximum)
9. Minimum share quantity (≥ 1 share)
10. Valid symbol syntax
11. Risk-budgeted position sizing
12. 1-bar execution delay with next-bar open fill (zero look-ahead)
13. Model disagreement exposure cut (circuit breaker)

---

## 4. Manual Promotion Protocol (14 Hard Gates)

A strategy CANNOT be promoted to live trading automatically.

A human quant researcher must review the challenger metrics against the 14 mandatory gates:
1. Minimum sample size: ≥ 30 closed trades
2. Out-of-sample Sharpe ratio: ≥ 1.20
3. Out-of-sample Sortino ratio: ≥ 1.40
4. Calmar ratio: ≥ 1.00
5. Profit factor: ≥ 1.30
6. Maximum historical drawdown: ≤ 12.0%
7. Minimum win rate: ≥ 45.0%
8. Parameter sensitivity score: ≤ 0.50 (low fragility)
9. Cost-stressed net P&L: ≥ ₹0.00 under 3x statutory brokerage & taxes
10. Slippage-stressed net P&L: ≥ ₹0.00 under 3x slippage
11. Walk-forward fold consistency: profitable in ≥ 60% of chronological test windows
12. Protected holdout evaluation: clean audit record (evaluations ≤ 5, no contamination)
13. Model calibration: Brier score ≤ 0.22, calibration gap ≤ 0.05
14. Shadow mode consistency: ≤ 5% divergence between simulated paper fills and live price path over 30 days

**Any single gate failure causes outright rejection.**

---

## 5. Failure Recovery Runbook

| Scenario | System Response |
|---|---|
| Broker API outage | Fall back to paper simulation; pause queue; alert researcher |
| Stale quote feed | Reject execution; flag `STALE` in validator; halt order generation |
| Risk engine trip | Trigger emergency halt; cancel open orders; freeze further submissions |
| High model disagreement | Meta Model automatically reduces exposure to 0.0 or issues `HOLD` |
| Extreme price gap (>2.5%) | Simulator applies gap penalty; RiskEngine rejects if volatility exceeds bounds |
| Holdout reuse limit | Immediately flags dataset as `CONTAMINATED` and alerts auditor |

---

## 6. Current Verdict

**Current Status**: System architecture is 100% complete and passing all 18 automated test suites. Live execution remains intentionally disabled.
**Operational Mode**: `BROKER=paper`, `TRADE_MODE=paper`.
