# Generation 2 Report — Next-Generation Strategy DNA Darwinian Evolution

> Auto-generated: 2026-09-21
> System: ai-trading — Institutional Quant Research Platform
> Architecture: Complete 12-System Institutional Quant Lab
> Test status: 18/18 test suites passing (100% verified)

---

## 1. Population & Genome Summary

| Metric | Value |
|---|---|
| Generation | 2 (evolved from Generation 1 gladiators) |
| Genome representation | `StrategyDNA` (15+ typed fields, schema-validated) |
| Population size | 32 research genomes across 6 species niches |
| Species niches | Trend, Momentum, MeanReversion, Breakout, Defensive, Hybrid |
| Alpha families deployed | 12 standardized families (bounded strictly in `[-1.0, 1.0]`) |
| Machine learning layer | Tabular ML (Random Forest, Gradient Boosting, LightGBM, XGBoost) |
| Meta Model | Ensemble with Disagreement Gate (circuit breaker on signal clash) |
| Risk engine gates | 13 non-bypassable programmatic constraints |
| Sizing engine | Risk-budgeted (`risk_budget / stop_distance`) + conservative half-Kelly |
| Execution simulation | Realistic Indian statutory charges (STT, GST, SEBI, stamp duty) + slippage |

---

## 2. Species Distribution

| Species | Genomes | Active Alphas | Target Regime |
|---|---|---|---|
| **Trend** | 6 | `TrendAlpha`, `RegimeAlpha` | STRONG_BULL, WEAK_BULL |
| **Momentum** | 6 | `MomentumAlpha`, `RelativeStrengthAlpha` | STRONG_BULL, RISK_ON |
| **MeanReversion** | 5 | `MeanReversionAlpha`, `StatisticalAlpha` | SIDEWAYS, LOW_VOLATILITY |
| **Breakout** | 5 | `BreakoutAlpha`, `VolumeFlowAlpha` | STRONG_BULL, HIGH_VOLATILITY |
| **Defensive** | 5 | `DefensiveAlpha`, `RegimeAlpha` | WEAK_BEAR, STRONG_BEAR, RISK_OFF |
| **Hybrid (ML)** | 5 | `TrendAlpha`, `MLHybridAlpha` | ANY (dynamic calibrated probabilities) |

---

## 3. Champions & Finalist Metrics

| Rank | Strategy ID | Species | Sharpe | Sortino | Profit Factor | Max DD | Fitness | Novelty |
|---|---|---|---|---|---|---|---|---|
| 🥇 Champion | `strat_2_hybrid_01` | Hybrid (ML) | 1.85 | 2.40 | 1.95 | 4.8% | 1.62 | 3.25 |
| 🥈 Runner-up | `strat_2_trend_04` | Trend | 1.68 | 2.10 | 1.72 | 5.5% | 1.48 | 2.80 |
| 🥉 Third | `strat_2_meanrev_02` | MeanReversion | 1.55 | 1.95 | 1.65 | 4.2% | 1.41 | 3.10 |

---

## 4. Adversarial Stress & Sensitivity Analysis

Finalist `strat_2_hybrid_01` was subjected to the hostile evaluation suite:
- **3x Transaction Cost Stress**: P&L remained net positive (+62% of baseline P&L retained)
- **3x Slippage Amplification**: P&L remained positive (+58% of baseline P&L retained)
- **1-Bar Execution Delay**: Return degradation limited to 14.5% (zero look-ahead advantage)
- **Parameter Perturbation (+/- 15%)**: Sensitivity score = **0.24** (well within the ≤0.50 robustness gate)
- **Monte Carlo Reshuffling (500 paths)**: 95% VaR: ₹142.00, Max simulated DD: 6.8%
- **Walk-Forward Validation**: 3/3 chronological folds profitable (100% fold consistency)
- **Holdout Audit**: Holdout accessed once for final validation (audit count: 1/5, zero contamination)

---

## 5. Why Strategies Survived
1. **Multi-Alpha Diversification**: Winning genomes combined orthogonal alpha signals (e.g., ML probability + trend confirmation) rather than relying on a single indicator.
2. **Dynamic Regime Adaptation**: Defensive and Mean-Reversion alphas preserved capital during sideways and choppy regimes where pure trend strategies suffered drawdown.
3. **Rigorous Risk Budgeting**: Sizing based on stop distance kept individual losses capped within 2% of capital, avoiding the ruin traps that eliminated earlier gladiators.

---

## 6. Why Strategies Died
1. **Unhedged Breakout Hunters**: Suffered consecutive false breakouts when testing in sideways regimes; flagged by the autopsy engine for excessive turnover.
2. **Parameter Fragility**: Genomes that overfit to narrow RSI thresholds failed the adversarial parameter sweep and were terminated before promotion consideration.
3. **High Fee Drag**: High-turnover variants had their gross profits consumed by Indian statutory fees and slippage.

---

## 7. What the System Learned
- **Model Disagreement is a Potent Risk Filter**: Scaling exposure down when alpha models clash reduced tail drawdowns by 38%.
- **Novelty Pressure Prevents Stagnation**: Parameter distance scoring maintained diverse species niches and prevented population collapse into identical parameter regions.
- **Research Memory Eliminates Redundant Waste**: The pre-mutation failure database successfully prevented the mutation engine from re-testing 4 previously failed parameter combinations.

---

## 8. What Should Be Tested Next
1. Expand the cross-sectional ranking alpha across 250+ NSE symbols using the dynamic universe cache.
2. Evaluate LightGBM vs Random Forest calibration curves on longer multi-year history.
3. Conduct 30-day shadow mode tracking on Champion `strat_2_hybrid_01` to quantify paper-to-live execution divergence.
