# Quantitative Platform Baseline Specification (BASELINE_V1)

**Status:** FROZEN & IMMUTABLE CONTROL BASELINE  
**Git Commit SHA:** `ce33fc082d61b8c1bc26af566f343a679523df29`  
**Date Frozen:** 2026-09-26  
**Active Environment:** US Equities (NYSE / NASDAQ)  
**Denomination:** USD ($)  

---

## 1. System Configuration
- **Active Market:** `MARKET=US`
- **Exchange:** NYSE / NASDAQ
- **Timezone:** `America/New_York`
- **Trading Hours:** 09:30 - 16:00 EST
- **Active Core Watchlist:** `SPY, AAPL, NVDA, MSFT, TSLA, AMZN, GOOGL, META`
- **Data Source:** Yahoo Finance Data Provider with Parquet Lakehouse caching
- **Broker Interface:** `BROKER=paper`, `TRADE_MODE=paper` (Live trading strictly disabled)

---

## 2. Research Data Lake & Lakehouse Versions
- **Storage Layer:** Columnar Parquet (`pyarrow`) partitioned by `/market/frequency/symbol.parquet`
- **Analytical Query Engine:** DuckDB 1.5.5 zero-copy SQL analytics
- **High-Throughput Feature Engine:** Polars vectorized multi-threaded pipeline
- **Dataset Registry Version:** `v1.0` (Cryptographic SHA256 manifest tracking)
- **Feature Registry Version:** `v1.0` (Semantic feature keys: `ret_1d:v1`, `ret_5d:v1`, `ret_20d:v1`, `rsi_14:v1`, `vol_20d:v1`, `atr_14:v1`, `f_quality_score:v1`, `market_risk_on_off:v1`)
- **Universe Registry Version:** `v1.0` (Dual-market Point-in-Time listing & delisting tracking)

---

## 3. Alpha & Machine Learning Architecture
- **Standardized Alpha Families (12):**
  1. `MomentumAlpha`
  2. `MeanReversionAlpha`
  3. `TrendAlpha`
  4. `BreakoutAlpha`
  5. `VolatilityAlpha`
  6. `VolumeFlowAlpha`
  7. `RelativeStrengthAlpha`
  8. `StatisticalAlpha`
  9. `DefensiveAlpha`
  10. `RegimeAlpha`
  11. `CrossSectionalAlpha`
  12. `MLHybridAlpha`
- **Meta-Model:** Calibrated LightGBM Tabular Classifier with Platt scaling / Isotonic regression
- **Regime Detector:** Deterministic Volatility / Trend Persistence Engine (Trending Bull, Trending Bear, Mean Reverting, High Volatility Choppy)

---

## 4. Portfolio Construction & Factor Intelligence
- **Primary Method:** True Equal Risk Contribution (ERC) / Risk Parity via SciPy SLSQP convex solver
- **Marginal Risk Contribution (MRC):** Active per-asset risk attribution
- **Risk Factor Model:** 6 Style Factors (Momentum, Size, Value, Quality, Volatility, Growth) + Market Beta ($\beta$)
- **Alpha / Beta Decomposition:** $R_p = \beta_{port} \cdot R_m + \alpha_{selection}$
- **Concentration Metric:** Herfindahl-Hirschman Index (HHI) + Effective $N$
- **Capacity Engine:** Almgren-Chriss Square-Root Market Impact Law ($Y = 0.60$)

---

## 5. Immutable RiskEngine Hard Gates (13 Non-Bypassable Rules)
1. Max Single Position Weight: 25.0%
2. Max Total Exposure: 95.0%
3. Min Cash Reserve: 5.0%
4. Max Sector Weight: 40.0%
5. Max Correlation Cluster Weight: 40.0%
6. Max Daily Portfolio Loss: 3.0%
7. Max Trailing Peak Drawdown: 8.0%
8. Max Single Trade Risk: 1.5% of equity
9. Min Liquidity / ADV Threshold: $500,000 daily dollar volume
10. Strict Pre-Trade Capital Sufficiency Check
11. Circuit Breaker Emergency Halt on 3 consecutive anomalies
12. Zero Broker Access Air-Gap (Paper execution only)
13. Strict No-Bypass Invariant ("AI Proposes, RiskEngine Disposes")

---

## 6. Baseline Validation Suite Status
- `trading/tests/test_full_architecture.py`: **12 / 12 PASSED (100%)**
- `trading/tests/test_phase1_foundation.py`: **14 / 14 PASSED (100%)**
- Golden Benchmarks: `GOLDEN_MOMENTUM_US_2023`, `GOLDEN_VOL_ARBITRAGE_US_2022` verified passing within tolerances.
