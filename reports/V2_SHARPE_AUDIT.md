# PHASE 3 AUDIT REPORT: Deconstruction of the V2 Sharpe Ratio & Observation Frequency

**Audit Target:** Laya Financial Specialist V2 Sharpe Calculation  
**Published Claim in V2 Report:** Sharpe = 8.32  
**Audit Finding:** The published 8.32 Sharpe is an **over-annualization artifact** arising from omitting uninvested cash days and assigning 10-bar multi-day holding period returns onto a single entry date.  
**True Economically Sound Portfolio Sharpe:** **4.04** (all-days daily portfolio) or **2.09** (holding-cycle annualized).

---

## 1. Mathematical Breakdown of the V2 Evaluator Flaw

In `trading/laya_specialist_v2/evaluator.py`, lines 98–106:
```python
unique_dates = sorted(daily_pnl.keys())
daily_returns = np.array([np.mean(daily_pnl[d]) for d in unique_dates])
daily_sharpe = float((np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252))
```

### Forensic Defect Analysis:
1. **Omission of Cash Days (24 out of 43 days):**
   - The Dev Holdout window (2026-07-01 to 2026-08-28) contains **43 business trading days**.
   - Laya V2 approved trades on only **19 unique entry days**.
   - The original code calculated `np.mean` and `np.std` over ONLY those 19 days, discarding the 24 days where the system held cash (return = 0.0).
   - This artificially suppressed the real volatility of sitting in cash and inflated the daily return frequency.
2. **Horizon-Frequency Mismatch (10-Bar Return Treated as 1-Day Return):**
   - Every candidate trade is evaluated over a **10-bar forward holding horizon**.
   - The realized net return (e.g. +3.55%) is the cumulative return earned across 10 bars.
   - By assigning the entire 10-bar return to day $t$ and then multiplying by $\sqrt{252}$, the evaluator implicitly assumed that the strategy executes 252 independent trades per year, earning +3.55% every single day!
   - In reality, a 10-bar trade cycle allows a single capital slot to turn over at most $252 / 10 = \mathbf{25.2\text{ times per year}}$.

---

## 2. Quantitative Comparison of Sharpe Methodologies

| Sharpe Calculation Methodology | Observation Basis | Observation Count ($N$) | Cash Days Included | Annualization Factor | Annualized Mean Return | Annualized Std Dev | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Published V2 Evaluator** | Active entry days (unnormalized) | 19 | No | $\sqrt{252}$ | +84.5% | 10.15% | **8.32** *(Inflated)* |
| **Slot-Normalized Active Days** | Active entry days | 19 | No | $\sqrt{252}$ | +56.8% | 8.94% | **6.35** |
| **True All-Calendar Days (Portfolio)** | **All 43 business days in holdout** | **43** | **Yes (24 cash days)** | **$\sqrt{252}$** | **+184.2%** | **45.54%** | **4.04 (Honest Daily Portfolio)** |
| **Holding-Cycle Trade Sharpe** | **28 approved 10-bar trades** | **28** | N/A | **$\sqrt{25.2}$** | **+89.6%** | **42.81%** | **2.09 (Honest Per-Cycle)** |
| *Naive Trade Annualization* | 28 approved trades treated as 1-day | 28 | N/A | $\sqrt{252}$ | +283.4% | 42.81% | **6.62** *(Methodologically Invalid)* |

---

## 3. Sortino, Calmar, and Drawdown Audit

When evaluated honestly across all 43 calendar trading days with cash drag:
- **Daily Portfolio Sharpe Ratio:** **4.04**
- **Daily Portfolio Sortino Ratio:** **7.09** (reflecting strongly positive upside skew with minimal downside deviation)
- **Maximum Drawdown:** **-6.06%**
- **Cumulative Return (Holdout Period):** **+31.43%**
- **Annualized Calmar Ratio:** **5.19**

---

## 4. Phase 4 Audit: True Unit of Observation

| Unit Level | Count in Holdout | Dependence Structure | Independent Observations |
| :--- | :--- | :--- | :--- |
| **Raw Typed Decisions** | 2,226 | High (6 choice questions per candidate) | Non-independent (intra-candidate) |
| **Candidate Proposals** | 371 | Moderate (cross-sectional market beta) | Dependent across simultaneous candidates |
| **Approved Trades** | 28 | Low (7.55% selective subset) | Partially clustered in 19 trading sessions |
| **Active Trading Sessions** | 19 | Minimal (spread across July–August 2026) | Clustered on volatile setup days |
| **Calendar Portfolio-Days** | **43** | **True independent economic unit** | **43 independent daily portfolio states** |

### Critical Auditor Rule:
Never cite 30,000 typed decisions or 371 candidates as 371 independent trades. In the out-of-sample holdout, the strategy produced **28 approved trades spanning 19 active sessions across 43 calendar portfolio-days**.
The genuine risk-adjusted economic performance is **Sharpe = 4.04 (Daily Portfolio)** or **Sharpe = 2.09 (Holding Cycle)**, NOT 8.32.
