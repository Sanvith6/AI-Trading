# Institutional Forward Paper-Trading Protocol

**Protocol Version**: `v1.0-ForwardAudit`  
**Governing Standard**: Institutional Quantitative Verification Policy  
**Effective Date**: 2026-09-21  
**Current Status**: **FORWARD VALIDATION = PENDING (Calendar time cannot be fabricated retrospectively)**

---

## 1. Objective

Before any quantitative strategy or machine learning meta-model can be considered eligible for live capital allocation, it must prove its stability in an autonomous, forward paper-trading environment.

Backtests prove **historical hypothesis viability**. Forward paper trading proves **operational resilience, execution realism, and absence of forward leakage**.

---

## 2. Hard Eligibility Gates for Live Review

Live capital review CANNOT be requested unless the forward paper journal strictly satisfies all six mandatory criteria:

| Gate | Criterion | Minimum Requirement | Audit Mechanism |
| :---: | :--- | :---: | :--- |
| **G-01** | **Forward Duration** | $\ge 30$ consecutive active trading days | Verified via timestamped daily rollups in `daily_rollups.json` |
| **G-02** | **Trade Sample Size** | $\ge 50$ completed round-trip trades | Count of executed orders in `paper_orders_ledger.jsonl` |
| **G-03** | **Statutory Net Profit Factor** | $\ge 1.25$ strictly AFTER statutory fees | Gross wins / Gross losses including STT, GST, SEBI, Stamp Duty |
| **G-04** | **Execution Tracking Error** | $\le 15.0\%$ divergence | $|\\text{Predicted Fill} - \\text{Simulated Fill}| / \\text{Price}$ |
| **G-05** | **Volatility Calibration** | Within $\pm 25\%$ of predicted $\\sigma$ | Ex-ante predicted volatility vs ex-post realized daily volatility |
| **G-06** | **Zero Risk Bypasses** | $0$ breaches | Strict invariant verification across all 13 immutable risk gates |

---

## 3. Operational Invariants During Forward Trial

1. **Frozen Code & Genome**: The champion strategy genome (`StrategyDNA`) and model weights cannot be modified, tuned, or re-optimized during the 30-day forward evaluation window.
2. **Dynamic Universe Ingestion**: Candidate symbols must be discovered and screened daily using [`trading/universe/discovery.py`](file:///c:/GRAG/AI/ai-trading/trading/universe/discovery.py) without hardcoded watchlists.
3. **Execution Delay Enforcement**: Every trade decision generated at market close (bar $t$) must be executed strictly at the subsequent market open (bar $t+1$) with simulated slippage.
4. **Immutable Ledger**: Entries in [`trading/paper/journal_data/paper_orders_ledger.jsonl`](file:///c:/GRAG/AI/ai-trading/trading/paper/journal_data/) are append-only. Any retroactive alteration invalidates the entire trial.

---

## 4. Current Platform Assessment

Because the forward calendar window must occur in genuine forward time and cannot be artificially accelerated or fabricated, the platform status is certified as:

# **FORWARD VALIDATION = PENDING**

- **Completed Research Layers**: Code-verified, unit-tested, behaviorally stress-tested, statistically validated across 2021–2026 historical data.
- **Pending Forward Layer**: Accumulation of 30 daily forward sessions in the forward journal.
- **Production Safety Mandate**: Live broker execution remains permanently isolated until this protocol is formally certified.
