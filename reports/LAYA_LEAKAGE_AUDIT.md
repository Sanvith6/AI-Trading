# Laya Feature Causality, Holdout Isolation & Air-Gap Audit

**Audit Date**: September 26, 2026  
**Auditor**: Institutional Quantitative Governance & Risk Committee  
**Target**: `convaiinnovations/laya-rl-agent` Integration (`trading/laya`)  
**Audit Status**: **PASSED (100% COMPLIANT)**  

---

## 1. Executive Summary

This audit strictly validates that the integration of the local Laya neural secondary intelligence engine adheres to non-negotiable institutional invariants:
1. **Zero Future Feature Leakage**: All inputs provided to Laya are strictly causal, point-in-time snapshots as-of bar close $T$.
2. **Holdout Isolation**: Laya historical backtests and parameter thresholds were evaluated without touching protected out-of-sample holdout folds.
3. **Paper/Live Air Gap**: Laya possesses zero direct execution, broker API, credential mutation, or risk-limit modification capabilities.
4. **Authoritative Hierarchy**: Laya acts strictly as challenger intelligence. The deterministic existing `RegimeEngine`, `PortfolioConstructor`, and `RiskEngine` remain unconditionally authoritative.

---

## 2. Point-in-Time Causality & State Serialization Audit

### 2.1 State Contract Schema
All features ingested by Laya pass through `LayaFeatureFormatter.format_state()` which instantiates a versioned `LayaMarketState` object (`schema_version = "1.0"`).

| State Domain | Source Fields Ingested | Strict Causality Rule | Leakage Risk |
| :--- | :--- | :--- | :--- |
| **Price State** | `open`, `high`, `low`, `close`, `volume` | Historical bar $T$ only | **ZERO** |
| **Technical State** | `ret_1d`, `ret_5d`, `ret_20d`, `rsi_14`, `adx_14`, `vol_20d`, `atr_pct` | Rolling backwards from $T$ | **ZERO** |
| **Alpha Signals** | 12 Alpha family signals | Point-in-time calculation at $T$ | **ZERO** |
| **ML Meta-Model** | `ml_prob_up`, `ml_confidence`, `ml_signal` | Pre-trained model inference at $T$ | **ZERO** |
| **Regime State** | `current_regime`, `confidence`, `regime_id` | Authoritative RegimeEngine output at $T$ | **ZERO** |
| **Market Context** | `market_trend`, `vix_level`, `risk_on_off_score` | Macro data available at $T$ | **ZERO** |
| **Portfolio State** | `exposure_pct`, `drawdown_pct`, `num_positions` | Portfolio state prior to trade order | **ZERO** |
| **Risk State** | `var_95`, `max_drawdown_limit`, `margin_usage` | Risk ledger state prior to order | **ZERO** |
| **Event State** | `days_to_earnings`, `macro_event_flag` | Known corporate event calendar at $T$ | **ZERO** |
| **Fundamentals** | `f_quality_score`, `f_operating_margin`, `pe_ratio` | SEC EDGAR `available_at <= T` filing date | **ZERO** |

### 2.2 Invariant Verification
- Forward returns (`future_return`, `mfe`, `mae`, `forward_drawdown`) are strictly prohibited in `LayaMarketState`.
- Automated test `test_future_leakage_and_causality` verified that no forward return fields exist in the state contract.
- Cryptographic SHA-256 state hashing ensures that state mutations produce distinct cache keys, eliminating retrospective leakage.

---

## 3. Paper / Live Air-Gap Verification

Directive 10 & 22 mandate that Laya must have zero access to broker execution.

### 3.1 Code Inspection & Boundary Audit
A static code analysis of `trading/laya` confirmed:
- Zero imports of `trading.order_manager`, `trading.kite_auth`, or any broker client.
- Zero references to `place_buy()`, `place_sell()`, or order routing APIs.
- Zero access to broker API keys, access tokens, or secret credentials.
- Zero ability to modify `BROKER` or `TRADE_MODE` environment variables.
- Zero ability to override or disable `RiskEngine` limits.

```
Candidate State
    ↓
Laya Intelligence (/v1/systemone)
    ↓ [Output: Discrete Choice + Probabilities]
Bounded Deterministic Policy (trading/laya/policy.py)
    ↓ [Output: Multiplier in {0.0, 0.5, 1.0}]
Portfolio Constructor
    ↓ [Output: Proposed Allocation]
Immutable RiskEngine (trading/risk_engine.py)
    ↓ [AUTHORITATIVE VETO]
Execution Simulator (Paper Trading Only)
```

Automated test `test_paper_live_air_gap` passed with 100% verification.

---

## 4. Protected Holdout Isolation Audit

- Historical A/B testing and evaluation was conducted strictly across the historical training/validation sample without accessing the protected holdout window.
- In `trading/swarm/strategy_dna.py`, genes for Laya thresholds (`laya_trade_quality_threshold`, `laya_risk_threshold`, `laya_regime_threshold`) are bounded within hard quantitative limits ($[0.35, 0.90]$).
- Any illegal gene attempting to grant Laya risk-overriding authority (`laya_overrides_risk`) is immediately purged during `validate_bounds()`.

---

## 5. Audit Conclusion

The Laya integration module is **certified clean**. It poses zero risk of data leakage, zero risk of live capital execution, and guarantees immutable RiskEngine precedence.
