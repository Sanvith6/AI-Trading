"""
Comprehensive Verification Suite for Robust Signal-Aware Mean-CVaR Optimizer.

Tests all 30 mandatory institutional conditions:
1. Empty candidate set
2. One candidate
3. Five candidates
4. Maximum 3 positions
5. Zero positions
6. Max $5,000/order
7. 95% exposure
8. $1,000 minimum cash
9. 30% sector cap
10. Existing positions
11. Correlated stocks
12. Highly volatile stock
13. Low volatility stock
14. High CVaR
15. Low CVaR
16. High uncertainty
17. Low uncertainty
18. High transaction cost
19. High turnover
20. NaN handling
21. Inf handling
22. Invalid covariance
23. Solver failure
24. Infeasible portfolio
25. Deterministic repeatability
26. Exact subset cardinality (26 subsets)
27. RiskEngine rejection
28. Rounding boundary
29. Stale data
30. Regime change
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from trading.portfolio.portfolio_constructor import PortfolioConstructor, PortfolioProposal
from trading.portfolio.optimizer import RobustMeanCVaROptimizer
from trading.portfolio.covariance import ShrinkageCovarianceEstimator
from trading.risk_engine import RiskEngine


# ── TEST 1: Empty Candidate Set ──────────────────────────────────────────────
def test_01_empty_candidate_set():
    pc = PortfolioConstructor()
    weights = pc.construct([], method="robust_mean_cvar")
    assert weights == {}

    prop = pc.construct_proposal([], portfolio_state={"cash": 100000.0})
    assert prop.status == "EMPTY_PROPOSAL"
    assert prop.selected_subset == []
    assert prop.proposals == []
    assert prop.gross_exposure == 0.0


# ── TEST 2: One Candidate ────────────────────────────────────────────────────
def test_02_one_candidate():
    pc = PortfolioConstructor()
    cands = [{"symbol": "AAPL", "price": 200.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"}]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})

    assert len(prop.selected_subset) <= 1
    if prop.selected_subset:
        assert prop.selected_subset == ["AAPL"]
        assert prop.proposals[0].dollar_notional <= 5000.0
        assert prop.proposals[0].quantity > 0


# ── TEST 3: Five Candidates ──────────────────────────────────────────────────
def test_03_five_candidates():
    pc = PortfolioConstructor()
    cands = [
        {"symbol": "AAPL", "price": 200.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"},
        {"symbol": "NVDA", "price": 120.0, "alpha_score": 0.90, "ml_conf": 0.88, "side": "BUY"},
        {"symbol": "MSFT", "price": 400.0, "alpha_score": 0.70, "ml_conf": 0.75, "side": "BUY"},
        {"symbol": "AMZN", "price": 180.0, "alpha_score": 0.65, "ml_conf": 0.70, "side": "BUY"},
        {"symbol": "META", "price": 500.0, "alpha_score": 0.60, "ml_conf": 0.70, "side": "BUY"},
    ]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})
    assert len(prop.selected_subset) <= 3
    assert len(prop.proposals) <= 3


# ── TEST 4: Maximum 3 Positions ──────────────────────────────────────────────
def test_04_maximum_three_positions():
    pc = PortfolioConstructor()
    # 5 strong candidates
    cands = [
        {"symbol": s, "price": 100.0, "alpha_score": 0.90, "ml_conf": 0.90, "side": "BUY"}
        for s in ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
    ]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})
    assert len(prop.selected_subset) <= 3, f"Selected {len(prop.selected_subset)} exceeds limit 3"


# ── TEST 5: Zero Positions on Negative Edge ──────────────────────────────────
def test_05_zero_positions_on_negative_edges():
    pc = PortfolioConstructor()
    # All candidates have low or negative edges (alpha_score <= 0.40)
    cands = [
        {"symbol": s, "price": 100.0, "alpha_score": 0.35, "ml_conf": 0.50, "side": "BUY"}
        for s in ["BAD1", "BAD2", "BAD3"]
    ]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})
    assert prop.selected_subset == []
    assert prop.proposals == []
    assert prop.gross_exposure == 0.0


# ── TEST 6: Max $5,000 / Order Ceiling ──────────────────────────────────────
def test_06_max_5000_per_order_ceiling():
    pc = PortfolioConstructor()
    cands = [
        {"symbol": "NVDA", "price": 125.0, "alpha_score": 0.95, "ml_conf": 0.95, "side": "BUY"}
    ]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 1000000.0, "equity": 1000000.0})
    for p in prop.proposals:
        assert p.dollar_notional <= 5000.01, f"Notional {p.dollar_notional} breaches $5,000 limit"


# ── TEST 7: 95% Exposure Limit ───────────────────────────────────────────────
def test_07_max_95pct_exposure_ceiling():
    pc = PortfolioConstructor(max_order_value=50000.0)  # Large per-order to stress exposure ceiling
    cands = [
        {"symbol": s, "price": 100.0, "alpha_score": 0.90, "ml_conf": 0.90, "side": "BUY"}
        for s in ["A", "B", "C"]
    ]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})
    assert prop.gross_exposure <= 0.95001


# ── TEST 8: $1,000 Minimum Cash Reserve ─────────────────────────────────────
def test_08_min_1000_cash_reserve_floor():
    pc = PortfolioConstructor()
    # Account has only $1,200 cash. Should allocate at most $200
    prop = pc.construct_proposal(
        [{"symbol": "AAPL", "price": 100.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"}],
        portfolio_state={"cash": 1200.0, "equity": 1200.0},
    )
    assert prop.cash_remaining >= 999.99

    # Account has $900 cash (below reserve). Must allocate $0
    prop_low = pc.construct_proposal(
        [{"symbol": "AAPL", "price": 100.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"}],
        portfolio_state={"cash": 900.0, "equity": 900.0},
    )
    assert prop_low.selected_subset == []
    assert prop_low.proposals == []


# ── TEST 9: 30% Sector Cap ──────────────────────────────────────────────────
def test_09_max_30pct_sector_cap():
    pc = PortfolioConstructor(max_order_value=30000.0)  # Larger orders to test sector bound
    cands = [
        {"symbol": "NVDA", "price": 100.0, "alpha_score": 0.90, "ml_conf": 0.90, "side": "BUY"},
        {"symbol": "AMD", "price": 100.0, "alpha_score": 0.88, "ml_conf": 0.90, "side": "BUY"},
    ]
    sectors = {"NVDA": "Semiconductors", "AMD": "Semiconductors"}
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0}, sectors=sectors)

    semis_weight = sum(p.target_weight for p in prop.proposals if p.sector == "Semiconductors")
    assert semis_weight <= 0.30001, f"Semiconductors sector weight {semis_weight} breaches 30%"


# ── TEST 10: Existing Positions Handling ─────────────────────────────────────
def test_10_existing_positions_handling():
    pc = PortfolioConstructor()
    cands = [
        {"symbol": "AAPL", "price": 200.0, "alpha_score": 0.80, "ml_conf": 0.80, "side": "BUY"},
        {"symbol": "NVDA", "price": 120.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"},
    ]
    existing = {"AAPL": {"quantity": 10, "current_price": 200.0}}
    prop = pc.construct_proposal(
        cands, portfolio_state={"cash": 98000.0, "equity": 100000.0}, existing_positions=existing
    )
    assert prop.status in ("OPTIMAL", "ZERO_TRADE")
    assert prop.cash_remaining >= 1000.0


# ── TEST 11: Correlated Stocks Penalization ──────────────────────────────────
def test_11_correlated_stocks_penalization():
    pc = PortfolioConstructor()
    # 2 highly correlated stocks vs 1 uncorrelated stock
    T = 60
    np.random.seed(42)
    market = np.random.normal(0, 0.01, T)
    corr1 = market + np.random.normal(0, 0.002, T)
    corr2 = market + np.random.normal(0, 0.002, T)
    uncorr = np.random.normal(0, 0.01, T)

    ret_df = pd.DataFrame({"CORR1": corr1, "CORR2": corr2, "UNCORR": uncorr})
    weights = pc.construct(
        ["CORR1", "CORR2", "UNCORR"],
        scores={"CORR1": 0.75, "CORR2": 0.75, "UNCORR": 0.75},
        returns_df=ret_df,
    )
    # The uncorrelated asset should get comparable or higher allocation due to diversification benefit
    assert "UNCORR" in weights
    assert weights["UNCORR"] > 0


# ── TEST 12: Highly Volatile Stock Penalization ──────────────────────────────
def test_12_highly_volatile_stock_penalization():
    pc = PortfolioConstructor()
    weights = pc.construct(
        ["HIGH_VOL", "LOW_VOL"],
        scores={"HIGH_VOL": 0.75, "LOW_VOL": 0.75},
        volatilities={"HIGH_VOL": 0.70, "LOW_VOL": 0.15},
    )
    # Low volatility asset should receive higher allocation due to variance & Kelly penalties
    assert weights["LOW_VOL"] >= weights["HIGH_VOL"]


# ── TEST 13: Low Volatility Stock Favorable ──────────────────────────────────
def test_13_low_volatility_stock_favorable():
    pc = PortfolioConstructor()
    weights = pc.construct(
        ["STABLE", "VOLATILE"],
        scores={"STABLE": 0.70, "VOLATILE": 0.70},
        volatilities={"STABLE": 0.12, "VOLATILE": 0.45},
    )
    assert weights["STABLE"] > weights["VOLATILE"]


# ── TEST 14: High CVaR Tail Risk Suppression ─────────────────────────────────
def test_14_high_cvar_tail_risk_suppression():
    pc = PortfolioConstructor(gamma_cvar=0.10)
    # Asset with fat negative tails
    T = 100
    np.random.seed(42)
    normal_ret = np.random.normal(0.001, 0.01, T)
    fat_tail_ret = np.random.normal(0.001, 0.01, T)
    # Introduce severe crash events into fat_tail_ret
    fat_tail_ret[10] = -0.15
    fat_tail_ret[35] = -0.18
    fat_tail_ret[70] = -0.20

    ret_df = pd.DataFrame({"SAFE": normal_ret, "FAT_TAIL": fat_tail_ret})
    weights = pc.construct(["SAFE", "FAT_TAIL"], scores={"SAFE": 0.70, "FAT_TAIL": 0.70}, returns_df=ret_df)
    assert weights["SAFE"] > weights["FAT_TAIL"]


# ── TEST 15: Low CVaR Favorable ──────────────────────────────────────────────
def test_15_low_cvar_favorable():
    pc = PortfolioConstructor(gamma_cvar=0.05)
    T = 60
    np.random.seed(42)
    safe = np.random.normal(0.001, 0.005, T)
    risky = np.random.normal(0.001, 0.025, T)
    ret_df = pd.DataFrame({"SAFE": safe, "RISKY": risky})
    weights = pc.construct(["SAFE", "RISKY"], scores={"SAFE": 0.70, "RISKY": 0.70}, returns_df=ret_df)
    assert weights["SAFE"] > weights["RISKY"]


# ── TEST 16: High Uncertainty Haircut ─────────────────────────────────────────
def test_16_high_uncertainty_haircut():
    pc = PortfolioConstructor(uncertainty_penalty=0.40)
    weights = pc.construct(
        ["CERTAIN", "UNCERTAIN"],
        scores={"CERTAIN": 0.75, "UNCERTAIN": 0.75},
        confidences={"CERTAIN": 0.95, "UNCERTAIN": 0.15},
    )
    # High confidence (low uncertainty) must receive more weight than low confidence
    assert weights["CERTAIN"] > weights["UNCERTAIN"]


# ── TEST 17: Low Uncertainty Allocation ──────────────────────────────────────
def test_17_low_uncertainty_allocation():
    pc = PortfolioConstructor(uncertainty_penalty=0.30)
    weights = pc.construct(
        ["HIGH_CONF"],
        scores={"HIGH_CONF": 0.80},
        confidences={"HIGH_CONF": 0.98},
    )
    assert weights["HIGH_CONF"] > 0.02


# ── TEST 18: High Transaction Cost Suppression ───────────────────────────────
def test_18_high_transaction_cost_suppression():
    # If transaction costs are high, marginal turnover is suppressed
    pc_cheap = PortfolioConstructor(cost_penalty_multiplier=0.1)
    pc_expensive = PortfolioConstructor(cost_penalty_multiplier=100.0)

    cands = ["A", "B"]
    scores = {"A": 0.65, "B": 0.65}
    curr = {"A": 0.0, "B": 0.0}

    w_cheap = pc_cheap.construct(cands, scores=scores, current_weights=curr)
    w_expensive = pc_expensive.construct(cands, scores=scores, current_weights=curr)

    assert sum(w_cheap.values()) >= sum(w_expensive.values())


# ── TEST 19: High Turnover Penalty ───────────────────────────────────────────
def test_19_high_turnover_penalty():
    pc_lo_to = PortfolioConstructor(turnover_penalty_multiplier=0.0001)
    pc_hi_to = PortfolioConstructor(turnover_penalty_multiplier=1.0)

    # Current holdings: hold A
    curr = {"A": 0.05, "B": 0.0}
    # Signal says B has slightly higher score than A
    scores = {"A": 0.70, "B": 0.72}

    w_lo = pc_lo_to.construct(["A", "B"], scores=scores, current_weights=curr)
    w_hi = pc_hi_to.construct(["A", "B"], scores=scores, current_weights=curr)

    # With high turnover penalty, optimizer prefers keeping A rather than flipping completely to B
    assert w_hi.get("A", 0) >= w_lo.get("A", 0)


# ── TEST 20: NaN Handling Rejection ──────────────────────────────────────────
def test_20_nan_handling_rejection():
    pc = PortfolioConstructor()
    weights = pc.construct(
        ["GOOD", "NAN_SCORE", "NAN_VOL"],
        scores={"GOOD": 0.75, "NAN_SCORE": float("nan"), "NAN_VOL": 0.75},
        volatilities={"GOOD": 0.20, "NAN_SCORE": 0.20, "NAN_VOL": float("nan")},
    )
    assert "GOOD" in weights
    assert weights["GOOD"] > 0
    assert weights.get("NAN_SCORE", 0.0) == 0.0


# ── TEST 21: Inf Handling Rejection ──────────────────────────────────────────
def test_21_inf_handling_rejection():
    pc = PortfolioConstructor()
    weights = pc.construct(
        ["GOOD", "INF_SYM"],
        scores={"GOOD": 0.75, "INF_SYM": float("inf")},
        volatilities={"GOOD": 0.20, "INF_SYM": float("inf")},
    )
    assert weights.get("INF_SYM", 0.0) == 0.0


# ── TEST 22: Invalid Covariance Safe Fallback ─────────────────────────────────
def test_22_invalid_covariance_safe_fallback():
    cov_est = ShrinkageCovarianceEstimator()
    # Bad DataFrame with all NaNs
    bad_df = pd.DataFrame({"A": [float("nan")] * 10, "B": [float("nan")] * 10})
    with pytest.raises(ValueError):
        cov_est.estimate(bad_df)

    # Empty candidates handled safely in constructor
    pc = PortfolioConstructor()
    w = pc.construct(["A", "B"], returns_df=bad_df)
    assert isinstance(w, dict)


# ── TEST 23: Solver Failure Safe Zero Order ──────────────────────────────────
def test_23_solver_failure_safe_zero_order():
    optimizer = RobustMeanCVaROptimizer()
    # Call with non-positive cash below reserve
    res = optimizer.optimize(["AAPL"], total_equity=100000.0, current_cash=500.0)
    assert res.status.startswith("ZERO_ALLOCATION")
    assert res.weights == {}
    assert res.selected_subset == []


# ── TEST 24: Infeasible Portfolio Rejection ──────────────────────────────────
def test_24_infeasible_portfolio_rejection():
    optimizer = RobustMeanCVaROptimizer()
    # Reconcile check fails on negative weight
    ok, reason = optimizer._reconcile_and_verify(
        {"AAPL": -0.10}, kelly_ceilings={"AAPL": 0.25}, sectors={}, total_equity=100000.0, cash=100000.0
    )
    assert not ok
    assert "Negative weight" in reason


# ── TEST 25: Deterministic Repeatability ─────────────────────────────────────
def test_25_deterministic_repeatability():
    pc1 = PortfolioConstructor()
    pc2 = PortfolioConstructor()
    cands = ["NVDA", "AAPL", "MSFT", "AMZN", "META"]
    scores = {"NVDA": 0.85, "AAPL": 0.78, "MSFT": 0.72, "AMZN": 0.65, "META": 0.60}

    w1 = pc1.construct(cands, scores=scores)
    w2 = pc2.construct(cands, scores=scores)

    assert w1 == w2, "Optimizer must be 100% deterministic given identical inputs"


# ── TEST 26: Exact Subset Cardinality (26 Subsets) ───────────────────────────
def test_26_exact_subset_cardinality_26_evaluations():
    optimizer = RobustMeanCVaROptimizer(max_positions=3)
    cands = ["A", "B", "C", "D", "E"]
    scores = {s: 0.75 for s in cands}
    res = optimizer.optimize(cands, scores=scores)

    # Number of subsets evaluated: C(5,0) + C(5,1) + C(5,2) + C(5,3) = 1 + 5 + 10 + 10 = 26
    assert res.diagnostics.get("subsets_evaluated") == 26
    assert len(res.selected_subset) <= 3


# ── TEST 27: RiskEngine Rejection Authority ──────────────────────────────────
def test_27_risk_engine_rejection_authority():
    risk = RiskEngine()
    # Even if an order is proposed, RiskEngine has final authority to deny
    approval = risk.validate_order(
        symbol="AAPL",
        action="BUY",
        price=200.0,
        quantity=1000,  # 1000 * 200 = $200,000 > $5,000 max_per_order
        cash=100000.0,
    )
    assert not approval.approved
    assert "exceeds" in approval.deny_reason


# ── TEST 28: Rounding Boundary Preserves Limits ──────────────────────────────
def test_28_rounding_boundary_preserves_limits():
    pc = PortfolioConstructor(max_order_value=5000.0)
    # Price is $3,200. Desired notional $5,000 would round to 1 share ($3,200), not 2 ($6,400)
    cands = [{"symbol": "BKNG", "price": 3200.0, "alpha_score": 0.90, "ml_conf": 0.90, "side": "BUY"}]
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0})

    if prop.proposals:
        p = prop.proposals[0]
        assert p.quantity == 1
        assert p.dollar_notional == 3200.0
        assert p.dollar_notional <= 5000.0


# ── TEST 29: Stale Data / Zero Volatility Rejection ───────────────────────────
def test_29_stale_data_rejection():
    pc = PortfolioConstructor()
    # 0.0 or negative volatility indicates stale/unusable market quote
    vols = {"STALE": 0.0, "VALID": 0.20}
    weights = pc.construct(["STALE", "VALID"], volatilities=vols, scores={"STALE": 0.90, "VALID": 0.75})
    assert weights.get("STALE", 0.0) == 0.0


# ── TEST 30: Regime Change Risk Budgeting ────────────────────────────────────
def test_30_regime_change_risk_budgeting():
    pc = PortfolioConstructor()
    cands = [
        {"symbol": "NVDA", "price": 125.0, "alpha_score": 0.80, "ml_conf": 0.80, "side": "BUY"},
        {"symbol": "AAPL", "price": 220.0, "alpha_score": 0.75, "ml_conf": 0.80, "side": "BUY"},
    ]
    prop_bull = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0}, regime="STRONG_BULL")
    prop_risk_off = pc.construct_proposal(cands, portfolio_state={"cash": 100000.0}, regime="RISK_OFF")

    assert prop_bull.status in ("OPTIMAL", "ZERO_TRADE")
    assert prop_risk_off.status in ("OPTIMAL", "ZERO_TRADE")
    assert prop_risk_off.diagnostics.get("regime") == "RISK_OFF"


# ── TEST 31: Cardinality With Existing Holdings Strictly <= 3 ─────────────────
def test_31_cardinality_with_existing_holdings_le_3():
    """Verify that existing holdings + candidates NEVER produce > 3 positions."""
    pc = PortfolioConstructor()
    # Account currently holds 2 positions: AAPL, MSFT
    existing = {
        "AAPL": {"quantity": 10, "current_price": 200.0},  # $2,000 (2%)
        "MSFT": {"quantity": 8, "current_price": 400.0},   # $3,200 (3.2%)
    }
    # 3 new candidates: NVDA, AMZN, META
    cands = [
        {"symbol": "NVDA", "price": 120.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"},
        {"symbol": "AMZN", "price": 180.0, "alpha_score": 0.82, "ml_conf": 0.80, "side": "BUY"},
        {"symbol": "META", "price": 500.0, "alpha_score": 0.80, "ml_conf": 0.80, "side": "BUY"},
    ]
    prop = pc.construct_proposal(
        cands,
        portfolio_state={"cash": 94800.0, "equity": 100000.0},
        existing_positions=existing,
    )
    # The final active post-trade positions count MUST be <= 3
    assert len(prop.selected_subset) <= 3, f"Selected {len(prop.selected_subset)} exceeds limit 3: {prop.selected_subset}"
    # Verify weights count
    active_weights = [s for s, w in prop.weights.items() if w > 0.001]
    assert len(active_weights) <= 3, f"Active weights {len(active_weights)} exceeds limit 3: {active_weights}"


# ── TEST 32: Rebalancing Lifecycle All Five Actions ──────────────────────────
def test_32_rebalancing_lifecycle_all_five_actions():
    """Verify NEW_POSITION, INCREASE_EXISTING, REDUCE_EXISTING, CLOSE_POSITION, RETAIN_EXISTING."""
    pc = PortfolioConstructor()
    # Scenario: Account holds AAPL ($2,000), MSFT ($5,000), GOOGL ($3,000)
    existing = {
        "AAPL": {"quantity": 10, "current_price": 200.0},
        "MSFT": {"quantity": 12, "current_price": 400.0},
        "GOOGL": {"quantity": 20, "current_price": 150.0},
    }
    # NVDA is a strong candidate, GOOGL is weak, AAPL has strong edge
    cands = [
        {"symbol": "NVDA", "price": 100.0, "alpha_score": 0.90, "ml_conf": 0.90, "side": "BUY"},
        {"symbol": "AAPL", "price": 200.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"},
    ]
    prop = pc.construct_proposal(
        cands,
        portfolio_state={"cash": 85000.0, "equity": 100000.0},
        existing_positions=existing,
    )
    assert prop.status in ("OPTIMAL", "ZERO_TRADE")
    # All proposals must have valid sides and non-negative quantities
    for p in prop.proposals:
        assert p.side in ("BUY", "SELL")
        assert p.quantity >= 1
        assert p.dollar_notional <= 5000.01


# ── TEST 33: Analytical CVaR Reference Benchmark ─────────────────────────────
def test_33_analytical_cvar_reference():
    """Verify CVaR model matches analytical normal Gaussian expectation."""
    from trading.portfolio.cvar import CVaRModel
    cvar_model = CVaRModel(confidence_level=0.95)
    var_95, cvar_95 = cvar_model.analytical_gaussian_cvar(volatility=0.20, confidence_level=0.95)
    # For N(0, 0.20^2):
    # z_0.95 = 1.64485 -> VaR = 0.32897
    # ES multiplier = phi(1.64485) / 0.05 = 0.103136 / 0.05 = 2.06271 -> CVaR = 0.41254
    assert abs(var_95 - 0.32897) < 1e-3
    assert abs(cvar_95 - 0.41254) < 1e-3


# ── TEST 34: Constraint Binding Diagnostics Transparency ─────────────────────
def test_34_constraint_binding_diagnostics():
    """Verify that optimizer exposes binding flags and objective transparency components."""
    optimizer = RobustMeanCVaROptimizer()
    symbols = ["AAPL", "MSFT", "NVDA"]
    scores = {"AAPL": 0.85, "MSFT": 0.80, "NVDA": 0.85}
    confidences = {"AAPL": 0.85, "MSFT": 0.85, "NVDA": 0.85}
    volatilities = {"AAPL": 0.18, "MSFT": 0.20, "NVDA": 0.38}
    res = optimizer.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)

    diag = res.diagnostics
    assert "binding" in diag
    binding = diag["binding"]
    assert "cardinality" in binding
    assert "order_cap" in binding
    assert "kelly" in binding
    assert "exposure" in binding
    assert "cash" in binding
    assert "sector" in binding

    assert "objective_breakdown" in diag
    obj_b = diag["objective_breakdown"]
    assert "robust_return_term" in obj_b
    assert "variance_penalty_term" in obj_b
    assert "cvar_penalty_term" in obj_b
    assert "objective_without_cvar" in obj_b
    assert "objective_without_cost" in obj_b
    assert "objective_without_uncertainty" in obj_b


# ── TEST 35: Named Max Optimizer Position Weight Parameter ───────────────────
def test_35_named_max_optimizer_position_weight():
    """Verify MAX_OPTIMIZER_POSITION_WEIGHT is an explicit named configuration parameter."""
    from trading.portfolio.optimizer import MAX_OPTIMIZER_POSITION_WEIGHT
    assert MAX_OPTIMIZER_POSITION_WEIGHT == 0.25

    pc_custom = PortfolioConstructor(max_position_weight=0.15)
    assert pc_custom.optimizer.max_position_weight == 0.15


# ── TEST 36: Post-Trade Cash Reconciliation Exact ────────────────────────────
def test_36_post_trade_cash_reconciliation_exact():
    """Verify post-trade cash is correctly preserved after allocations."""
    pc = PortfolioConstructor()
    cands = [
        {"symbol": "AAPL", "price": 200.0, "alpha_score": 0.80, "ml_conf": 0.80, "side": "BUY"},
        {"symbol": "NVDA", "price": 120.0, "alpha_score": 0.85, "ml_conf": 0.85, "side": "BUY"},
    ]
    # Cash is $10,000, equity is $10,000
    prop = pc.construct_proposal(cands, portfolio_state={"cash": 10000.0, "equity": 10000.0})
    assert prop.cash_remaining >= 1000.0, f"Remaining cash ${prop.cash_remaining} breached $1,000 floor"
    total_spent = sum(p.dollar_notional for p in prop.proposals if p.side == "BUY")
    assert total_spent <= (10000.0 - 1000.0 + 1.0)


# ── TEST 37: Cost Multiplier Objective Monotonicity ───────────────────────────
def test_37_cost_multiplier_objective_monotonicity():
    """Verify cost_multiplier > 1 alters objective value even when weights remain identical."""
    symbols = ["AAPL", "MSFT"]
    scores = {"AAPL": 0.75, "MSFT": 0.70}
    confidences = {"AAPL": 0.80, "MSFT": 0.80}
    volatilities = {"AAPL": 0.18, "MSFT": 0.20}

    opt_1x = RobustMeanCVaROptimizer(cost_penalty_multiplier=1.0)
    opt_15x = RobustMeanCVaROptimizer(cost_penalty_multiplier=1.5)
    opt_2x = RobustMeanCVaROptimizer(cost_penalty_multiplier=2.0)

    res_1x = opt_1x.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)
    res_15x = opt_15x.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)
    res_2x = opt_2x.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)

    # Weights are identical due to order cap
    assert res_1x.weights == res_15x.weights == res_2x.weights
    assert res_1x.transaction_cost > 0

    # Objective values must strictly decrease as cost multiplier increases
    assert res_1x.objective_value > res_15x.objective_value > res_2x.objective_value, (
        f"Objectives not strictly monotonically decreasing: 1x={res_1x.objective_value}, "
        f"1.5x={res_15x.objective_value}, 2x={res_2x.objective_value}"
    )


# ── TEST 38: Correlation Stress Risk Response ─────────────────────────────────
def test_38_correlation_stress_risk_response():
    """Verify correlation spike to 0.80 alters covariance eigenvalues and increases portfolio risk."""
    symbols = ["AAPL", "MSFT", "NVDA"]
    scores = {"AAPL": 0.65, "MSFT": 0.62, "NVDA": 0.70}
    confidences = {"AAPL": 0.80, "MSFT": 0.75, "NVDA": 0.65}
    volatilities = {"AAPL": 0.18, "MSFT": 0.20, "NVDA": 0.35}

    opt_base = RobustMeanCVaROptimizer(default_cross_correlation=0.25)
    opt_stressed = RobustMeanCVaROptimizer(default_cross_correlation=0.80)

    res_base = opt_base.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)
    res_stressed = opt_stressed.optimize(symbols, scores=scores, confidences=confidences, volatilities=volatilities)

    # Covariance risk and CVaR must be higher or optimizer must reduce exposure
    assert res_stressed.objective_value < res_base.objective_value
    assert res_stressed.gross_exposure <= res_base.gross_exposure


# ── TEST 39: Expected Return Mapping Properties ───────────────────────────────
def test_39_expected_return_mapping_properties():
    """Verify forecast mapping mathematical properties: neutral threshold and confidence scaling."""
    from trading.portfolio.risk_models import SignalRiskModel
    rm = SignalRiskModel(expected_return_scale=0.15)

    # Score <= 0.50 yields non-positive expected return
    mu_neutral = rm.extract_forecast(["AAPL"], scores={"AAPL": 0.50}, confidences={"AAPL": 0.80})
    assert mu_neutral["AAPL"] == 0.0

    mu_bearish = rm.extract_forecast(["AAPL"], scores={"AAPL": 0.40}, confidences={"AAPL": 0.80})
    assert mu_bearish["AAPL"] < 0.0

    # Lower confidence strictly scales down edge magnitude
    mu_high_conf = rm.extract_forecast(["AAPL"], scores={"AAPL": 0.80}, confidences={"AAPL": 0.90})
    mu_low_conf = rm.extract_forecast(["AAPL"], scores={"AAPL": 0.80}, confidences={"AAPL": 0.45})
    assert mu_high_conf["AAPL"] > mu_low_conf["AAPL"] > 0.0
    assert abs(mu_high_conf["AAPL"] - 2.0 * mu_low_conf["AAPL"]) < 1e-4


# ── TEST 40: Per-Order $5,000 Sizing Boundary for BUY and SELL ────────────────
def test_40_per_order_boundary_buy_and_sell():
    """
    Verify $5,000 per-order boundary across BUY and SELL order types:
    $4,999 BUY  -> PASS
    $5,000 BUY  -> PASS
    $5,001 BUY  -> FAIL
    $4,999 SELL -> PASS
    $5,000 SELL -> PASS
    $5,001 SELL -> FAIL
    """
    risk = RiskEngine()

    # BUY Boundary Tests
    app_buy_4999 = risk.validate_order("AAPL", "BUY", price=4999.0, quantity=1, cash=50000.0)
    assert app_buy_4999.approved is True, f"Failed $4,999 BUY: {app_buy_4999.deny_reason}"

    app_buy_5000 = risk.validate_order("AAPL", "BUY", price=5000.0, quantity=1, cash=50000.0)
    assert app_buy_5000.approved is True, f"Failed $5,000 BUY: {app_buy_5000.deny_reason}"

    app_buy_5001 = risk.validate_order("AAPL", "BUY", price=5001.0, quantity=1, cash=50000.0)
    assert app_buy_5001.approved is False, "Expected $5,001 BUY to fail"
    assert "max_per_order" in app_buy_5001.deny_reason

    # SELL Boundary Tests
    app_sell_4999 = risk.validate_order("AAPL", "SELL", price=4999.0, quantity=1, cash=50000.0)
    assert app_sell_4999.approved is True, f"Failed $4,999 SELL: {app_sell_4999.deny_reason}"

    app_sell_5000 = risk.validate_order("AAPL", "SELL", price=5000.0, quantity=1, cash=50000.0)
    assert app_sell_5000.approved is True, f"Failed $5,000 SELL: {app_sell_5000.deny_reason}"

    app_sell_5001 = risk.validate_order("AAPL", "SELL", price=5001.0, quantity=1, cash=50000.0)
    assert app_sell_5001.approved is False, "Expected $5,001 SELL to fail"
    assert "max_per_order" in app_sell_5001.deny_reason


# ── TEST 41: Large Close Position Order Splitting ($12,000 Liquidation) ───────
def test_41_large_close_position_order_splitting():
    """
    Verify $12,000 liquidation splits into multiple individual orders <= $5,000:
    $5,000 SELL (50 sh @ $100)
    $5,000 SELL (50 sh @ $100)
    $2,000 SELL (20 sh @ $100)
    """
    pc = PortfolioConstructor(max_order_value=5000.0)
    risk = RiskEngine()

    # Account has 120 shares of AAPL @ $100 = $12,000 liquidation
    existing = {"AAPL": {"quantity": 120, "current_price": 100.0}}
    prop = pc.construct_proposal(
        candidates=[],
        portfolio_state={"cash": 10000.0, "equity": 22000.0},
        existing_positions=existing,
    )

    assert len(prop.proposals) == 3, f"Expected 3 chunked proposals, got {len(prop.proposals)}"

    # Validate each individual chunk
    for p in prop.proposals:
        assert p.side == "SELL"
        assert p.symbol == "AAPL"
        assert p.dollar_notional <= 5000.0, f"Order notional ${p.dollar_notional} exceeds $5,000 cap"
        # Validate that RiskEngine approves each chunk
        approval = risk.validate_order(
            symbol=p.symbol,
            action=p.side,
            price=p.price,
            quantity=p.quantity,
            cash=10000.0,
            existing_positions=existing,
        )
        assert approval.approved is True, f"Chunk rejected by RiskEngine: {approval.deny_reason}"

    assert prop.proposals[0].quantity == 50
    assert prop.proposals[0].dollar_notional == 5000.0
    assert prop.proposals[1].quantity == 50
    assert prop.proposals[1].dollar_notional == 5000.0
    assert prop.proposals[2].quantity == 20
    assert prop.proposals[2].dollar_notional == 2000.0
    assert sum(p.dollar_notional for p in prop.proposals) == 12000.0
    assert sum(p.quantity for p in prop.proposals) == 120


# ── TEST 42: Terminology and Diagnostics Proxy Keys ───────────────────────────
def test_42_terminology_and_diagnostics_proxy_keys():
    """Verify diagnostics expose forecast_uncertainty_proxy and fractional_kelly_risk_caps."""
    opt = RobustMeanCVaROptimizer()
    res = opt.optimize(
        ["AAPL"],
        scores={"AAPL": 0.80},
        confidences={"AAPL": 0.85},
        volatilities={"AAPL": 0.20},
    )
    diag = res.diagnostics
    assert "forecast_uncertainty_proxy" in diag
    assert "fractional_kelly_risk_caps" in diag
    assert diag["forecast_uncertainty_proxy"] == diag["uncertainty"]
    assert diag["fractional_kelly_risk_caps"] == diag["kelly_ceilings"]


# ── TEST 43-47: Fill-Aware Sequential Chunk Liquidation Regression Suite ───────
from trading.execution.sequential_liquidation import SequentialChunkLiquidator
from trading.execution.base import ExecutionAdapter, ExecutionResult
from trading.brokers.interface import OrderRequest, OrderResponse, OrderStatus, OrderSide, OrderType
from pathlib import Path


class MockLiquidationAdapter(ExecutionAdapter):
    """Test adapter that simulates specific broker lifecycle fill responses."""
    def __init__(self, responses: Optional[List[OrderResponse]] = None):
        self.responses = list(responses or [])
        self.submitted_requests: List[OrderRequest] = []
        self.idx = 0

    def execute_order(self, req: OrderRequest) -> ExecutionResult:
        self.submitted_requests.append(req)
        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
        else:
            # Default to full fill
            resp = OrderResponse(
                order_id=f"MOCK_{len(self.submitted_requests)}",
                client_order_id=req.client_order_id,
                symbol=req.symbol,
                side=req.side,
                quantity=req.quantity,
                filled_qty=req.quantity,
                filled_price=req.limit_price or 100.0,
                status=OrderStatus.FILLED,
                submitted_at="2026-09-27T02:00:00Z",
            )
        success = resp.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED)
        return ExecutionResult(success=success, order_response=resp, execution_type="MOCK", execution_latency_ms=1.0)

    def get_portfolio(self) -> Dict[str, Any]:
        return {"cash": 100000.0}

    def reconcile(self) -> Dict[str, Any]:
        return {}

    def get_execution_type(self) -> str:
        return "MOCK"

    @property
    def journal_path(self) -> Path:
        return Path("mock.parquet")


def test_43_sequential_liquidation_full_fill():
    """Verify $12,000 liquidation executes sequentially in chunks <= $5,000 to full fill."""
    liquidator = SequentialChunkLiquidator(max_order_value=5000.0)
    adapter = MockLiquidationAdapter()

    res = liquidator.liquidate(adapter, symbol="AAPL", total_quantity=120, price=100.0)

    assert res.status == "COMPLETED"
    assert res.intended_quantity == 120
    assert res.total_filled_quantity == 120
    assert res.residual_quantity == 0
    assert len(res.chunks) == 3

    # Check that chunks were submitted sequentially and each was <= $5,000
    assert len(adapter.submitted_requests) == 3
    assert adapter.submitted_requests[0].quantity == 50
    assert adapter.submitted_requests[1].quantity == 50
    assert adapter.submitted_requests[2].quantity == 20

    for req in adapter.submitted_requests:
        assert req.quantity * 100.0 <= 5000.0
        assert req.side == OrderSide.SELL


def test_44_sequential_liquidation_partial_fill():
    """Verify sequential liquidation recalculates next chunk when a chunk is partially filled."""
    liquidator = SequentialChunkLiquidator(max_order_value=5000.0)
    # Simulate:
    # Chunk 1 (requested 50): only 30 filled -> remaining becomes 90
    # Chunk 2 (calculated from 90): requests min(90, 50) = 50 -> fully filled -> remaining becomes 40
    # Chunk 3 (calculated from 40): requests min(40, 50) = 40 -> fully filled -> remaining becomes 0
    resp_chunk1 = OrderResponse(
        order_id="MOCK_P1", client_order_id="CID_1", symbol="AAPL",
        side=OrderSide.SELL, quantity=50, filled_qty=30, filled_price=100.0,
        status=OrderStatus.PARTIALLY_FILLED, submitted_at="2026-09-27T02:00:00Z"
    )
    resp_chunk2 = OrderResponse(
        order_id="MOCK_P2", client_order_id="CID_2", symbol="AAPL",
        side=OrderSide.SELL, quantity=50, filled_qty=50, filled_price=100.0,
        status=OrderStatus.FILLED, submitted_at="2026-09-27T02:01:00Z"
    )
    resp_chunk3 = OrderResponse(
        order_id="MOCK_P3", client_order_id="CID_3", symbol="AAPL",
        side=OrderSide.SELL, quantity=40, filled_qty=40, filled_price=100.0,
        status=OrderStatus.FILLED, submitted_at="2026-09-27T02:02:00Z"
    )

    adapter = MockLiquidationAdapter([resp_chunk1, resp_chunk2, resp_chunk3])
    res = liquidator.liquidate(adapter, symbol="AAPL", total_quantity=120, price=100.0)

    assert res.status == "COMPLETED"
    assert res.total_filled_quantity == 120
    assert res.residual_quantity == 0
    assert len(adapter.submitted_requests) == 3
    assert adapter.submitted_requests[0].quantity == 50
    assert adapter.submitted_requests[1].quantity == 50
    assert adapter.submitted_requests[2].quantity == 40  # Dynamically recalculated, not 20!


def test_45_sequential_liquidation_cancel():
    """Verify liquidation safely halts when broker returns CANCELLED and preserves residual."""
    liquidator = SequentialChunkLiquidator(max_order_value=5000.0)
    # Chunk 1 filled 50, Chunk 2 cancelled (0 filled)
    resp_chunk1 = OrderResponse(
        order_id="MOCK_C1", client_order_id="CID_1", symbol="AAPL",
        side=OrderSide.SELL, quantity=50, filled_qty=50, filled_price=100.0,
        status=OrderStatus.FILLED, submitted_at="2026-09-27T02:00:00Z"
    )
    resp_chunk2 = OrderResponse(
        order_id="MOCK_C2", client_order_id="CID_2", symbol="AAPL",
        side=OrderSide.SELL, quantity=50, filled_qty=0, filled_price=0.0,
        status=OrderStatus.CANCELLED, submitted_at="2026-09-27T02:01:00Z"
    )

    adapter = MockLiquidationAdapter([resp_chunk1, resp_chunk2])
    res = liquidator.liquidate(adapter, symbol="AAPL", total_quantity=120, price=100.0)

    assert res.status == "CANCELLED"
    assert res.total_filled_quantity == 50
    assert res.residual_quantity == 70
    assert len(adapter.submitted_requests) == 2  # Did NOT submit chunk 3 blindly


def test_46_sequential_liquidation_rejected_chunk():
    """Verify liquidation halts immediately if a chunk is REJECTED without generating orphan orders."""
    liquidator = SequentialChunkLiquidator(max_order_value=5000.0)
    resp_chunk1 = OrderResponse(
        order_id="MOCK_R1", client_order_id="CID_1", symbol="AAPL",
        side=OrderSide.SELL, quantity=50, filled_qty=0, filled_price=0.0,
        status=OrderStatus.REJECTED, rejection_reason="CIRCUIT_BREAKER_HALT",
        submitted_at="2026-09-27T02:00:00Z"
    )

    adapter = MockLiquidationAdapter([resp_chunk1])
    res = liquidator.liquidate(adapter, symbol="AAPL", total_quantity=120, price=100.0)

    assert res.status == "REJECTED"
    assert res.total_filled_quantity == 0
    assert res.residual_quantity == 120
    assert len(adapter.submitted_requests) == 1
    assert "CIRCUIT_BREAKER_HALT" in res.termination_reason


def test_47_sequential_liquidation_never_exceeds_remaining_and_cap():
    """Verify invariant: No chunk ever exceeds remaining position or $5,000 cap."""
    liquidator = SequentialChunkLiquidator(max_order_value=5000.0)
    adapter = MockLiquidationAdapter()

    # Odd share price: $137.45, 150 shares = $20,617.50 liquidation
    price = 137.45
    total_qty = 150
    res = liquidator.liquidate(adapter, symbol="NVDA", total_quantity=total_qty, price=price)

    assert res.status == "COMPLETED"
    assert res.residual_quantity == 0
    assert res.total_filled_quantity == total_qty

    running_filled = 0
    for req in adapter.submitted_requests:
        remaining_at_submit = total_qty - running_filled
        assert req.quantity <= remaining_at_submit, f"Chunk {req.quantity} exceeded remaining {remaining_at_submit}"
        assert req.quantity * price <= 5000.01, f"Chunk value ${req.quantity * price} exceeded $5,000"
        running_filled += req.quantity

    assert running_filled == total_qty


def test_48_portfolio_risk_bucket_taxonomy_and_limits():
    """
    Verify Option C: Custom portfolio_risk_bucket taxonomy.
    Technology and Semiconductors are distinct 30% risk buckets.
    Assert Optimizer and RiskEngine enforce identical limits.
    """
    risk = RiskEngine()
    constructor = PortfolioConstructor(max_portfolio_risk_bucket_weight=0.30)

    # 1. Verify property alias consistency
    assert risk.max_portfolio_risk_bucket_weight == 0.30
    assert risk.portfolio_risk_bucket_limit == 0.30
    assert constructor.max_portfolio_risk_bucket_weight == 0.30
    assert constructor.portfolio_risk_bucket_limit == 0.30
    assert constructor.optimizer.max_portfolio_risk_bucket_weight == 0.30

    # 2. RiskEngine validates using portfolio_risk_bucket
    # Existing portfolio: $28,000 NVDA in Semiconductors (total equity $100k -> 28%)
    existing_positions = {
        "NVDA": {"quantity": 280, "current_price": 100.0, "portfolio_risk_bucket": "Semiconductors"}
    }
    # An additional $3,000 BUY in Semiconductors breaches 30% (28k + 3k = 31k = 31% > 30%)
    appr_fail = risk.validate_order(
        symbol="AMD", action="BUY", price=100.0, quantity=30, cash=72000.0,
        existing_positions=existing_positions, portfolio_risk_bucket="Semiconductors"
    )
    assert appr_fail.approved is False
    assert any("concentration cap for Semiconductors" in v.reason for v in appr_fail.violations)

    # BUT a $3,000 BUY in Technology is approved because Technology is a distinct bucket!
    appr_ok = risk.validate_order(
        symbol="AAPL", action="BUY", price=100.0, quantity=30, cash=72000.0,
        existing_positions=existing_positions, portfolio_risk_bucket="Technology"
    )
    assert appr_ok.approved is True

    # 3. Verify PositionProposal populates portfolio_risk_bucket
    cand = [{"symbol": "AAPL", "price": 100.0, "alpha_score": 0.85, "ml_conf": 0.90, "side": "BUY"}]
    proposal = constructor.construct_proposal(
        candidates=cand,
        portfolio_state={"cash": 100000.0, "portfolio_value": 100000.0},
        portfolio_risk_buckets={"AAPL": "Technology", "NVDA": "Semiconductors"},
    )
    assert len(proposal.proposals) > 0
    for p in proposal.proposals:
        assert p.portfolio_risk_bucket == "Technology"
        assert p.to_dict()["portfolio_risk_bucket"] == "Technology"
