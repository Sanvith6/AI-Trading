"""
Robust Signal-Aware Mean-CVaR Portfolio Optimizer.

Implements the exact-cardinality robust convex optimizer over decision universe U = E union C
(where E is the set of existing holdings and C is the Top-5 candidate set):
- Subsets S subseteq U with |S| <= min(max_positions, |U|) (<= 3 positions post-trade)
- Maximum 93 subsets evaluated for |U| <= 8
- Objective (minimized as negative net utility):
    maximize: w^T mu_robust
              - lambda_variance * (w^T Sigma_shrink w)
              - gamma_cvar * CVaR_alpha(w)
              - cost_penalty * transaction_cost(w)
              - turnover_penalty * turnover(w)
- Smooth Rockafellar-Uryasev CVaR formulation for stable C^inf convex optimization
- Fractional-Kelly capital ceilings as preliminary asset constraints
- Strict institutional boundaries:
    1. Maximum 3 simultaneous post-trade positions (|K| <= 3)
    2. Maximum 95% gross exposure
    3. Minimum $1,000 cash reserve floor (accounting for current holdings and proceeds)
    4. Maximum 30% single-sector weight
    5. Maximum $5,000 per new order notional cap
    6. Named MAX_OPTIMIZER_POSITION_WEIGHT (default 0.25)
- Independent recalculation and constraint reconciliation gate
- Full constraint-binding diagnostics and counterfactual objective transparency
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
import logging

from trading.portfolio.covariance import ShrinkageCovarianceEstimator
from trading.portfolio.risk_models import SignalRiskModel
from trading.portfolio.kelly import FractionalKellyCeiling
from trading.portfolio.cvar import CVaRModel
from trading.portfolio.transaction_costs import TransactionCostModel

logger = logging.getLogger(__name__)

# Named configuration constant for maximum single-position weight
# Active in smaller accounts (< $20,000) or position accumulation; redundant for single orders when equity >= $20,000
MAX_OPTIMIZER_POSITION_WEIGHT: float = 0.25


@dataclass
class OptimizerResult:
    weights: Dict[str, float]
    selected_subset: List[str]
    gross_exposure: float
    cash_remaining: float
    portfolio_variance: float
    portfolio_volatility: float
    portfolio_cvar: float
    turnover: float
    transaction_cost: float
    objective_value: float
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RobustMeanCVaROptimizer:
    """Exact-cardinality robust Signal-Aware Mean-CVaR portfolio optimizer."""

    def __init__(
        self,
        lambda_variance: float = 0.25,
        gamma_cvar: float = 0.02,
        cvar_confidence: float = 0.95,
        uncertainty_penalty: float = 0.15,
        fractional_kelly_multiplier: float = 0.25,
        turnover_penalty_multiplier: float = 0.002,
        cost_penalty_multiplier: float = 1.0,
        max_positions: int = 3,
        max_total_exposure: float = 0.95,
        min_cash_reserve: float = 1000.0,
        max_sector_weight: float = 0.30,
        max_portfolio_risk_bucket_weight: Optional[float] = None,
        max_order_value: float = 5000.0,
        max_position_weight: float = MAX_OPTIMIZER_POSITION_WEIGHT,
        expected_return_scale: float = 0.15,
        min_volatility: float = 0.02,
        max_volatility: float = 2.00,
        default_cross_correlation: float = 0.25,
    ):
        self.lambda_variance = lambda_variance
        self.gamma_cvar = gamma_cvar
        self.cvar_confidence = cvar_confidence
        self.uncertainty_penalty = uncertainty_penalty
        self.fractional_kelly_multiplier = fractional_kelly_multiplier
        self.turnover_penalty_multiplier = turnover_penalty_multiplier
        self.cost_penalty_multiplier = cost_penalty_multiplier
        self.max_positions = max_positions
        self.max_total_exposure = max_total_exposure
        self.min_cash_reserve = min_cash_reserve
        sec_cap = max_portfolio_risk_bucket_weight if max_portfolio_risk_bucket_weight is not None else max_sector_weight
        self.max_sector_weight = sec_cap
        self.max_portfolio_risk_bucket_weight = sec_cap
        self.portfolio_risk_bucket_limit = sec_cap
        self.max_order_value = max_order_value
        self.max_position_weight = max_position_weight
        self.default_cross_correlation = default_cross_correlation

        # Sub-components
        self.cov_estimator = ShrinkageCovarianceEstimator()
        self.risk_model = SignalRiskModel(
            uncertainty_penalty=uncertainty_penalty,
            expected_return_scale=expected_return_scale,
            min_volatility=min_volatility,
            max_volatility=max_volatility,
        )
        self.kelly_calculator = FractionalKellyCeiling(
            fractional_multiplier=fractional_kelly_multiplier,
            max_position_weight=max_position_weight,
        )
        self.cvar_model = CVaRModel(confidence_level=cvar_confidence)
        self.cost_model = TransactionCostModel(
            turnover_penalty_multiplier=turnover_penalty_multiplier,
            cost_penalty_multiplier=cost_penalty_multiplier,
        )

    def optimize(
        self,
        candidates: List[str],
        scores: Optional[Dict[str, float]] = None,
        confidences: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        sectors: Optional[Dict[str, str]] = None,
        portfolio_risk_buckets: Optional[Dict[str, str]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        total_equity: float = 100000.0,
        current_cash: Optional[float] = None,
        current_prices: Optional[Dict[str, float]] = None,
        regime: str = "SIDEWAYS",
        custom_cov_matrix: Optional[np.ndarray] = None,
    ) -> OptimizerResult:
        """
        Solves the exact-cardinality robust portfolio optimization problem.
        
        Evaluates all subsets S subseteq U with |S| <= max_positions, where
        U = existing holdings union Top-5 candidates (|U| <= 8).
        Guarantees final post-trade portfolio has at most max_positions (3) active holdings.
        """
        # 1. Input Sanitization & Empty/Trivial Guards
        top_candidates = [s.upper() for s in (candidates or [])[:5]]
        current_weights = {s.upper(): float(w) for s, w in (current_weights or {}).items() if w > 0.0001}
        existing_syms = list(current_weights.keys())

        # Decision Universe: union of active existing holdings and top candidates
        decision_universe = list(dict.fromkeys(existing_syms + top_candidates))
        if not decision_universe:
            return self._empty_result(total_equity, current_cash, reason="EMPTY_UNIVERSE")

        scores = {s.upper(): float(v) for s, v in (scores or {}).items()}
        confidences = {s.upper(): float(v) for s, v in (confidences or {}).items()}
        volatilities = {s.upper(): float(v) for s, v in (volatilities or {}).items()}
        bucket_map = portfolio_risk_buckets if portfolio_risk_buckets is not None else sectors
        sectors = {s.upper(): str(sec) for s, sec in (bucket_map or {}).items()}
        cash = float(current_cash if current_cash is not None else total_equity)

        # Baseline scores for existing holdings not in candidates
        for s in existing_syms:
            if s not in scores:
                scores[s] = 0.50  # Neutral baseline
            if s not in confidences:
                confidences[s] = 0.70
            if s not in sectors:
                sectors[s] = "UNKNOWN"

        for s in top_candidates:
            if s not in sectors:
                sectors[s] = "UNKNOWN"

        # Guard: Cash below reserve floor AND no existing positions to liquidate
        if cash <= self.min_cash_reserve and not existing_syms:
            logger.warning(
                "Cash $%.2f is at or below minimum reserve floor $%.2f with 0 positions. Rejecting new allocations.",
                cash, self.min_cash_reserve
            )
            return self._empty_result(total_equity, cash, reason="INSUFFICIENT_CASH_RESERVE")

        # 2. Volatility Validation & Estimation across decision universe
        valid_vols, valid_syms = self.risk_model.validate_and_estimate_volatilities(
            decision_universe, volatilities=volatilities, returns_df=returns_df
        )
        if not valid_syms:
            logger.warning("No symbols in decision universe have valid volatility. Yielding 0 allocation.")
            return self._empty_result(total_equity, cash, reason="INVALID_VOLATILITIES")

        # 3. Forecast Extraction & Uncertainty Haircut
        mu_raw = self.risk_model.extract_forecast(valid_syms, scores=scores, confidences=confidences)
        uncertainty = self.risk_model.estimate_uncertainty(
            valid_syms, valid_vols, confidences=confidences, returns_df=returns_df
        )
        mu_robust = self.risk_model.compute_robust_mu(
            valid_syms, mu_raw, uncertainty, uncertainty_penalty=self.uncertainty_penalty
        )

        # 4. Shrinkage Covariance Estimation
        try:
            if custom_cov_matrix is not None:
                cov_matrix = custom_cov_matrix
                shrinkage_intensity = 0.0
            elif returns_df is not None and len([s for s in valid_syms if s in returns_df.columns]) >= 2:
                cov_matrix, shrinkage_intensity = self.cov_estimator.estimate(returns_df, symbols=valid_syms)
            else:
                cov_matrix = self.cov_estimator.build_from_volatilities(
                    valid_vols, valid_syms, cross_correlation=self.default_cross_correlation
                )
                shrinkage_intensity = 1.0
        except Exception as e:
            logger.error("Covariance estimation failed: %s. Safe halting to zero new orders.", e)
            return self._empty_result(total_equity, cash, reason="COVARIANCE_ESTIMATION_FAILED")

        # 5. Fractional Kelly Preliminary Ceilings
        kelly_ceilings = self.kelly_calculator.compute_ceilings(
            symbols=valid_syms,
            robust_mu=mu_robust,
            volatilities=valid_vols,
            total_equity=total_equity,
            max_order_value=self.max_order_value,
            multiplier_override=self.fractional_kelly_multiplier,
        )

        # 6. Extract Return Scenarios for CVaR
        scenarios = self.cvar_model.extract_scenarios(
            symbols=valid_syms,
            returns_df=returns_df,
            cov_matrix=cov_matrix,
            volatilities=valid_vols,
        )

        # 7. Exact Subset Enumeration over sizes [0, 1, ..., min(max_positions, len(valid_syms))]
        # For |U| <= 8 and max 3 positions, at most C(8,0)+C(8,1)+C(8,2)+C(8,3) = 1+8+28+56 = 93 subsets.
        all_subsets = []
        max_cardinality = min(self.max_positions, len(valid_syms))
        for k in range(max_cardinality + 1):
            for subset in itertools.combinations(valid_syms, k):
                all_subsets.append(list(subset))

        best_subset: List[str] = []
        best_weights: Dict[str, float] = {s: 0.0 for s in valid_syms}
        best_objective: float = -1e9
        best_metrics: Dict[str, float] = {}

        # Baseline: Retain current positions as-is (if current positions count <= max_positions)
        # Check current portfolio objective
        curr_active_valid = [s for s in existing_syms if s in valid_syms]
        if len(curr_active_valid) <= self.max_positions:
            w_curr_baseline = np.array([current_weights.get(s, 0.0) for s in valid_syms])
            var_curr = float(w_curr_baseline @ cov_matrix @ w_curr_baseline)
            _, cvar_curr = self.cvar_model.calculate_cvar(w_curr_baseline, scenarios, self.cvar_confidence)
            ret_curr = float(np.dot(w_curr_baseline, [mu_robust.get(s, 0.0) for s in valid_syms]))
            # Turnover/cost to stay in current portfolio is 0
            base_obj = ret_curr - (self.lambda_variance * var_curr) - (self.gamma_cvar * cvar_curr)
            best_objective = base_obj
            best_subset = list(curr_active_valid)
            best_weights = {s: current_weights.get(s, 0.0) for s in valid_syms}
            best_metrics = {
                "gross_exposure": float(np.sum(w_curr_baseline)),
                "variance": var_curr,
                "volatility": float(np.sqrt(max(var_curr, 0.0))),
                "cvar": cvar_curr,
                "turnover": 0.0,
                "cost": 0.0,
                "objective": base_obj,
            }
        else:
            # Over-cardinalized current state: zero-allocation fallback baseline
            best_objective = -1e6

        # 8. Solve continuous convex sub-problem for each candidate subset
        max_order_weight = self.max_order_value / max(total_equity, 1000.0)
        sum_curr_w = sum(current_weights.get(s, 0.0) for s in valid_syms)
        max_cash_weight = sum_curr_w + max(0.0, (cash - self.min_cash_reserve) / max(total_equity, 1000.0))
        exposure_limit = min(self.max_total_exposure, max_cash_weight)

        for subset in all_subsets:
            k_len = len(subset)
            if k_len == 0:
                # 0-position portfolio (liquidate all existing positions)
                zero_w = {s: 0.0 for s in valid_syms}
                c_drag, t_fric, total_pen = self.cost_model.compute_cost_drag(zero_w, current_weights)
                zero_obj = 0.0 - total_pen
                if zero_obj > best_objective:
                    best_objective = zero_obj
                    best_subset = []
                    best_weights = zero_w
                    best_metrics = {
                        "gross_exposure": 0.0,
                        "variance": 0.0,
                        "volatility": 0.0,
                        "cvar": 0.0,
                        "turnover": t_fric / max(self.turnover_penalty_multiplier, 1e-6),
                        "cost": c_drag,
                        "objective": zero_obj,
                    }
                continue

            sub_indices = [valid_syms.index(s) for s in subset]
            sub_mu = np.array([mu_robust[s] for s in subset])
            sub_cov = cov_matrix[np.ix_(sub_indices, sub_indices)]
            sub_scenarios = scenarios[:, sub_indices] if scenarios.size > 0 else np.empty((0, k_len))
            sub_curr_w = np.array([current_weights.get(s, 0.0) for s in subset])
            sub_ceilings = [kelly_ceilings.get(s, self.max_position_weight) for s in subset]

            # Bounds per asset in subset
            # New asset: w_i in [0, min(kelly, max_pos, max_order)]
            # Existing asset: w_i in [0, min(kelly, max_pos, w_curr + max_order)]
            bounds = []
            skip_subset = False
            for i, s in enumerate(subset):
                w_c = sub_curr_w[i]
                k_ceil = sub_ceilings[i]
                if w_c <= 1e-4:
                    # New position: requires positive robust edge
                    if k_ceil <= 1e-5:
                        skip_subset = True
                        break
                    u_bound = min(k_ceil, self.max_position_weight, max_order_weight)
                else:
                    # Existing position: can be held or increased by up to max_order_weight
                    u_bound = min(k_ceil, self.max_position_weight, w_c + max_order_weight)
                bounds.append((0.0, max(0.0, u_bound)))

            if skip_subset:
                continue

            # Linear Constraints:
            # 1. Total gross exposure <= exposure_limit
            cons = [
                {"type": "ineq", "fun": lambda w, lim=exposure_limit: lim - np.sum(w)},
            ]

            # 2. Sector Exposure Cap (max 30% per sector)
            unique_sectors = set(sectors.get(s, "UNKNOWN") for s in subset)
            for sec in unique_sectors:
                if sec == "UNKNOWN":
                    continue
                sec_mask = np.array([1.0 if sectors.get(s, "UNKNOWN") == sec else 0.0 for s in subset])
                if np.sum(sec_mask) > 0:
                    cons.append({
                        "type": "ineq",
                        "fun": lambda w, mask=sec_mask, cap=self.max_sector_weight: cap - np.sum(w * mask)
                    })

            # Smooth Rockafellar-Uryasev Loss Function for SLSQP
            def loss_function(w: np.ndarray) -> float:
                ret_term = float(np.dot(w, sub_mu))
                var_term = float(w @ sub_cov @ w)
                cvar_val = self.cvar_model.rockafellar_uryasev_objective(
                    w, sub_scenarios, alpha=self.cvar_confidence
                )
                cost_penalty = self.cost_model.evaluate_vectorized(w, sub_curr_w)
                net_utility = (
                    ret_term
                    - (self.lambda_variance * var_term)
                    - (self.gamma_cvar * cvar_val)
                    - cost_penalty
                )
                return -net_utility

            # Initial starting guess: clamped to bounds
            w0 = np.array([min(bounds[i][1], sub_curr_w[i] if sub_curr_w[i] > 0 else (exposure_limit / k_len) * 0.5) for i in range(k_len)])
            if np.sum(w0) > exposure_limit:
                w0 = w0 * (exposure_limit / max(np.sum(w0), 1e-6))

            try:
                res = minimize(
                    loss_function,
                    w0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"ftol": 1e-7, "maxiter": 150},
                )
                if not res.success or np.isnan(res.fun):
                    continue

                w_opt = np.maximum(0.0, res.x)
                cand_obj = -float(res.fun)

                if cand_obj > best_objective:
                    full_w_cand = {s: 0.0 for s in valid_syms}
                    for i, s in enumerate(subset):
                        full_w_cand[s] = float(w_opt[i])

                    w_vec = np.array([full_w_cand[s] for s in valid_syms])
                    port_var = float(w_vec @ cov_matrix @ w_vec)
                    port_vol = float(np.sqrt(max(port_var, 0.0)))
                    _, port_cvar = self.cvar_model.calculate_cvar(w_vec, scenarios, self.cvar_confidence)
                    c_drag, t_friction, _ = self.cost_model.compute_cost_drag(full_w_cand, current_weights)

                    best_objective = cand_obj
                    best_subset = list(subset)
                    best_weights = full_w_cand
                    best_metrics = {
                        "gross_exposure": float(np.sum(w_vec)),
                        "variance": port_var,
                        "volatility": port_vol,
                        "cvar": port_cvar,
                        "turnover": t_friction / max(self.turnover_penalty_multiplier, 1e-6),
                        "cost": c_drag,
                        "objective": cand_obj,
                    }
            except Exception as opt_err:
                logger.debug("Subset %s optimization error: %s", subset, opt_err)
                continue

        # 9. Clean insignificant weights & Rounding
        cleaned_weights: Dict[str, float] = {}
        for s in valid_syms:
            w = best_weights.get(s, 0.0)
            cleaned_weights[s] = round(w, 4) if w >= 0.005 else 0.0

        final_subset = [s for s, w in cleaned_weights.items() if w > 0.0]

        # 10. Independent Numerical Reconciliation & Feasibility Layer
        reconciled_ok, reconcile_reason = self._reconcile_and_verify(
            weights=cleaned_weights,
            kelly_ceilings=kelly_ceilings,
            sectors=sectors,
            total_equity=total_equity,
            cash=cash,
            current_weights=current_weights,
        )

        if not reconciled_ok:
            logger.error("INDEPENDENT RECONCILIATION FAILED: %s. Safely rejecting portfolio to 0 new orders.", reconcile_reason)
            return self._empty_result(total_equity, cash, reason=f"RECONCILIATION_FAILED: {reconcile_reason}")

        # Compute post-trade remaining cash
        net_capital_delta = sum((cleaned_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in valid_syms) * total_equity
        cost_dollars = best_metrics.get("cost", 0.0) * total_equity
        cash_remaining = round(max(0.0, cash - net_capital_delta - cost_dollars), 2)

        # 11. Constraint Binding Analysis & Objective Transparency
        w_final_vec = np.array([cleaned_weights.get(s, 0.0) for s in valid_syms])
        tot_exposure = float(np.sum(w_final_vec))

        # Check constraint bindings
        active_pos_count = len(final_subset)
        cardinality_bound = (active_pos_count == self.max_positions)
        exposure_bound = (tot_exposure >= self.max_total_exposure - 1e-3)
        cash_bound = (tot_exposure >= max_cash_weight - 1e-3 and max_cash_weight < self.max_total_exposure)
        sector_bound = False
        for sec in set(sectors.get(s, "UNKNOWN") for s in final_subset):
            sec_w = sum(cleaned_weights.get(s, 0.0) for s in final_subset if sectors.get(s, "UNKNOWN") == sec)
            if sec != "UNKNOWN" and sec_w >= self.max_sector_weight - 1e-3:
                sector_bound = True
                break

        order_cap_bound = any(
            (cleaned_weights.get(s, 0.0) - current_weights.get(s, 0.0)) * total_equity >= self.max_order_value - 1.0
            for s in valid_syms
        )
        position_cap_bound = any(cleaned_weights.get(s, 0.0) >= self.max_position_weight - 1e-3 for s in valid_syms)
        kelly_bound = any(
            cleaned_weights.get(s, 0.0) >= kelly_ceilings.get(s, 0.25) - 1e-3
            and kelly_ceilings.get(s, 0.25) < min(self.max_position_weight, max_order_weight)
            for s in valid_syms
        )
        is_no_trade = (
            set(final_subset) == set(existing_syms)
            and all(abs(cleaned_weights.get(s, 0.0) - current_weights.get(s, 0.0)) < 0.005 for s in valid_syms)
        )

        # Objective component breakdown
        ret_term = float(np.dot(w_final_vec, [mu_robust.get(s, 0.0) for s in valid_syms]))
        var_term = float(w_final_vec @ cov_matrix @ w_final_vec)
        cvar_term = best_metrics.get("cvar", 0.0)
        cost_term = float(best_metrics.get("cost", 0.0) * self.cost_penalty_multiplier)
        turnover_term = float(best_metrics.get("turnover", 0.0) * self.turnover_penalty_multiplier)

        # Counterfactual objective evaluations
        obj_val = best_objective
        obj_without_cvar = obj_val + (self.gamma_cvar * cvar_term)
        obj_without_cost = obj_val + cost_term
        obj_without_turnover = obj_val + turnover_term
        raw_ret_term = float(np.dot(w_final_vec, [mu_raw.get(s, 0.0) for s in valid_syms]))
        obj_without_uncertainty = (raw_ret_term - (self.lambda_variance * var_term) - (self.gamma_cvar * cvar_term) - cost_term - turnover_term)

        diagnostics = {
            "subsets_evaluated": len(all_subsets),
            "decision_universe": decision_universe,
            "existing_positions": existing_syms,
            "mu_raw": mu_raw,
            "uncertainty": uncertainty,
            "forecast_uncertainty_proxy": uncertainty,
            "mu_robust": mu_robust,
            "volatilities": valid_vols,
            "kelly_ceilings": kelly_ceilings,
            "fractional_kelly_risk_caps": kelly_ceilings,
            "reconciliation": "PASSED",
            "regime": regime,
            "binding": {
                "cardinality": cardinality_bound,
                "exposure": exposure_bound,
                "cash": cash_bound,
                "sector": sector_bound,
                "portfolio_risk_bucket": sector_bound,
                "order_cap": order_cap_bound,
                "position_cap": position_cap_bound,
                "kelly": kelly_bound,
                "no_trade": is_no_trade,
            },
            "objective_breakdown": {
                "robust_return_term": round(ret_term, 6),
                "variance_penalty_term": round(self.lambda_variance * var_term, 6),
                "cvar_penalty_term": round(self.gamma_cvar * cvar_term, 6),
                "cost_penalty_term": round(cost_term, 6),
                "turnover_penalty_term": round(turnover_term, 6),
                "objective_value": round(obj_val, 6),
                "objective_without_cvar": round(obj_without_cvar, 6),
                "objective_without_cost": round(obj_without_cost, 6),
                "objective_without_turnover": round(obj_without_turnover, 6),
                "objective_without_uncertainty": round(obj_without_uncertainty, 6),
            },
        }

        return OptimizerResult(
            weights=cleaned_weights,
            selected_subset=final_subset,
            gross_exposure=round(tot_exposure, 4),
            cash_remaining=cash_remaining,
            portfolio_variance=round(best_metrics.get("variance", 0.0), 6),
            portfolio_volatility=round(best_metrics.get("volatility", 0.0), 4),
            portfolio_cvar=round(best_metrics.get("cvar", 0.0), 4),
            turnover=round(best_metrics.get("turnover", 0.0), 4),
            transaction_cost=round(best_metrics.get("cost", 0.0), 4),
            objective_value=round(best_objective, 6),
            status="OPTIMAL" if final_subset else ("NO_TRADE" if is_no_trade else "ZERO_ALLOCATION"),
            diagnostics=diagnostics,
        )

    def _reconcile_and_verify(
        self,
        weights: Dict[str, float],
        kelly_ceilings: Dict[str, float],
        sectors: Dict[str, str],
        total_equity: float,
        cash: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """Independent verification of all portfolio constraints outside the optimizer."""
        current_weights = current_weights or {}

        # 1. Non-negative weights
        for s, w in weights.items():
            if w < -1e-6:
                return False, f"Negative weight {w} for {s}"

        # 2. Maximum positions <= max_positions (3) on FINAL portfolio
        active_count = sum(1 for w in weights.values() if w > 0.001)
        if active_count > self.max_positions:
            return False, f"Final position count {active_count} exceeds maximum {self.max_positions}"

        # 3. Maximum total gross exposure <= 95%
        tot_exp = sum(weights.values())
        if tot_exp > self.max_total_exposure + 1e-4:
            return False, f"Total exposure {tot_exp:.4f} exceeds ceiling {self.max_total_exposure}"

        # 4. Post-trade cash floor reserve ($1,000)
        net_spent = sum((weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in set(weights).union(current_weights)) * total_equity
        post_cash = cash - net_spent
        if post_cash < (self.min_cash_reserve - 1.0):
            return False, f"Post-trade cash ${post_cash:.2f} breaches reserve floor ${self.min_cash_reserve}"

        # 5. Sector cap <= 30%
        sec_totals: Dict[str, float] = {}
        for s, w in weights.items():
            sec = sectors.get(s, "UNKNOWN")
            sec_totals[sec] = sec_totals.get(sec, 0.0) + w

        for sec, tot_w in sec_totals.items():
            if tot_w > self.max_sector_weight + 1e-4 and sec != "UNKNOWN":
                return False, f"Sector {sec} weight {tot_w:.4f} exceeds sector limit {self.max_sector_weight}"

        # 6. Kelly ceilings, Position Caps, & Max Order Value
        for s, w in weights.items():
            k_ceil = kelly_ceilings.get(s, self.max_position_weight)
            if w > (k_ceil + 1e-4):
                return False, f"Asset {s} weight {w:.4f} exceeds Kelly ceiling {k_ceil:.4f}"

            if w > (self.max_position_weight + 1e-4):
                return False, f"Asset {s} weight {w:.4f} exceeds position cap {self.max_position_weight}"

            w_curr = current_weights.get(s, 0.0)
            if w > w_curr:
                incremental_notional = (w - w_curr) * total_equity
                if incremental_notional > (self.max_order_value + 1.0):
                    return False, f"Asset {s} order notional ${incremental_notional:.2f} exceeds per-order cap ${self.max_order_value}"

        return True, "ALL_CONSTRAINTS_SATISFIED"

    def _empty_result(self, total_equity: float, cash: Optional[float], reason: str) -> OptimizerResult:
        """Returns safe zero-allocation result."""
        c = cash if cash is not None else total_equity
        return OptimizerResult(
            weights={},
            selected_subset=[],
            gross_exposure=0.0,
            cash_remaining=round(c, 2),
            portfolio_variance=0.0,
            portfolio_volatility=0.0,
            portfolio_cvar=0.0,
            turnover=0.0,
            transaction_cost=0.0,
            objective_value=0.0,
            status=f"ZERO_ALLOCATION_{reason}",
            diagnostics={"reason": reason, "reconciliation": "SKIPPED"},
        )
