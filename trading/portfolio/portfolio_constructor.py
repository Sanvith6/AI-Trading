"""
Unified Portfolio Construction Framework.

Houses the production ROBUST SIGNAL-AWARE MEAN-CVaR OPTIMIZER while preserving
complete interface compatibility with existing risk parity, equal weight, and
factor-weighting benchmarks.

Key Features:
1. Exact cardinality convex optimizer over U = E union C (<= 3 positions post-trade)
2. Shrinkage covariance estimation (Ledoit-Wolf)
3. Forecast extraction (mu_i) with confidence uncertainty haircut
4. Fractional-Kelly preliminary allocation ceiling
5. Rockafellar-Uryasev CVaR tail-loss constraint
6. Realistic transaction cost and turnover friction penalties
7. Strict institutional limits:
   - 95% max exposure
   - $1,000 cash floor
   - 30% sector cap
   - $5,000 max per new order
   - Named MAX_OPTIMIZER_POSITION_WEIGHT (0.25)
8. Comprehensive rebalancing lifecycle:
   - NEW_POSITION (BUY)
   - INCREASE_EXISTING (BUY)
   - REDUCE_EXISTING (SELL)
   - CLOSE_POSITION (SELL)
   - RETAIN_EXISTING (HOLD / no order)
9. Independent reconciliation layer before order proposal generation
10. 100% backward-compatible fallback to legacy methods for research/benchmarks
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from trading.portfolio.optimizer import RobustMeanCVaROptimizer, OptimizerResult, MAX_OPTIMIZER_POSITION_WEIGHT
from trading.portfolio.covariance import ShrinkageCovarianceEstimator
from trading.portfolio.legacy_constructor import LegacyPortfolioConstructor

logger = logging.getLogger(__name__)


@dataclass
class PositionProposal:
    symbol: str
    target_weight: float
    dollar_notional: float
    quantity: int
    side: str
    price: float
    sector: str
    robust_mu: float
    uncertainty: float
    volatility: float
    kelly_ceiling: float
    cvar_contribution: float
    transaction_cost: float
    reason: str
    portfolio_risk_bucket: str = ""

    def __post_init__(self):
        if not self.portfolio_risk_bucket:
            self.portfolio_risk_bucket = self.sector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "dollar_notional": self.dollar_notional,
            "quantity": self.quantity,
            "side": self.side,
            "price": self.price,
            "sector": self.sector,
            "portfolio_risk_bucket": self.portfolio_risk_bucket or self.sector,
            "robust_mu": self.robust_mu,
            "uncertainty": self.uncertainty,
            "volatility": self.volatility,
            "kelly_ceiling": self.kelly_ceiling,
            "cvar_contribution": self.cvar_contribution,
            "transaction_cost": self.transaction_cost,
            "reason": self.reason,
        }


@dataclass
class PortfolioProposal:
    proposals: List[PositionProposal]
    weights: Dict[str, float]
    selected_subset: List[str]
    gross_exposure: float
    net_exposure: float
    cash_remaining: float
    portfolio_variance: float
    portfolio_volatility: float
    portfolio_cvar: float
    turnover: float
    total_cost: float
    objective_value: float
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "weights": self.weights,
            "selected_subset": self.selected_subset,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "cash_remaining": self.cash_remaining,
            "portfolio_variance": self.portfolio_variance,
            "portfolio_volatility": self.portfolio_volatility,
            "portfolio_cvar": self.portfolio_cvar,
            "turnover": self.turnover,
            "total_cost": self.total_cost,
            "objective_value": self.objective_value,
            "status": self.status,
            "diagnostics": self.diagnostics,
        }


class PortfolioConstructor:
    """Institutional portfolio construction engine featuring Robust Signal-Aware Mean-CVaR Optimization."""

    def __init__(
        self,
        max_position_weight: float = MAX_OPTIMIZER_POSITION_WEIGHT,
        max_sector_weight: float = 0.30,
        max_portfolio_risk_bucket_weight: Optional[float] = None,
        max_total_exposure: float = 0.95,
        max_positions: int = 40,
        min_cash_pct: float = 0.05,
        correlation_threshold: float = 0.70,
        max_cluster_weight: float = 0.40,
        # Optimizer specific hyperparameters
        lambda_variance: float = 0.25,
        gamma_cvar: float = 0.02,
        cvar_confidence: float = 0.95,
        uncertainty_penalty: float = 0.15,
        fractional_kelly_multiplier: float = 0.25,
        turnover_penalty_multiplier: float = 0.002,
        cost_penalty_multiplier: float = 1.0,
        max_order_value: float = 5000.0,
        min_cash_reserve: float = 1000.0,
    ):
        self.max_position_weight = max_position_weight
        sec_cap = max_portfolio_risk_bucket_weight if max_portfolio_risk_bucket_weight is not None else max_sector_weight
        self.max_sector_weight = sec_cap
        self.max_portfolio_risk_bucket_weight = sec_cap
        self.portfolio_risk_bucket_limit = sec_cap
        self.max_total_exposure = max_total_exposure
        self.max_positions = max_positions
        self.min_cash_pct = min_cash_pct
        self.correlation_threshold = correlation_threshold
        self.max_cluster_weight = max_cluster_weight
        self.max_order_value = max_order_value
        self.min_cash_reserve = min_cash_reserve

        # Robust Mean-CVaR Optimizer Core (strictly capped at max 3 positions)
        self.optimizer = RobustMeanCVaROptimizer(
            lambda_variance=lambda_variance,
            gamma_cvar=gamma_cvar,
            cvar_confidence=cvar_confidence,
            uncertainty_penalty=uncertainty_penalty,
            fractional_kelly_multiplier=fractional_kelly_multiplier,
            turnover_penalty_multiplier=turnover_penalty_multiplier,
            cost_penalty_multiplier=cost_penalty_multiplier,
            max_positions=min(max_positions, 3),
            max_total_exposure=max_total_exposure,
            min_cash_reserve=min_cash_reserve,
            max_sector_weight=sec_cap,
            max_portfolio_risk_bucket_weight=sec_cap,
            max_order_value=max_order_value,
            max_position_weight=max_position_weight,
        )

        # Legacy fallback engine for benchmarking and regression
        self.legacy_engine = LegacyPortfolioConstructor(
            max_position_weight=max_position_weight,
            max_sector_weight=max_sector_weight,
            max_total_exposure=max_total_exposure,
            max_positions=max_positions,
            min_cash_pct=min_cash_pct,
            correlation_threshold=correlation_threshold,
            max_cluster_weight=max_cluster_weight,
        )

        self.last_proposal: Optional[PortfolioProposal] = None

    def construct(
        self,
        candidates: List[str],
        scores: Optional[Dict[str, float]] = None,
        confidences: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        method: str = "robust_mean_cvar",
        sectors: Optional[Dict[str, str]] = None,
        portfolio_risk_buckets: Optional[Dict[str, str]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        total_equity: float = 100000.0,
        current_cash: Optional[float] = None,
        current_prices: Optional[Dict[str, float]] = None,
        regime: str = "SIDEWAYS",
    ) -> Dict[str, float]:
        """
        Public Portfolio Construction API.

        Supported methods:
        - 'robust_mean_cvar' (Default Production Optimizer)
        - 'risk_parity' (Equal Risk Contribution benchmark)
        - 'equal_weight' (Baseline benchmark)
        - 'score_weight' (Alpha-proportional benchmark)
        - 'confidence_weight' (Confidence-proportional benchmark)
        - 'inverse_vol' (Inverse volatility benchmark)
        - 'vol_target' (Target volatility benchmark)
        """
        if not candidates and not current_weights:
            return {}

        symbols = [s.upper() for s in (candidates or [])]
        bucket_map = portfolio_risk_buckets if portfolio_risk_buckets is not None else sectors

        # 1. Primary Path: Robust Signal-Aware Mean-CVaR Optimizer
        if method == "robust_mean_cvar":
            opt_res = self.optimizer.optimize(
                candidates=symbols,
                scores=scores,
                confidences=confidences,
                volatilities=volatilities,
                returns_df=returns_df,
                sectors=bucket_map,
                portfolio_risk_buckets=bucket_map,
                current_weights=current_weights,
                total_equity=total_equity,
                current_cash=current_cash,
                current_prices=current_prices,
                regime=regime,
            )

            # Store structured proposal
            proposal = self._build_proposal_from_result(
                opt_res,
                total_equity=total_equity,
                cash=current_cash if current_cash is not None else total_equity,
                prices=current_prices or {},
                sectors=sectors or {},
                current_weights=current_weights or {},
                existing_quantities={},
            )
            self.last_proposal = proposal

            # Return symbol -> weight mapping for all symbols requested
            return {s: opt_res.weights.get(s, 0.0) for s in symbols}

        # 2. Legacy / Benchmark Method Dispatch
        return self.legacy_engine.construct(
            candidates=symbols,
            scores=scores,
            confidences=confidences,
            volatilities=volatilities,
            returns_df=returns_df,
            method=method,
            sectors=sectors,
        )

    def construct_proposal(
        self,
        candidates: List[Dict[str, Any]],
        portfolio_state: Dict[str, Any],
        existing_positions: Optional[Dict[str, Any]] = None,
        current_prices: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        sectors: Optional[Dict[str, str]] = None,
        portfolio_risk_buckets: Optional[Dict[str, str]] = None,
        regime: str = "SIDEWAYS",
    ) -> PortfolioProposal:
        """
        High-level runtime proposal generator directly consumed by TradingOrchestrator.

        Args:
            candidates: List of candidate dicts with keys: symbol, price, alpha_score, ml_conf, side
            portfolio_state: Dict with 'cash', 'portfolio_value' / 'equity'
            existing_positions: Dict of current positions
            current_prices: Dict of current market prices
            returns_df: Historical daily returns DataFrame
            sectors: Dict mapping symbol -> sector
            portfolio_risk_buckets: Dict mapping symbol -> portfolio risk bucket
            regime: Current market regime string
        """
        # Parse portfolio state
        cash = float(portfolio_state.get("cash", 100000.0))
        equity = float(portfolio_state.get("portfolio_value", portfolio_state.get("equity", cash)))
        if equity <= 0:
            equity = cash

        prices = dict(current_prices or {})
        for c in (candidates or []):
            sym = c["symbol"].upper()
            if sym not in prices and c.get("price"):
                prices[sym] = float(c["price"])

        # Calculate current weights and quantities from existing positions
        current_weights: Dict[str, float] = {}
        existing_quantities: Dict[str, int] = {}
        if existing_positions:
            for s, pos in existing_positions.items():
                sym = s.upper()
                if isinstance(pos, dict):
                    qty = int(pos.get("quantity", pos.get("qty", 0)))
                    p = float(prices.get(sym, pos.get("current_price", pos.get("price", 0.0))))
                else:
                    qty = int(getattr(pos, "quantity", getattr(pos, "qty", 0)))
                    p = float(prices.get(sym, getattr(pos, "current_price", getattr(pos, "price", 0.0))))
                if sym not in prices and p > 0:
                    prices[sym] = p
                if qty > 0 and p > 0:
                    pos_val = qty * p
                    current_weights[sym] = round(pos_val / max(equity, 1.0), 6)
                    existing_quantities[sym] = qty

        if not candidates and not existing_positions:
            return self._empty_proposal(portfolio_state)

        # Extract symbols, scores, confidences
        symbols = [c["symbol"].upper() for c in (candidates or [])[:5]]
        scores = {c["symbol"].upper(): float(c.get("alpha_score", 0.50)) for c in (candidates or [])}
        confidences = {c["symbol"].upper(): float(c.get("ml_conf", 0.75)) for c in (candidates or [])}
        bucket_map = portfolio_risk_buckets if portfolio_risk_buckets is not None else sectors

        opt_res = self.optimizer.optimize(
            candidates=symbols,
            scores=scores,
            confidences=confidences,
            returns_df=returns_df,
            sectors=bucket_map,
            portfolio_risk_buckets=bucket_map,
            current_weights=current_weights,
            total_equity=equity,
            current_cash=cash,
            current_prices=prices,
            regime=regime,
        )

        proposal = self._build_proposal_from_result(
            result=opt_res,
            total_equity=equity,
            cash=cash,
            prices=prices,
            sectors=bucket_map or {},
            current_weights=current_weights,
            existing_quantities=existing_quantities,
            candidate_sides={c["symbol"].upper(): c.get("side", "BUY") for c in (candidates or [])},
        )
        self.last_proposal = proposal
        return proposal

    def _build_proposal_from_result(
        self,
        result: OptimizerResult,
        total_equity: float,
        cash: float,
        prices: Dict[str, float],
        sectors: Dict[str, str],
        current_weights: Dict[str, float],
        existing_quantities: Dict[str, int],
        candidate_sides: Optional[Dict[str, str]] = None,
    ) -> PortfolioProposal:
        """
        Converts OptimizerResult into executable PositionProposals handling all 5 cases:
        1. NEW_POSITION (BUY)
        2. INCREASE_EXISTING (BUY)
        3. REDUCE_EXISTING (SELL)
        4. CLOSE_POSITION (SELL)
        5. RETAIN_EXISTING (HOLD / no trade)
        """
        candidate_sides = candidate_sides or {}
        proposals: List[PositionProposal] = []
        diag = result.diagnostics

        allocated_dollars = 0.0
        remaining_cash = cash

        # Union of all symbols in target result and active existing holdings
        all_symbols = list(dict.fromkeys(
            list(existing_quantities.keys()) +
            list(result.weights.keys()) +
            result.selected_subset
        ))

        # First pass: Process SELLS / LIQUIDATIONS to raise cash and free cardinality
        for sym in all_symbols:
            w_curr = current_weights.get(sym, 0.0)
            w_target = result.weights.get(sym, 0.0)
            price = prices.get(sym, 100.0)
            if price <= 0:
                continue

            delta_w = w_target - w_curr

            if delta_w < -0.002 and w_curr > 0.001:
                # Need to SELL
                q_exist = existing_quantities.get(sym, 0)
                is_close = (w_target <= 0.001 or q_exist <= 0)
                if is_close:
                    total_sell_qty = q_exist
                    target_w = 0.0
                else:
                    sell_dollars = abs(delta_w) * total_equity
                    total_sell_qty = min(q_exist, max(1, int(sell_dollars // price)))
                    target_w = round((q_exist - total_sell_qty) * price / max(total_equity, 1.0), 4)

                max_chunk_qty = max(1, int(self.max_order_value // price)) if price > 0 else 1
                remaining_sell = total_sell_qty
                chunk_num = 1
                total_chunks = max(1, int(np.ceil(total_sell_qty / max_chunk_qty))) if max_chunk_qty > 0 else 1

                while remaining_sell > 0:
                    chunk_qty = min(remaining_sell, max_chunk_qty)
                    chunk_notional = round(chunk_qty * price, 2)
                    while chunk_qty > 1 and chunk_notional > self.max_order_value:
                        chunk_qty -= 1
                        chunk_notional = round(chunk_qty * price, 2)

                    remaining_cash += chunk_notional
                    remaining_sell -= chunk_qty

                    if is_close:
                        if total_chunks == 1:
                            reason = f"CLOSE_POSITION (liquidate {chunk_qty} sh @ ${price:.2f})"
                        else:
                            reason = f"CLOSE_POSITION chunk {chunk_num}/{total_chunks} (liquidate {chunk_qty} sh @ ${price:.2f})"
                    else:
                        if total_chunks == 1:
                            reason = f"REDUCE_EXISTING (sell {chunk_qty} sh, target w={target_w*100:.1f}%)"
                        else:
                            reason = f"REDUCE_EXISTING chunk {chunk_num}/{total_chunks} (sell {chunk_qty} sh, target w={target_w*100:.1f}%)"

                    proposals.append(PositionProposal(
                        symbol=sym,
                        target_weight=target_w,
                        dollar_notional=chunk_notional,
                        quantity=chunk_qty,
                        side="SELL",
                        price=price,
                        sector=sectors.get(sym, "UNKNOWN"),
                        portfolio_risk_bucket=sectors.get(sym, "UNKNOWN"),
                        robust_mu=diag.get("mu_robust", {}).get(sym, 0.0),
                        uncertainty=diag.get("uncertainty", {}).get(sym, 0.0),
                        volatility=diag.get("volatilities", {}).get(sym, 0.0),
                        kelly_ceiling=diag.get("kelly_ceilings", {}).get(sym, 0.25),
                        cvar_contribution=result.portfolio_cvar,
                        transaction_cost=result.transaction_cost,
                        reason=reason,
                    ))
                    chunk_num += 1

        # Second pass: Process BUYS (New Positions & Increases)
        for sym in all_symbols:
            w_curr = current_weights.get(sym, 0.0)
            w_target = result.weights.get(sym, 0.0)
            price = prices.get(sym, 100.0)
            if price <= 0:
                continue

            delta_w = w_target - w_curr

            if delta_w > 0.002:
                # Target dollar increment
                desired_order_notional = delta_w * total_equity
                # Cap at hard max $5,000 per order
                notional = min(desired_order_notional, self.max_order_value)

                # Ensure we do not breach cash reserve floor ($1,000)
                available_cash_for_order = max(0.0, remaining_cash - self.min_cash_reserve)
                notional = min(notional, available_cash_for_order)

                quantity = int(notional // price)
                if quantity < 1:
                    if notional >= price and (remaining_cash - price) >= self.min_cash_reserve:
                        quantity = 1
                    else:
                        logger.info("Notional $%.2f insufficient for 1 share of %s @ $%.2f. Skipping.", notional, sym, price)
                        continue

                actual_notional = round(quantity * price, 2)
                # Final check against per-order limit
                if actual_notional > self.max_order_value + 0.01:
                    quantity = int(self.max_order_value // price)
                    actual_notional = round(quantity * price, 2)

                remaining_cash -= actual_notional
                allocated_dollars += actual_notional
                is_new = (w_curr < 0.001)
                order_reason = "NEW_POSITION" if is_new else "INCREASE_EXISTING"

                post_trade_w = round((w_curr * total_equity + actual_notional) / max(total_equity, 1.0), 4)

                proposals.append(PositionProposal(
                    symbol=sym,
                    target_weight=post_trade_w,
                    dollar_notional=actual_notional,
                    quantity=quantity,
                    side="BUY",
                    price=price,
                    sector=sectors.get(sym, "UNKNOWN"),
                    portfolio_risk_bucket=sectors.get(sym, "UNKNOWN"),
                    robust_mu=diag.get("mu_robust", {}).get(sym, 0.0),
                    uncertainty=diag.get("uncertainty", {}).get(sym, 0.0),
                    volatility=diag.get("volatilities", {}).get(sym, 0.0),
                    kelly_ceiling=diag.get("kelly_ceilings", {}).get(sym, self.max_position_weight),
                    cvar_contribution=result.portfolio_cvar,
                    transaction_cost=result.transaction_cost,
                    reason=f"{order_reason} w={post_trade_w*100:.1f}%, ${actual_notional:.1f} ({quantity} sh)",
                ))

        # Compute post-trade active weights and exposure
        final_post_trade_weights: Dict[str, float] = {}
        for sym in all_symbols:
            w_curr = current_weights.get(sym, 0.0)
            # Find proposal if any
            prop = next((p for p in proposals if p.symbol == sym), None)
            if prop:
                if prop.side == "BUY":
                    final_post_trade_weights[sym] = round(w_curr + (prop.dollar_notional / max(total_equity, 1.0)), 4)
                elif prop.side == "SELL":
                    final_post_trade_weights[sym] = max(0.0, round(w_curr - (prop.dollar_notional / max(total_equity, 1.0)), 4))
            else:
                if w_curr > 0.001:
                    final_post_trade_weights[sym] = w_curr

        final_active_subset = [s for s, w in final_post_trade_weights.items() if w > 0.001]
        gross_exp = round(sum(final_post_trade_weights.values()), 4)

        # Independent Post-Trade Verification Gate (Mandatory Section 10)
        invariants_ok, invariant_reason = self.verify_post_trade_invariants(
            final_weights=final_post_trade_weights,
            proposals=proposals,
            cash_remaining=remaining_cash,
            total_equity=total_equity,
            sectors=sectors,
        )
        if not invariants_ok:
            logger.error("POST-TRADE INVARIANT BREACH: %s. Safely halting to ZERO NEW ORDERS.", invariant_reason)
            return self._empty_proposal({"cash": cash, "portfolio_value": total_equity})

        return PortfolioProposal(
            proposals=proposals,
            weights=final_post_trade_weights,
            selected_subset=final_active_subset,
            gross_exposure=gross_exp,
            net_exposure=gross_exp,
            cash_remaining=round(remaining_cash, 2),
            portfolio_variance=result.portfolio_variance,
            portfolio_volatility=result.portfolio_volatility,
            portfolio_cvar=result.portfolio_cvar,
            turnover=result.turnover,
            total_cost=result.transaction_cost,
            objective_value=result.objective_value,
            status=result.status if (proposals or final_active_subset) else "ZERO_TRADE",
            diagnostics=diag,
        )

    def verify_post_trade_invariants(
        self,
        final_weights: Dict[str, float],
        proposals: List[PositionProposal],
        cash_remaining: float,
        total_equity: float,
        sectors: Dict[str, str],
        portfolio_risk_buckets: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        Mandatory Independent Post-Trade Verification Gate.
        Calculates and asserts ALL 5 hard institutional bounds after proposed trades:
        1. final_positions <= 3
        2. every new order <= $5,000
        3. final_cash >= $1,000
        4. final gross exposure <= 95%
        5. every portfolio_risk_bucket / sector <= 30%
        """
        # 1. Final positions <= 3
        active_count = sum(1 for w in final_weights.values() if w > 0.001)
        if active_count > 3:
            return False, f"Post-trade positions count {active_count} exceeds limit 3"

        # 2. Every new order <= $5,000
        for p in proposals:
            if p.side == "BUY" and p.dollar_notional > self.max_order_value + 1.0:
                return False, f"Order {p.symbol} notional ${p.dollar_notional:.2f} exceeds cap ${self.max_order_value}"

        # 3. Final cash >= $1,000
        if cash_remaining < self.min_cash_reserve - 1.0:
            return False, f"Post-trade cash ${cash_remaining:.2f} breaches reserve floor ${self.min_cash_reserve}"

        # 4. Final gross exposure <= 95%
        gross_exp = sum(final_weights.values())
        if gross_exp > self.max_total_exposure + 1e-4:
            return False, f"Post-trade gross exposure {gross_exp:.4f} exceeds ceiling {self.max_total_exposure}"

        # 5. Every portfolio_risk_bucket / sector <= 30%
        bucket_map = portfolio_risk_buckets if portfolio_risk_buckets is not None else sectors
        sec_weights: Dict[str, float] = {}
        for s, w in final_weights.items():
            sec = bucket_map.get(s, "UNKNOWN")
            sec_weights[sec] = sec_weights.get(sec, 0.0) + w

        for sec, sw in sec_weights.items():
            if sec != "UNKNOWN" and sw > self.max_portfolio_risk_bucket_weight + 1e-4:
                return False, f"Post-trade portfolio risk bucket / sector {sec} exposure {sw:.4f} exceeds ceiling {self.max_portfolio_risk_bucket_weight}"

        return True, "ALL_POST_TRADE_INVARIANTS_VERIFIED"


    def _empty_proposal(self, portfolio_state: Dict[str, Any]) -> PortfolioProposal:
        cash = float(portfolio_state.get("cash", 100000.0))
        return PortfolioProposal(
            proposals=[],
            weights={},
            selected_subset=[],
            gross_exposure=0.0,
            net_exposure=0.0,
            cash_remaining=round(cash, 2),
            portfolio_variance=0.0,
            portfolio_volatility=0.0,
            portfolio_cvar=0.0,
            turnover=0.0,
            total_cost=0.0,
            objective_value=0.0,
            status="EMPTY_PROPOSAL",
            diagnostics={"reason": "No candidates or positions provided"},
        )

    def diagnose_portfolio(
        self,
        weights: Dict[str, float],
        returns_df: Optional[pd.DataFrame] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Provides portfolio diagnostics (HHI, Effective N, Volatility, Marginal Risk Contribution)."""
        return self.legacy_engine.diagnose_portfolio(
            weights=weights,
            returns_df=returns_df,
            sectors=sectors,
        )
