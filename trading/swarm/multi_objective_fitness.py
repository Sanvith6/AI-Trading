"""
Multi-Objective Fitness & Novelty Scoring — Cost-Aware Institutional Robustness Engine.

Evaluates:
- Net Out-of-Sample Sharpe, Sortino, Calmar (strictly post-statutory costs)
- Net Edge After Costs (`net_edge_after_costs = gross_pnl - statutory_fees`)
- Cost-to-Expected-Edge Ratio (`cost_to_edge_ratio`)
- Turnover & Rebalancing Frequency Drag Penalty
- Complexity Penalty (penalizes excess parameters / components without incremental return)
- Max Drawdown Penalty (exponential over 8%)
- Parameter Sensitivity & Fragility Penalty
- Novelty Score (rewards strategies exploring non-redundant parameter space)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from trading.swarm.strategy_dna import StrategyDNA


class MultiObjectiveFitness:
    """Computes composite cost-aware institutional fitness and novelty pressure."""

    @staticmethod
    def compute_fitness(
        metrics: Dict[str, Any],
        sensitivity_score: float = 0.0,
        dna: Optional[StrategyDNA] = None,
    ) -> float:
        """
        Composite cost-aware fitness formula:
        + Net Sharpe Ratio * 0.35
        + Net Sortino Ratio * 0.25
        + Net Calmar Ratio * 0.20
        + Net Profit Factor * 0.10
        - Severe Net Negative Edge Penalty (if net_edge_after_costs < 0)
        - Cost-to-Edge Ratio Penalty (penalizes fee-heavy turnover)
        - Drawdown Penalty (exponential over 8%)
        - Complexity Penalty (penalizes excessive alphas/features without edge)
        - Parameter Fragility Penalty
        """
        sharpe = float(metrics.get("sharpe", 0.0))
        sortino = float(metrics.get("sortino", 0.0))
        calmar = float(metrics.get("calmar", 0.0))
        profit_factor = float(metrics.get("profit_factor", 1.0))
        max_dd_pct = float(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0)))
        total_costs = float(metrics.get("costs", metrics.get("total_costs", 0.0)))
        net_pnl = float(metrics.get("net_pnl", metrics.get("pnl", 0.0)))
        trades = int(metrics.get("trades", 0))

        # 1. Base risk-adjusted performance (post-cost)
        base = (
            max(-2.0, min(4.0, sharpe)) * 0.35 +
            max(-2.0, min(4.0, sortino)) * 0.25 +
            max(-1.0, min(3.0, calmar)) * 0.20 +
            max(0.0, min(3.0, profit_factor - 1.0)) * 0.10
        )

        # 2. Net edge after costs & cost-to-edge ratio
        gross_pnl = net_pnl + total_costs
        cost_to_edge_ratio = total_costs / max(100.0, abs(net_pnl) + total_costs)

        # Heavy penalty if costs devour more than 30% of gross edge
        cost_ratio_penalty = 0.0
        if cost_to_edge_ratio > 0.30:
            cost_ratio_penalty = (cost_to_edge_ratio - 0.30) * 1.5

        # Catastrophic penalty if strategy is net negative after statutory fees
        net_edge_penalty = 0.0
        if net_pnl < 0:
            net_edge_penalty = 1.5 + abs(net_pnl) / 10000.0

        # 3. Drawdown penalty (exponential over 8%)
        dd_penalty = 0.0
        if max_dd_pct > 8.0:
            dd_penalty = (max_dd_pct - 8.0) * 0.15
        if max_dd_pct > 15.0:
            dd_penalty += (max_dd_pct - 15.0) * 0.35

        # 4. Sensitivity penalty (fragility under parameter perturbation)
        sensitivity_penalty = float(max(0.0, sensitivity_score - 0.25) * 1.2)

        # 5. Complexity penalty (Occam's razor: penalize excess genome complexity)
        complexity_penalty = 0.0
        if dna:
            n_alphas = len(dna.active_alphas)
            if n_alphas > 3:
                complexity_penalty += (n_alphas - 3) * 0.05
            if dna.rebalance_bars < 3:
                # Direct turnover penalty for rapid whipsaw trading
                complexity_penalty += (3 - dna.rebalance_bars) * 0.25

        final_fitness = (
            base
            - cost_ratio_penalty
            - net_edge_penalty
            - dd_penalty
            - sensitivity_penalty
            - complexity_penalty
        )
        return round(float(final_fitness), 4)

    @staticmethod
    def compute_novelty(candidate: StrategyDNA, population: List[StrategyDNA]) -> float:
        """
        Measures Euclidean parameter distance between candidate and the population.
        High novelty prevents genetic collapse into identical parameter basins.
        """
        if len(population) <= 1:
            return 1.0

        distances = []
        c_p = candidate.to_dict()
        numeric_keys = ["stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "risk_per_trade_pct", "lookback_bars"]

        for other in population:
            if other.strategy_id == candidate.strategy_id:
                continue
            o_p = other.to_dict()
            diff_sq = 0.0
            for k in numeric_keys:
                diff_sq += (c_p.get(k, 0.0) - o_p.get(k, 0.0)) ** 2
            set_c = set(candidate.active_alphas)
            set_o = set(other.active_alphas)
            jaccard_dist = 1.0 - (len(set_c & set_o) / max(1, len(set_c | set_o)))
            dist = np.sqrt(diff_sq) + (jaccard_dist * 5.0)
            distances.append(dist)

        distances.sort()
        k_nearest = distances[: min(5, len(distances))]
        novelty = float(np.mean(k_nearest)) if k_nearest else 1.0
        return round(novelty, 3)
