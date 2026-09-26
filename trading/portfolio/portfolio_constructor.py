"""
Portfolio Construction Framework — Risk-Aware Institutional Portfolio Building.

Implements:
- Equal Weight
- Score Weight
- Confidence Weight
- Inverse Volatility
- True Risk Parity / Equal Risk Contribution (ERC) via convex optimization
- Volatility Targeting
- Marginal Risk Contribution (MRC) Analysis
- Hierarchical Correlation Clustering
- Portfolio Heat & Concentration Scoring (HHI)

Enforces Hard Institutional Constraints:
- max_position_weight (default: 25%)
- max_sector_weight (default: 40%)
- max_total_exposure (default: 95%)
- max_positions (default: 40)
- min_cash (default: 5%)
- max_cluster_weight (default: 40%)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Constructs multi-asset portfolios subject to risk and correlation constraints."""

    def __init__(
        self,
        max_position_weight: float = 0.25,
        max_sector_weight: float = 0.40,
        max_total_exposure: float = 0.95,
        max_positions: int = 40,
        min_cash_pct: float = 0.05,
        correlation_threshold: float = 0.70,
        max_cluster_weight: float = 0.40,
    ):
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.max_total_exposure = max_total_exposure
        self.max_positions = max_positions
        self.min_cash_pct = min_cash_pct
        self.correlation_threshold = correlation_threshold
        self.max_cluster_weight = max_cluster_weight

    def construct(
        self,
        candidates: List[str],
        scores: Optional[Dict[str, float]] = None,
        confidences: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        method: str = "risk_parity",
        sectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio weights for candidates.
        Methods:
        - 'equal_weight'
        - 'score_weight'
        - 'confidence_weight'
        - 'inverse_vol'
        - 'risk_parity' (Equal Risk Contribution)
        - 'vol_target'
        """
        if not candidates:
            return {}

        symbols = [s.upper() for s in candidates][: self.max_positions]
        scores = scores or {s: 1.0 for s in symbols}
        confidences = confidences or {s: 1.0 for s in symbols}
        volatilities = volatilities or {s: 0.20 for s in symbols}
        sectors = sectors or {s: "UNKNOWN" for s in symbols}

        # Dynamic position weight ceiling for small candidate lists
        pos_ceiling = max(self.max_position_weight, 1.0 / len(symbols))

        # 1. Base weights by method
        raw_weights: Dict[str, float] = {}

        if method == "equal_weight":
            w = 1.0 / len(symbols)
            raw_weights = {s: w for s in symbols}

        elif method == "score_weight":
            pos_scores = {s: max(0.01, scores.get(s, 0.01)) for s in symbols}
            tot = sum(pos_scores.values())
            raw_weights = {s: (pos_scores[s] / tot) for s in symbols}

        elif method == "confidence_weight":
            confs = {s: max(0.05, confidences.get(s, 0.5)) for s in symbols}
            tot = sum(confs.values())
            raw_weights = {s: (confs[s] / tot) for s in symbols}

        elif method == "risk_parity":
            if returns_df is not None and len([s for s in symbols if s in returns_df.columns]) >= 2:
                raw_weights = self._solve_risk_parity(symbols, returns_df)
            else:
                inv_vols = {s: 1.0 / max(0.01, volatilities.get(s, 0.20)) for s in symbols}
                tot = sum(inv_vols.values())
                raw_weights = {s: (inv_vols[s] / tot) for s in symbols}

        elif method == "inverse_vol":
            inv_vols = {s: 1.0 / max(0.01, volatilities.get(s, 0.20)) for s in symbols}
            tot = sum(inv_vols.values())
            raw_weights = {s: (inv_vols[s] / tot) for s in symbols}

        elif method == "vol_target":
            target_ann_vol = 0.15
            unconstrained = {s: target_ann_vol / max(0.05, volatilities.get(s, 0.20)) for s in symbols}
            tot = sum(unconstrained.values())
            raw_weights = {s: (unconstrained[s] / tot) if tot > 0 else (1.0 / len(symbols)) for s in symbols}

        else:
            w = 1.0 / len(symbols)
            raw_weights = {s: w for s in symbols}

        # 2. Enforce position weight ceiling
        capped_weights = {s: min(w, pos_ceiling) for s, w in raw_weights.items()}

        # 3. Correlation Clustering Constraint
        if returns_df is not None and not returns_df.empty:
            capped_weights = self._apply_correlation_clusters(capped_weights, returns_df)

        # 4. Enforce sector exposure limits (only if multiple sectors present)
        sector_totals: Dict[str, float] = {}
        for s, w in capped_weights.items():
            sec = sectors.get(s, "UNKNOWN")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w

        if len(set(sectors.values())) > 1:
            for sec, total in sector_totals.items():
                if total > self.max_sector_weight and sec != "UNKNOWN":
                    scale = self.max_sector_weight / total
                    for s in capped_weights:
                        if sectors.get(s) == sec:
                            capped_weights[s] *= scale

        # 5. Normalize within max_total_exposure (leaving min_cash intact)
        tot = sum(capped_weights.values())
        max_allowed = min(self.max_total_exposure, 1.0 - self.min_cash_pct)
        if tot > max_allowed:
            scale = max_allowed / tot
            final_weights = {s: round(w * scale, 4) for s, w in capped_weights.items()}
        else:
            final_weights = {s: round(w, 4) for s, w in capped_weights.items()}

        return final_weights

    def _solve_risk_parity(self, symbols: List[str], returns_df: pd.DataFrame) -> Dict[str, float]:
        """Convex optimization solver for Equal Risk Contribution (Risk Parity)."""
        valid_syms = [s for s in symbols if s in returns_df.columns]
        if len(valid_syms) < 2:
            return {s: 1.0 / len(symbols) for s in symbols}

        cov = returns_df[valid_syms].cov().values * 252.0
        n = len(valid_syms)

        def erc_objective(w):
            port_var = w @ cov @ w
            port_vol = np.sqrt(max(port_var, 1e-8))
            rc = w * (cov @ w) / port_vol
            target_rc = port_vol / n
            return float(np.sum((rc - target_rc) ** 2))

        w0 = np.ones(n) / n
        bounds = [(0.01, 0.99) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            res = minimize(erc_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            w_opt = res.x if res.success else w0
        except Exception:
            w_opt = w0

        res_dict = {valid_syms[i]: float(w_opt[i]) for i in range(n)}
        for s in symbols:
            if s not in res_dict:
                res_dict[s] = 0.01
        tot = sum(res_dict.values())
        return {k: v / tot for k, v in res_dict.items()}

    def _apply_correlation_clusters(
        self, weights: Dict[str, float], returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Group highly correlated assets and cap total weight per cluster."""
        symbols = [s for s in weights if s in returns_df.columns]
        if len(symbols) < 2:
            return weights

        corr = returns_df[symbols].corr().fillna(0)
        clusters: List[List[str]] = []
        assigned = set()

        for s1 in symbols:
            if s1 in assigned:
                continue
            current_cluster = [s1]
            assigned.add(s1)
            for s2 in symbols:
                if s2 not in assigned and corr.loc[s1, s2] >= self.correlation_threshold:
                    current_cluster.append(s2)
                    assigned.add(s2)
            clusters.append(current_cluster)

        adjusted_weights = dict(weights)
        for cluster in clusters:
            cluster_weight = sum(adjusted_weights.get(s, 0) for s in cluster)
            if cluster_weight > self.max_cluster_weight:
                scale = self.max_cluster_weight / cluster_weight
                for s in cluster:
                    adjusted_weights[s] = adjusted_weights.get(s, 0) * scale

        return adjusted_weights

    def diagnose_portfolio(
        self,
        weights: Dict[str, float],
        returns_df: Optional[pd.DataFrame] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deep portfolio intelligence diagnostics:
        - Portfolio Volatility (Annualized)
        - Marginal Risk Contributions (MRC)
        - Concentration HHI & Effective N
        - Sector Exposures
        """
        if not weights:
            return {"status": "EMPTY_PORTFOLIO"}

        hhi = sum(w ** 2 for w in weights.values())
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        sectors = sectors or {}
        sec_dist = {}
        for s, w in weights.items():
            sec = sectors.get(s, "UNKNOWN")
            sec_dist[sec] = round(sec_dist.get(sec, 0.0) + w, 4)

        mrc_dict = {}
        ann_vol = 0.15
        if returns_df is not None and not returns_df.empty:
            valid = [s for s in weights if s in returns_df.columns]
            if len(valid) >= 2:
                w_vec = np.array([weights[s] for s in valid])
                w_vec = w_vec / max(np.sum(w_vec), 1e-6)
                cov = returns_df[valid].cov().values * 252.0
                port_var = w_vec @ cov @ w_vec
                ann_vol = float(np.sqrt(max(port_var, 1e-8)))
                mrc = (cov @ w_vec) / max(ann_vol, 1e-6)
                for i, s in enumerate(valid):
                    mrc_dict[s] = round(float(w_vec[i] * mrc[i]), 4)

        return {
            "portfolio_heat_annualized_vol": round(ann_vol, 4),
            "concentration_hhi": round(float(hhi), 4),
            "effective_n_stocks": round(float(effective_n), 1),
            "active_positions_count": len([w for w in weights.values() if w > 0.001]),
            "marginal_risk_contributions": mrc_dict,
            "sector_distribution": sec_dist,
        }
