"""
Multi-Factor Risk & Exposure Engine.

Quantifies institutional style factor exposures across:
1. MARKET BETA: Sensitivity to broad market (SPY for US, Nifty for IN)
2. MOMENTUM: 12-1m / 20d relative momentum
3. SIZE: Log market-cap proxy (Large vs Small cap tilt)
4. VALUE: Earnings yield (E/P) and book-to-market
5. QUALITY: ROE and operating profitability
6. VOLATILITY: 60d realized annualized volatility
7. GROWTH: 1-year sales growth rate

Answers the fundamental research question:
'Did the trading strategy generate idiosyncratic stock-selection alpha,
or simply ride systematic market/momentum factor exposure?'
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioFactorProfile:
    market_beta: float
    factor_exposures: Dict[str, float]  # momentum, size, value, quality, volatility, growth
    sector_exposures: Dict[str, float]
    concentration_hhi: float  # Herfindahl-Hirschman Index (0.0 to 1.0)
    effective_n_stocks: float  # 1 / sum(w_i^2)
    style_classification: str  # e.g., "Large-Cap Growth", "Quality Momentum"
    alpha_beta_decomposition: Dict[str, float]  # {"total_return": ..., "factor_beta_return": ..., "selection_alpha": ...}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactorModel:
    """
    Computes cross-sectional standardized factor exposures and portfolio attribution.
    """

    FACTORS = ["momentum", "size", "value", "quality", "volatility", "growth"]

    def __init__(self):
        pass

    @staticmethod
    def compute_asset_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate market beta via covariance / variance."""
        common = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(common) < 15:
            return 1.0
        cov = np.cov(common.iloc[:, 0], common.iloc[:, 1])[0, 1]
        var = np.var(common.iloc[:, 1])
        if var <= 1e-8:
            return 1.0
        return float(np.clip(cov / var, -0.5, 3.0))

    def compute_factor_exposures(
        self,
        symbol_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Z-score standardize factor metrics cross-sectionally across universe.
        symbol_metrics format: {
            "AAPL": {"momentum": 0.15, "size": 3.0e12, "value": 0.03, "quality": 0.45, "volatility": 0.22, "growth": 0.05},
            ...
        }
        """
        if not symbol_metrics:
            return {}

        df = pd.DataFrame.from_dict(symbol_metrics, orient="index")
        z_df = pd.DataFrame(index=df.index)

        for col in self.FACTORS:
            if col in df.columns:
                vals = df[col].astype(float)
                mean = vals.mean()
                std = vals.std()
                if std > 1e-6:
                    # For size, standard is log(cap)
                    if col == "size":
                        vals = np.log(np.maximum(vals, 1e6))
                        z_df[col] = (vals - vals.mean()) / max(vals.std(), 1e-6)
                    else:
                        z_df[col] = (vals - mean) / std
                else:
                    z_df[col] = 0.0
            else:
                z_df[col] = 0.0

        return z_df.round(3).to_dict(orient="index")

    def profile_portfolio(
        self,
        weights: Dict[str, float],
        symbol_exposures: Dict[str, Dict[str, float]],
        betas: Dict[str, float],
        sectors: Dict[str, str],
        portfolio_return: float = 0.0,
        benchmark_return: float = 0.0,
    ) -> PortfolioFactorProfile:
        """
        Aggregates security-level exposures into a comprehensive portfolio factor profile.
        """
        if not weights:
            return PortfolioFactorProfile(
                market_beta=1.0, factor_exposures={f: 0.0 for f in self.FACTORS},
                sector_exposures={}, concentration_hhi=0.0, effective_n_stocks=0.0,
                style_classification="Neutral", alpha_beta_decomposition={}
            )

        # Normalize weights
        tot_w = sum(weights.values())
        norm_weights = {k: v / tot_w for k, v in weights.items()} if tot_w > 0 else weights

        # 1. Market Beta
        port_beta = sum(norm_weights.get(s, 0.0) * betas.get(s, 1.0) for s in norm_weights)

        # 2. Factor Exposures
        port_factors = {f: 0.0 for f in self.FACTORS}
        for s, w in norm_weights.items():
            s_exp = symbol_exposures.get(s, {})
            for f in self.FACTORS:
                port_factors[f] += w * s_exp.get(f, 0.0)

        # 3. Sector Exposures
        sector_exp: Dict[str, float] = {}
        for s, w in norm_weights.items():
            sec = sectors.get(s, "Other")
            sector_exp[sec] = sector_exp.get(sec, 0.0) + w

        # 4. Concentration & Effective N
        hhi = sum(w ** 2 for w in norm_weights.values())
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        # 5. Style classification
        mom_tag = "Momentum" if port_factors["momentum"] > 0.3 else ("Reversal" if port_factors["momentum"] < -0.3 else "")
        val_tag = "Growth" if port_factors["value"] < -0.3 else ("Value" if port_factors["value"] > 0.3 else "Core")
        size_tag = "Large" if port_factors["size"] > 0.2 else ("Small" if port_factors["size"] < -0.2 else "Mid")
        style = f"{size_tag}-{val_tag} {mom_tag}".strip()

        # 6. Alpha / Beta decomposition
        factor_beta_ret = port_beta * benchmark_return
        selection_alpha = portfolio_return - factor_beta_ret

        return PortfolioFactorProfile(
            market_beta=round(float(port_beta), 3),
            factor_exposures={k: round(float(v), 3) for k, v in port_factors.items()},
            sector_exposures={k: round(float(v), 3) for k, v in sector_exp.items()},
            concentration_hhi=round(float(hhi), 4),
            effective_n_stocks=round(float(effective_n), 1),
            style_classification=style,
            alpha_beta_decomposition={
                "total_return": round(float(portfolio_return), 4),
                "factor_beta_return": round(float(factor_beta_ret), 4),
                "selection_alpha": round(float(selection_alpha), 4),
            }
        )
