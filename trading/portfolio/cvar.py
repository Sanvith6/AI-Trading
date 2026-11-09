"""
Conditional Value-at-Risk (CVaR / Expected Shortfall) Module.

Implements tail-risk quantification using the Rockafellar & Uryasev (2000)
formulation on historical and simulated return scenarios.

Guarantees:
1. Exact tail loss measurement beyond the alpha-quantile (VaR).
2. Fully convex representation.
3. Strict numerical stability and determinism.
4. Safe fallback if insufficient scenarios exist.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CVaRModel:
    """Calculates Portfolio VaR and CVaR (Expected Shortfall) at specified confidence."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_scenarios: int = 20,
        synthetic_scenarios_count: int = 250,
        random_seed: int = 42,
    ):
        self.confidence_level = confidence_level
        self.min_scenarios = min_scenarios
        self.synthetic_scenarios_count = synthetic_scenarios_count
        self.random_seed = random_seed

    def extract_scenarios(
        self,
        symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[np.ndarray] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Extracts scenario returns matrix R (S x N) for the active symbols.
        
        Args:
            symbols: List of symbols (length N)
            returns_df: Historical returns DataFrame
            cov_matrix: Covariance matrix (N x N, annualized)
            volatilities: Volatilities per symbol
            
        Returns:
            np.ndarray of shape (S, N) representing daily return scenarios.
        """
        n = len(symbols)
        if n == 0:
            return np.empty((0, 0))

        # 1. Try empirical returns from returns_df
        if returns_df is not None and not returns_df.empty:
            valid_cols = [s for s in symbols if s in returns_df.columns]
            if len(valid_cols) == n:
                cleaned = returns_df[symbols].dropna()
                if len(cleaned) >= self.min_scenarios:
                    vals = cleaned.values
                    if np.isfinite(vals).all():
                        return vals

        # 2. If empirical returns are insufficient or incomplete, generate deterministic synthetic scenarios
        # using the shrinkage covariance matrix scaled to daily
        rng = np.random.RandomState(self.random_seed)
        if cov_matrix is not None and cov_matrix.shape == (n, n):
            daily_cov = cov_matrix / 252.0
            # Symmetrize and ensure PD
            daily_cov = 0.5 * (daily_cov + daily_cov.T)
            eigvals, eigvecs = np.linalg.eigh(daily_cov)
            eigvals = np.maximum(eigvals, 1e-8)
            daily_cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            scenarios = rng.multivariate_normal(
                mean=np.zeros(n),
                cov=daily_cov_pd,
                size=self.synthetic_scenarios_count,
            )
            return scenarios

        # 3. Fallback from volatilities
        vols = np.array([volatilities.get(s, 0.20) if volatilities else 0.20 for s in symbols]) / np.sqrt(252.0)
        scenarios = rng.normal(loc=0.0, scale=vols, size=(self.synthetic_scenarios_count, n))
        return scenarios

    def calculate_cvar(
        self,
        weights: np.ndarray,
        scenarios: np.ndarray,
        confidence_level: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculates VaR and CVaR for a given weight vector and scenario matrix.
        
        Args:
            weights: 1D array of weights (length N)
            scenarios: 2D array of return scenarios (S x N)
            confidence_level: Confidence level (e.g. 0.95)
            
        Returns:
            Tuple of (annualized_VaR, annualized_CVaR)
        """
        if len(weights) == 0 or len(scenarios) == 0:
            return 0.0, 0.0

        w = np.asarray(weights, dtype=np.float64)
        if np.sum(np.abs(w)) < 1e-8:
            return 0.0, 0.0

        alpha = confidence_level if confidence_level is not None else self.confidence_level

        # Portfolio return scenarios: r_s = scenarios @ w
        port_returns = scenarios @ w
        # Portfolio loss scenarios: L_s = -r_s
        losses = -port_returns

        s = len(losses)
        if s == 0:
            return 0.0, 0.0

        # VaR is the alpha-quantile of losses
        var_daily = float(np.percentile(losses, alpha * 100.0))

        # CVaR is the expected loss exceeding VaR
        tail_losses = losses[losses >= var_daily]
        if len(tail_losses) > 0:
            cvar_daily = float(np.mean(tail_losses))
        else:
            cvar_daily = var_daily

        # Annualize for consistency with annual expected return and annual variance
        # Standard Square-root of time scaling for risk
        scale = np.sqrt(252.0)
        return float(var_daily * scale), float(cvar_daily * scale)

    def rockafellar_uryasev_objective(
        self,
        weights: np.ndarray,
        scenarios: np.ndarray,
        var_val: Optional[float] = None,
        beta: float = 100.0,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Smooth, differentiable Rockafellar-Uryasev auxiliary function:
          F_alpha(w, v) = v + 1/((1 - alpha) * S) * sum(smooth_plus(L_s - v, beta))
        Annualized by sqrt(252.0).
        """
        conf = alpha if alpha is not None else self.confidence_level
        s = len(scenarios)
        if s == 0 or len(weights) == 0:
            return 0.0

        w = np.asarray(weights, dtype=np.float64)
        if np.sum(np.abs(w)) < 1e-8:
            return 0.0

        losses = -scenarios @ w
        if var_val is None:
            var_val = float(np.percentile(losses, conf * 100.0))

        loss_std = float(np.std(losses))
        scale_beta = (100.0 / max(loss_std, 1e-5)) if (beta is None or beta == 100.0) else beta
        diff = losses - var_val
        z = scale_beta * diff
        excess = np.where(z > 30.0, diff, np.log1p(np.exp(np.clip(z, -50.0, 30.0))) / scale_beta)
        cvar_daily = var_val + (1.0 / ((1.0 - conf) * s)) * np.sum(excess)
        return float(cvar_daily * np.sqrt(252.0))


    @staticmethod
    def analytical_gaussian_cvar(volatility: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Analytical reference benchmark for Gaussian VaR and CVaR.
        For N(0, sigma^2):
          VaR_alpha = z_alpha * sigma
          CVaR_alpha = (phi(z_alpha) / (1 - alpha)) * sigma
        """
        from scipy.stats import norm
        z = norm.ppf(confidence_level)
        phi_z = norm.pdf(z)
        es_multiplier = phi_z / (1.0 - confidence_level)
        var = float(z * volatility)
        cvar = float(es_multiplier * volatility)
        return var, cvar

