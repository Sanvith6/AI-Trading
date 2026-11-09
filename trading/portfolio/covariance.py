"""
Shrinkage Covariance Module.

Implements robust covariance estimation using Ledoit-Wolf shrinkage,
OAS (Oracle Approximating Shrinkage), and spectral regularized PSD projection.

Guarantees:
1. Symmetric matrix: Sigma = 0.5 * (Sigma + Sigma.T)
2. Positive semi-definite / Positive definite (all eigenvalues >= eps)
3. Finite values (no NaN, Inf)
4. Deterministic and numerically stable
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
import logging

logger = logging.getLogger(__name__)


class ShrinkageCovarianceEstimator:
    """Estimates robust, positive-definite covariance matrices via Ledoit-Wolf shrinkage."""

    def __init__(
        self,
        method: str = "ledoit_wolf",
        min_eigenvalue: float = 1e-6,
        annualize_factor: float = 252.0,
    ):
        self.method = method
        self.min_eigenvalue = min_eigenvalue
        self.annualize_factor = annualize_factor

    def estimate(
        self,
        returns_df: pd.DataFrame,
        symbols: Optional[list] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate shrinkage covariance matrix from returns.
        
        Args:
            returns_df: DataFrame of asset returns (T x N)
            symbols: Optional list of symbols to filter / order columns
            
        Returns:
            Tuple of (annualized_covariance_matrix, shrinkage_intensity)
        """
        if returns_df is None or returns_df.empty:
            raise ValueError("Empty or None returns DataFrame passed to covariance estimator")

        cols = [s for s in symbols if s in returns_df.columns] if symbols else list(returns_df.columns)
        if not cols:
            raise ValueError("No matching symbols found in returns DataFrame")

        if returns_df[cols].isna().all().any():
            raise ValueError("Returns data contains column(s) with all NaN values")

        data = returns_df[cols].copy()
        
        # Check for invalid values
        if data.isna().any().any() or np.isinf(data.values).any():
            data = data.ffill().bfill().fillna(0.0)
            if data.isna().any().any() or np.isinf(data.values).any():
                raise ValueError("Returns data contains non-finite values that could not be cleaned")

        n_samples, n_features = data.shape

        if n_samples < 5:
            # Insufficient samples: fall back to diagonal sample variance with conservative correlation
            variances = data.var().values
            variances = np.maximum(variances, 1e-6)
            std = np.sqrt(variances)
            cov = np.outer(std, std) * 0.20  # conservative 0.20 cross-correlation
            np.fill_diagonal(cov, variances)
            shrinkage = 1.0
        else:
            X = data.values
            if self.method == "oas" and n_samples > 2:
                model = OAS().fit(X)
                cov = model.covariance_
                shrinkage = float(getattr(model, "shrinkage_", 0.0))
            else:
                model = LedoitWolf().fit(X)
                cov = model.covariance_
                shrinkage = float(getattr(model, "shrinkage_", 0.0))

        # Annualize
        cov = cov * self.annualize_factor

        # Guarantee symmetry
        cov = 0.5 * (cov + cov.T)

        # Spectral projection to guarantee strict positive definiteness
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals_clipped = np.maximum(eigvals, self.min_eigenvalue)
        cov_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        # Re-symmetrize after reconstruction
        cov_pd = 0.5 * (cov_pd + cov_pd.T)

        # Final numerical safety check
        if not np.isfinite(cov_pd).all():
            raise ValueError("Covariance matrix contains non-finite elements after shrinkage")

        return cov_pd, shrinkage

    def build_from_volatilities(
        self,
        volatilities: dict,
        symbols: list,
        cross_correlation: float = 0.25,
    ) -> np.ndarray:
        """
        Synthesize a regularized shrinkage covariance matrix from annualized volatilities
        when full historical return series is unavailable.
        """
        n = len(symbols)
        vols = np.array([max(1e-4, float(volatilities.get(s, 0.20))) for s in symbols])
        
        # Base correlation matrix: identity shrunk towards cross_correlation
        corr = np.full((n, n), cross_correlation)
        np.fill_diagonal(corr, 1.0)
        
        cov = np.outer(vols, vols) * corr
        cov = 0.5 * (cov + cov.T)
        
        # Ensure positive definiteness
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals_clipped = np.maximum(eigvals, self.min_eigenvalue)
        cov_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        cov_pd = 0.5 * (cov_pd + cov_pd.T)
        return cov_pd
