"""
Risk Models Module.

Handles:
1. Forecast extraction (mu_i) from upstream alpha score and ML confidence.
2. Forecast uncertainty estimation (uncertainty_i).
3. Conservative robust expected return:
     mu_robust_i = mu_i - uncertainty_penalty * uncertainty_i
4. Volatility estimation and validation (handling NaN, Inf, non-positive, extreme values).
5. Volatility normalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalRiskModel:
    """Extracts expected returns, estimates forecast uncertainties, and validates volatilities."""

    def __init__(
        self,
        uncertainty_penalty: float = 0.50,
        expected_return_scale: float = 0.15,  # Annualized scale for alpha signal
        min_volatility: float = 0.02,
        max_volatility: float = 2.00,
        default_annual_vol: float = 0.20,
    ):
        self.uncertainty_penalty = uncertainty_penalty
        self.expected_return_scale = expected_return_scale
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.default_annual_vol = default_annual_vol

    def extract_forecast(
        self,
        symbols: List[str],
        scores: Optional[Dict[str, float]] = None,
        confidences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Maps upstream alpha score and ML confidence into expected annualized edge mu_i.
        
        Alpha score in [0.10, 0.95] with 0.50 baseline neutral:
          edge = (alpha_score - 0.50) * 2.0 (range ~ [-0.8, 0.9])
        If score is already centered around 0 (e.g. min < 0), use score directly.
        Expected return mu_i = edge * expected_return_scale * ml_conf
        """
        mu_dict: Dict[str, float] = {}
        scores = scores or {}
        confidences = confidences or {}

        # Detect whether scores are [0, 1] probability/score or raw returns
        vals = [scores[s] for s in symbols if s in scores]
        is_probability_scale = True
        if vals and (min(vals) < -0.05 or max(vals) > 1.5):
            is_probability_scale = False

        for s in symbols:
            raw_score = scores.get(s, 0.50 if is_probability_scale else 0.0)
            conf = confidences.get(s, 0.75)

            # Numerical validation
            if raw_score is None or np.isnan(raw_score) or np.isinf(raw_score):
                logger.warning("Invalid score for %s: %s. Setting mu to 0.0", s, raw_score)
                mu_dict[s] = 0.0
                continue

            if conf is None or np.isnan(conf) or np.isinf(conf) or conf <= 0:
                conf = 0.50

            conf = float(np.clip(conf, 0.05, 1.0))

            if is_probability_scale:
                # 0.50 is neutral midpoint
                edge = (raw_score - 0.50) * 2.0
            else:
                edge = raw_score

            mu_i = float(edge * self.expected_return_scale * conf)
            mu_dict[s] = round(mu_i, 6)

        return mu_dict

    def estimate_uncertainty(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
        confidences: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Estimates forecast_uncertainty_proxy for each candidate.
        
        This is a constructed uncertainty proxy:
          u = sqrt(((1-conf)*sigma)^2 + (sigma/sqrt(T))^2)
        Because it is a constructed heuristic proxy rather than an OOF predictive-uncertainty
        calibration, it does not claim calibrated predictive confidence intervals.
        Preserves robust formulation: mu_robust = mu - kappa_unc * u.
        """
        uncertainty_dict: Dict[str, float] = {}
        confidences = confidences or {}

        for s in symbols:
            vol = volatilities.get(s, self.default_annual_vol)
            conf = confidences.get(s, 0.75)
            if conf is None or np.isnan(conf) or np.isinf(conf):
                conf = 0.50
            conf = float(np.clip(conf, 0.05, 1.0))

            # Primary uncertainty: confidence gap scaled by asset volatility
            model_dispersion = (1.0 - conf) * vol

            # Sample dispersion if returns are available
            if returns_df is not None and s in returns_df.columns:
                n_obs = len(returns_df[s].dropna())
                sample_se = vol / np.sqrt(max(n_obs, 10))
            else:
                sample_se = vol * 0.10

            unc = float(np.sqrt(model_dispersion ** 2 + sample_se ** 2))
            uncertainty_dict[s] = round(unc, 6)

        return uncertainty_dict

    # Alias reflecting constructed proxy terminology
    estimate_forecast_uncertainty_proxy = estimate_uncertainty

    def compute_robust_mu(
        self,
        symbols: List[str],
        mu: Dict[str, float],
        uncertainty: Dict[str, float],
        uncertainty_penalty: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculates conservative robust expected return:
        mu_robust_i = mu_i - uncertainty_penalty * uncertainty_i
        """
        penalty = uncertainty_penalty if uncertainty_penalty is not None else self.uncertainty_penalty
        robust_mu: Dict[str, float] = {}

        for s in symbols:
            m = mu.get(s, 0.0)
            u = uncertainty.get(s, 0.0)
            # Conservative haircut: subtract uncertainty penalty
            # For long allocations, positive mu is discounted by uncertainty
            r_mu = m - penalty * u
            robust_mu[s] = round(r_mu, 6)

        return robust_mu

    def validate_and_estimate_volatilities(
        self,
        symbols: List[str],
        volatilities: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Estimates and validates annualized volatilities for symbols.
        Rejects invalid symbols (NaN, Inf, <= 0, or extreme).
        
        Returns:
            Tuple of (valid_volatilities_dict, valid_symbols_list)
        """
        valid_vols: Dict[str, float] = {}
        valid_syms: List[str] = []
        volatilities = volatilities or {}

        for s in symbols:
            vol = None
            if s in volatilities:
                v = volatilities[s]
                if v is None or not np.isfinite(v) or v <= 0:
                    logger.warning("REJECTED: Symbol %s has non-positive or non-finite volatility %s", s, v)
                    continue
                vol = float(v)
            elif returns_df is not None and s in returns_df.columns:
                col = returns_df[s].dropna()
                if len(col) >= 10:
                    std_daily = float(col.std())
                    if np.isfinite(std_daily) and std_daily > 0:
                        vol = std_daily * np.sqrt(252.0)
                    else:
                        logger.warning("REJECTED: Symbol %s has non-positive or non-finite std %s", s, std_daily)
                        continue
                else:
                    logger.warning("REJECTED: Symbol %s has insufficient history (%d bars)", s, len(col))
                    continue
            else:
                vol = self.default_annual_vol

            # Strict bounds validation
            if not np.isfinite(vol) or vol <= 0:
                logger.error("REJECTED: Symbol %s has non-positive or non-finite volatility %s", s, vol)
                continue

            if vol < self.min_volatility:
                logger.warning("Symbol %s volatility %s below minimum %s, clamping", s, vol, self.min_volatility)
                vol = self.min_volatility

            if vol > self.max_volatility:
                logger.warning("REJECTED: Symbol %s volatility %s exceeds maximum ceiling %s", s, vol, self.max_volatility)
                continue

            valid_vols[s] = round(vol, 6)
            valid_syms.append(s)

        return valid_vols, valid_syms
