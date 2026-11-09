"""
Fractional-Kelly-Inspired Risk Cap Module.

Implements Kelly-inspired preliminary position risk caps:
  kelly_raw_i = mu_robust_i / (sigma_i^2)   (for mu_robust_i > 0, else 0.0)
  k_ceiling_i = fractional_kelly_multiplier * kelly_raw_i

The cap uses Kelly's mu/sigma^2 functional form as a conservative risk-sizing heuristic.
Because mu is a model objective proxy rather than a statistically calibrated
expected-return estimate, the cap is not presented as a theoretically exact
Kelly-optimal fraction.

Guarantees:
1. Strictly serves as an upper bound (risk cap), never the sole allocator.
2. Clamped >= 0.0 (negative or zero edge yields 0.0 cap).
3. Never exceeds hard max_position_weight or per-order dollar cap.
4. Scale-consistent with annualized return and annualized variance.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FractionalKellyCeiling:
    """Calculates fractional-Kelly preliminary allocation ceilings per asset."""

    def __init__(
        self,
        fractional_multiplier: float = 0.25,
        max_position_weight: float = 0.25,
        min_ceiling: float = 0.0,
    ):
        self.fractional_multiplier = fractional_multiplier
        self.max_position_weight = max_position_weight
        self.min_ceiling = min_ceiling

    def compute_ceilings(
        self,
        symbols: List[str],
        robust_mu: Dict[str, float],
        volatilities: Dict[str, float],
        total_equity: float = 100000.0,
        max_order_value: float = 5000.0,
        multiplier_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute preliminary fractional-Kelly capital ceiling for each candidate.
        
        Args:
            symbols: List of candidate symbols
            robust_mu: Robust expected return per symbol (annualized)
            volatilities: Annualized volatility per symbol
            total_equity: Current portfolio total equity (cash + position value)
            max_order_value: Hard dollar cap per order (e.g. $5,000)
            multiplier_override: Optional override for the Kelly multiplier
            
        Returns:
            Dict mapping symbol -> capital weight ceiling in [0.0, max_position_weight]
        """
        multiplier = multiplier_override if multiplier_override is not None else self.fractional_multiplier
        ceilings: Dict[str, float] = {}

        # Dollar cap as a portfolio weight
        dollar_cap_weight = (max_order_value / max(total_equity, 1000.0))

        # Absolute upper bound on individual position weight
        hard_weight_cap = min(self.max_position_weight, dollar_cap_weight)

        for s in symbols:
            mu = robust_mu.get(s, 0.0)
            vol = volatilities.get(s, 0.20)

            # Check validity
            if not np.isfinite(mu) or not np.isfinite(vol) or vol <= 1e-4:
                ceilings[s] = 0.0
                continue

            if mu <= 0.0:
                # No positive edge after uncertainty penalty -> ceiling is 0
                ceilings[s] = 0.0
                continue

            # Annualized Kelly formula: mu / (vol^2)
            var = vol ** 2
            raw_kelly = mu / var

            # Apply fractional multiplier (e.g. 0.25 for quarter-Kelly)
            fractional_kelly = multiplier * raw_kelly

            # Clamp between 0.0 and hard weight cap
            ceiling = float(np.clip(fractional_kelly, self.min_ceiling, hard_weight_cap))
            ceilings[s] = round(ceiling, 6)

        return ceilings
