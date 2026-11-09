r"""
Financial Specialist Decision Policy & Explicit Abstention Engine.

Enforces Phases 18 and 19:
- Derives deterministic decision policy using expected economic utility.
- Evaluates:
    - Expected Net Outcome ($E[R]$)
    - Downside Probability ($P(\text{Loss})$)
    - Upside Probability ($P(\text{Gain})$)
    - Risk Asymmetry Ratio ($P(\text{Gain}) / P(\text{Loss})$)
    - Expected Economic Utility ($U = E[R] - \lambda \cdot \text{Downside Risk} - \text{Friction}$)
    - Explicit Abstention: Maps uncertain, high-entropy distributions to WAIT/PASS.
- Invariant: Laya itself never selects position sizes or notional quantities.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np


@dataclass
class MetaDecisionOutput:
    """Actionable output from the economic decision policy."""
    action: str  # "TAKE", "WAIT", "PASS"
    expected_net_outcome_pct: float
    downside_probability: float
    upside_probability: float
    risk_asymmetry_ratio: float
    economic_utility_score: float
    confidence_entropy: float
    is_abstaining: bool
    position_scale_multiplier: float  # [0.0, 1.0] bounded multiplier
    rationale: str
    policy_version: str = "economic_utility_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EconomicDecisionPolicy:
    """
    Deterministic utility-maximizing policy mapper.
    Translates soft probability distributions into disciplined trading actions.
    """

    POLICY_VERSION = "economic_utility_v1.0"

    def __init__(
        self,
        min_expected_utility: float = 0.005,  # +0.50% net expected gain
        max_downside_prob: float = 0.40,      # Max 40% loss probability allowed
        min_risk_asymmetry: float = 1.30,     # At least 1.3:1 upside vs downside probability
        max_entropy_for_take: float = 0.90,   # High entropy triggers abstention (WAIT)
        friction_drag_bps: float = 15.0,      # 15 bps round-trip friction
    ):
        self.min_expected_utility = min_expected_utility
        self.max_downside_prob = max_downside_prob
        self.min_risk_asymmetry = min_risk_asymmetry
        self.max_entropy_for_take = max_entropy_for_take
        self.friction_drag_bps = friction_drag_bps

    def evaluate_candidate(
        self,
        outcome_10d_probs: Dict[str, float],
        meta_decision_probs: Dict[str, float],
        atr_pct: float = 0.025,
    ) -> MetaDecisionOutput:
        """
        Computes expected economic utility and maps to TAKE, WAIT, or PASS.
        """
        # 1. Expected return from 10-bar outcome distribution
        # Class midpoints in terms of ATR
        class_weights = {
            "STRONGLY_NEGATIVE": -2.0,
            "NEGATIVE": -0.8,
            "NEUTRAL": 0.0,
            "POSITIVE": +0.8,
            "STRONGLY_POSITIVE": +2.0,
        }

        e_atr_mult = sum(
            outcome_10d_probs.get(k, 0.0) * weight
            for k, weight in class_weights.items()
        )
        expected_raw_ret = e_atr_mult * atr_pct
        friction_cost = self.friction_drag_bps / 10000.0
        expected_net_ret = expected_raw_ret - friction_cost

        # 2. Downside vs Upside Probability
        downside_prob = (
            outcome_10d_probs.get("STRONGLY_NEGATIVE", 0.0)
            + outcome_10d_probs.get("NEGATIVE", 0.0)
        )
        upside_prob = (
            outcome_10d_probs.get("POSITIVE", 0.0)
            + outcome_10d_probs.get("STRONGLY_POSITIVE", 0.0)
        )
        asymmetry = upside_prob / max(1e-4, downside_prob)

        # 3. Normalized Entropy for Abstention (Phase 19)
        meta_p = list(meta_decision_probs.values())
        k = len(meta_p)
        entropy = -sum(p * math.log(max(1e-6, p)) for p in meta_p) / math.log(k) if k > 1 else 0.0

        # 4. Economic Utility: E[R] penalized by downside probability and tail risk
        tail_risk_pen = outcome_10d_probs.get("STRONGLY_NEGATIVE", 0.0) * 0.05
        utility = expected_net_ret - (downside_prob * 0.02) - tail_risk_pen

        # Direct model meta-decision votes
        p_take = meta_decision_probs.get("TAKE", 0.33)
        p_wait = meta_decision_probs.get("WAIT", 0.33)
        p_pass = meta_decision_probs.get("PASS", 0.33)

        # 5. Deterministic Policy Rules
        # Rule A: High uncertainty or neutral outcome -> Abstain (WAIT)
        if entropy > self.max_entropy_for_take or p_wait > max(p_take, p_pass):
            return MetaDecisionOutput(
                action="WAIT",
                expected_net_outcome_pct=round(expected_net_ret * 100, 2),
                downside_probability=round(downside_prob, 4),
                upside_probability=round(upside_prob, 4),
                risk_asymmetry_ratio=round(asymmetry, 2),
                economic_utility_score=round(utility * 100, 3),
                confidence_entropy=round(entropy, 3),
                is_abstaining=True,
                position_scale_multiplier=0.50,  # Scaled down if baseline forces entry
                rationale=f"Explicit Abstention: High entropy ({entropy:.2f}) or WAIT plurality ({p_wait:.2f})",
            )

        # Rule B: Favorable expectancy and strong risk asymmetry -> TAKE
        if (
            utility >= self.min_expected_utility
            and downside_prob <= self.max_downside_prob
            and asymmetry >= self.min_risk_asymmetry
            and p_take >= p_pass
        ):
            return MetaDecisionOutput(
                action="TAKE",
                expected_net_outcome_pct=round(expected_net_ret * 100, 2),
                downside_probability=round(downside_prob, 4),
                upside_probability=round(upside_prob, 4),
                risk_asymmetry_ratio=round(asymmetry, 2),
                economic_utility_score=round(utility * 100, 3),
                confidence_entropy=round(entropy, 3),
                is_abstaining=False,
                position_scale_multiplier=1.0,
                rationale=(
                    f"Favorable utility ({utility*100:+.2f}%) with upside asymmetry ({asymmetry:.2f}:1) "
                    f"and low downside prob ({downside_prob*100:.1f}%)"
                ),
            )

        # Rule C: Inadequate expectancy, excessive risk, or high PASS probability -> PASS
        return MetaDecisionOutput(
            action="PASS",
            expected_net_outcome_pct=round(expected_net_ret * 100, 2),
            downside_probability=round(downside_prob, 4),
            upside_probability=round(upside_prob, 4),
            risk_asymmetry_ratio=round(asymmetry, 2),
            economic_utility_score=round(utility * 100, 3),
            confidence_entropy=round(entropy, 3),
            is_abstaining=False,
            position_scale_multiplier=0.0,
            rationale=(
                f"Negative or inadequate expected utility ({utility*100:+.2f}%) "
                f"or elevated downside prob ({downside_prob*100:.1f}%)"
            ),
        )
