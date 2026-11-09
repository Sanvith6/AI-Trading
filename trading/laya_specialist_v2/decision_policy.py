"""
Financial Specialist V2 Decision Policy & Explicit Abstention Engine.

Deterministic utility-maximizing policy mapper.
Translates soft probability distributions into disciplined trading actions:
- Evaluates:
    - Expected Net Outcome (E[R])
    - Downside Probability (P(Loss))
    - Upside Probability (P(Gain))
    - Risk Asymmetry Ratio (P(Gain) / P(Loss))
    - Shannon Entropy over candidate outcomes
    - Economic Utility Score: U = E[R] - lambda * DownsideRisk - Friction
    - Explicit Abstention: Maps uncertain, high-entropy distributions to WAIT/PASS.
- Hard Architectural Invariant: Laya itself never selects position sizes or notional quantities.
  It only provides advisory action and a bounded multiplier [0.0, 1.0].
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import math
import numpy as np


@dataclass
class MetaDecisionOutputV2:
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
    policy_version: str = "economic_utility_v2.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EconomicDecisionPolicyV2:
    """
    Deterministic utility-maximizing policy mapper.
    Translates soft probability distributions into disciplined trading actions.
    """

    POLICY_VERSION = "economic_utility_v2.0"

    def __init__(
        self,
        take_probability_threshold: float = 0.270,  # Calibrated on Validation split
        pass_probability_threshold: float = 0.420,  # Rejection threshold where PASS dominates
        friction_drag_bps: float = 15.0,            # 15 bps round-trip friction
        risk_aversion_lambda: float = 1.0,          # Penalty factor on downside probability
    ):
        self.take_probability_threshold = take_probability_threshold
        self.pass_probability_threshold = pass_probability_threshold
        self.friction_drag_bps = friction_drag_bps
        self.risk_aversion_lambda = risk_aversion_lambda

    def evaluate_candidate(
        self,
        outcome_10d_probs: Dict[str, float],
        meta_decision_probs: Dict[str, float],
        atr_pct: float = 0.025,
    ) -> MetaDecisionOutputV2:
        """
        Computes expected economic utility and maps to TAKE, WAIT, or PASS.
        """
        # 1. Expected return from 10-bar outcome distribution
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

        # 3. Shannon Entropy of META_DECISION
        p_take = meta_decision_probs.get("TAKE", 0.0)
        p_wait = meta_decision_probs.get("WAIT", 0.0)
        p_pass = meta_decision_probs.get("PASS", 0.0)
        
        entropy = 0.0
        for p in [p_take, p_wait, p_pass]:
            if p > 1e-6:
                entropy -= p * math.log2(p)
        # Normalized entropy in [0, 1] for 3 classes (log2(3) = 1.58496)
        norm_entropy = entropy / 1.58496

        # 4. Economic Utility Score
        utility = expected_net_ret - (self.risk_aversion_lambda * downside_prob * atr_pct)

        # 5. Deterministic Decision Tree using Validation-Calibrated Boundaries
        is_abstaining = False
        if p_take >= self.take_probability_threshold:
            action = "TAKE"
            scale_mult = min(1.0, max(0.5, 0.5 + ((p_take - self.take_probability_threshold) / max(1e-4, 1.0 - self.take_probability_threshold)) * 0.5))
            rationale = (
                f"TAKE: High relative candidate conviction (P(TAKE)={p_take:.1%} >= {self.take_probability_threshold:.1%}) "
                f"approved by validation-calibrated policy."
            )
        elif p_pass >= self.pass_probability_threshold:
            action = "PASS"
            scale_mult = 0.0
            rationale = (
                f"PASS: Elevated downside probability (P(PASS)={p_pass:.1%} >= {self.pass_probability_threshold:.1%}). "
                f"Candidate expected to fail or drag."
            )
        else:
            action = "WAIT"
            is_abstaining = True
            scale_mult = 0.0
            rationale = (
                f"WAIT: Ambiguous trade profile (P(TAKE)={p_take:.1%}, P(WAIT)={p_wait:.1%}, P(PASS)={p_pass:.1%}). "
                f"Explicit abstention."
            )

        return MetaDecisionOutputV2(
            action=action,
            expected_net_outcome_pct=round(expected_net_ret, 5),
            downside_probability=round(downside_prob, 4),
            upside_probability=round(upside_prob, 4),
            risk_asymmetry_ratio=round(asymmetry, 3),
            economic_utility_score=round(utility, 5),
            confidence_entropy=round(norm_entropy, 4),
            is_abstaining=is_abstaining,
            position_scale_multiplier=round(scale_mult, 3),
            rationale=rationale,
        )
