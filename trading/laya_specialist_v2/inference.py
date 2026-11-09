"""
Financial Specialist V2 Shadow Inference & Pipeline Adapter.

Authoritative Trading Hierarchy:
Market Data -> PIT Universe -> Features -> Alpha/ML -> Existing RegimeEngine ->
Candidate Ranking -> Laya V2 Specialist (Advisory Filter) -> Portfolio Constructor ->
Immutable RiskEngine -> Execution Simulator.

Hard Architectural Invariants:
1. Regime Engine remains authoritative (Laya provides secondary confirmation only).
2. Laya outputs advisory meta-decision (TAKE, WAIT, PASS) and bounded multiplier [0.0, 1.0].
3. Laya never directly selects position sizes or notional quantities.
4. Laya operates strictly in SHADOW MODE by default.
"""

from typing import Dict, Any, Optional
import os
import logging
from dataclasses import dataclass
import numpy as np

from .config import LayaV2Config
from .decision_policy import EconomicDecisionPolicyV2, MetaDecisionOutputV2
from .tasks import SpecialistTasksV2

logger = logging.getLogger(__name__)


class SpecialistInferenceEngineV2:
    """
    Shadow inference adapter for Laya Financial Specialist V2.
    Safely encapsulates model inference and decision policy within the trading loop.
    """

    def __init__(
        self,
        config: Optional[LayaV2Config] = None,
        policy: Optional[EconomicDecisionPolicyV2] = None,
        shadow_mode: bool = True,
    ):
        self.config = config or LayaV2Config()
        self.policy = policy or EconomicDecisionPolicyV2()
        self.shadow_mode = shadow_mode
        self.trainer = None  # Loaded on demand or injected

    def evaluate_candidate_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        sector: str,
        cap_tier: str,
        beta: float,
        features: Dict[str, float],
        regime: str,
        candidate_rank: int = 1,
        baseline_score: float = 0.75,
    ) -> MetaDecisionOutputV2:
        """
        Advisory evaluation of candidate trade setup.
        Constructs de-biased narrative state, predicts forward distribution, and evaluates policy.
        """
        # Build de-biased state narrative (omits raw symbol string)
        ret_1d = features.get("ret_1d", 0.0)
        ret_5d = features.get("ret_5d", 0.0)
        ret_20d = features.get("ret_20d", 0.0)
        rsi = features.get("rsi_14", 50.0)
        adx = features.get("adx_14", 25.0)
        atr_pct = features.get("atr_pct", 0.025)
        vol_20d = features.get("vol_20d", 0.20)
        vix = features.get("vix", 18.0)
        ml_prob = features.get("ml_prob_up", 0.60)
        ml_conf = features.get("ml_confidence", 0.75)
        alpha_t = features.get("alpha_trend", 0.5)
        alpha_m = features.get("alpha_momentum", 0.5)
        alpha_mr = features.get("alpha_mean_reversion", 0.5)

        narrative = (
            f"Candidate: Profile[{sector}, Cap-Tier:{cap_tier}, Beta:{beta:.2f}] {side} proposal at price level {entry_price:.2f}. "
            f"Rank #{candidate_rank}, BaselineScore {baseline_score:.2f}. "
            f"Returns: 1d {ret_1d*100:+.2f}%, 5d {ret_5d*100:+.2f}%, 20d {ret_20d*100:+.2f}%. "
            f"RSI(14) {rsi:.1f}, ADX {adx:.1f}, ATR% {atr_pct*100:.2f}%. "
            f"Regime: {regime}, VIX {vix:.1f}, Vol(20d) {vol_20d*100:.1f}%. "
            f"ML Up Prob {ml_prob:.2f} (Conf {ml_conf:.2f}). "
            f"Alphas: Trend {alpha_t:.2f}, Mom {alpha_m:.2f}, MR {alpha_mr:.2f}."
        )

        # Mock/fast forward probabilities when model is not physically instantiated
        # (Allows unit testing and air-gapped simulation)
        if self.trainer is None or self.trainer.model is None:
            # Deterministic heuristic proxy for lightweight testing
            score = (ret_5d * 50.0) + (baseline_score * 2.0) - (vol_20d * 3.0)
            if "BEAR" in regime or "OFF" in regime:
                score -= 1.5
            p_take = float(1.0 / (1.0 + np.exp(-score)))
            p_pass = float(1.0 / (1.0 + np.exp(score * 1.2)))
            p_wait = max(0.05, 1.0 - p_take - p_pass)
            tot = p_take + p_wait + p_pass
            meta_probs = {"TAKE": p_take / tot, "WAIT": p_wait / tot, "PASS": p_pass / tot}
            outcome_10d = {
                "STRONGLY_NEGATIVE": 0.05,
                "NEGATIVE": 0.15,
                "NEUTRAL": 0.30,
                "POSITIVE": 0.35,
                "STRONGLY_POSITIVE": 0.15,
            }
        else:
            # Full model forward pass
            preds = self.trainer.predict_examples([
                FinancialCandidateExampleV2(
                    example_id="INFERENCE_CALL",
                    symbol_reference=symbol,
                    market="US",
                    sector=sector,
                    cap_tier=cap_tier,
                    timestamp="2026-09-26T14:30:00Z",
                    split="inference",
                    proposed_side=side,
                    entry_price=entry_price,
                    candidate_rank=candidate_rank,
                    baseline_score=baseline_score,
                    expected_holding_horizon=10,
                    estimated_transaction_cost_bps=5.0,
                    estimated_slippage_bps=10.0,
                    state_narrative=narrative,
                    state_features=features,
                    state_hash="",
                    outcomes={},
                    targets_label_a={},
                    targets_label_b={},
                    targets={},
                )
            ])
            meta_probs = preds["INFERENCE_CALL"]["META_DECISION"]
            outcome_10d = preds["INFERENCE_CALL"]["OUTCOME_10BAR"]

        decision = self.policy.evaluate_candidate(
            outcome_10d_probs=outcome_10d,
            meta_decision_probs=meta_probs,
            atr_pct=atr_pct,
        )

        logger.info(
            f"[LAYA V2 SHADOW] Symbol={symbol} Side={side} Action={decision.action} "
            f"Utility={decision.economic_utility_score*100:+.2f}% ScaleMult={decision.position_scale_multiplier}"
        )

        return decision
