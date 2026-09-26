"""
Laya Integrated Decision Engine & Candidate Pipeline Orchestrator.

Enforces Directives 1, 2, 3, 5, 6, 9, 10, 11, and 25:
- Preserves the authoritative institutional pipeline:
    Market Data
    ↓
    Dynamic Universe
    ↓
    Feature Engine
    ↓
    Alpha + ML
    ↓
    Existing Regime Engine
    ↓
    Candidate Ranking
    ↓
    Top N Filter (Directive 5)
    ↓
    Laya Secondary Intelligence (Directives 1, 2, 3)
    ↓
    Portfolio Construction
    ↓
    Immutable RiskEngine (Directive 11)
    ↓
    Execution (Paper Only, Directive 10, 22)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

from .schemas import LayaMarketState, LayaDecision
from .client import laya_client
from .policy import LayaPolicyEngine, PolicyEvaluationResult
from .fallback import LayaFailSafeEngine
from .config import laya_config

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    """Complete multi-task evaluation bundle for a single candidate."""
    symbol: str
    market_state: LayaMarketState
    baseline_rank: int
    baseline_score: float
    trade_quality_decision: Optional[LayaDecision] = None
    risk_action_decision: Optional[LayaDecision] = None
    regime_opinion_decision: Optional[LayaDecision] = None
    trade_quality_policy: Optional[PolicyEvaluationResult] = None
    risk_action_policy: Optional[PolicyEvaluationResult] = None
    regime_opinion_policy: Optional[PolicyEvaluationResult] = None
    final_allowed: bool = True
    effective_position_multiplier: float = 1.0
    rejection_reason: Optional[str] = None


class LayaDecisionEngine:
    """
    Central orchestration engine for challenger intelligence.
    Ensures bounded inference volume, bounded risk sizing, and immutable RiskEngine precedence.
    """

    def __init__(self, config=None):
        self.config = config or laya_config
        self.client = laya_client

    def process_candidate_scan(
        self,
        ranked_candidates: List[Dict[str, Any]],
        authoritative_regime: str,
        regime_confidence: float = 0.80,
        disagreement_policy: str = "BASELINE_WINS",
        trade_quality_threshold: Optional[float] = None,
        risk_threshold: Optional[float] = None,
        regime_threshold: Optional[float] = None,
    ) -> List[CandidateEvaluation]:
        """
        Executes Top N bounded evaluation over pre-ranked candidates.
        Candidates below Top N bypass Laya and retain baseline parameters.
        """
        tq_thresh = trade_quality_threshold or self.config.trade_quality_threshold
        r_thresh = risk_threshold or self.config.risk_threshold
        reg_thresh = regime_threshold or self.config.regime_threshold

        evaluations: List[CandidateEvaluation] = []

        # 1. Pipeline filtering: Top N candidates only (Directive 5)
        top_n = self.config.max_candidates_per_scan
        candidates_to_evaluate = ranked_candidates[:top_n]
        remaining_candidates = ranked_candidates[top_n:]

        # Extract market states for Top N
        states: List[LayaMarketState] = [c["market_state"] for c in candidates_to_evaluate]
        tasks = ["TRADE_QUALITY", "RISK_ACTION", "REGIME_OPINION"]

        # 2. Bounded inference across Top N
        inference_results = self.client.evaluate_candidates_bounded(states, tasks)

        # 3. Policy mapping for Top N candidates (Directives 1, 2, 3)
        for idx, (cand, dec_dict) in enumerate(zip(candidates_to_evaluate, inference_results)):
            state = cand["market_state"]
            tq_dec = dec_dict.get("TRADE_QUALITY")
            ra_dec = dec_dict.get("RISK_ACTION")
            reg_dec = dec_dict.get("REGIME_OPINION")

            # Evaluate Trade Quality
            tq_pol = LayaPolicyEngine.evaluate_trade_quality(
                decision=tq_dec,
                threshold=tq_thresh,
                disagreement_policy=disagreement_policy,
                baseline_signal=cand.get("score", 1.0),
            )

            # Evaluate Risk Action
            ra_pol = LayaPolicyEngine.evaluate_risk_action(
                decision=ra_dec,
                threshold=r_thresh,
                reduce_multiplier=self.config.reduce_position_multiplier,
            )

            # Evaluate Regime Secondary Opinion
            reg_pol = LayaPolicyEngine.evaluate_regime_secondary_opinion(
                decision=reg_dec,
                authoritative_regime=authoritative_regime,
                authoritative_confidence=regime_confidence,
                threshold=reg_thresh,
            )

            # Composite policy: Laya can only reduce or block, never increase (Directive 11)
            is_allowed = not (tq_pol.is_blocked or ra_pol.is_blocked)
            effective_mult = min(
                tq_pol.position_size_multiplier,
                ra_pol.position_size_multiplier,
                reg_pol.position_size_multiplier,
            )

            rejection_reason = None
            if not is_allowed:
                rejection_reason = tq_pol.policy_reason if tq_pol.is_blocked else ra_pol.policy_reason

            evaluations.append(CandidateEvaluation(
                symbol=state.symbol,
                market_state=state,
                baseline_rank=idx + 1,
                baseline_score=cand.get("score", 1.0),
                trade_quality_decision=tq_dec,
                risk_action_decision=ra_dec,
                regime_opinion_decision=reg_dec,
                trade_quality_policy=tq_pol,
                risk_action_policy=ra_pol,
                regime_opinion_policy=reg_pol,
                final_allowed=is_allowed,
                effective_position_multiplier=round(effective_mult, 4),
                rejection_reason=rejection_reason,
            ))

        # 4. For remaining candidates beyond Top N, retain baseline without Laya overhead
        for idx, cand in enumerate(remaining_candidates):
            state = cand["market_state"]
            evaluations.append(CandidateEvaluation(
                symbol=state.symbol,
                market_state=state,
                baseline_rank=top_n + idx + 1,
                baseline_score=cand.get("score", 0.0),
                final_allowed=True,
                effective_position_multiplier=1.0,
                rejection_reason="Ranked beyond Top N; bypasses Laya inference",
            ))

        return evaluations


# Global singleton engine
laya_decision_engine = LayaDecisionEngine()
