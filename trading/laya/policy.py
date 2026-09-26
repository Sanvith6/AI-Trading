"""
Laya Bounded Deterministic Policy Engine.

Enforces Directives 1, 2, 11, and 12:
1. Regime Engine remains authoritative (Directive 1):
   Laya provides secondary opinion/confirmation only. Laya cannot directly replace or override RegimeEngine.
2. Laya must NOT determine position size (Directive 2):
   Maps RISK_ACTION:
     NORMAL -> 1.0
     REDUCE_POSITION -> configured multiplier (e.g., 0.50)
     BLOCK -> 0.0
   Pipeline: Laya recommendation -> bounded policy -> Portfolio Constructor -> RiskEngine.
   Laya cannot directly select dollar/notional quantity.
3. Precedence hierarchy (Directive 11):
   RiskEngine > Portfolio constraints > Laya recommendation.
   Laya can only reduce or block risk, never increase it.
4. StrategyDNA disagreement policies (Directive 12):
   BASELINE_WINS, REDUCE_CONFIDENCE, HALVE_SIZE.
   Under no circumstances is "Laya overrides risk" allowed.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

from .schemas import LayaDecision
from .config import laya_config

logger = logging.getLogger(__name__)


# Supported disagreement policies
LEGAL_DISAGREEMENT_POLICIES = ["BASELINE_WINS", "REDUCE_CONFIDENCE", "HALVE_SIZE"]


@dataclass
class PolicyEvaluationResult:
    """Bounded, deterministic evaluation result from Laya intelligence."""
    task: str
    laya_choice: str
    laya_probability: float
    confidence_multiplier: float  # Multiplier on alpha/signal confidence [0.0, 1.0]
    position_size_multiplier: float  # Multiplier on target weight [0.0, 1.0] (never > 1.0)
    is_blocked: bool
    policy_reason: str
    disagreement_detected: bool
    effective_regime_opinion: Optional[str] = None


class LayaPolicyEngine:
    """
    Deterministic rule-bound policy mapper that bridges neural output to immutable risk controls.
    Guarantees Laya never possesses sizing or broker authority.
    """

    @classmethod
    def evaluate_trade_quality(
        cls,
        decision: LayaDecision,
        threshold: float = laya_config.trade_quality_threshold,
        disagreement_policy: str = "BASELINE_WINS",
        baseline_signal: float = 1.0,
    ) -> PolicyEvaluationResult:
        """
        Evaluates TRADE_QUALITY decision against task-specific threshold.
        Choices: ENTER, WAIT, REJECT.
        """
        if not decision.is_valid or decision.fallback_triggered:
            # Safe fallback: Non-interfering baseline continuation
            return PolicyEvaluationResult(
                task="TRADE_QUALITY",
                laya_choice=decision.choice,
                laya_probability=decision.decision_probability,
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason="Laya fallback triggered; baseline signal retained",
                disagreement_detected=False,
            )

        prob = decision.decision_probability
        choice = decision.choice.upper()

        # Check threshold qualification
        meets_threshold = prob >= threshold

        if choice == "ENTER" and meets_threshold:
            return PolicyEvaluationResult(
                task="TRADE_QUALITY",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason=f"Laya ENTER confirmed (prob={prob:.3f} >= thresh={threshold:.2f})",
                disagreement_detected=False,
            )
        elif choice == "ENTER" and not meets_threshold:
            # Low confidence approval -> apply disagreement policy
            disagreement = True
            pos_mult, conf_mult = cls._apply_disagreement(disagreement_policy)
            return PolicyEvaluationResult(
                task="TRADE_QUALITY",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=conf_mult,
                position_size_multiplier=pos_mult,
                is_blocked=False,
                policy_reason=f"Laya ENTER below threshold ({prob:.3f} < {threshold:.2f}); applied {disagreement_policy}",
                disagreement_detected=disagreement,
            )
        elif choice == "WAIT":
            # Defer / reduce size
            pos_mult = 0.5 if disagreement_policy == "HALVE_SIZE" else (1.0 if disagreement_policy == "BASELINE_WINS" else 0.75)
            conf_mult = 0.7 if disagreement_policy == "REDUCE_CONFIDENCE" else 1.0
            return PolicyEvaluationResult(
                task="TRADE_QUALITY",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=conf_mult,
                position_size_multiplier=pos_mult,
                is_blocked=(disagreement_policy not in ["BASELINE_WINS", "REDUCE_CONFIDENCE", "HALVE_SIZE"]),
                policy_reason=f"Laya recommended WAIT (prob={prob:.3f})",
                disagreement_detected=True,
            )
        else:  # REJECT
            if disagreement_policy == "BASELINE_WINS":
                return PolicyEvaluationResult(
                    task="TRADE_QUALITY",
                    laya_choice=choice,
                    laya_probability=prob,
                    confidence_multiplier=1.0,
                    position_size_multiplier=1.0,
                    is_blocked=False,
                    policy_reason="Laya REJECT overridden by BASELINE_WINS policy",
                    disagreement_detected=True,
                )
            elif disagreement_policy == "HALVE_SIZE":
                return PolicyEvaluationResult(
                    task="TRADE_QUALITY",
                    laya_choice=choice,
                    laya_probability=prob,
                    confidence_multiplier=0.5,
                    position_size_multiplier=0.5,
                    is_blocked=False,
                    policy_reason="Laya REJECT scaled down by HALVE_SIZE policy",
                    disagreement_detected=True,
                )
            else:
                # Default safety: block trade entry
                return PolicyEvaluationResult(
                    task="TRADE_QUALITY",
                    laya_choice=choice,
                    laya_probability=prob,
                    confidence_multiplier=0.0,
                    position_size_multiplier=0.0,
                    is_blocked=True,
                    policy_reason=f"Laya REJECT enforced (prob={prob:.3f})",
                    disagreement_detected=True,
                )

    @classmethod
    def evaluate_risk_action(
        cls,
        decision: LayaDecision,
        threshold: float = laya_config.risk_threshold,
        reduce_multiplier: float = laya_config.reduce_position_multiplier,
    ) -> PolicyEvaluationResult:
        """
        Evaluates RISK_ACTION decision (Directive 2).
        NORMAL -> 1.0x
        REDUCE_POSITION -> configured multiplier (e.g. 0.5x, max 1.0x)
        BLOCK -> 0.0x (is_blocked = True)
        """
        if not decision.is_valid or decision.fallback_triggered:
            return PolicyEvaluationResult(
                task="RISK_ACTION",
                laya_choice=decision.choice,
                laya_probability=decision.decision_probability,
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason="Laya fallback triggered; standard sizing preserved",
                disagreement_detected=False,
            )

        choice = decision.choice.upper()
        prob = decision.decision_probability

        # Enforce bounded multiplier: never allow > 1.0 (Directive 11: Laya must never increase exposure)
        bounded_reduce = max(0.10, min(0.90, reduce_multiplier))

        if choice == "NORMAL":
            return PolicyEvaluationResult(
                task="RISK_ACTION",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason=f"Laya NORMAL risk confirmed (prob={prob:.3f})",
                disagreement_detected=False,
            )
        elif choice == "REDUCE_POSITION":
            return PolicyEvaluationResult(
                task="RISK_ACTION",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=bounded_reduce,
                position_size_multiplier=bounded_reduce,
                is_blocked=False,
                policy_reason=f"Laya REDUCE_POSITION applied ({bounded_reduce:.2f}x multiplier, prob={prob:.3f})",
                disagreement_detected=True,
            )
        elif choice == "BLOCK":
            return PolicyEvaluationResult(
                task="RISK_ACTION",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=0.0,
                position_size_multiplier=0.0,
                is_blocked=True,
                policy_reason=f"Laya BLOCK recommendation enforced (prob={prob:.3f})",
                disagreement_detected=True,
            )
        else:
            # Unknown choice -> fail-safe default (never fail open)
            return PolicyEvaluationResult(
                task="RISK_ACTION",
                laya_choice=choice,
                laya_probability=prob,
                confidence_multiplier=0.5,
                position_size_multiplier=0.5,
                is_blocked=False,
                policy_reason=f"Unknown choice '{choice}'; fail-safe 0.5x applied",
                disagreement_detected=True,
            )

    @classmethod
    def evaluate_regime_secondary_opinion(
        cls,
        decision: LayaDecision,
        authoritative_regime: str,
        authoritative_confidence: float = 0.80,
        threshold: float = laya_config.regime_threshold,
    ) -> PolicyEvaluationResult:
        """
        Evaluates REGIME_OPINION as a secondary opinion/confirmation (Directive 1).
        The deterministic existing RegimeEngine remains authoritative.
        Laya may: agree, disagree, reduce confidence, recommend HOLD.
        Laya may NOT directly replace or override the existing regime engine.
        """
        if not decision.is_valid or decision.fallback_triggered:
            return PolicyEvaluationResult(
                task="REGIME_OPINION",
                laya_choice=decision.choice,
                laya_probability=decision.decision_probability,
                confidence_multiplier=authoritative_confidence,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason="Laya fallback triggered; authoritative regime preserved",
                disagreement_detected=False,
                effective_regime_opinion=authoritative_regime,
            )

        laya_opinion = decision.choice.upper()
        prob = decision.decision_probability
        auth_upper = authoritative_regime.upper()

        # Mapping for agreement check
        # Existing regimes: STRONG_BULL, WEAK_BULL, SIDEWAYS, WEAK_BEAR, STRONG_BEAR, HIGH_VOLATILITY, LOW_VOLATILITY, RISK_ON, RISK_OFF
        # Laya choices: TRENDING, RANGE_BOUND, HIGH_VOLATILITY, RISK_OFF, UNCERTAIN
        is_agreement = False
        if laya_opinion == "TRENDING" and ("BULL" in auth_upper or "TREND" in auth_upper):
            is_agreement = True
        elif laya_opinion == "RANGE_BOUND" and ("SIDEWAYS" in auth_upper or "LOW_VOL" in auth_upper):
            is_agreement = True
        elif laya_opinion == "HIGH_VOLATILITY" and "HIGH_VOLATILITY" in auth_upper:
            is_agreement = True
        elif laya_opinion == "RISK_OFF" and ("BEAR" in auth_upper or "RISK_OFF" in auth_upper):
            is_agreement = True

        if is_agreement:
            # Agreement reinforces confidence slightly (capped at 1.0)
            combined_conf = min(1.0, authoritative_confidence + 0.05)
            return PolicyEvaluationResult(
                task="REGIME_OPINION",
                laya_choice=laya_opinion,
                laya_probability=prob,
                confidence_multiplier=combined_conf,
                position_size_multiplier=1.0,
                is_blocked=False,
                policy_reason=f"Laya confirms authoritative regime '{authoritative_regime}' (prob={prob:.3f})",
                disagreement_detected=False,
                effective_regime_opinion=authoritative_regime,
            )
        else:
            # Disagreement: RegimeEngine remains authoritative, but Laya reduces confidence
            if laya_opinion in ["RISK_OFF", "HIGH_VOLATILITY"]:
                # High severity disagreement: recommend reduced sizing / hold
                reduced_conf = max(0.30, authoritative_confidence * 0.70)
                pos_mult = 0.75
                reason = f"Laya signals {laya_opinion} against authoritative {authoritative_regime}; confidence reduced"
            elif laya_opinion == "UNCERTAIN":
                reduced_conf = max(0.40, authoritative_confidence * 0.85)
                pos_mult = 0.90
                reason = f"Laya uncertain on {authoritative_regime}; slight confidence reduction"
            else:
                reduced_conf = max(0.50, authoritative_confidence * 0.80)
                pos_mult = 1.0
                reason = f"Laya expresses secondary opinion {laya_opinion} vs authoritative {authoritative_regime}"

            return PolicyEvaluationResult(
                task="REGIME_OPINION",
                laya_choice=laya_opinion,
                laya_probability=prob,
                confidence_multiplier=reduced_conf,
                position_size_multiplier=pos_mult,
                is_blocked=False,
                policy_reason=reason,
                disagreement_detected=True,
                effective_regime_opinion=authoritative_regime,  # Authoritative engine stays!
            )

    @classmethod
    def apply_risk_engine_precedence(
        cls,
        laya_eval: PolicyEvaluationResult,
        risk_engine_allowed: bool,
        portfolio_target_weight: float,
    ) -> Tuple[bool, float, str]:
        """
        Enforces Directive 11:
        Precedence: RiskEngine > Portfolio constraints > Laya recommendation.

        Examples:
          Laya = NORMAL, RiskEngine = BLOCK -> BLOCK
          Laya = REDUCE, RiskEngine = BLOCK -> BLOCK
          Laya = BLOCK,  RiskEngine = ALLOW -> BLOCK
          Laya must NEVER increase exposure.
        """
        # If RiskEngine rejects, ALWAYS BLOCK
        if not risk_engine_allowed:
            return False, 0.0, "RiskEngine authoritative rejection (overrides all)"

        # If Laya blocked the trade, BLOCK
        if laya_eval.is_blocked or laya_eval.position_size_multiplier <= 0.0:
            return False, 0.0, f"Laya policy block: {laya_eval.policy_reason}"

        # Otherwise, scale portfolio target weight strictly by Laya multiplier (<= 1.0)
        final_multiplier = min(1.0, max(0.0, laya_eval.position_size_multiplier))
        effective_weight = portfolio_target_weight * final_multiplier

        return True, effective_weight, f"Approved with Laya size multiplier {final_multiplier:.2f} ({laya_eval.policy_reason})"

    @staticmethod
    def _apply_disagreement(policy: str) -> Tuple[float, float]:
        """Helper to return (position_mult, confidence_mult) under disagreement."""
        if policy == "HALVE_SIZE":
            return 0.50, 0.50
        elif policy == "REDUCE_CONFIDENCE":
            return 0.75, 0.60
        else:  # BASELINE_WINS
            return 1.0, 1.0
