"""
Laya Data Contracts, State Schemas, & Response Validation.

Defines:
1. `LayaMarketState`: Strongly typed, versioned structured input contract (Directive 8).
2. `LayaDecision`: Validated response contract storing decision probabilities and latency (Directives 8, 9).
3. Deterministic state hashing for reproducible historical caching (Directive 4).
4. Strict response validation: finite numbers, probability range [0, 1], sum ~ 1, known choices.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json
import hashlib
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayaMarketState:
    """Versioned point-in-time state contract presented to Laya."""
    market: str
    exchange: str
    symbol: str
    timestamp: str  # ISO 8601 UTC
    price_state: Dict[str, float]
    technical_state: Dict[str, float]
    alpha_state: Dict[str, float]
    ml_state: Dict[str, float]
    regime_state: Dict[str, Any]
    market_context: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    risk_state: Dict[str, Any]
    event_state: Dict[str, Any]
    fundamental_state: Dict[str, Any]
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_state_hash(self) -> str:
        """Deterministic cryptographic hash for caching and audit trails."""
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_narrative_context(self) -> str:
        """Constructs concise structured context string for Laya SystemOne prompt."""
        p = self.price_state
        t = self.technical_state
        m = self.market_context
        r = self.regime_state
        ml = self.ml_state
        f = self.fundamental_state

        return (
            f"Symbol: {self.symbol} ({self.exchange}, {self.market}) at ${p.get('close', 0.0):.2f}. "
            f"Returns: 1d {t.get('ret_1d', 0.0)*100:+.2f}%, 5d {t.get('ret_5d', 0.0)*100:+.2f}%, 20d {t.get('ret_20d', 0.0)*100:+.2f}%. "
            f"RSI(14): {t.get('rsi_14', 50.0):.1f}, Vol(20d): {t.get('vol_20d', 0.20)*100:.1f}%. "
            f"Market Context: SPY trend {m.get('market_trend', 'UNKNOWN')}, VIX {m.get('vix_level', 16.0):.1f}, Risk Score {m.get('risk_on_off_score', 0.0):+.2f}. "
            f"Regime: {r.get('current_regime', 'UNKNOWN')} (Confidence: {r.get('confidence', 0.5):.2f}). "
            f"ML Meta-Model: Up Prob {ml.get('ml_prob_up', 0.5):.2f}, Conf {ml.get('ml_confidence', 0.5):.2f}. "
            f"Fundamentals: Quality Score {f.get('f_quality_score', 0.5):.2f}, Operating Margin {f.get('f_operating_margin', 0.2)*100:.1f}%. "
            f"Portfolio: Exposure {self.portfolio_state.get('exposure_pct', 0.0)*100:.1f}%, Drawdown {self.portfolio_state.get('drawdown_pct', 0.0)*100:.1f}%."
        )


@dataclass
class LayaDecision:
    """Validated structured decision produced by Laya."""
    task: str  # "TRADE_QUALITY", "RISK_ACTION", "REGIME_OPINION"
    choice: str  # Selected discrete option
    decision_probability: float  # Raw model probability for the selected choice (Directive 8)
    selected_choice_probability: float  # Alias for independent calibration tests
    probabilities: Dict[str, float]  # Full probability distribution across all choices
    confidence: float  # Model reported confidence
    latency_ms: float  # Inference latency in milliseconds
    model_version: str
    state_hash: str
    timestamp: str
    is_valid: bool = True
    fallback_triggered: bool = False
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResponseValidator:
    """Strict response validator ensuring numerical stability and fail-safe handling (Directive 9)."""

    VALID_CHOICES_BY_TASK = {
        "TRADE_QUALITY": {"ENTER", "WAIT", "REJECT"},
        "RISK_ACTION": {"NORMAL", "REDUCE_POSITION", "BLOCK"},
        "REGIME_OPINION": {"TRENDING", "RANGE_BOUND", "HIGH_VOLATILITY", "RISK_OFF", "UNCERTAIN"},
    }

    @classmethod
    def validate(
        cls,
        task: str,
        raw_answer: Dict[str, Any],
        model_version: str,
        state_hash: str,
        latency_ms: float,
    ) -> LayaDecision:
        """
        Validates raw response dictionary from Laya service.
        Rejects NaNs, infinities, out-of-range probabilities, or invalid choices.
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        expected_choices = cls.VALID_CHOICES_BY_TASK.get(task, set())

        if not isinstance(raw_answer, dict):
            return LayaDecision(
                task=task, choice="FALLBACK", decision_probability=0.0,
                selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                timestamp=now_ts, is_valid=False, fallback_triggered=True,
                validation_error=f"Expected answer dict, got {type(raw_answer).__name__}"
            )

        choice = str(raw_answer.get("choice", "")).strip().upper()
        if choice not in expected_choices:
            return LayaDecision(
                task=task, choice="FALLBACK", decision_probability=0.0,
                selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                timestamp=now_ts, is_valid=False, fallback_triggered=True,
                validation_error=f"Invalid choice '{choice}' for task {task}. Expected one of {expected_choices}"
            )

        raw_probs = raw_answer.get("probabilities", {})
        if not isinstance(raw_probs, dict):
            return LayaDecision(
                task=task, choice="FALLBACK", decision_probability=0.0,
                selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                timestamp=now_ts, is_valid=False, fallback_triggered=True,
                validation_error="Missing or non-dict 'probabilities' field"
            )

        # Validate numerical soundness of probabilities
        clean_probs = {}
        for k, v in raw_probs.items():
            k_clean = str(k).strip().upper()
            try:
                v_float = float(v)
            except (ValueError, TypeError):
                return LayaDecision(
                    task=task, choice="FALLBACK", decision_probability=0.0,
                    selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                    latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                    timestamp=now_ts, is_valid=False, fallback_triggered=True,
                    validation_error=f"Non-numeric probability for '{k}': {v}"
                )

            if math.isnan(v_float) or math.isinf(v_float):
                return LayaDecision(
                    task=task, choice="FALLBACK", decision_probability=0.0,
                    selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                    latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                    timestamp=now_ts, is_valid=False, fallback_triggered=True,
                    validation_error=f"NaN or Inf probability encountered for '{k}'"
                )

            if v_float < -0.01 or v_float > 1.01:
                return LayaDecision(
                    task=task, choice="FALLBACK", decision_probability=0.0,
                    selected_choice_probability=0.0, probabilities={}, confidence=0.0,
                    latency_ms=latency_ms, model_version=model_version, state_hash=state_hash,
                    timestamp=now_ts, is_valid=False, fallback_triggered=True,
                    validation_error=f"Probability out of bounds [0, 1] for '{k}': {v_float}"
                )

            clean_probs[k_clean] = max(0.0, min(1.0, v_float))

        prob_sum = sum(clean_probs.values())
        if prob_sum < 0.85 or prob_sum > 1.15:
            logger.warning(f"Probabilities for task {task} sum to {prob_sum:.4f} (expected ~1.0). Normalizing.")
            clean_probs = {k: v / max(prob_sum, 1e-6) for k, v in clean_probs.items()}

        selected_prob = clean_probs.get(choice, float(raw_answer.get("answer_confidence", 0.5)))
        raw_conf = float(raw_answer.get("confidence", selected_prob))
        clean_conf = max(0.0, min(1.0, raw_conf)) if not (math.isnan(raw_conf) or math.isinf(raw_conf)) else 0.5

        return LayaDecision(
            task=task,
            choice=choice,
            decision_probability=round(selected_prob, 4),
            selected_choice_probability=round(selected_prob, 4),
            probabilities={k: round(v, 4) for k, v in clean_probs.items()},
            confidence=round(clean_conf, 4),
            latency_ms=round(latency_ms, 2),
            model_version=model_version,
            state_hash=state_hash,
            timestamp=now_ts,
            is_valid=True,
            fallback_triggered=False,
            validation_error=None,
        )
