"""
Laya Fail-Safe Fallback & Circuit Breaker Engine.

Enforces Directive 9:
If Laya:
- times out
- crashes
- returns malformed data
- returns NaN
- returns invalid probabilities
- loses GPU
- becomes unavailable
- exceeds latency budget

then:
SHADOW: baseline continues unchanged (non-interfering).
ADVISORY: baseline continues using deterministic fallback.
GATE: HOLD / safe configured fallback.

NEVER FAIL OPEN.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import logging

from .schemas import LayaDecision
from .config import laya_config

logger = logging.getLogger(__name__)


@dataclass
class FallbackResolution:
    """Actionable resolution produced when Laya fails or experiences degraded state."""
    mode: str
    action: str  # "CONTINUE_BASELINE", "DETERMINISTIC_FALLBACK", "HOLD_BLOCK"
    position_multiplier: float
    confidence_multiplier: float
    reason: str
    is_safe: bool = True


class LayaCircuitBreaker:
    """
    In-memory circuit breaker protecting trading operations from downstream neural service instability.
    States: CLOSED (normal), OPEN (tripped due to repeated errors), HALF_OPEN (probing).
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self) -> None:
        """Resets breaker on successful inference."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self, error_message: str) -> None:
        """Records failure and trips breaker if threshold exceeded."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        logger.warning(f"Laya failure recorded ({self.failure_count}/{self.failure_threshold}): {error_message}")

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Laya Circuit Breaker TRIPPED to OPEN. Service calls paused for {self.reset_timeout_seconds}s.")

    def can_attempt(self) -> bool:
        """Checks if inference request is permitted to run."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time and datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.reset_timeout_seconds):
                self.state = "HALF_OPEN"
                logger.info("Laya Circuit Breaker transitioned to HALF_OPEN (probing single request).")
                return True
            return False

        if self.state == "HALF_OPEN":
            return True

        return False


# Global singleton breaker
laya_circuit_breaker = LayaCircuitBreaker()


class LayaFailSafeEngine:
    """
    Implements Directive 9 fail-safe behavior based on operational mode.
    Guarantees the system never fails open into an unverified or unhedged position.
    """

    @classmethod
    def resolve_failure(
        cls,
        task: str,
        reason: str,
        mode: Optional[str] = None,
    ) -> FallbackResolution:
        """
        Determines the safe response when Laya fails.
        """
        active_mode = (mode or laya_config.mode).lower()

        logger.warning(f"Resolving Laya failure in '{active_mode}' mode: {reason}")

        if active_mode == "shadow":
            # SHADOW: Baseline continues completely unchanged without modification
            return FallbackResolution(
                mode="shadow",
                action="CONTINUE_BASELINE",
                position_multiplier=1.0,
                confidence_multiplier=1.0,
                reason=f"Shadow mode fallback: baseline unaffected ({reason})",
            )
        elif active_mode == "advisory":
            # ADVISORY: Baseline continues using deterministic fallback (no Laya sizing modification)
            return FallbackResolution(
                mode="advisory",
                action="DETERMINISTIC_FALLBACK",
                position_multiplier=1.0,
                confidence_multiplier=1.0,
                reason=f"Advisory mode fallback: using deterministic baseline rule ({reason})",
            )
        elif active_mode == "gate":
            # GATE: Strict fail-safe HOLD / BLOCK entry. Never fail open!
            return FallbackResolution(
                mode="gate",
                action="HOLD_BLOCK",
                position_multiplier=0.0,
                confidence_multiplier=0.0,
                reason=f"Gate mode safe fallback: trade blocked due to Laya failure ({reason})",
            )
        else:
            # Default safest action: HOLD
            return FallbackResolution(
                mode=active_mode,
                action="HOLD_BLOCK",
                position_multiplier=0.0,
                confidence_multiplier=0.0,
                reason=f"Unrecognized mode '{active_mode}'; enforcing fail-safe block ({reason})",
            )

    @classmethod
    def create_fallback_decision(
        cls,
        task: str,
        state_hash: str,
        error_msg: str,
        latency_ms: float = 0.0,
    ) -> LayaDecision:
        """Creates a standardized fallback LayaDecision object."""
        return LayaDecision(
            task=task,
            choice="FALLBACK",
            decision_probability=0.0,
            selected_choice_probability=0.0,
            probabilities={},
            confidence=0.0,
            latency_ms=latency_ms,
            model_version=laya_config.model_version,
            state_hash=state_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_valid=False,
            fallback_triggered=True,
            validation_error=error_msg,
        )
