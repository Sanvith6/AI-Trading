"""
Risk Engine — Immutable Security Boundary

The Risk Engine is stronger than the AI. The AI CANNOT override these limits.

This module enforces ALL hard risk constraints. Every trading decision must pass
through RiskEngine validation before execution. The LLM output is a proposal,
never an order.

Chain: AI decision → Portfolio construction → Risk validation → Execution validation → Broker adapter
"""

from trading.risk_engine.validators import RiskValidator
from trading.risk_engine.risk_state import RiskState

__all__ = ["RiskValidator", "RiskState"]