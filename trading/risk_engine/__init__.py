"""
Risk Engine — Immutable Security Boundary

The Risk Engine is stronger than the AI. The AI CANNOT override these limits.

This module enforces ALL hard risk constraints. Every trading decision must pass
through RiskEngine validation before execution. The LLM output is a proposal,
never an order.

Chain: AI decision → Portfolio construction → Risk validation → Execution validation → Broker adapter
"""

import importlib.util
from pathlib import Path

from trading.risk_engine.validators import RiskValidator
from trading.risk_engine.risk_state import RiskState

# Load core RiskEngine class from standalone module
_py_file = Path(__file__).parent.parent / "risk_engine.py"
if _py_file.exists():
    _spec = importlib.util.spec_from_file_location("trading._risk_engine_standalone", _py_file)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    RiskEngine = getattr(_mod, "RiskEngine", None)
    RiskApproval = getattr(_mod, "RiskApproval", None)
    RiskViolation = getattr(_mod, "RiskViolation", None)
else:
    RiskEngine = None
    RiskApproval = None
    RiskViolation = None

__all__ = ["RiskValidator", "RiskState", "RiskEngine", "RiskApproval", "RiskViolation"]