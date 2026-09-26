"""
Laya Task Definitions & Choice Question Contracts.

Formulates question definitions for the 3 core tasks:
1. `TRADE_QUALITY`: Choices: ENTER, WAIT, REJECT (Directive 5)
2. `RISK_ACTION`: Choices: NORMAL, REDUCE_POSITION, BLOCK (Directive 6)
3. `REGIME_OPINION`: Choices: TRENDING, RANGE_BOUND, HIGH_VOLATILITY, RISK_OFF, UNCERTAIN (Directive 7)
"""

from typing import Dict, Any, List


class LayaTasks:
    """Standardized task specifications matching the local /v1/systemone choice schema."""

    TASKS: Dict[str, Dict[str, Any]] = {
        "TRADE_QUALITY": {
            "type": "choice",
            "instructions": "Evaluate the quantitative setup quality for this candidate trade entry.",
            "criteria": {
                "ENTER": "Setup demonstrates favorable risk-reward alignment and multi-factor confirmation to enter.",
                "WAIT": "Setup is ambiguous, consolidating, or awaiting macro confirmation; defer entry.",
                "REJECT": "Risk is excessive, negative expectancy, or structural breakdown; reject trade."
            }
        },
        "RISK_ACTION": {
            "type": "choice",
            "instructions": "Recommend appropriate risk management action based on portfolio state and market context.",
            "criteria": {
                "NORMAL": "Portfolio risk is within standard tolerances; proceed with standard position sizing.",
                "REDUCE_POSITION": "Elevated market volatility or sector concentration; recommend reducing position size.",
                "BLOCK": "Severe tail risk, macro stress, or critical event; recommend blocking entry."
            }
        },
        "REGIME_OPINION": {
            "type": "choice",
            "instructions": "Provide secondary opinion on the prevailing market regime environment.",
            "criteria": {
                "TRENDING": "Persistent directional momentum with low mean-reverting noise.",
                "RANGE_BOUND": "Consolidating sideways market oscillating between defined boundaries.",
                "HIGH_VOLATILITY": "Erratic intraday price swings with elevated volatility index.",
                "RISK_OFF": "Broad market liquidation and defensive sector outperformance.",
                "UNCERTAIN": "Conflicting cross-asset signals with low directional confidence."
            }
        }
    }

    @classmethod
    def get_task_payload(cls, task_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Constructs questions dictionary payload for the specified tasks."""
        payload = {}
        for t in task_names:
            t_upper = t.upper()
            if t_upper in cls.TASKS:
                payload[t_upper.lower()] = cls.TASKS[t_upper]
        return payload
