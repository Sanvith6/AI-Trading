"""
Financial Specialist V2 Choice Tasks & Prompts.

Matches Laya SystemOne choice contract:
1. TRADE_QUALITY (EXCEPTIONAL, STRONG, VALID, WEAK, AVOID)
2. OUTCOME_5BAR (STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE)
3. OUTCOME_10BAR (STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE)
4. OUTCOME_20BAR (STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE)
5. META_DECISION (TAKE, WAIT, PASS)
6. RISK_CONDITION (NORMAL, ELEVATED, DANGEROUS)
"""

from typing import Dict, Any, List


class SpecialistTasksV2:
    """Primary choice question definitions for Laya Financial Specialist V2."""

    TASKS: Dict[str, Dict[str, Any]] = {
        "TRADE_QUALITY": {
            "type": "choice",
            "instructions": "Evaluate the risk-reward setup quality for this candidate trade entry.",
            "criteria": {
                "EXCEPTIONAL": "Setup has outstanding asymmetric expectancy with multi-factor and regime confluence.",
                "STRONG": "Setup has robust edge and favorable risk-reward exceeding trading friction.",
                "VALID": "Setup meets baseline entry criteria with standard positive expectation.",
                "WEAK": "Setup is marginal, high friction, or lacking factor confirmation.",
                "AVOID": "Setup has unfavorable expectancy, negative risk-reward, or breakdown risk."
            }
        },
        "OUTCOME_5BAR": {
            "type": "choice",
            "instructions": "Forecast the cost-adjusted outcome distribution over the next 5 bars.",
            "criteria": {
                "STRONGLY_NEGATIVE": "Net return < -1.5x ATR (severe immediate loss).",
                "NEGATIVE": "Net return between -1.5x and -0.3x ATR (moderate loss).",
                "NEUTRAL": "Net return within +/- 0.3x ATR (sideways/flat).",
                "POSITIVE": "Net return between +0.3x and +1.5x ATR (profitable gain).",
                "STRONGLY_POSITIVE": "Net return > +1.5x ATR (strong impulsive move)."
            }
        },
        "OUTCOME_10BAR": {
            "type": "choice",
            "instructions": "Forecast the cost-adjusted outcome distribution over the next 10 bars (primary holding horizon).",
            "criteria": {
                "STRONGLY_NEGATIVE": "Net return < -1.5x ATR (severe adverse excursion).",
                "NEGATIVE": "Net return between -1.5x and -0.3x ATR (loss).",
                "NEUTRAL": "Net return within +/- 0.3x ATR (consolidation/friction drag).",
                "POSITIVE": "Net return between +0.3x and +1.5x ATR (target achieved).",
                "STRONGLY_POSITIVE": "Net return > +1.5x ATR (trend expansion)."
            }
        },
        "OUTCOME_20BAR": {
            "type": "choice",
            "instructions": "Forecast the cost-adjusted outcome distribution over the next 20 bars.",
            "criteria": {
                "STRONGLY_NEGATIVE": "Net return < -1.5x ATR (persistent breakdown).",
                "NEGATIVE": "Net return between -1.5x and -0.3x ATR (adverse trend).",
                "NEUTRAL": "Net return within +/- 0.3x ATR (rangebound consolidation).",
                "POSITIVE": "Net return between +0.3x and +1.5x ATR (sustained advance).",
                "STRONGLY_POSITIVE": "Net return > +1.5x ATR (extended momentum run)."
            }
        },
        "META_DECISION": {
            "type": "choice",
            "instructions": "Recommend secondary meta-decision given the baseline system's proposed trade entry.",
            "criteria": {
                "TAKE": "Approve trade entry: expected net return meaningfully exceeds downside risk and transaction friction.",
                "WAIT": "Defer entry: setup is consolidating, volatile, or awaiting macro confirmation.",
                "PASS": "Reject trade entry: high probability of adverse excursion or insufficient reward-to-risk."
            }
        },
        "RISK_CONDITION": {
            "type": "choice",
            "instructions": "Provide advisory assessment of prevailing market risk and tail exposure.",
            "criteria": {
                "NORMAL": "Market context is within standard risk parameters and volatility limits.",
                "ELEVATED": "Heightened volatility, macro uncertainty, or sector concentration risks.",
                "DANGEROUS": "Severe systemic stress, liquidation risk, or extreme tail hazard."
            }
        }
    }

    @classmethod
    def get_task_dict(cls, task_name: str) -> Dict[str, Any]:
        """Returns question definition formatted for sequence construction."""
        t_key = task_name.upper()
        if t_key not in cls.TASKS:
            raise KeyError(f"Unknown task: {task_name}. Expected one of {list(cls.TASKS.keys())}")
        info = cls.TASKS[t_key]
        return {
            "t": info["type"],
            "ins": info["instructions"],
            "crit": info["criteria"]
        }

    @classmethod
    def get_task_names(cls) -> List[str]:
        return list(cls.TASKS.keys())
