"""
Laya Forward Counterfactual Analysis & Multi-Horizon Outcome Tracking.

Enforces Directives 7 & 16:
- Forward counterfactual evaluation uses explicit horizons:
    e.g., 5 bars, 10 bars, 20 bars (and actual holding period).
- For every decision and disagreement, records:
    1. future_return (at horizon h)
    2. MFE (Maximum Favorable Excursion)
    3. MAE (Maximum Adverse Excursion)
    4. future_volatility
    5. future_drawdown
    6. benchmark_relative_return (excess return over SPY)
- Classifies decision quality:
    - BAD TRADE AVOIDED: Laya blocked/reduced a trade that subsequently lost money (Value Add).
    - GOOD TRADE MISSED: Laya blocked/reduced a trade that subsequently made money (Opportunity Cost).
    - BAD TRADE ACCEPTED: Laya approved a trade that subsequently lost money (False Approval).
    - GOOD TRADE ACCEPTED: Laya approved a trade that subsequently made money.
- Calculates:
    - False Approval Rate
    - False Rejection Rate
    - Laya Disagreement Rate
    - Laya Abstention Rate
    - Laya Decision Stability
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class HorizonOutcome:
    """Outcome metrics measured over an explicit forward horizon."""
    horizon_bars: int
    future_return: float
    mfe: float  # Maximum Favorable Excursion (max high relative to entry)
    mae: float  # Maximum Adverse Excursion (min low relative to entry)
    future_volatility: float
    future_drawdown: float
    benchmark_relative_return: float


@dataclass
class CounterfactualRecord:
    """Detailed record linking baseline decision, Laya decision, and forward reality."""
    timestamp: str
    symbol: str
    baseline_action: str  # "ENTER", "HOLD", "BLOCK"
    laya_task: str
    laya_choice: str
    laya_probability: float
    is_disagreement: bool
    policy_action_taken: str  # "APPROVED", "REDUCED", "BLOCKED"
    entry_price: float
    horizons: Dict[str, HorizonOutcome] = field(default_factory=dict)
    classification: str = "PENDING"  # "BAD_TRADE_AVOIDED", "GOOD_TRADE_MISSED", "BAD_TRADE_ACCEPTED", "GOOD_TRADE_ACCEPTED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizons"] = {k: asdict(v) for k, v in self.horizons.items()}
        return d


class LayaCounterfactualEngine:
    """Computes rigorous counterfactual metrics and decision quality tables."""

    DEFAULT_HORIZONS = [5, 10, 20]

    @classmethod
    def compute_horizon_outcome(
        cls,
        entry_price: float,
        future_closes: List[float],
        future_highs: List[float],
        future_lows: List[float],
        benchmark_closes: Optional[List[float]] = None,
        horizon_bars: int = 10,
    ) -> HorizonOutcome:
        """Computes return, MFE, MAE, volatility, drawdown, and benchmark relative return."""
        if entry_price <= 0.0 or not future_closes:
            return HorizonOutcome(
                horizon_bars=horizon_bars,
                future_return=0.0,
                mfe=0.0,
                mae=0.0,
                future_volatility=0.0,
                future_drawdown=0.0,
                benchmark_relative_return=0.0,
            )

        n = min(len(future_closes), horizon_bars)
        sub_closes = future_closes[:n]
        sub_highs = future_highs[:n] if future_highs else sub_closes
        sub_lows = future_lows[:n] if future_lows else sub_closes

        exit_price = sub_closes[-1]
        f_ret = (exit_price - entry_price) / entry_price

        # MFE: highest point reached / entry
        max_high = max(sub_highs)
        mfe = max(0.0, (max_high - entry_price) / entry_price)

        # MAE: lowest point reached / entry (represented as positive or negative drawdown)
        min_low = min(sub_lows)
        mae = min(0.0, (min_low - entry_price) / entry_price)

        # Rolling volatility over the forward window
        if len(sub_closes) > 1:
            rets = [
                (sub_closes[i] - sub_closes[i - 1]) / sub_closes[i - 1]
                for i in range(1, len(sub_closes))
            ]
            f_vol = float(np.std(rets) * np.sqrt(252))
        else:
            f_vol = 0.0

        # Maximum drawdown within the horizon
        peak = entry_price
        max_dd = 0.0
        for p in sub_closes:
            if p > peak:
                peak = p
            dd = (p - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Benchmark excess return
        bm_ret = 0.0
        if benchmark_closes and len(benchmark_closes) >= n and benchmark_closes[0] > 0:
            bm_ret = (benchmark_closes[n - 1] - benchmark_closes[0]) / benchmark_closes[0]
        bm_excess = f_ret - bm_ret

        return HorizonOutcome(
            horizon_bars=horizon_bars,
            future_return=round(f_ret, 4),
            mfe=round(mfe, 4),
            mae=round(mae, 4),
            future_volatility=round(f_vol, 4),
            future_drawdown=round(max_dd, 4),
            benchmark_relative_return=round(bm_excess, 4),
        )

    @classmethod
    def classify_decision(
        cls,
        baseline_action: str,
        policy_action_taken: str,
        future_return: float,
    ) -> str:
        """
        Classifies the decision quadrant at the primary horizon (Directive 7):
        - BAD TRADE AVOIDED: Trade blocked/reduced, future return <= 0. (Positive value add)
        - GOOD TRADE MISSED: Trade blocked/reduced, future return > 0. (Opportunity cost)
        - BAD TRADE ACCEPTED: Trade approved, future return <= 0. (False approval)
        - GOOD TRADE ACCEPTED: Trade approved, future return > 0. (True positive)
        """
        is_blocked_or_reduced = policy_action_taken in ["BLOCKED", "REDUCED"]

        if is_blocked_or_reduced:
            if future_return <= 0.0:
                return "BAD_TRADE_AVOIDED"
            else:
                return "GOOD_TRADE_MISSED"
        else:
            if future_return <= 0.0:
                return "BAD_TRADE_ACCEPTED"
            else:
                return "GOOD_TRADE_ACCEPTED"

    @classmethod
    def evaluate_decision_quality_metrics(
        cls,
        records: List[CounterfactualRecord],
        primary_horizon_key: str = "10_bars",
    ) -> Dict[str, Any]:
        """
        Computes aggregate decision quality metrics (Directive 16):
        - False Approval Rate
        - False Rejection Rate
        - Bad Trades Avoided vs Good Trades Missed
        - Laya Disagreement Rate
        - Laya Abstention Rate
        - Laya Decision Stability
        """
        if not records:
            return {
                "total_records": 0,
                "disagreement_rate": 0.0,
                "abstention_rate": 0.0,
                "bad_trades_avoided_count": 0,
                "good_trades_missed_count": 0,
                "bad_trades_accepted_count": 0,
                "good_trades_accepted_count": 0,
                "false_approval_rate": 0.0,
                "false_rejection_rate": 0.0,
                "net_avoidance_ratio": 0.0,
                "decision_stability": 1.0,
            }

        total = len(records)
        disagreements = sum(1 for r in records if r.is_disagreement)
        abstentions = sum(1 for r in records if r.laya_choice in ["WAIT", "UNCERTAIN"])

        counts = {
            "BAD_TRADE_AVOIDED": 0,
            "GOOD_TRADE_MISSED": 0,
            "BAD_TRADE_ACCEPTED": 0,
            "GOOD_TRADE_ACCEPTED": 0,
        }

        for r in records:
            outcome = r.horizons.get(primary_horizon_key)
            ret = outcome.future_return if outcome else 0.0
            classification = cls.classify_decision(r.baseline_action, r.policy_action_taken, ret)
            r.classification = classification
            counts[classification] = counts.get(classification, 0) + 1

        total_accepted = counts["GOOD_TRADE_ACCEPTED"] + counts["BAD_TRADE_ACCEPTED"]
        total_blocked = counts["BAD_TRADE_AVOIDED"] + counts["GOOD_TRADE_MISSED"]

        false_approval_rate = (
            counts["BAD_TRADE_ACCEPTED"] / total_accepted if total_accepted > 0 else 0.0
        )
        false_rejection_rate = (
            counts["GOOD_TRADE_MISSED"] / total_blocked if total_blocked > 0 else 0.0
        )

        net_avoidance_ratio = (
            counts["BAD_TRADE_AVOIDED"] / max(1, counts["GOOD_TRADE_MISSED"])
        )

        return {
            "total_records": total,
            "disagreement_rate": round(disagreements / total, 4),
            "abstention_rate": round(abstentions / total, 4),
            "bad_trades_avoided_count": counts["BAD_TRADE_AVOIDED"],
            "good_trades_missed_count": counts["GOOD_TRADE_MISSED"],
            "bad_trades_accepted_count": counts["BAD_TRADE_ACCEPTED"],
            "good_trades_accepted_count": counts["GOOD_TRADE_ACCEPTED"],
            "false_approval_rate": round(false_approval_rate, 4),
            "false_rejection_rate": round(false_rejection_rate, 4),
            "net_avoidance_ratio": round(net_avoidance_ratio, 2),
            "decision_stability": round(1.0 - (disagreements / max(1, total * 2)), 4),
        }
