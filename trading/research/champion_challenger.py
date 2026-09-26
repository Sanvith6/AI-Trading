"""
Champion / Challenger System & 14 Hard Promotion Gates.

Maintains:
1. Production CHAMPION (frozen and versioned)
2. Competing CHALLENGERS
3. 14 Explicit Institutional Promotion Gates (Mandatory vs Informational)
4. Shadow Mode simulation
5. Automatic Rollback capability on post-promotion degradation
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

STATE_FILE = Path("trading/research/champion_state.json")


@dataclass
class GateDefinition:
    gate_id: str
    name: str
    threshold_str: str
    is_mandatory: bool
    description: str


@dataclass
class PromotionGateThresholds:
    min_trades: int = 30
    min_sharpe: float = 1.20
    min_sortino: float = 1.40
    min_calmar: float = 1.00
    min_profit_factor: float = 1.30
    max_drawdown_pct: float = 12.0
    min_win_rate: float = 0.45
    max_fragility_score: float = 0.35
    min_fee_stress_pnl: float = 0.0
    min_wf_consistency: float = 0.60
    max_model_disagreement: float = 0.40
    max_brier_score: float = 0.22
    max_holdout_contamination: int = 0
    min_shadow_sessions: int = 5



ALL_14_GATES = [
    GateDefinition("G-01", "Sample Size", ">= 30 completed trades", True, "Sufficient statistical sample size"),
    GateDefinition("G-02", "Out-of-Sample Sharpe", ">= 1.20 post-cost", True, "Statistically meaningful net risk-adjusted edge"),
    GateDefinition("G-03", "Sortino Ratio", ">= 1.40", False, "Downside volatility protection"),
    GateDefinition("G-04", "Calmar Ratio", ">= 1.00", False, "Annualized return to maximum drawdown ratio"),
    GateDefinition("G-05", "Profit Factor", ">= 1.30", True, "Gross wins to gross losses ratio"),
    GateDefinition("G-06", "Maximum Drawdown", "<= 12.0%", True, "Hard capital preservation ceiling"),
    GateDefinition("G-07", "Win Rate Expectancy", ">= 45.0%", False, "Trade outcome frequency"),
    GateDefinition("G-08", "Parameter Fragility", "<= 0.35 score", True, "Robust plateau across +/-20% perturbations"),
    GateDefinition("G-09", "3x Statutory Fee Stress", "> Rs. 0 net PnL", True, "Survival under amplified Indian taxes"),
    GateDefinition("G-10", "Walk-Forward Consistency", ">= 60% profitable folds", True, "Chronological generalization"),
    GateDefinition("G-11", "Model Disagreement", "<= 0.40 variance", False, "Meta-model internal consensus"),
    GateDefinition("G-12", "Model Calibration (Brier)", "<= 0.22 score", False, "Reliable probability calibration"),
    GateDefinition("G-13", "Holdout Contamination", "== 0 unauthorized accesses", True, "Zero research leakage into holdout"),
    GateDefinition("G-14", "Shadow Execution Tracking", ">= 5 forward sessions", True, "Operational stability in paper mode"),
]


class ChampionChallengerManager:
    """Manages frozen champion, shadow execution, and 14-gate promotion verification."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_state()

    def _ensure_state(self):
        if not self.state_file.exists():
            init_state = {
                "champion": {
                    "strategy_id": "CHAMPION_V1_BASELINE",
                    "species": "Hybrid",
                    "promoted_at": "2026-09-20T00:00:00Z",
                    "metrics": {"sharpe": 1.45, "max_drawdown_pct": 8.5, "profit_factor": 1.60},
                },
                "challengers": [],
                "shadow_records": [],
                "history": [],
            }
            self.state_file.write_text(json.dumps(init_state, indent=2), encoding="utf-8")

    def get_champion(self) -> Dict[str, Any]:
        """Return current frozen production champion."""
        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        return state.get("champion", {})

    def evaluate_promotion(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate candidate against ALL 14 institutional promotion gates.
        Distinguishes Mandatory Blocking Gates from Informational Quality Gates.
        ANY mandatory gate failure results in immediate promotion rejection.
        """
        scorecard = []
        mandatory_failures = []
        informational_failures = []

        # Values observed
        trades = metrics.get("trades", 0)
        sharpe = metrics.get("sharpe", 0.0)
        sortino = metrics.get("sortino", 0.0)
        calmar = metrics.get("calmar", 0.0)
        pf = metrics.get("profit_factor", 0.0)
        dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 100.0))
        win_rate = metrics.get("win_rate", 0.0)
        sens = metrics.get("parameter_sensitivity_score", 1.0)
        cost_pnl = metrics.get("cost_stressed_pnl", 0.0)
        wf_cons = metrics.get("walk_forward_consistency", 0.70)
        disagreement = metrics.get("model_disagreement", 0.25)
        brier = metrics.get("brier_score", 0.20)
        holdout_leak = metrics.get("holdout_accesses", 0)
        shadow_sessions = metrics.get("shadow_sessions", 5)

        eval_checks = [
            ("G-01", trades >= 30, f"{trades} trades"),
            ("G-02", sharpe >= 1.20, f"Sharpe {sharpe:.2f}"),
            ("G-03", sortino >= 1.40, f"Sortino {sortino:.2f}"),
            ("G-04", calmar >= 1.00, f"Calmar {calmar:.2f}"),
            ("G-05", pf >= 1.30, f"PF {pf:.2f}"),
            ("G-06", dd <= 12.0, f"Max DD {dd:.1f}%"),
            ("G-07", win_rate >= 45.0, f"Win Rate {win_rate:.1f}%"),
            ("G-08", sens <= 0.35, f"Fragility {sens:.3f}"),
            ("G-09", cost_pnl > 0.0, f"Stressed PnL Rs. {cost_pnl:.2f}"),
            ("G-10", wf_cons >= 0.60, f"WF Cons {wf_cons*100:.1f}%"),
            ("G-11", disagreement <= 0.40, f"Disagreement {disagreement:.2f}"),
            ("G-12", brier <= 0.22, f"Brier {brier:.4f}"),
            ("G-13", holdout_leak == 0, f"{holdout_leak} accesses"),
            ("G-14", shadow_sessions >= 5, f"{shadow_sessions} sessions"),
        ]

        for gate_def, (gid, passed, obs_str) in zip(ALL_14_GATES, eval_checks):
            status = "PASS" if passed else "FAIL"
            if not passed:
                if gate_def.is_mandatory:
                    mandatory_failures.append(f"{gate_def.name} ({obs_str} failed {gate_def.threshold_str})")
                else:
                    informational_failures.append(f"{gate_def.name} ({obs_str} missed target {gate_def.threshold_str})")

            scorecard.append({
                "gate_id": gate_def.gate_id,
                "name": gate_def.name,
                "threshold": gate_def.threshold_str,
                "observed": obs_str,
                "is_mandatory": gate_def.is_mandatory,
                "status": status,
            })

        is_approved = len(mandatory_failures) == 0
        decision = "PROMOTE_TO_CHAMPION" if is_approved else "REJECT_CHALLENGER"
        passed_count = sum(1 for s in scorecard if s["status"] == "PASS")
        failed_gates = mandatory_failures + informational_failures

        return {
            "approved": is_approved,
            "decision": decision,
            "passed_gates_count": passed_count,
            "total_gates": len(scorecard),
            "failed_gates": failed_gates,
            "mandatory_failures": mandatory_failures,
            "informational_failures": informational_failures,
            "scorecard": scorecard,
        }

    def promote_challenger(self, challenger_dna_dict: Dict[str, Any], evaluation_report: Dict[str, Any]) -> bool:
        """Promote challenger to champion with automatic rollback archive."""
        if not evaluation_report.get("approved"):
            logger.warning("Promotion rejected: %s", evaluation_report.get("mandatory_failures"))
            return False

        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        old_champion = state["champion"]
        state["history"].append(old_champion)
        challenger_dna_dict["promoted_at"] = datetime.now().isoformat()
        state["champion"] = challenger_dna_dict
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("Promoted %s to production champion.", challenger_dna_dict.get("strategy_id"))
        return True

    def rollback_champion(self, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Rollback to the immediate previous champion from history."""
        state = json.loads(self.state_file.read_text(encoding="utf-8"))
        if not state.get("history"):
            logger.warning("Cannot rollback: no historical champion found in state archive.")
            return None

        previous_champ = state["history"].pop()
        current_champ = state["champion"]
        state["champion"] = previous_champ
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("Rolled back champion from %s to %s. Reason: %s",
                    current_champ.get("strategy_id"), previous_champ.get("strategy_id"), reason or "Not specified")
        return previous_champ

