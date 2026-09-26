"""
Laya Forward Shadow Evaluation & 30-Session Paper Validation Tracker.

Enforces Directives 18, 19, and 21:
- Initial default: LAYA_MODE=shadow.
- Laya observes live/paper candidates in parallel while baseline paper trading continues 100% unchanged.
- Non-interfering: Records both BASELINE DECISION and LAYA DECISION.
- Freezes: model, task, thresholds, and configuration for the 30-session window.
- Logs GPU telemetry, latency metrics, and candidate evaluations to JSONL for auditability.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

from .decision_engine import CandidateEvaluation, laya_decision_engine
from .gpu_monitor import gpu_monitor
from .client import laya_client
from .config import laya_config

logger = logging.getLogger(__name__)


@dataclass
class ForwardShadowLogEntry:
    """Audit log entry capturing parallel baseline vs Laya observation."""
    timestamp: str
    session_id: str
    symbol: str
    baseline_decision: str
    baseline_score: float
    laya_tq_choice: Optional[str]
    laya_tq_prob: Optional[float]
    laya_ra_choice: Optional[str]
    laya_ra_prob: Optional[float]
    laya_regime_choice: Optional[str]
    laya_regime_prob: Optional[float]
    laya_policy_allowed: bool
    laya_size_multiplier: float
    is_disagreement: bool
    disagreement_type: Optional[str]
    gpu_vram_used_mb: float
    gpu_temp_c: float
    latency_ms: float
    state_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LayaForwardShadowTracker:
    """
    Manages non-interfering shadow tracking during 30-session paper validation.
    Guarantees baseline execution is never modified.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or (Path(__file__).parent.parent.parent / "data_lake" / "forward_paper")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "laya_shadow_log.jsonl"
        self._entries: List[ForwardShadowLogEntry] = []

    def record_shadow_observation(
        self,
        session_id: str,
        evaluation: CandidateEvaluation,
        baseline_action: str = "ENTER",
    ) -> ForwardShadowLogEntry:
        """
        Records parallel evaluation without executing or modifying baseline orders.
        """
        now_ts = datetime.utcnow().isoformat()
        gpu = gpu_monitor.check_telemetry()

        tq = evaluation.trade_quality_decision
        ra = evaluation.risk_action_decision
        reg = evaluation.regime_opinion_decision

        # Determine disagreement
        is_disagreement = False
        disagreement_type = None

        if baseline_action == "ENTER" and not evaluation.final_allowed:
            is_disagreement = True
            disagreement_type = "BASELINE_BUY_LAYA_BLOCK"
        elif baseline_action == "ENTER" and evaluation.effective_position_multiplier < 0.99:
            is_disagreement = True
            disagreement_type = "BASELINE_BUY_LAYA_REDUCE"

        latency = tq.latency_ms if tq else 0.0

        entry = ForwardShadowLogEntry(
            timestamp=now_ts,
            session_id=session_id,
            symbol=evaluation.symbol,
            baseline_decision=baseline_action,
            baseline_score=round(evaluation.baseline_score, 4),
            laya_tq_choice=tq.choice if tq else None,
            laya_tq_prob=tq.decision_probability if tq else None,
            laya_ra_choice=ra.choice if ra else None,
            laya_ra_prob=ra.decision_probability if ra else None,
            laya_regime_choice=reg.choice if reg else None,
            laya_regime_prob=reg.decision_probability if reg else None,
            laya_policy_allowed=evaluation.final_allowed,
            laya_size_multiplier=evaluation.effective_position_multiplier,
            is_disagreement=is_disagreement,
            disagreement_type=disagreement_type,
            gpu_vram_used_mb=gpu.vram_used_mb,
            gpu_temp_c=gpu.temperature_c,
            latency_ms=latency,
            state_hash=evaluation.market_state.compute_state_hash(),
        )

        self._entries.append(entry)

        # Append to JSONL log
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to append shadow log: {e}")

        return entry

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates aggregated forward validation metrics."""
        if not self._entries:
            return {
                "total_observations": 0,
                "disagreement_rate": 0.0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "gpu_vram_avg_mb": 0.0,
            }

        total = len(self._entries)
        disagreements = sum(1 for e in self._entries if e.is_disagreement)
        lats = [e.latency_ms for e in self._entries if e.latency_ms > 0]
        vrams = [e.gpu_vram_used_mb for e in self._entries if e.gpu_vram_used_mb > 0]

        import numpy as np
        return {
            "total_observations": total,
            "disagreements": disagreements,
            "disagreement_rate": round(disagreements / total, 4),
            "latency_p50": round(float(np.percentile(lats, 50)), 2) if lats else 0.0,
            "latency_p95": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
            "gpu_vram_avg_mb": round(float(np.mean(vrams)), 1) if vrams else 0.0,
        }


# Global singleton shadow tracker
laya_forward_shadow = LayaForwardShadowTracker()
